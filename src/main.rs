use learst::funcs::*;
use learst::graph::{Graph, TensorId};
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{combine_map, shuffle_batch, Tensor, TensorElement, TensorMutOps, TensorOps};
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

fn mnist_images(
    img_path: &PathBuf,
    label_path: &PathBuf,
) -> std::io::Result<(Tensor<f32>, Tensor<u32>)> {
    let mut img_file = File::open(img_path)?;
    let mut img_bytes = Vec::new();
    img_file.read_to_end(&mut img_bytes)?;
    let mut label_file = File::open(label_path)?;
    let mut label_bytes = Vec::new();
    label_file.read_to_end(&mut label_bytes)?;

    let num_img_samples = u32::from_be_bytes(img_bytes[4..8].try_into().unwrap()) as usize;
    let num_label_samples = u32::from_be_bytes(label_bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(num_img_samples, num_label_samples);

    let mut images = Tensor::<f32>::zeros(&[num_img_samples, 784]);
    let mut labels = Tensor::<u32>::zeros(&[num_img_samples]);
    for (i, (img, label)) in img_bytes[8..]
        .chunks(784)
        .zip(label_bytes[8..].iter())
        .enumerate()
    {
        images
            .get_mut(i)
            .blob_mut()
            .clone_from_slice(&img.iter().map(|b| *b as f32 / 255.).collect::<Vec<_>>());
        labels.get_mut(i).set(Tensor::scalar(*label as u32));
    }

    Ok((images, labels))
}

fn xor_dataset() -> (Tensor<f32>, Tensor<u32>) {
    let xs = Tensor::<f32>::raw(
        &[16, 4],
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ],
    );
    let ys = Tensor::<u32>::raw(&[16], vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    (xs, ys)
}

fn read_ppm(path: &PathBuf) -> std::io::Result<Tensor<f32>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let parts = bytes.split(|b| *b == b'\n').collect::<Vec<_>>();
    let data = parts
        .last()
        .unwrap()
        .chunks(3)
        .map(|ch| (1. - ch.iter().map(|b| *b as f32 / 255. / 3.).sum::<f32>()))
        .collect::<Vec<_>>();
    Ok(Tensor::raw(&[1, 784], data))
}

fn nn_mnist() {
    let mut g = Graph::new();

    let (xs, ys) = mnist_images(
        &"train-images.idx3-ubyte".into(),
        &"train-labels.idx1-ubyte".into(),
    )
    .unwrap();
    let (xs_test, ys_test) = mnist_images(
        &"t10k-images.idx3-ubyte".into(),
        &"t10k-labels.idx1-ubyte".into(),
    )
    .unwrap();

    let samples = g.alloc_input(&[784]);
    let mut rng = rand::thread_rng();

    let lin1 = g.alloc_param(&mut rng, &[784, 200]);
    let lin1_bias = g.alloc_param(&mut rng, &[200]);
    let lin2 = g.alloc_param(&mut rng, &[200, 100]);
    let lin2_bias = g.alloc_param(&mut rng, &[100]);
    let lin3 = g.alloc_param(&mut rng, &[100, 50]);
    let lin3_bias = g.alloc_param(&mut rng, &[50]);
    let lin4 = g.alloc_param(&mut rng, &[50, 10]);
    let lin4_bias = g.alloc_param(&mut rng, &[10]);

    let out1 = g.call(MatMul::new(), &[samples, lin1]);
    let out1_bias = g.call(Add::new(), &[out1, lin1_bias]);
    let out1_bias_sigm = g.call(Sigmoid::new(), &[out1_bias]);
    let out2 = g.call(MatMul::new(), &[out1_bias_sigm, lin2]);
    let out2_bias = g.call(Add::new(), &[out2, lin2_bias]);
    let out2_bias_sigm = g.call(Sigmoid::new(), &[out2_bias]);
    let out3 = g.call(MatMul::new(), &[out2_bias_sigm, lin3]);
    let out3_bias = g.call(Add::new(), &[out3, lin3_bias]);
    let out3_bias_sigm = g.call(Sigmoid::new(), &[out3_bias]);
    let out4 = g.call(MatMul::new(), &[out3_bias_sigm, lin4]);
    let out4_bias = g.call(Add::new(), &[out4, lin4_bias]);
    let out4_bias_sigm = g.call(Softmax::new(), &[out4_bias]);

    let params = vec![
        lin1, lin2, lin3, lin4, lin1_bias, lin2_bias, lin3_bias, lin4_bias,
    ];

    for p in params.iter() {
        let mut tensor_file = File::open(format!("tensor_{}.dat", p)).unwrap();
        let mut bytes = Vec::new();
        tensor_file.read_to_end(&mut bytes).unwrap();
        let t: Tensor<f32> = bincode::deserialize(&bytes).unwrap();
        g.load(*p, &t);
    }

    let mut opt = NaiveOptimizer::new(0.001);
    for epoch in 0..60 {
        g.load(samples, &xs_test);
        g.forward();
        let predictions = g.get(out4_bias_sigm).argmax();
        println!(
            "Accuracy: {}",
            predictions
                .equals(&ys_test)
                .blob()
                .iter()
                .map(|v| v.as_f32())
                .sum::<f32>()
                / predictions.size() as f32
        );

        println!(
            "Train on image {} to {}...",
            epoch * 1000,
            epoch * 1000 + 1000
        );
        let xs: Tensor<f32> = xs.get_slice(epoch * 1000, 1000).into();
        let ys: Tensor<u32> = ys.get_slice(epoch * 1000, 1000).into();

        g.load(samples, &xs);
        g.forward();
        g.zero_grad();
        let err = g.backward_all(out4_bias_sigm, CrossEntropy::new(10, ys.clone()));
        println!("Loss: {}", err.mean());

        for p in params.iter() {
            let data = bincode::serialize(g.get(*p)).unwrap();
            fs::write(format!("tensor_{}.dat", p), &data).expect("Unable to write file");
        }

        g.optimize(&mut opt, &params.iter().cloned().collect());
    }
}

fn convo() {
    let mut rng = rand::thread_rng();
    let mut g = Graph::new();

    let (mut xs, mut ys) = mnist_images(
        &"train-images.idx3-ubyte".into(),
        &"train-labels.idx1-ubyte".into(),
    )
    .unwrap();
    let (xs_test, ys_test) = mnist_images(
        &"t10k-images.idx3-ubyte".into(),
        &"t10k-labels.idx1-ubyte".into(),
    )
    .unwrap();
    let xs_test: Tensor<f32> = xs_test.reshape(&[10000, 1, 28, 28]).into();
    let ys_test: Tensor<u32> = ys_test.reshape(&[10000]).into();
    let batch_size = 1000;

    let inp = g.alloc_input(&[1, 28, 28]);
    let conv1 = g.alloc_param(&mut rng, &[5, 1, 3, 3]);
    let conv2 = g.alloc_param(&mut rng, &[10, 5, 3, 3]);
    let lin = g.alloc_param(&mut rng, &[490, 200]);
    let lin_bias = g.alloc_param(&mut rng, &[200]);
    let lin2 = g.alloc_param(&mut rng, &[200, 10]);
    let lin2_bias = g.alloc_param(&mut rng, &[10]);
    let out1 = g.call(Convolution::new(3, 1, 1, 5), &[inp, conv1]);
    let sigm1 = g.call(Relu::new(), &[out1]);
    let max1 = g.call(MaxPool::new(2), &[sigm1]);
    let norm1 = g.call(LayerNorm::new(2), &[max1]);
    let out2 = g.call(Convolution::new(3, 1, 5, 10), &[norm1, conv2]);
    let sigm2 = g.call(Relu::new(), &[out2]);
    let max2 = g.call(MaxPool::new(2), &[sigm2]);
    let norm2 = g.call(LayerNorm::new(2), &[max2]);
    let flat = g.call(Flatten::new(2), &[norm2]);
    let out = g.call(MatMul::new(), &[flat, lin]);
    let out_bias = g.call(Add::new(), &[out, lin_bias]);
    let out_bias_relu = g.call(Relu::new(), &[out_bias]);
    let out_bias_relu_norm = g.call(LayerNorm::new(2), &[out_bias_relu]);
    let out2 = g.call(MatMul::new(), &[out_bias_relu_norm, lin2]);
    let out2_bias = g.call(Add::new(), &[out2, lin2_bias]);
    let params = vec![conv1, conv2, lin, lin_bias, lin2, lin2_bias];
    for p in params.iter() {
        let mut tensor_file = File::open(format!("tensor_{}.dat", p)).unwrap();
        let mut bytes = Vec::new();
        tensor_file.read_to_end(&mut bytes).unwrap();
        let t: Tensor<f32> = bincode::deserialize(&bytes).unwrap();
        g.load(*p, &t);
    }
    let mut opt = NaiveOptimizer::new(0.000002);
    println!("=============");
    loop {
        (xs, ys) = shuffle_batch(&mut rng, &xs, &ys);
        for epoch in 0..60 {
            g.load(inp, &xs_test);
            g.forward();
            let predictions = g.get(out2_bias).argmax();
            println!(
                "Accuracy: {}",
                predictions
                    .equals(&ys_test)
                    .blob()
                    .iter()
                    .map(|v| v.as_f32())
                    .sum::<f32>()
                    / predictions.size() as f32
            );

            let xs: Tensor<f32> = xs
                .get_slice(epoch * batch_size, batch_size)
                .reshape(&[batch_size, 1, 28, 28])
                .into();
            let ys: Tensor<u32> = ys
                .get_slice(epoch * batch_size, batch_size)
                .reshape(&[batch_size])
                .into();

            g.load(inp, &xs);
            g.forward();
            g.zero_grad();
            let err = g.backward_all(out2_bias, CrossEntropy::new(10, ys.clone()));
            println!("Loss: {}", err.mean());
            for p in params.iter() {
                let data = bincode::serialize(g.get(*p)).unwrap();
                fs::write(format!("tensor_{}.dat", p), &data).expect("Unable to write file");
            }

            g.optimize(&mut opt, &params.iter().cloned().collect());
        }
    }
}

fn xor() {
    let mut rng = rand::thread_rng();
    let mut g = Graph::new();

    let (xs, ys) = xor_dataset();

    let inp = g.alloc_input(&[4]);
    let lin1 = g.alloc_param(&mut rng, &[4, 1]);
    let lin1_bias = g.alloc_param(&mut rng, &[1]);
    let lin2 = g.alloc_param(&mut rng, &[1, 4]);
    let lin2_bias = g.alloc_param(&mut rng, &[4]);
    let out1 = g.call(MatMul::new(), &[inp, lin1]);
    let out1_bias = g.call(Add::new(), &[out1, lin1_bias]);
    let out1_bias_sigm = g.call(Sigmoid::new(), &[out1_bias]);
    let out2 = g.call(MatMul::new(), &[out1_bias_sigm, lin2]);
    let out2_bias = g.call(Add::new(), &[out2, lin2_bias]);
    let mut opt = NaiveOptimizer::new(0.1);
    let params = vec![lin1, lin2, lin1_bias, lin2_bias];

    loop {
        g.load(inp, &xs);
        g.forward();
        g.zero_grad();
        let err = g.backward_all(out2_bias, CrossEntropy::new(4, ys.clone()));
        println!("Loss: {}", err.mean());
        g.optimize(&mut opt, &params.iter().cloned().collect());
    }
}
use rand::Rng;
use std::collections::{HashMap, HashSet};
fn sample_dataset<R: Rng>(
    dataset: &[u32],
    batch_size: usize,
    context_size: usize,
    rng: &mut R,
) -> (Tensor<u32>, Tensor<u32>) {
    let mut xs: Vec<u32> = Vec::new();
    let mut ys: Vec<u32> = Vec::new();
    for _i in 0..batch_size {
        let start: usize = rng.gen_range(0..dataset.len());
        let all = dataset
            .iter()
            .cycle()
            .skip(start)
            .take(context_size + 1)
            .cloned()
            .collect::<Vec<_>>();
        xs.extend(&all[0..context_size]);
        ys.extend(&all[1..context_size + 1]);
    }

    (
        Tensor::raw(&[batch_size, context_size], xs),
        Tensor::raw(&[batch_size, context_size], ys),
    )
}

fn embed(s: &Tensor<u32>, embedding: &Tensor<f32>) -> Tensor<f32> {
    s.map(0, |s| embedding.get(s.scalar() as usize).into())
}

fn unembed(s: &Tensor<u32>, s_result: &Tensor<f32>, embedding: &mut Tensor<f32>) -> Tensor<f32> {
    let degree = s_result.shape()[s_result.dim() - 1];
    for (ch, embed) in s.blob().iter().zip(s_result.keep_right(1).inners().iter()) {
        let mut t = embedding.get_mut(*ch as usize);
        t.set(embed.clone());
    }
    Tensor::scalar(0.)
}

fn gpt() {
    let mut rng = rand::thread_rng();

    let mut g = Graph::new();
    let dataset_char =
        fs::read_to_string("dataset.txt").expect("Should have been able to read the file");
    let mut chars = dataset_char
        .chars()
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    chars.sort();
    let int_to_ch = chars
        .iter()
        .enumerate()
        .map(|(i, ch)| (i as u32, *ch))
        .collect::<HashMap<u32, char>>();
    let ch_to_int = chars
        .iter()
        .enumerate()
        .map(|(i, ch)| (*ch, i as u32))
        .collect::<HashMap<char, u32>>();
    let dataset = dataset_char
        .chars()
        .map(|ch| ch_to_int.get(&ch).unwrap().clone())
        .collect::<Vec<_>>();

    let batch_size = 3;
    let num_tokens = 64;
    let vocab_size = chars.len();
    let embedding_degree = 64;

    let num_attentions = 4;
    let num_heads = 4;
    let head_size = 16;
    let head_size_sqrt_inv = 0.25;

    let mut embedding = Tensor::<f32>::rand(&mut rng, &[vocab_size, embedding_degree]);
    let mut pos_embedding = Tensor::<f32>::rand(&mut rng, &[vocab_size, embedding_degree]);

    let char_inp = g.alloc_input(&[num_tokens, embedding_degree]);
    let pos_inp = g.alloc_input(&[num_tokens, embedding_degree]);
    let inp = g.call(Add::new(), &[char_inp, pos_inp]);

    let mut params: Vec<TensorId> = Vec::new();

    params.extend(&[inp]);

    let mut curr_inp = inp;
    for _ in 0..num_attentions {
        let norm_inp = g.call(LayerNorm::new(1), &[curr_inp]);
        let mut heads = Vec::new();
        for _i in 0..num_heads {
            let k_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            let q_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            let v_params = g.alloc_param(&mut rng, &[embedding_degree, head_size]);
            params.extend(&[k_params, q_params, v_params]);
            let k = g.call(MatMul::new(), &[norm_inp, k_params]);
            let q = g.call(MatMul::new(), &[norm_inp, q_params]);
            let v = g.call(MatMul::new(), &[norm_inp, v_params]);
            let q_t = g.call(Transpose::new(), &[q]);
            let kq = g.call(MatMul::new(), &[k, q_t]);
            let kq_coeff = g.call(Coeff::new(head_size_sqrt_inv), &[kq]);

            let masked_kq = g.call(
                Mask::new(!&Tensor::<bool>::tril(num_tokens), f32::NEG_INFINITY),
                &[kq_coeff],
            );
            let soft_masked_kq = g.call(Softmax::new(), &[masked_kq]);
            let dropped_soft_masked_kq = g.call(Dropout::new(0.2), &[soft_masked_kq]);
            let atten = g.call(MatMul::new(), &[dropped_soft_masked_kq, v]);
            heads.push(atten);
        }
        let cat = g.call(Cat::new(), &heads);

        let proj_params = g.alloc_param(&mut rng, &[num_heads * head_size, embedding_degree]);
        let proj_bias_params = g.alloc_param(&mut rng, &[embedding_degree]);
        let proj_cat = g.call(MatMul::new(), &[cat, proj_params]);

        let proj_cat_bias = g.call(Add::new(), &[proj_cat, proj_bias_params]);
        let dropped_proj_cat_bias = g.call(Dropout::new(0.2), &[proj_cat_bias]);

        let add_atten = g.call(Add::new(), &[norm_inp, dropped_proj_cat_bias]);
        let add_atten_norm = g.call(LayerNorm::new(1), &[add_atten]);

        let lin1_params = g.alloc_param(&mut rng, &[embedding_degree, 4 * embedding_degree]);
        let bias1_params = g.alloc_param(&mut rng, &[4 * embedding_degree]);
        let lin2_params = g.alloc_param(&mut rng, &[4 * embedding_degree, embedding_degree]);
        let bias2_params = g.alloc_param(&mut rng, &[embedding_degree]);

        let lin1_result = g.call(MatMul::new(), &[add_atten_norm, lin1_params]);
        let lin1_bias_result = g.call(Add::new(), &[lin1_result, bias1_params]);
        let lin1_act = g.call(Relu::new(), &[lin1_bias_result]);
        let lin2_result = g.call(MatMul::new(), &[lin1_act, lin2_params]);
        let lin2_bias_result = g.call(Add::new(), &[lin2_result, bias2_params]);

        params.extend(&[
            proj_params,
            proj_bias_params,
            lin1_params,
            bias1_params,
            lin2_params,
            bias2_params,
        ]);

        curr_inp = g.call(Add::new(), &[add_atten_norm, lin2_bias_result]);
    }

    let norm_out = g.call(LayerNorm::new(1), &[curr_inp]);
    let to_vocab = g.alloc_param(&mut rng, &[embedding_degree, vocab_size]);
    let to_vocab_bias = g.alloc_param(&mut rng, &[vocab_size]);
    let result_lin = g.call(MatMul::new(), &[norm_out, to_vocab]);
    let result = g.call(Add::new(), &[result_lin, to_vocab_bias]);
    params.extend(&[to_vocab, to_vocab_bias]);

    {
        for p in params.iter() {
            let mut tensor_file = File::open(format!("tensor_{}.dat", p)).unwrap();
            let mut bytes = Vec::new();
            tensor_file.read_to_end(&mut bytes).unwrap();
            let t: Tensor<f32> = bincode::deserialize(&bytes).unwrap();
            g.load(*p, &t);
        }
        let mut embed_data = File::open("embedding.dat").unwrap();
        let mut bytes = Vec::new();
        embed_data.read_to_end(&mut bytes).unwrap();
        embedding = bincode::deserialize(&bytes).unwrap();

        let mut pos_embed_data = File::open("pos_embedding.dat").unwrap();
        let mut bytes = Vec::new();
        pos_embed_data.read_to_end(&mut bytes).unwrap();
        pos_embedding = bincode::deserialize(&bytes).unwrap();
    }

    let mut opt = NaiveOptimizer::new(0.0001);
    loop {
        let poses = Tensor::raw(
            &[batch_size, num_tokens],
            (0..num_tokens as u32)
                .cycle()
                .take(num_tokens * batch_size)
                .collect(),
        );
        let (xs, ys) = sample_dataset(&dataset, batch_size, num_tokens, &mut rng);
        g.load(char_inp, &embed(&xs, &embedding));
        g.load(pos_inp, &embed(&poses, &pos_embedding));
        g.forward();
        g.zero_grad();
        let err = g.backward_all(result, CrossEntropy::new(vocab_size as u32, ys.clone()));
        println!("Loss: {}", err.mean());
        if err.mean().is_nan() {
            break;
        }
        {
            for p in params.iter() {
                let data = bincode::serialize(g.get(*p)).unwrap();
                fs::write(format!("tensor_{}.dat", p), &data).expect("Unable to write file");
            }
            let embed_data = bincode::serialize(&embedding).unwrap();
            fs::write("embedding.dat", &embed_data).expect("Unable to write file");
            let pos_embed_data = bincode::serialize(&pos_embedding).unwrap();
            fs::write("pos_embedding.dat", &pos_embed_data).expect("Unable to write file");
        }
        g.optimize(&mut opt, &params.iter().cloned().collect());
        unembed(&xs, g.get(char_inp), &mut embedding);
        unembed(&poses, g.get(pos_inp), &mut pos_embedding);
        /*{
            let mut cnt = 1;
            let mut context = vec![0; num_tokens];
            for _ in 0..30 {
                g.load(
                    inp,
                    &embed(&Tensor::raw(&[1, num_tokens], context.clone()), &embedding),
                );
                g.forward();
                let next_ch = g.get(result).argmax().blob()[cnt - 1];
                println!("{:?}", context.iter().map(|i|int_to_ch.get(i).unwrap()).collect::<Vec<_>>());
                context[cnt] = next_ch;
                cnt += 1;
            }
            println!();
        }*/
    }
}

fn main() {
    gpt();
}
