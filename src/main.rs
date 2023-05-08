use learst::funcs::*;
use learst::graph::Graph;
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{Tensor, TensorMutOps, TensorOps};
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::PathBuf;

fn mnist_images() -> std::io::Result<(Tensor<f32>, Tensor<u32>)> {
    let mut img_file = File::open("train-images.idx3-ubyte")?;
    let mut img_bytes = Vec::new();
    img_file.read_to_end(&mut img_bytes)?;
    let mut label_file = File::open("train-labels.idx1-ubyte")?;
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
    let xs = Tensor::<f32>::raw(&[4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let ys = Tensor::<u32>::raw(&[4], vec![0, 1, 0, 1]);
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

    let (xs, ys) = mnist_images().unwrap();

    let samples = g.alloc_input(&[784]);

    let lin1 = g.alloc_param(&[784, 200]);
    let lin1_bias = g.alloc_param(&[200]);
    let lin2 = g.alloc_param(&[200, 100]);
    let lin2_bias = g.alloc_param(&[100]);
    let lin3 = g.alloc_param(&[100, 50]);
    let lin3_bias = g.alloc_param(&[50]);
    let lin4 = g.alloc_param(&[50, 10]);
    let lin4_bias = g.alloc_param(&[10]);

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

    let mut opt = NaiveOptimizer::new(0.0001);
    for epoch in 0..60 {
        let handdrawn = read_ppm(&"Untitled.ppm".into()).unwrap();
        g.load(samples, &handdrawn);
        g.forward();
        println!("{:?}", g.get(out4_bias_sigm));

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

fn main() {
    let mut g = Graph::new();

    let inp = g.alloc_input(&[28, 28, 3]);
    let lin1 = g.alloc_param(&[27, 5]);
    let out = g.call(Convolution::new(3, 0, 3, 5), &[inp, lin1]);
    println!("{:?}", g.get(out).shape());
}
