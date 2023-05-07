use learst::funcs::*;
use learst::graph::Graph;
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{Tensor, TensorMutOps};
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

fn mnist_images(limit: Option<usize>) -> std::io::Result<(Tensor<f32>, Tensor<u32>)> {
    let mut img_file = File::open("train-images.idx3-ubyte")?;
    let mut img_bytes = Vec::new();
    img_file.read_to_end(&mut img_bytes)?;
    let mut label_file = File::open("train-labels.idx1-ubyte")?;
    let mut label_bytes = Vec::new();
    label_file.read_to_end(&mut label_bytes)?;

    let num_img_samples = u32::from_be_bytes(img_bytes[4..8].try_into().unwrap()) as usize;
    let num_label_samples = u32::from_be_bytes(label_bytes[4..8].try_into().unwrap()) as usize;
    assert_eq!(num_img_samples, num_label_samples);

    let final_num_samples = limit.unwrap_or(num_img_samples);

    let mut images = Tensor::<f32>::zeros(&[final_num_samples, 784]);
    let mut labels = Tensor::<u32>::zeros(&[final_num_samples]);
    for (i, (img, label)) in img_bytes[8..]
        .chunks(784)
        .zip(label_bytes[8..].iter())
        .take(final_num_samples)
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

fn main() {
    let mut rng = thread_rng();
    let mut g = Graph::new();

    let (xs, ys) = mnist_images(Some(1000)).unwrap();

    let samples = g.alloc(xs);

    let lin1 = g.alloc(Tensor::<f32>::rand(&mut rng, &[784, 200]));
    let lin1_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[200]));
    let lin2 = g.alloc(Tensor::<f32>::rand(&mut rng, &[200, 100]));
    let lin2_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[100]));
    let lin3 = g.alloc(Tensor::<f32>::rand(&mut rng, &[100, 50]));
    let lin3_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[50]));
    let lin4 = g.alloc(Tensor::<f32>::rand(&mut rng, &[50, 10]));
    let lin4_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[10]));

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

    let error = g.call(CrossEntropy::new(10, ys), &[out4_bias_sigm]);
    let mean_error = g.call(Mean::new(), &[error]);

    let mut opt = NaiveOptimizer::new(0.1);
    loop {
        g.forward();
        g.zero_grad();
        g.backward_all(mean_error);
        println!("{:?}", g.get(mean_error));
        g.optimize(
            &mut opt,
            &[
                lin1, lin2, lin3, lin4, lin1_bias, lin2_bias, lin3_bias, lin4_bias,
            ]
            .into_iter()
            .collect(),
        );
    }
}
