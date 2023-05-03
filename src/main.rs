use learst::graph::{Graph, TensorId};
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{Tensor, TensorMutOps, TensorOps};
use learst::*;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

fn mse(g: &mut Graph, out: TensorId, target: TensorId) -> TensorId {
    let diff = g.call(Sub::new(), &[out, target]);
    let diff_sqr = g.call(Pow::new(2.), &[diff]);
    diff_sqr
}

fn mnist_images() -> std::io::Result<(Tensor<f32>, Tensor<u32>)> {
    let mut img_file = File::open("train-images-idx3-ubyte")?;
    let mut img_bytes = Vec::new();
    img_file.read_to_end(&mut img_bytes);
    let mut label_file = File::open("train-labels-idx1-ubyte")?;
    let mut label_bytes = Vec::new();
    label_file.read_to_end(&mut label_bytes);

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

fn main() {
    let mut rng = thread_rng();
    let mut g = Graph::new();

    let samples = Tensor::fill_by(&[10, 100], |pos| (pos[0] + pos[1]) as f32);
    let outputs = Tensor::fill_by(&[10], |_| 3);

    let inp = g.alloc(samples);
    let lin1 = g.alloc(Tensor::<f32>::rand(&mut rng, &[100, 50]));
    let lin2 = g.alloc(Tensor::<f32>::rand(&mut rng, &[50, 20]));
    let lin3 = g.alloc(Tensor::<f32>::rand(&mut rng, &[20, 10]));

    let post_lin1 = g.call(MatMul::new(), &[inp, lin1]);
    let post_sigm1 = g.call(Sigmoid::new(), &[post_lin1]);

    let post_lin2 = g.call(MatMul::new(), &[post_sigm1, lin2]);
    let post_sigm2 = g.call(Sigmoid::new(), &[post_lin2]);

    let post_lin3 = g.call(MatMul::new(), &[post_sigm2, lin3]);
    let post_sigm3 = g.call(Sigmoid::new(), &[post_lin3]);

    let soft = g.call(Softmax::new(), &[post_sigm3]);
    let loss = g.call(CrossEntropy::new(10, outputs), &[soft]);

    let mut opt = NaiveOptimizer::new(0.1);
    for _ in 0..1000 {
        g.forward();
        g.zero_grad();
        g.backward_all(loss);
        println!("{:?}", g.get(loss));
        g.optimize(&mut opt, &[lin1, lin2, lin3].into_iter().collect());
    }
}
