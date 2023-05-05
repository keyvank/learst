use learst::funcs::*;
use learst::graph::{Graph, TensorId};
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{Tensor, TensorMutOps, TensorOps};
use learst::*;
use rand::prelude::*;
use std::fs::File;
use std::io::prelude::*;

fn mse(g: &mut Graph, out: TensorId, target: TensorId) -> TensorId {
    let diff = g.call(Sub::new(), &[out, target]);
    let diff_sqr = g.call(Square::new(), &[diff]);
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

    let xs = Tensor::<f32>::rand(&mut rng, &[10, 2]);
    let ys = xs.map(1, |l| Tensor::<u32>::scalar(0));
    //print

    let samples = g.alloc(xs);

    let lin1 = g.alloc(Tensor::<f32>::rand(&mut rng, &[2, 2]));
    let lin1_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[2]));
    let lin2 = g.alloc(Tensor::<f32>::rand(&mut rng, &[2, 2]));
    let lin2_bias = g.alloc(Tensor::<f32>::rand(&mut rng, &[2]));

    let out1 = g.call(MatMul::new(), &[samples, lin1]);
    let out1_bias = g.call(Add::new(), &[out1, lin1_bias]);
    let out1_bias_sigm = g.call(Sigmoid::new(), &[out1_bias]);
    let out2 = g.call(MatMul::new(), &[out1_bias_sigm, lin2]);
    let out2_bias = g.call(Add::new(), &[out2, lin2_bias]);
    let out = g.call(Softmax::new(), &[out2_bias]);

    let error = g.call(CrossEntropy::new(2, ys), &[out]);
    //let squared_error = g.call(Square::new(), &[error]);
    let mean_error = g.call(Mean::new(), &[error]);

    let mut opt = NaiveOptimizer::new(0.001);
    loop {
        g.forward();
        g.zero_grad();
        g.backward_all(mean_error);
        println!("{:?}", g.get(mean_error));
        g.optimize(
            &mut opt,
            &[lin1, lin2, lin1_bias, lin2_bias].into_iter().collect(),
        );
    }
}
