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

    let samples = g.alloc(Tensor::fill_by(&[10, 2], |pos| {
        (pos[0] * 2 + pos[1]) as f32
    }));
    let expected = g.alloc(Tensor::fill_by(&[10, 2], |pos| {
        let v = (pos[0] * 2 + pos[1]) as f32;
        if pos[1] == 0 {
            v * 3.
        } else {
            v * 4.
        }
    }));

    let lin1 = g.alloc(Tensor::<f32>::rand(&mut rng, &[2, 2]));

    let out = g.call(MatMul::new(), &[samples, lin1]);

    let loss_sqrt = g.call(Sub::new(), &[out, expected]);
    let loss = g.call(Square::new(), &[loss_sqrt]);

    let mut opt = NaiveOptimizer::new(0.0002);
    for _ in 0..10000 {
        g.forward();
        g.zero_grad();
        g.backward_all(loss);
        println!("{:?}", g.get(lin1));
        g.optimize(&mut opt, &[lin1].into_iter().collect());
    }
}
