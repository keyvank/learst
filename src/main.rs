use learst::graph::Graph;
use learst::optimizer::NaiveOptimizer;
use learst::tensor::{Tensor, TensorOps};
use learst::*;
use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    let mut g = Graph::new();

    let samples = Tensor::fill_by(&[20, 100], |pos| (pos[0] + pos[1]) as f32);

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

    println!("{:?}", g.get(soft));

    /*let t3 = g.call(Add::new(), &[t2, neg_expected]);
    let t4 = g.call(Pow::new(2.), &[t3]);
    let t5 = g.call(Mul::new(0.05), &[t4]);
    let mut opt = NaiveOptimizer::new(0.0001);
    for _ in 0..1000 {
        g.forward();
        g.zero_grad();
        g.backward_all(t5);
        g.optimize(&mut opt, &[t1].into_iter().collect());
    }
    println!("{:?}", g.get(t1));*/
}
