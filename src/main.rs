use learst::graph::Graph;
use learst::tensor::Tensor;
use learst::*;
use rand::prelude::*;

fn main() {
    let mut rng = thread_rng();
    let mut g = Graph::new();

    let samples = Tensor::fill_by(&[20, 2], |pos| (pos[0] + pos[1]) as f32);

    let t0 = g.alloc(samples);
    let t1 = g.alloc(Tensor::rand(&mut rng, &[2, 1]));
    let neg_expected = g.alloc(Tensor::fill_by(&[20, 1], |pos| {
        -((pos[0] * 10 + (pos[0] + 1) * 19) as f32)
    }));
    let t2 = g.call(MatMul::new(), &[t0, t1]);
    let t3 = g.call(Add::new(), &[t2, neg_expected]);
    let t4 = g.call(Pow::new(2.), &[t3]);
    let t5 = g.call(Mul::new(0.05), &[t4]);
    println!("{:?}", g.get(t5));
    for i in 0..100 {
        g.forward();
        g.zero_grad();
        g.backward_all(t4);
        println!("{:?}", g.grads.get(&t1).unwrap());
        let new_t1 = g.get_mut(t1);
        *new_t1 = apply(
            AddOp {},
            new_t1,
            &apply(MulOp {}, g.grads.get(&t1).unwrap(), &Tensor::scalar(-0.001)),
        );
    }
    println!("{:?}", g.get(t1));
}
