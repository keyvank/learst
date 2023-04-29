use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, params: Vec<&mut Tensor>, grads: Vec<&Tensor>);
}

pub struct NaiveOptimizer {
    learning_rate: f32,
}

impl NaiveOptimizer {
    pub fn new(learning_rate: f32) -> Box<dyn Optimizer> {
        Box::new(Self { learning_rate })
    }
}

impl Optimizer for NaiveOptimizer {
    fn step(&self, params: Vec<&mut Tensor>, grads: Vec<&Tensor>) {
        for (param, grad) in params.into_iter().zip(grads.into_iter()) {
            *param = &*param + &(grad * &Tensor::scalar(-self.learning_rate));
        }
    }
}
