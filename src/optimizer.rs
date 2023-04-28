use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&self, params: &mut [Tensor], grads: &[Tensor]);
}

pub struct NaiveOptimizer {
    learning_rate: f32,
}

impl Optimizer for NaiveOptimizer {
    fn step(&self, params: &mut [Tensor], grads: &[Tensor]) {}
}
