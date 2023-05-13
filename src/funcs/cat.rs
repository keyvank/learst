use super::{Function, Tensor, TensorOps};

pub struct Cat {}
impl Cat {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Cat {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        Tensor::cat(inps)
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        Tensor::split(out_grad, inps.len())
    }
}
