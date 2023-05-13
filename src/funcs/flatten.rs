use super::{Function, Tensor, TensorOps};

pub struct Flatten {}
impl Flatten {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Flatten {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].reshape(&[inps[0].len(), 0]).into()
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        vec![out_grad.reshape(inps[0].shape()).into()]
    }
}
