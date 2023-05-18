use super::{Function, Tensor, TensorOps};

pub struct Flatten {}
impl Flatten {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}

impl Function for Flatten {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].keep_left(1).into()
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
