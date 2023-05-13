use super::{Function, Tensor, TensorOps};

pub struct Dropout {
    rate: f32,
}
impl Dropout {
    pub fn new(rate: f32) -> Box<dyn Function> {
        Box::new(Self { rate })
    }
}

impl Function for Dropout {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        todo!()
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        todo!()
    }
}
