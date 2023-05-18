use super::{Function, Tensor, TensorOps};

pub struct Mean;
impl Mean {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Mean {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        Tensor::scalar(inps[0].blob().iter().sum::<f32>() / inps[0].blob().len() as f32)
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        vec![out_grad * &Tensor::<f32>::scalar(1. / inps[0].blob().len() as f32)]
    }
}
