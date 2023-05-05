use super::{Function, Tensor, TensorOps};

pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 2);
        inps[0] ^ inps[1]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![
            out_grad ^ &inps[1].transpose(),
            &inps[0].transpose() ^ out_grad,
        ]
    }
}
