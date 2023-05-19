use super::{Function, Tensor};

pub struct Mul;
impl Mul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Mul {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 2);
        inps[0] * inps[1]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![inps[1].clone(), inps[0].clone()]
    }
}
