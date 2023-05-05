use super::{Function, Tensor};

pub struct Sub;
impl Sub {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Sub {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 2);
        inps[0] - inps[1]
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![out_grad.clone(), -out_grad]
    }
}
