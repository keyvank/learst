use super::{Function, Tensor, TensorOps};

pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Sigmoid {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].map_values(|f| 1. / (1. + (-f).exp()))
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        let der = out.map_values(|f| f * (1. - f));
        vec![&der * out_grad]
    }
}
