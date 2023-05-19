use super::{Function, Tensor, TensorOps};

pub struct Coeff {
    coeff: f32,
}
impl Coeff {
    pub fn new(coeff: f32) -> Box<dyn Function> {
        Box::new(Self { coeff })
    }
}
impl Function for Coeff {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].map_values(|f| f * self.coeff)
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        vec![out_grad.map_values(|d| d * self.coeff)]
    }
}
