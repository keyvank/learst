use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
pub struct Square;
impl Square {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Square {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].map_values(|f| f.powf(2.))
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        vec![&(inps[0] * out_grad) * &Tensor::<f32>::scalar(2.)]
    }
}
