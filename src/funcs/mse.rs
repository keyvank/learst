use super::{Loss, Tensor};

pub struct MeanSquaredError {
    target: Tensor<f32>,
}
impl MeanSquaredError {
    pub fn new(target: Tensor<f32>) -> Box<dyn Loss> {
        Box::new(Self { target })
    }
}
impl Loss for MeanSquaredError {
    fn run(&self, inp: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        let diff = inp - &self.target;
        (&diff * &diff, &diff * &Tensor::<f32>::scalar(2.))
    }
}
