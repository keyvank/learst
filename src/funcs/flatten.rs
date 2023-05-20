use super::{Function, Tensor, TensorOps};

#[derive(Debug)]
pub struct Flatten {
    dims: usize,
}
impl Flatten {
    pub fn new(dims: usize) -> Box<dyn Function> {
        Box::new(Self { dims })
    }
}

impl Function for Flatten {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].map(self.dims, |t| t.reshape(&[0]).into())
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
