mod add;
mod crossentropy;
mod matmul;
mod mean;
mod sigmoid;
mod softmax;
mod square;
mod sub;

pub use add::*;
pub use crossentropy::*;
pub use matmul::*;
pub use mean::*;
pub use sigmoid::*;
pub use softmax::*;
pub use square::*;
pub use sub::*;

use super::tensor::*;

pub trait Function {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32>;
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>>;
}

pub trait Loss {
    fn run(&self, inp: &Tensor<f32>) -> Tensor<f32>;
    fn grad(&self, inp: &Tensor<f32>, out: &Tensor<f32>) -> Tensor<f32>;
}
