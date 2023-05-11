mod add;
mod convolution;
mod crossentropy;
mod flatten;
mod matmul;
mod maxpool;
mod mean;
mod mse;
mod relu;
mod sigmoid;
mod softmax;
mod square;
mod sub;

pub use add::*;
pub use convolution::*;
pub use crossentropy::*;
pub use flatten::*;
pub use matmul::*;
pub use maxpool::*;
pub use mean::*;
pub use mse::*;
pub use relu::*;
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
