mod add;
mod cat;
mod convolution;
mod crossentropy;
mod dropout;
mod flatten;
mod layer_norm;
mod mask;
mod matmul;
mod maxpool;
mod mean;
mod mse;
mod relu;
mod sigmoid;
mod softmax;
mod square;
mod sub;
mod transpose;

pub use add::*;
pub use cat::*;
pub use convolution::*;
pub use crossentropy::*;
pub use dropout::*;
pub use flatten::*;
pub use layer_norm::*;
pub use mask::*;
pub use matmul::*;
pub use maxpool::*;
pub use mean::*;
pub use mse::*;
pub use relu::*;
pub use sigmoid::*;
pub use softmax::*;
pub use square::*;
pub use sub::*;
pub use transpose::*;

use super::tensor::*;

pub trait Function {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32>;
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>>;
}

pub trait Loss {
    fn run(&mut self, inp: &Tensor<f32>) -> Tensor<f32>;
    fn grad(&self, inp: &Tensor<f32>, out: &Tensor<f32>) -> Tensor<f32>;
}
