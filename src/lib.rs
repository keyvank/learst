pub mod graph;
pub mod optimizer;
pub mod tensor;

use graph::*;
use tensor::*;

use std::collections::HashMap;

pub trait Function {
    fn run(&self, inps: &[&Tensor]) -> Tensor;
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    );
}

pub struct Add;
impl Add {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Add {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 2);
        inps[0] + inps[1]
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        _tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], &grads[&inps[0]] + &grads[&out]);
        grads.insert(inps[1], &grads[&inps[1]] + &grads[&out]);
    }
}

pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 2);
        inps[0] ^ inps[1]
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], &grads[&out] ^ &tensors[&inps[1]].transpose());
        grads.insert(inps[1], &tensors[&inps[0]].transpose() ^ &grads[&out]);
    }
}

pub struct Pow(f32);
impl Pow {
    pub fn new(p: f32) -> Box<dyn Function> {
        Box::new(Self(p))
    }
}
impl Function for Pow {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 1);
        inps[0].map(|f| f.powf(self.0))
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        grads.insert(
            inps[0],
            &(&tensors[&inps[0]].transpose() ^ &grads[&out]) * &Tensor::scalar(self.0),
        );
    }
}

pub struct Mul(f32);
impl Mul {
    pub fn new(p: f32) -> Box<dyn Function> {
        Box::new(Self(p))
    }
}
impl Function for Mul {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 1);
        inps[0].map(|f| f * self.0)
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        _tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        grads.insert(inps[0], &grads[&out] * &Tensor::scalar(self.0));
    }
}

pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Sigmoid {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 1);
        inps[0].map(|f| 1. / (1. + (-f).exp()))
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        let der = tensors[&out].map(|f| f * (1. - f));
        grads.insert(inps[0], &der * &grads[&out]);
    }
}
