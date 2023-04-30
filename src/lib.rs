pub mod graph;
pub mod optimizer;
pub mod tensor;

use graph::*;
use tensor::*;

use std::collections::HashMap;

pub trait Function {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32>;
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
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
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 2);
        inps[0] + inps[1]
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        _tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], &grads[&inps[0]] + &grads[&out]);
        grads.insert(inps[1], &grads[&inps[1]] + &grads[&out]);
    }
}

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
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        _tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], &grads[&inps[0]] - &grads[&out]);
        grads.insert(inps[1], &grads[&inps[1]] - &grads[&out]);
    }
}

pub struct MatMul;
impl MatMul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for MatMul {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 2);
        inps[0] ^ inps[1]
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
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
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].mapf(|f| f.powf(self.0))
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        grads.insert(
            inps[0],
            &(&tensors[&inps[0]].transpose() ^ &grads[&out]) * &Tensor::<f32>::scalar(self.0),
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
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].mapf(|f| f * self.0)
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        _tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        grads.insert(inps[0], &grads[&out] * &Tensor::<f32>::scalar(self.0));
    }
}

pub struct Sigmoid;
impl Sigmoid {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Sigmoid {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].mapf(|f| 1. / (1. + (-f).exp()))
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 1);
        let der = tensors[&out].mapf(|f| f * (1. - f));
        grads.insert(inps[0], &der * &grads[&out]);
    }
}

pub struct Softmax;
impl Softmax {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Softmax {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].map(1, |l| {
            let sum = l.mapf(|f| f.exp()).iter().map(|t| t.scalar()).sum::<f32>();
            l.mapf(|f| f.exp() / sum)
        })
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        let jacobian = tensors[&out].map(1, |l| {
            let n = l.shape()[0];
            let mut jacobian = Tensor::<f32>::zeros(&[n, n]);
            for i in 0..n {
                for j in 0..n {
                    jacobian
                        .get_mut(i)
                        .get_mut(j)
                        .set(Tensor::scalar(if i == j {
                            let sij = l.get(i).scalar();
                            sij * (1. - sij)
                        } else {
                            let si = l.get(i).scalar();
                            let sj = l.get(j).scalar();
                            -si * sj
                        }));
                }
            }
            jacobian
        });
        grads.insert(inps[0], &jacobian ^ &grads[&out]);
    }
}

pub struct CrossEntropy;
impl CrossEntropy {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for CrossEntropy {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        unimplemented!();
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor<f32>>,
        tensors: &HashMap<TensorId, Tensor<f32>>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        unimplemented!();
    }
}
