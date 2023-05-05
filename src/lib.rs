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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>>;
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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![out_grad.copy(), out_grad.copy()]
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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![out_grad.copy(), -out_grad]
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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 2);
        vec![
            out_grad ^ &inps[1].transpose(),
            &inps[0].transpose() ^ out_grad,
        ]
    }
}

pub struct Square;
impl Square {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Square {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].mapf(|f| f.powf(2.))
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        vec![&(inps[0] * out_grad) * &Tensor::<f32>::scalar(2.)]
    }
}

pub struct Mean;
impl Mean {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Mean {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        assert_eq!(inps[0].dim(), 2);
        Tensor::scalar(inps[0].blob().iter().sum::<f32>() / inps[0].blob().len() as f32)
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        vec![out_grad * &Tensor::<f32>::scalar(1. / inps[0].blob().len() as f32)]
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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);
        let der = out.mapf(|f| f * (1. - f));
        vec![&der * out_grad]
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
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let jacobian = out.map(1, |l| {
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
        let out = &out_grad.unsqueeze(-2) ^ &jacobian;
        vec![out.squeeze(-2).copy()]
    }
}

pub struct CrossEntropy {
    classes: u32,
    target: Tensor<u32>,
}
impl CrossEntropy {
    pub fn new(classes: u32, target: Tensor<u32>) -> Box<dyn Function> {
        Box::new(Self { classes, target })
    }
}
impl Function for CrossEntropy {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        let mut expected_shape = self.target.shape().to_vec();
        expected_shape.push(self.classes as usize);
        assert_eq!(inps[0].shape(), expected_shape);
        let mut out = Tensor::<f32>::zeros(self.target.shape());
        for ((mut r, o), t) in out
            .reshape_mut(&[0])
            .iter_mut()
            .zip(inps[0].reshape(&[0, self.classes as usize]).iter())
            .zip(self.target.reshape(&[0]).iter())
        {
            r.set(Tensor::scalar(-(o.get(t.scalar() as usize).scalar().ln())));
        }
        out
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);

        let mut result = Tensor::<f32>::zeros(self.target.shape());
        for ((mut r, o), t) in result
            .reshape_mut(&[0])
            .iter_mut()
            .zip(inps[0].reshape(&[0, self.classes as usize]).iter())
            .zip(self.target.reshape(&[0]).iter())
        {
            r.set(Tensor::scalar(-1. / (o.get(t.scalar() as usize).scalar())));
        }

        vec![&result * out_grad]
    }
}
