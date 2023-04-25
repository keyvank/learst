use rand::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Tensor {
    blob: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorView<'a> {
    mirror: &'a Tensor,
    offset: usize,
    shape: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorMutView<'a> {
    mirror: &'a mut Tensor,
    offset: usize,
    shape: Vec<usize>,
}

impl<'a> From<&TensorView<'a>> for Tensor {
    fn from(view: &TensorView<'a>) -> Self {
        Self {
            blob: view.blob().to_vec(),
            shape: view.shape().to_vec(),
        }
    }
}

impl<'a> From<&TensorMutView<'a>> for Tensor {
    fn from(view: &TensorMutView<'a>) -> Self {
        Self {
            blob: view.blob().to_vec(),
            shape: view.shape().to_vec(),
        }
    }
}

pub struct TensorIter<'a, T: TensorOps> {
    target: &'a T,
    index: usize,
}
impl<'a, T: TensorOps> Iterator for TensorIter<'a, T> {
    type Item = TensorView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.index < self.target.len() {
            Some(self.target.get(self.index))
        } else {
            None
        };
        self.index += 1;
        ret
    }
}

pub struct TensorIterMut<'a, T: TensorMutOps> {
    target: &'a mut T,
    index: usize,
}
impl<'a, T: TensorMutOps> Iterator for TensorIterMut<'a, T> {
    type Item = TensorMutView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.index < self.target.len() {
            unsafe {
                let t = self.target.get_mut(self.index);
                Some(TensorMutView {
                    mirror: &mut *(t.mirror as *mut Tensor),
                    offset: t.offset,
                    shape: t.shape,
                })
            }
        } else {
            None
        };
        self.index += 1;
        ret
    }
}

pub trait TensorMutOps: TensorOps {
    fn blob_mut(&mut self) -> &mut [f32];
    fn tensor_mut(&mut self) -> &mut Tensor;

    fn set_scalar(&mut self, f: f32) {
        if self.dim() == 0 {
            self.blob_mut()[0] = f;
        } else {
            panic!("Tensor is not a scalar!")
        }
    }
    fn fill(&mut self, v: f32) {
        self.blob_mut().fill(v);
    }
    fn set<T: TensorOps>(&mut self, t: &T) {
        assert_eq!(self.shape(), t.shape());
        self.blob_mut().clone_from_slice(t.blob());
    }
    fn reshape_mut(&mut self, shape: &[usize]) -> TensorMutView {
        let final_shape = reshape(self.size(), shape);
        let new_size = final_shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        let offset = self.offset();
        TensorMutView {
            mirror: self.tensor_mut(),
            offset: offset,
            shape: final_shape.to_vec(),
        }
    }
    fn get_mut(&mut self, ind: usize) -> TensorMutView {
        let sub_size = self.size() / self.len();
        let shape = self.shape()[1..].to_vec();
        let offset = self.offset();
        TensorMutView {
            mirror: self.tensor_mut(),
            offset: offset + sub_size * ind,
            shape,
        }
    }
    fn iter_mut<'a>(&'a mut self) -> TensorIterMut<'a, Self> {
        TensorIterMut {
            target: self,
            index: 0,
        }
    }
}

fn reshape(size: usize, shape: &[usize]) -> Vec<usize> {
    let mut final_shape = shape.to_vec();
    if shape[0] == 0 && shape[1..].iter().all(|s| *s != 0) {
        let mul = shape[1..].iter().fold(1, |c, s| c * s);
        final_shape[0] = size / mul;
    } else if shape[shape.len() - 1] == 0 && shape[0..shape.len() - 1].iter().all(|s| *s != 0) {
        let mul = shape[..shape.len() - 1].iter().fold(1, |c, s| c * s);
        final_shape[shape.len() - 1] = size / mul;
    } else {
        assert!(shape.iter().all(|s| *s != 0));
    };
    final_shape
}

pub trait TensorOps: Sized {
    fn shape(&self) -> &[usize];
    fn blob(&self) -> &[f32];
    fn tensor(&self) -> &Tensor;
    fn offset(&self) -> usize;
    fn sum(&self) -> f32 {
        self.blob().iter().sum()
    }
    fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        Tensor {
            blob: self.blob().iter().map(|v| f(*v)).collect(),
            shape: self.shape().to_vec(),
        }
    }

    fn scalar(&self) -> f32 {
        if self.dim() == 0 {
            self.blob()[0]
        } else {
            panic!("Tensor is not a scalar!")
        }
    }
    fn iter<'a>(&'a self) -> TensorIter<'a, Self> {
        TensorIter {
            target: self,
            index: 0,
        }
    }
    fn dim(&self) -> usize {
        self.shape().len()
    }
    fn len(&self) -> usize {
        *self
            .shape()
            .get(0)
            .expect("Scalar values don't have a size!")
    }
    fn size(&self) -> usize {
        self.shape().iter().fold(1, |curr, s| curr * s)
    }
    fn reshape(&self, shape: &[usize]) -> TensorView {
        let final_shape = reshape(self.size(), shape);
        let new_size = final_shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        let offset = self.offset();
        TensorView {
            mirror: self.tensor(),
            offset: offset,
            shape: final_shape,
        }
    }

    fn get(&self, ind: usize) -> TensorView {
        let sub_size = self.size() / self.len();
        let shape = self.shape()[1..].to_vec();
        let offset = self.offset();
        TensorView {
            mirror: self.tensor(),
            offset: offset + sub_size * ind,
            shape,
        }
    }
}

impl TensorMutOps for Tensor {
    fn tensor_mut(&mut self) -> &mut Tensor {
        self
    }
    fn blob_mut(&mut self) -> &mut [f32] {
        &mut self.blob
    }
}

impl TensorOps for Tensor {
    fn tensor(&self) -> &Tensor {
        self
    }

    fn offset(&self) -> usize {
        0
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[f32] {
        &self.blob
    }
}

impl TensorMutOps for TensorMutView<'_> {
    fn tensor_mut(&mut self) -> &mut Tensor {
        &mut self.mirror
    }
    fn blob_mut(&mut self) -> &mut [f32] {
        let sz = self.size();
        &mut self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl TensorOps for TensorMutView<'_> {
    fn tensor(&self) -> &Tensor {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[f32] {
        let sz = self.size();
        &self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl TensorOps for TensorView<'_> {
    fn tensor(&self) -> &Tensor {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[f32] {
        &self.mirror.blob[self.offset..self.offset + self.size()]
    }
}

impl Tensor {
    pub fn matmul<A: TensorOps, B: TensorOps>(graph: &mut Graph, a: &A, b: &B) -> Tensor {
        assert!(a.dim() >= 2);
        assert_eq!(b.dim(), 2);
        assert_eq!(a.shape()[a.dim() - 1], b.shape()[0]);

        let mut final_shape = a.shape().to_vec();
        final_shape[a.dim() - 1] = b.shape()[1];

        let reshaped_a = a.reshape(&[0, a.shape()[a.dim() - 2], a.shape()[a.dim() - 1]]);

        let mut result = Self::zeros(&final_shape);
        for (mut t, corr_a) in result
            .reshape_mut(&[
                0,
                final_shape[final_shape.len() - 2],
                final_shape[final_shape.len() - 1],
            ])
            .iter_mut()
            .zip(reshaped_a.iter())
        {
            for i in 0..final_shape[final_shape.len() - 2] {
                for j in 0..final_shape[final_shape.len() - 1] {
                    let aa = corr_a.blob();
                    let bb = b.blob();
                    let mut sum = 0.;
                    for k in 0..b.shape()[0] {
                        sum += aa[i * b.shape()[0] + k]
                            * bb[k * final_shape[final_shape.len() - 1] + j];
                    }
                    t.get_mut(i).get_mut(j).set_scalar(sum);
                }
            }
        }
        result
    }
    pub fn rand<R: Rng>(r: &mut R, shape: &[usize]) -> Self {
        let blob: Vec<f32> = (0..shape.iter().fold(1, |curr, s| curr * s))
            .map(|_| r.gen::<f32>() * 2. - 1.)
            .collect();
        Self {
            blob,
            shape: shape.to_vec(),
        }
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            blob: vec![0.; shape.iter().fold(1, |curr, s| curr * s)],
            shape: shape.to_vec(),
        }
    }
    pub fn iden(n: usize) -> Self {
        Self {
            blob: (0..n * n)
                .map(|i| if i / n == i % n { 1. } else { 0. })
                .collect(),
            shape: vec![n, n],
        }
    }
    pub fn scalar(v: f32) -> Self {
        Self {
            blob: vec![v],
            shape: vec![],
        }
    }
    pub fn add<A: TensorOps, B: TensorOps>(graph: &mut Graph, a: &A, b: &B) -> Tensor {
        let mut output = Tensor::zeros(a.shape());
        let mut shape = b.shape().to_vec();
        shape.insert(0, 0);
        for mut t in output.reshape_mut(&shape).iter_mut() {
            assert_eq!(t.shape(), b.shape());
            t.blob_mut()
                .iter_mut()
                .zip(b.blob().iter())
                .for_each(|(t, a)| *t += a);
        }
        output
    }
    pub fn mul_f32<A: TensorOps>(graph: &mut Graph, a: &A, b: f32) -> Tensor {
        let mut output = Tensor::zeros(a.shape());
        output.blob_mut().iter_mut().for_each(|t| *t *= b);
        output
    }
    pub fn mul_tensor<A: TensorOps, B: TensorOps>(graph: &mut Graph, a: &A, b: &B) -> Tensor {
        let mut output = Tensor::zeros(a.shape());
        let mut shape = b.shape().to_vec();
        shape.insert(0, 0);
        for mut t in output.reshape_mut(&shape).iter_mut() {
            assert_eq!(t.shape(), b.shape());
            t.blob_mut()
                .iter_mut()
                .zip(b.blob().iter())
                .for_each(|(t, a)| *t *= a);
        }
        output
    }
    pub fn sub<A: TensorOps, B: TensorOps>(graph: &mut Graph, a: &A, b: &B) -> Tensor {
        let neg_b = Self::mul_f32(graph, b, -1.);
        Self::add(graph, a, &neg_b)
    }
}

use std::ops::*;

trait Module {
    fn forward(&mut self, graph: &mut Graph, inp: &Tensor) -> Tensor;
}

trait Loss {
    fn loss(&mut self, graph: &mut Graph, inp: &Tensor, target: &Tensor) -> Tensor;
}

pub struct ReLU;

impl Module for ReLU {
    fn forward(&mut self, graph: &mut Graph, inp: &Tensor) -> Tensor {
        inp.map(|f| if f < 0.0 { 0.0 } else { f })
    }
}

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Module for Linear {
    fn forward(&mut self, graph: &mut Graph, inp: &Tensor) -> Tensor {
        let mul = Tensor::matmul(graph, inp, &self.weights);
        Tensor::add(graph, &mul, &self.bias)
    }
}

pub struct L2Loss;

impl Loss for L2Loss {
    fn loss(&mut self, graph: &mut Graph, inp: &Tensor, target: &Tensor) -> Tensor {
        let diff = Tensor::sub(graph, inp, target);
        Tensor::mul_tensor(graph, &diff, &diff)
    }
}

type TensorId = usize;

pub struct Graph {
    grads: HashMap<TensorId, Tensor>,
    tensors: HashMap<TensorId, Tensor>,
    parents: HashMap<TensorId, Vec<TensorId>>,
    children: HashMap<TensorId, Vec<TensorId>>,
    next_tensor_id: TensorId,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            grads: Default::default(),
            tensors: Default::default(),
            parents: Default::default(),
            children: Default::default(),
            next_tensor_id: Default::default(),
        }
    }
    pub fn alloc(&mut self, shape: &[usize]) -> TensorId {
        let id = self.next_tensor_id;
        self.tensors.insert(id, Tensor::zeros(shape));
        self.next_tensor_id += 1;
        id
    }
}

fn main() {
    let mut g = Graph::new();
}
