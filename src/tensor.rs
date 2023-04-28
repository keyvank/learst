use rand::prelude::*;

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

impl<'a> TensorView<'a> {
    pub fn zoom(&mut self, ind: usize) {
        assert!(ind < self.shape[0]);
        let sub_size = self.size() / self.len();
        self.shape.remove(0);
        self.offset += sub_size * ind;
    }
}

impl<'a> TensorMutView<'a> {
    pub fn zoom(&mut self, ind: usize) {
        assert!(ind < self.shape[0]);
        let sub_size = self.size() / self.len();
        self.shape.remove(0);
        self.offset += sub_size * ind;
    }
}

#[derive(Debug)]
pub struct TensorMutView<'a> {
    mirror: &'a mut Tensor,
    offset: usize,
    shape: Vec<usize>,
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

    fn view_mut(&mut self) -> TensorMutView {
        TensorMutView {
            offset: self.offset(),
            shape: self.shape().to_vec(),
            mirror: self.tensor_mut(),
        }
    }
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
    fn copy<T: TensorOps>(&mut self, from: &T) {
        assert_eq!(self.shape(), from.shape());
        self.blob_mut().copy_from_slice(from.blob())
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
        let mut v = self.view_mut();
        v.zoom(ind);
        v
    }
    fn iter_mut<'a>(&'a mut self) -> TensorIterMut<'a, Self> {
        TensorIterMut {
            target: self,
            index: 0,
        }
    }
}

pub fn reshape(size: usize, shape: &[usize]) -> Vec<usize> {
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
    fn view(&self) -> TensorView {
        TensorView {
            mirror: self.tensor(),
            offset: self.offset(),
            shape: self.shape().to_vec(),
        }
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
        let mut v = self.view();
        v.zoom(ind);
        v
    }

    fn transpose(&self) -> Tensor {
        let dim = self.dim();
        let mut shape = self.shape().to_vec();
        let temp = shape[dim - 2];
        shape[dim - 2] = shape[dim - 1];
        shape[dim - 1] = temp;
        let mut t = Tensor::zeros(&shape);
        for (mut dst, src) in t
            .reshape_mut(&[0, shape[dim - 2], shape[dim - 1]])
            .iter_mut()
            .zip(self.reshape(&[0, shape[dim - 1], shape[dim - 2]]).iter())
        {
            for i in 0..shape[dim - 2] {
                for j in 0..shape[dim - 1] {
                    let d = src.get(j).get(i).scalar();
                    dst.get_mut(i).get_mut(j).set_scalar(d);
                }
            }
        }
        t
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
    pub fn raw(shape: &[usize], blob: Vec<f32>) -> Self {
        Self {
            blob,
            shape: shape.to_vec(),
        }
    }
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Self {
            blob: self.blob().iter().map(|v| f(*v)).collect(),
            shape: self.shape().to_vec(),
        }
    }
    pub fn fill_by<F: Fn(&[usize]) -> f32>(shape: &[usize], f: F) -> Tensor {
        let mut curr = vec![0; shape.len()];
        let mut blob = Vec::new();
        let mut finished = false;
        while !finished {
            blob.push(f(&curr));
            if curr.len() == 0 {
                break;
            }
            curr[shape.len() - 1] += 1;
            for i in (0..curr.len()).rev() {
                if curr[i] == shape[i] {
                    if i == 0 {
                        finished = true;
                        break;
                    }
                    curr[i] = 0;
                    curr[i - 1] += 1;
                }
            }
        }
        Tensor {
            blob,
            shape: shape.to_vec(),
        }
    }
    pub fn iden(n: usize) -> Self {
        Tensor {
            blob: (0..n * n)
                .map(|i| if i / n == i % n { 1. } else { 0. })
                .collect(),
            shape: vec![n, n],
        }
    }
    pub fn scalar(v: f32) -> Self {
        Tensor {
            blob: vec![v],
            shape: vec![],
        }
    }
    pub fn constant(shape: &[usize], value: f32) -> Self {
        Tensor {
            blob: vec![value; shape.iter().fold(1, |curr, s| curr * s)],
            shape: shape.to_vec(),
        }
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self::constant(shape, 0.)
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self::constant(shape, 1.)
    }
    pub fn rand<R: Rng>(r: &mut R, shape: &[usize]) -> Self {
        Tensor {
            blob: (0..shape.iter().fold(1, |curr, s| curr * s))
                .map(|_| r.gen::<f32>() * 2. - 1.)
                .collect(),
            shape: shape.to_vec(),
        }
    }
}
