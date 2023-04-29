use rand::prelude::*;
use std::ops::*;
mod ops;
pub use ops::*;

pub trait TensorElement: Clone + Copy + Sized {
    fn zero() -> Self;
    fn one() -> Self;
}
impl TensorElement for f32 {
    fn zero() -> Self {
        0.
    }
    fn one() -> Self {
        1.
    }
}
impl TensorElement for u32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<V: TensorElement> {
    blob: Vec<V>,
    shape: Vec<usize>,
}

#[derive(Debug)]
pub struct TensorView<'a, V: TensorElement> {
    mirror: &'a Tensor<V>,
    offset: usize,
    shape: Vec<usize>,
}

impl<'a, V: TensorElement> TensorView<'a, V> {
    pub fn zoom(&mut self, ind: usize) {
        assert!(ind < self.shape[0]);
        let sub_size = self.size() / self.len();
        self.shape.remove(0);
        self.offset += sub_size * ind;
    }
}

impl<'a, V: TensorElement> TensorMutView<'a, V> {
    pub fn zoom(&mut self, ind: usize) {
        assert!(ind < self.shape[0]);
        let sub_size = self.size() / self.len();
        self.shape.remove(0);
        self.offset += sub_size * ind;
    }
}

#[derive(Debug)]
pub struct TensorMutView<'a, V: TensorElement> {
    mirror: &'a mut Tensor<V>,
    offset: usize,
    shape: Vec<usize>,
}

pub struct TensorIter<'a, V: TensorElement, T: TensorOps<V>> {
    target: &'a T,
    index: usize,
    _value_type: std::marker::PhantomData<V>,
}
impl<'a, V: 'a + TensorElement, T: TensorOps<V>> Iterator for TensorIter<'a, V, T> {
    type Item = TensorView<'a, V>;

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

pub struct TensorIterMut<'a, V: TensorElement, T: TensorMutOps<V>> {
    target: &'a mut T,
    index: usize,
    _value_type: std::marker::PhantomData<V>,
}
impl<'a, V: 'a + TensorElement, T: TensorMutOps<V>> Iterator for TensorIterMut<'a, V, T> {
    type Item = TensorMutView<'a, V>;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = if self.index < self.target.len() {
            unsafe {
                let t = self.target.get_mut(self.index);
                Some(TensorMutView {
                    mirror: &mut *(t.mirror as *mut Tensor<V>),
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

pub trait TensorMutOps<V: TensorElement>: TensorOps<V> {
    fn blob_mut(&mut self) -> &mut [V];
    fn tensor_mut(&mut self) -> &mut Tensor<V>;

    fn view_mut(&mut self) -> TensorMutView<V> {
        TensorMutView {
            offset: self.offset(),
            shape: self.shape().to_vec(),
            mirror: self.tensor_mut(),
        }
    }
    fn fill(&mut self, v: V) {
        self.blob_mut().fill(v);
    }
    fn set<T: TensorOps<V>>(&mut self, t: T) {
        assert_eq!(self.shape(), t.shape());
        self.blob_mut().clone_from_slice(t.blob());
    }
    fn reshape_mut(&mut self, shape: &[usize]) -> TensorMutView<V> {
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
    fn get_mut(&mut self, ind: usize) -> TensorMutView<V> {
        let mut v = self.view_mut();
        v.zoom(ind);
        v
    }
    fn iter_mut<'a>(&'a mut self) -> TensorIterMut<'a, V, Self> {
        TensorIterMut {
            target: self,
            index: 0,
            _value_type: std::marker::PhantomData::<V>,
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

pub trait TensorOps<V: TensorElement>: Sized {
    fn shape(&self) -> &[usize];
    fn blob(&self) -> &[V];
    fn tensor(&self) -> &Tensor<V>;
    fn offset(&self) -> usize;

    fn map<F: Fn(V) -> V>(&self, f: F) -> Tensor<V> {
        Tensor {
            blob: self.blob().iter().map(|v| f(*v)).collect(),
            shape: self.shape().to_vec(),
        }
    }

    fn scalar(&self) -> V {
        if self.dim() == 0 {
            self.blob()[0]
        } else {
            panic!("Tensor is not a scalar!")
        }
    }
    fn iter<'a>(&'a self) -> TensorIter<'a, V, Self> {
        TensorIter {
            target: self,
            index: 0,
            _value_type: std::marker::PhantomData::<V>,
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
    fn view(&self) -> TensorView<V> {
        TensorView {
            mirror: self.tensor(),
            offset: self.offset(),
            shape: self.shape().to_vec(),
        }
    }
    fn reshape(&self, shape: &[usize]) -> TensorView<V> {
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

    fn get(&self, ind: usize) -> TensorView<V> {
        let mut v = self.view();
        v.zoom(ind);
        v
    }

    fn transpose(&self) -> Tensor<V> {
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
                    dst.get_mut(i).get_mut(j).set(src.get(j).get(i));
                }
            }
        }
        t
    }
}

impl<V: TensorElement> TensorMutOps<V> for Tensor<V> {
    fn tensor_mut(&mut self) -> &mut Tensor<V> {
        self
    }
    fn blob_mut(&mut self) -> &mut [V] {
        &mut self.blob
    }
}

impl<V: TensorElement> TensorOps<V> for Tensor<V> {
    fn tensor(&self) -> &Tensor<V> {
        self
    }

    fn offset(&self) -> usize {
        0
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        &self.blob
    }
}

impl<V: TensorElement> TensorMutOps<V> for TensorMutView<'_, V> {
    fn tensor_mut(&mut self) -> &mut Tensor<V> {
        &mut self.mirror
    }
    fn blob_mut(&mut self) -> &mut [V] {
        let sz = self.size();
        &mut self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl<V: TensorElement> TensorOps<V> for TensorMutView<'_, V> {
    fn tensor(&self) -> &Tensor<V> {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        let sz = self.size();
        &self.mirror.blob[self.offset..self.offset + sz]
    }
}

impl<V: TensorElement> TensorOps<V> for TensorView<'_, V> {
    fn tensor(&self) -> &Tensor<V> {
        &self.mirror
    }
    fn offset(&self) -> usize {
        self.offset
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn blob(&self) -> &[V] {
        &self.mirror.blob[self.offset..self.offset + self.size()]
    }
}

impl<V: TensorElement> Tensor<V> {
    pub fn raw(shape: &[usize], blob: Vec<V>) -> Self {
        Self {
            blob,
            shape: shape.to_vec(),
        }
    }
    pub fn fill_by<F: Fn(&[usize]) -> V>(shape: &[usize], f: F) -> Self {
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
                .map(|i| if i / n == i % n { V::one() } else { V::zero() })
                .collect(),
            shape: vec![n, n],
        }
    }
    pub fn scalar(v: V) -> Self {
        Tensor {
            blob: vec![v],
            shape: vec![],
        }
    }
    pub fn constant(shape: &[usize], value: V) -> Self {
        Tensor {
            blob: vec![value; shape.iter().fold(1, |curr, s| curr * s)],
            shape: shape.to_vec(),
        }
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self::constant(shape, V::zero())
    }
    pub fn ones(shape: &[usize]) -> Self {
        Self::constant(shape, V::one())
    }
    pub fn rand<R: Rng>(r: &mut R, shape: &[usize]) -> Tensor<f32> {
        Tensor::<f32> {
            blob: (0..shape.iter().fold(1, |curr, s| curr * s))
                .map(|_| r.gen::<f32>() * 2. - 1.)
                .collect(),
            shape: shape.to_vec(),
        }
    }
}

fn combine_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let shape_len = std::cmp::max(a.len(), b.len());
    let mut shape = Vec::new();
    for i in 0..shape_len {
        shape.insert(
            0,
            if i >= a.len() {
                b[b.len() - 1 - i]
            } else if i >= b.len() {
                a[a.len() - 1 - i]
            } else {
                let (a, b) = (a[a.len() - 1 - i], b[b.len() - 1 - i]);
                if a == b {
                    a
                } else if a == 1 {
                    b
                } else if b == 1 {
                    a
                } else {
                    panic!("Cannot be combined!")
                }
            },
        );
    }
    shape
}

pub fn binary<F: FnMut(&[usize], &[usize], &[usize]) -> ()>(a: &[usize], b: &[usize], mut f: F) {
    let shape = combine_shapes(a, b);
    let mut curr = vec![0; shape.len()];
    fn calc_shape(pos: &[usize], shape: &[usize]) -> Vec<usize> {
        pos[pos.len() - shape.len()..]
            .iter()
            .zip(shape.iter())
            .map(|(p, s)| if *s == 1 { 0 } else { *p })
            .collect()
    }
    loop {
        let a_pos = calc_shape(&curr, a);
        let b_pos = calc_shape(&curr, b);
        f(&curr, &a_pos, &b_pos);
        if curr.len() == 0 {
            return;
        }
        curr[shape.len() - 1] += 1;
        for i in (0..curr.len()).rev() {
            if curr[i] == shape[i] {
                if i == 0 {
                    return;
                }
                curr[i] = 0;
                curr[i - 1] += 1;
            }
        }
    }
}
