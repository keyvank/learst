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

impl<'a> From<TensorView<'a>> for Tensor {
    fn from(view: TensorView<'a>) -> Self {
        Self {
            blob: view.blob().to_vec(),
            shape: view.shape().to_vec(),
        }
    }
}

impl<'a> From<TensorMutView<'a>> for Tensor {
    fn from(view: TensorMutView<'a>) -> Self {
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
    fn fill(&mut self, v: f32) {
        self.blob_mut().fill(v);
    }
    fn reshape_mut(&mut self, shape: &[usize]) -> TensorMutView {
        let new_size = shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        let offset = self.offset();
        TensorMutView {
            mirror: self.tensor_mut(),
            offset: offset,
            shape: shape.to_vec(),
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

pub trait TensorOps: Sized {
    fn shape(&self) -> &[usize];
    fn blob(&self) -> &[f32];

    fn tensor(&self) -> &Tensor;
    fn offset(&self) -> usize;
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
        let new_size = shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        let offset = self.offset();
        TensorView {
            mirror: self.tensor(),
            offset: offset,
            shape: shape.to_vec(),
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
    pub fn matmul<A: TensorOps, B: TensorOps>(a: &A, b: &B) -> Tensor {
        assert!(a.dim() >= 2);
        assert_eq!(b.dim(), 2);
        let a_1 = a.shape()[a.dim() - 2];
        let a_2 = a.shape()[a.dim() - 1];
        let b_1 = b.shape()[b.dim() - 2];
        let b_2 = b.shape()[b.dim() - 1];
        assert_eq!(a_2, b_1);
        let mut final_shape = a.shape().to_vec();
        final_shape[a.dim() - 1] = b_2;
        let mut result = Self::zeros(&final_shape);

        result
    }
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            blob: vec![0.; shape.iter().fold(1, |curr, s| curr * s)],
            shape: shape.to_vec(),
        }
    }
    pub fn scalar(v: f32) -> Self {
        Self {
            blob: vec![v],
            shape: vec![],
        }
    }
}

trait Module {
    fn forward(&mut self, inp: Tensor) -> Tensor;
}

pub struct Linear {
    weights: Tensor,
    bias: Tensor,
}

impl Module for Linear {
    fn forward(&mut self, inp: Tensor) -> Tensor {
        inp
    }
}

fn main() {
    let mut t = Tensor::zeros(&[3, 4, 5]);
    for (i, mut t) in t.iter_mut().enumerate() {
        t.fill(i as f32);
    }
    println!("{:?}", t);
}
