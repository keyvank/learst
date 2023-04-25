#[derive(Debug)]
pub enum Data<'a> {
    View(&'a [f32]),
    MutView(&'a mut [f32]),
    Blob(Vec<f32>),
}

impl<'a> AsRef<[f32]> for Data<'a> {
    fn as_ref(&self) -> &[f32] {
        match self {
            Data::Blob(v) => &v,
            Data::View(s) => &s,
            Data::MutView(s) => &s,
        }
    }
}

impl<'a> AsMut<[f32]> for Data<'a> {
    fn as_mut(&mut self) -> &mut [f32] {
        match self {
            Data::Blob(ref mut v) => v,
            Data::View(_) => panic!("Cannot mutate an immutable view!"),
            Data::MutView(ref mut s) => s,
        }
    }
}

impl<'a> Clone for Data<'a> {
    fn clone(&self) -> Self {
        Data::Blob(match self {
            Data::Blob(v) => v.clone(),
            Data::View(s) => s.to_vec(),
            Data::MutView(s) => s.to_vec(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    blob: Data<'a>,
    shape: Vec<usize>,
}

impl<'a> Tensor<'a> {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            blob: Data::Blob(vec![0.; shape.iter().fold(1, |curr, s| curr * s)]),
            shape: shape.to_vec(),
        }
    }
    pub fn fill(&mut self, v: f32) {
        self.blob.as_mut().fill(v);
    }
    pub fn size(&self) -> usize {
        self.shape.iter().fold(1, |curr, s| curr * s)
    }
    pub fn reshape(&self, shape: &[usize]) -> Tensor {
        let new_size = shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        Tensor {
            blob: Data::View(self.blob.as_ref()),
            shape: shape.to_vec(),
        }
    }
    pub fn reshape_mut(&mut self, shape: &[usize]) -> Tensor {
        let new_size = shape.iter().fold(1, |c, s| c * s);
        assert_eq!(new_size, self.size());
        Tensor {
            blob: Data::MutView(self.blob.as_mut()),
            shape: shape.to_vec(),
        }
    }
    pub fn get_mut(&mut self, ind: usize) -> Tensor {
        let dim = self.shape.get(0).expect("Cannot index into a scalar!");
        let sub_size = self.size() / dim;
        let shape = self.shape[1..].to_vec();
        Tensor {
            blob: Data::MutView(&mut self.blob.as_mut()[sub_size * ind..sub_size * (ind + 1)]),
            shape,
        }
    }
    pub fn get(&self, ind: usize) -> Tensor {
        let dim = self.shape.get(0).expect("Cannot index into a scalar!");
        let sub_size = self.size() / dim;
        let shape = self.shape[1..].to_vec();
        Tensor {
            blob: Data::View(&self.blob.as_ref()[sub_size * ind..sub_size * (ind + 1)]),
            shape,
        }
    }

    pub fn scalar(v: f32) -> Self {
        Self {
            blob: Data::Blob(vec![v]),
            shape: vec![],
        }
    }
}

fn main() {
    let mut t = Tensor::zeros(&[2, 3]);
    t.get_mut(0).fill(1.0);
    t.get_mut(1).fill(2.0);
    println!("{:?} {:?}", t.get(0), t.get(1));
    println!("{:?} {:?}", t.get(0).get(1), t.get(1).get(2));
    let mut t2 = t.reshape_mut(&[3, 2]);
    println!("{:?} {:?} {:?}", t2.get(0), t2.get(1), t2.get(2));
    t2.get_mut(2).fill(4.0);
    println!("{:?} {:?} {:?}", t2.get(0), t2.get(1), t2.get(2));
    println!("{:?} {:?}", t.get(0), t.get(1));
}
