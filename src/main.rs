use rand::prelude::*;
use std::collections::{HashMap, HashSet};

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
    fn zoom(&mut self, ind: usize) {
        assert!(ind < self.shape[0]);
        let sub_size = self.size() / self.len();
        self.shape.remove(0);
        self.offset += sub_size * ind;
    }
}

impl<'a> TensorMutView<'a> {
    fn zoom(&mut self, ind: usize) {
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

fn combine_shapes(a: &[usize], b: &[usize], op_dim: usize) -> Vec<usize> {
    assert!(a.len() >= op_dim);
    assert!(b.len() >= op_dim);
    let shape_len = std::cmp::max(a.len(), b.len());
    let mut shape = Vec::new();
    for i in op_dim..shape_len {
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

fn apply<Op: Operation>(a: &Tensor, b: &Tensor) -> Tensor {
    let shape = combine_shapes(a.shape(), b.shape(), Op::input_dim());

    let mut result_shape = shape.clone();
    let op_lhs_shape = a.shape()[a.shape().len() - Op::input_dim()..].to_vec();
    let op_rhs_shape = b.shape()[b.shape().len() - Op::input_dim()..].to_vec();
    result_shape.extend(Op::output_shape(&[&op_lhs_shape, &op_rhs_shape]));
    let mut result = Tensor::zeros(&result_shape);

    let mut curr = vec![0; shape.len()];
    let mut finished = false;
    while !finished {
        let mut rv = result.view_mut();
        let mut av = a.view();
        let mut bv = b.view();
        for (i, d) in curr.iter().enumerate() {
            rv.zoom(*d);
            if i >= shape.len() + Op::input_dim() - a.shape.len() {
                av.zoom(if av.shape()[0] == 1 { 0 } else { *d });
            }
            if i >= shape.len() + Op::input_dim() - b.shape.len() {
                bv.zoom(if bv.shape()[0] == 1 { 0 } else { *d });
            }
        }
        Op::apply(&[av, bv], &mut rv);
        if curr.len() == 0 {
            break;
        }
        curr[0] += 1;
        for i in 0..curr.len() {
            if curr[i] == shape[i] {
                if i == curr.len() - 1 {
                    finished = true;
                    break;
                }
                curr[i] = 0;
                curr[i + 1] += 1;
            }
        }
    }
    result
}

impl Tensor {
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
    pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
        let mut output = a.clone();
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
    /*pub fn mul_f32<A: TensorOps>(graph: &mut Graph, a: &A, b: f32) -> Tensor {
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
    }*/
}

trait Module {
    fn forward(&mut self, graph: &mut Graph, inp: &Tensor) -> Tensor;
}

trait Loss {
    fn loss(&mut self, graph: &mut Graph, inp: &Tensor, target: &Tensor) -> Tensor;
}

/*pub struct ReLU;

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
}*/

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
        apply::<AddOp>(inps[0], inps[1])
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], apply::<AddOp>(&grads[&inps[0]], &grads[&out]));
        grads.insert(inps[1], apply::<AddOp>(&grads[&inps[1]], &grads[&out]));
    }
}

pub struct Mul;
impl Mul {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Mul {
    fn run(&self, inps: &[&Tensor]) -> Tensor {
        assert_eq!(inps.len(), 2);
        apply::<MatMulOp>(inps[0], inps[1])
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(
            inps[0],
            apply::<MatMulOp>(&grads[&out], &tensors[&inps[1]].transpose()),
        );
        grads.insert(
            inps[1],
            apply::<MatMulOp>(&tensors[&inps[0]].transpose(), &grads[&out]),
        );
    }
}

struct Computation {
    inps: Vec<TensorId>,
    func: Box<dyn Function>,
}

pub type TensorId = usize;

pub struct Graph {
    grads: HashMap<TensorId, Tensor>,
    computations: HashMap<TensorId, Computation>,
    tensors: HashMap<TensorId, Tensor>,
    parents: HashMap<TensorId, HashSet<TensorId>>,
    next_tensor_id: TensorId,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            grads: Default::default(),
            tensors: Default::default(),
            parents: Default::default(),
            computations: Default::default(),
            next_tensor_id: Default::default(),
        }
    }
    pub fn alloc(&mut self, tensor: Tensor) -> TensorId {
        let id = self.next_tensor_id;
        self.tensors.insert(id, tensor);
        self.next_tensor_id += 1;
        id
    }
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }
    pub fn get(&self, id: TensorId) -> &Tensor {
        self.tensors.get(&id).expect("Tensor not found!")
    }
    pub fn topology(&self, root: TensorId) -> Vec<TensorId> {
        let mut visited = HashSet::<TensorId>::new();
        let mut to_visit = vec![root];
        let mut order = Vec::new();
        while let Some(id) = to_visit.pop() {
            if !visited.contains(&id) {
                visited.insert(id);
                for child in self.parents.get(&id).cloned().unwrap_or_default() {
                    to_visit.insert(0, child);
                }
                order.push(id);
            }
        }

        order
    }
    pub fn backward(&mut self, id: TensorId) {
        if let Some(comp) = self.computations.get(&id) {
            for inp in comp.inps.iter() {
                let shape = self.get(*inp).shape.clone();
                self.grads.entry(*inp).or_insert(Tensor::zeros(&shape));
            }
            comp.func
                .grad(&mut self.grads, &mut self.tensors, &comp.inps, id);
        }
    }
    pub fn backward_all(&mut self, id: TensorId) {
        let shape = &self.get(id).shape;
        self.grads.insert(id, Tensor::ones(&shape));

        for t in self.topology(id) {
            self.backward(t);
        }
    }
    pub fn call(&mut self, f: Box<dyn Function>, tensor_ids: &[TensorId]) -> TensorId {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.tensors.get(id).expect("Tensor not found!"))
            .collect::<Vec<_>>();
        let child = self.alloc(f.run(&tensors));
        self.computations.insert(
            child,
            Computation {
                func: f,
                inps: tensor_ids.to_vec(),
            },
        );
        for parent in tensor_ids {
            self.parents.entry(child).or_default().insert(*parent);
        }
        child
    }
}

trait Operation {
    fn input_dim() -> usize;
    fn output_shape(inp_shapes: &[&[usize]]) -> Vec<usize>;
    fn apply(inps: &[TensorView], out: &mut TensorMutView);
}

struct AddOp;
impl Operation for AddOp {
    fn input_dim() -> usize {
        0
    }
    fn output_shape(inp_shapes: &[&[usize]]) -> Vec<usize> {
        assert_eq!(inp_shapes.len(), 2);
        assert_eq!(inp_shapes[0].len(), 0);
        assert_eq!(inp_shapes[1].len(), 0);
        vec![]
    }
    fn apply(inps: &[TensorView], out: &mut TensorMutView) {
        out.set_scalar(inps[0].scalar() + inps[1].scalar());
    }
}

struct MatMulOp;
impl Operation for MatMulOp {
    fn input_dim() -> usize {
        2
    }
    fn output_shape(inp_shapes: &[&[usize]]) -> Vec<usize> {
        println!("{:?}", inp_shapes);
        assert_eq!(inp_shapes.len(), 2);
        assert_eq!(inp_shapes[0].len(), 2);
        assert_eq!(inp_shapes[1].len(), 2);
        assert_eq!(inp_shapes[0][1], inp_shapes[1][0]);
        vec![inp_shapes[0][0], inp_shapes[1][1]]
    }
    fn apply(inps: &[TensorView], out: &mut TensorMutView) {
        for i in 0..out.shape()[0] {
            for j in 0..out.shape()[1] {
                let mut sum = 0.;
                for k in 0..inps[1].shape()[0] {
                    sum += inps[0].get(i).get(k).scalar() * inps[1].get(k).get(j).scalar();
                }
                out.get_mut(i).get_mut(j).set_scalar(sum);
            }
        }
    }
}

fn main() {
    let mut rng = thread_rng();
    let mut g = Graph::new();

    let t0 = g.alloc(Tensor::rand(&mut rng, &[3, 4]));
    let t1 = g.alloc(Tensor::rand(&mut rng, &[4, 5]));
    let t2 = g.call(Mul::new(), &[t0, t1]);
    g.backward_all(t2);
    println!("{:?}", g.grads.get(&t1));
}
