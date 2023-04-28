pub mod graph;
pub mod optimizer;
pub mod tensor;

use graph::*;
use tensor::*;

use std::collections::HashMap;

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

pub fn apply<Op: Operation>(op: Op, a: &Tensor, b: &Tensor) -> Tensor {
    let shape = combine_shapes(a.shape(), b.shape(), op.input_dim());

    let mut result_shape = shape.clone();
    let op_lhs_shape = a.shape()[a.shape().len() - op.input_dim()..].to_vec();
    let op_rhs_shape = b.shape()[b.shape().len() - op.input_dim()..].to_vec();
    result_shape.extend(op.output_shape(&[&op_lhs_shape, &op_rhs_shape]));
    let mut result = Tensor::zeros(&result_shape);

    let mut curr = vec![0; shape.len()];
    let mut finished = false;
    while !finished {
        let mut rv = result.view_mut();
        let mut av = a.view();
        let mut bv = b.view();
        for (i, d) in curr.iter().enumerate() {
            rv.zoom(*d);
            if i >= shape.len() + op.input_dim() - a.dim() {
                av.zoom(if av.shape()[0] == 1 { 0 } else { *d });
            }
            if i >= shape.len() + op.input_dim() - b.dim() {
                bv.zoom(if bv.shape()[0] == 1 { 0 } else { *d });
            }
        }
        op.apply(&[av, bv], &mut rv);
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
        apply(AddOp {}, inps[0], inps[1])
    }
    fn grad(
        &self,
        grads: &mut HashMap<TensorId, Tensor>,
        _tensors: &mut HashMap<TensorId, Tensor>,
        inps: &[TensorId],
        out: TensorId,
    ) {
        assert_eq!(inps.len(), 2);
        grads.insert(inps[0], apply(AddOp {}, &grads[&inps[0]], &grads[&out]));
        grads.insert(inps[1], apply(AddOp {}, &grads[&inps[1]], &grads[&out]));
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
        apply(MatMulOp {}, inps[0], inps[1])
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
            apply(MatMulOp {}, &grads[&out], &tensors[&inps[1]].transpose()),
        );
        grads.insert(
            inps[1],
            apply(MatMulOp {}, &tensors[&inps[0]].transpose(), &grads[&out]),
        );
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
            apply(
                MulOp {},
                &apply(MatMulOp {}, &tensors[&inps[0]].transpose(), &grads[&out]),
                &Tensor::scalar(self.0),
            ),
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
        grads.insert(
            inps[0],
            apply(MulOp {}, &grads[&out], &Tensor::scalar(self.0)),
        );
    }
}

pub trait Operation {
    fn input_dim(&self) -> usize;
    fn output_shape(&self, inp_shapes: &[&[usize]]) -> Vec<usize>;
    fn apply(&self, inps: &[TensorView], out: &mut TensorMutView);
}

pub struct PowOp(f32);
impl Operation for PowOp {
    fn input_dim(&self) -> usize {
        0
    }
    fn output_shape(&self, inp_shapes: &[&[usize]]) -> Vec<usize> {
        assert_eq!(inp_shapes.len(), 1);
        assert_eq!(inp_shapes[0].len(), 0);
        vec![]
    }
    fn apply(&self, inps: &[TensorView], out: &mut TensorMutView) {
        out.set_scalar(inps[0].scalar().powf(self.0));
    }
}

pub struct AddOp;
impl Operation for AddOp {
    fn input_dim(&self) -> usize {
        0
    }
    fn output_shape(&self, inp_shapes: &[&[usize]]) -> Vec<usize> {
        assert_eq!(inp_shapes.len(), 2);
        assert_eq!(inp_shapes[0].len(), 0);
        assert_eq!(inp_shapes[1].len(), 0);
        vec![]
    }
    fn apply(&self, inps: &[TensorView], out: &mut TensorMutView) {
        out.set_scalar(inps[0].scalar() + inps[1].scalar());
    }
}

pub struct MulOp;
impl Operation for MulOp {
    fn input_dim(&self) -> usize {
        0
    }
    fn output_shape(&self, inp_shapes: &[&[usize]]) -> Vec<usize> {
        assert_eq!(inp_shapes.len(), 2);
        assert_eq!(inp_shapes[0].len(), 0);
        assert_eq!(inp_shapes[1].len(), 0);
        vec![]
    }
    fn apply(&self, inps: &[TensorView], out: &mut TensorMutView) {
        out.set_scalar(inps[0].scalar() * inps[1].scalar());
    }
}

pub struct MatMulOp;
impl Operation for MatMulOp {
    fn input_dim(&self) -> usize {
        2
    }
    fn output_shape(&self, inp_shapes: &[&[usize]]) -> Vec<usize> {
        println!("{:?}", inp_shapes);
        assert_eq!(inp_shapes.len(), 2);
        assert_eq!(inp_shapes[0].len(), 2);
        assert_eq!(inp_shapes[1].len(), 2);
        assert_eq!(inp_shapes[0][1], inp_shapes[1][0]);
        vec![inp_shapes[0][0], inp_shapes[1][1]]
    }
    fn apply(&self, inps: &[TensorView], out: &mut TensorMutView) {
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
