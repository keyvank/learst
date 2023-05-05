use crate::optimizer::Optimizer;
use crate::tensor::*;
use crate::Function;
use std::collections::{HashMap, HashSet};

pub type TensorId = usize;

struct Computation {
    inps: Vec<TensorId>,
    out: TensorId,
    func: Box<dyn Function>,
}

pub struct Graph {
    grads: HashMap<TensorId, Tensor<f32>>,
    computations: Vec<Computation>,
    tensors: HashMap<TensorId, Tensor<f32>>,
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
    pub fn alloc(&mut self, tensor: Tensor<f32>) -> TensorId {
        let id = self.next_tensor_id;
        self.tensors.insert(id, tensor);
        self.next_tensor_id += 1;
        id
    }
    pub fn zero_grad(&mut self) {
        self.grads.clear();
    }
    pub fn add_grad<T: TensorOps<f32>>(&mut self, id: TensorId, add: T) {
        let mut shape = self.get(id).shape().to_vec();
        let grad = self.grads.entry(id).or_insert(Tensor::zeros(&shape));
        if add.dim() >= shape.len() {
            shape.insert(0, 0);
            for t in add.reshape(&shape).iter() {
                *grad = &*grad + &t;
            }
        } else {
            *grad = &*grad + &add.view();
        }
    }
    pub fn get(&self, id: TensorId) -> &Tensor<f32> {
        self.tensors.get(&id).expect("Tensor not found!")
    }
    pub fn get_mut(&mut self, id: TensorId) -> &mut Tensor<f32> {
        self.tensors.get_mut(&id).expect("Tensor not found!")
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
        if let Some(comp) = self.computations.iter().find(|c| c.out == id) {
            for inp in comp.inps.iter() {
                let shape = self.get(*inp).shape().to_vec();
                self.grads
                    .entry(*inp)
                    .or_insert(Tensor::<f32>::zeros(&shape));
            }
            let inps = comp
                .inps
                .iter()
                .map(|id| &self.tensors[id])
                .collect::<Vec<_>>();
            let result_out = &self.tensors[&id];
            let grad_out = &self.grads[&id];
            let grads = comp.func.grad(&inps, result_out, grad_out);
            for (id, grad) in comp.inps.clone().into_iter().zip(grads.into_iter()) {
                self.add_grad(id, grad);
            }
        }
    }
    pub fn backward_all(&mut self, id: TensorId) {
        let shape = &self.get(id).shape();
        self.grads.insert(id, Tensor::<f32>::ones(&shape));

        for t in self.topology(id) {
            self.backward(t);
        }
    }
    pub fn forward(&mut self) {
        for c in self.computations.iter() {
            let tensors = c
                .inps
                .iter()
                .map(|id| self.tensors.get(id).expect("Tensor not found!"))
                .collect::<Vec<_>>();
            let result = c.func.run(&tensors);
            self.tensors
                .get_mut(&c.out)
                .expect("Tensor not found!")
                .set(result);
        }
    }
    pub fn call(&mut self, f: Box<dyn Function>, tensor_ids: &[TensorId]) -> TensorId {
        let tensors = tensor_ids
            .iter()
            .map(|id| self.tensors.get(id).expect("Tensor not found!"))
            .collect::<Vec<_>>();
        let child = self.alloc(f.run(&tensors));
        self.computations.push(Computation {
            func: f,
            out: child,
            inps: tensor_ids.to_vec(),
        });
        for parent in tensor_ids {
            self.parents.entry(child).or_default().insert(*parent);
        }
        child
    }
    pub fn optimize(&mut self, opt: &mut Box<dyn Optimizer>, params: &HashSet<TensorId>) {
        let (params, grads): (Vec<&mut Tensor<f32>>, Vec<&Tensor<f32>>) = self
            .tensors
            .iter_mut()
            .filter(|(id, _)| params.contains(id))
            .map(|(id, params)| {
                let grad = self.grads.get(id).expect("Tensor not found!");
                (params, grad)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();
        opt.step(params, grads);
    }
}
