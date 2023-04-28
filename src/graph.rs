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
    pub grads: HashMap<TensorId, Tensor>,
    computations: Vec<Computation>,
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
    pub fn get_mut(&mut self, id: TensorId) -> &mut Tensor {
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
                self.grads.entry(*inp).or_insert(Tensor::zeros(&shape));
            }
            comp.func
                .grad(&mut self.grads, &mut self.tensors, &comp.inps, id);
        }
    }
    pub fn backward_all(&mut self, id: TensorId) {
        let shape = &self.get(id).shape();
        self.grads.insert(id, Tensor::ones(&shape));

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
                .copy(&result);
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
        let (params, grads): (Vec<&mut Tensor>, Vec<&Tensor>) = self
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
