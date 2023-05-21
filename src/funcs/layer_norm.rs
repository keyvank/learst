use super::{Function, Tensor, TensorOps};
use rayon::prelude::*;

#[derive(Debug)]
pub struct LayerNorm {
    dims: usize,
}
impl LayerNorm {
    pub fn new(dims: usize) -> Box<dyn Function> {
        Box::new(Self { dims })
    }
}

const EPSILON: f32 = 1e-5;

impl Function for LayerNorm {
    fn run(&mut self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].map(self.dims, |l| {
            let avg = l.blob().iter().sum::<f32>() / l.size() as f32;
            let var = (l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / l.size() as f32
                + EPSILON)
                .sqrt();
            Tensor::raw(
                l.shape(),
                l.blob().iter().map(|v| (v - avg) / var).collect(),
            )
        })
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let jacobian = inps[0].map(1, |l| {
            let n = l.shape()[0];
            let avg = l.blob().iter().sum::<f32>() / l.size() as f32;
            let var = (l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / l.size() as f32
                + EPSILON)
                .sqrt();
            Tensor::raw(
                &[n, n],
                (0..n * n)
                    .into_par_iter()
                    .map(|work| {
                        let i = work / n;
                        let j = work % n;
                        if i == j {
                            let a = l.get(i).scalar();
                            ((1. - 1. / (n as f32)) * var - (a - avg).powi(2) / var) / var.powi(2)
                        } else {
                            let a = l.get(i).scalar();
                            let b = l.get(j).scalar();
                            (-1. / (n as f32) * var - (b - avg) * (a - avg) / var) / var.powi(2)
                        }
                    })
                    .collect(),
            )
        });

        let out = &out_grad.unsqueeze(-2) ^ &jacobian;
        vec![out.squeeze(-2).into()]
    }
}
