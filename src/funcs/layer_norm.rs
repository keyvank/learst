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
        let jacobian = inps[0].map(self.dims, |l| {
            let n = l.size();
            let nf = n as f32;
            let avg = l.blob().iter().sum::<f32>() / nf;
            let sigma2 = l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / nf + EPSILON;
            let sigma = sigma2.sqrt();
            Tensor::raw(
                &[n, n],
                (0..n * n)
                    .into_par_iter()
                    .map(|work| {
                        let i = work / n;
                        let j = work % n;
                        if i == j {
                            let a = l.blob()[i];
                            ((1. - 1. / nf) * sigma - (a - avg).powi(2) / sigma / nf) / sigma2
                        } else {
                            let a = l.blob()[i];
                            let b = l.blob()[j];
                            (-1. / nf * sigma - (b - avg) * (a - avg) / sigma / nf) / sigma2
                        }
                    })
                    .collect(),
            )
        });
        let mut kept_right_shape = out_grad.shape().to_vec();
        for _ in 0..self.dims - 1 {
            let last = kept_right_shape.pop().unwrap();
            *kept_right_shape.last_mut().unwrap() *= last;
        }
        let kept_right = out_grad.reshape(&kept_right_shape);
        let out = &kept_right.unsqueeze(-2) ^ &jacobian;
        let squeezed = out.squeeze(-2);
        vec![squeezed.reshape(out_grad.shape()).into()]
    }
}
