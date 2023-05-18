use super::{Function, Tensor, TensorOps};

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
        let vars = inps[0].map(self.dims, |l| {
            let avg = l.blob().iter().sum::<f32>() / l.size() as f32;
            let var = (l.blob().iter().map(|f| (f - avg).powi(2)).sum::<f32>() / l.size() as f32
                + EPSILON)
                .sqrt();
            Tensor::constant(l.shape(), var)
        });
        vec![out_grad * &vars]
    }
}
