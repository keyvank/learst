use super::{Function, Tensor, TensorMutOps, TensorOps};

pub struct Softmax;
impl Softmax {
    pub fn new() -> Box<dyn Function> {
        Box::new(Self {})
    }
}
impl Function for Softmax {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        inps[0].map(1, |l| {
            let sum = l.mapf(|f| f.exp()).iter().map(|t| t.scalar()).sum::<f32>();
            l.mapf(|f| f.exp() / sum)
        })
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let jacobian = out.map(1, |l| {
            let n = l.shape()[0];
            let mut jacobian = Tensor::<f32>::zeros(&[n, n]);
            for i in 0..n {
                for j in 0..n {
                    jacobian
                        .get_mut(i)
                        .get_mut(j)
                        .set(Tensor::scalar(if i == j {
                            let sij = l.get(i).scalar();
                            sij * (1. - sij)
                        } else {
                            let si = l.get(i).scalar();
                            let sj = l.get(j).scalar();
                            -si * sj
                        }));
                }
            }
            jacobian
        });
        let out = &out_grad.unsqueeze(-2) ^ &jacobian;
        vec![out.squeeze(-2).into()]
    }
}
