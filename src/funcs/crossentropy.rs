use super::{Loss, Tensor, TensorMutOps, TensorOps};

pub struct CrossEntropy {
    classes: u32,
    target: Tensor<u32>,
}
impl CrossEntropy {
    pub fn new(classes: u32, target: Tensor<u32>) -> Box<dyn Loss> {
        Box::new(Self { classes, target })
    }
}
impl Loss for CrossEntropy {
    fn run(&self, inp: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
        let mut loss = Tensor::<f32>::zeros(self.target.shape());
        let mut grad = Tensor::<f32>::zeros(inp.shape());
        for (((mut r, l), o), t) in grad
            .keep_right_mut(1)
            .iter_mut()
            .zip(loss.blob_mut().iter_mut())
            .zip(inp.keep_right(1).inners().iter())
            .zip(self.target.blob().iter())
        {
            let sum = o
                .map_values(|f| f.exp())
                .inners()
                .iter()
                .map(|t| t.scalar())
                .sum::<f32>();
            *l = sum.ln() - o.get(*t as usize).scalar();

            for c in 0..self.classes as usize {
                let val = o.get(c).scalar().exp();
                r.get_mut(c).set(Tensor::scalar(if *t as usize == c {
                    val / sum - 1.0
                } else {
                    val / sum
                }));
            }
        }

        (loss, grad)
    }
}
