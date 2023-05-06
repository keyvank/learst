use super::{Function, Tensor, TensorMutOps, TensorOps};

pub struct CrossEntropy {
    classes: u32,
    target: Tensor<u32>,
}
impl CrossEntropy {
    pub fn new(classes: u32, target: Tensor<u32>) -> Box<dyn Function> {
        Box::new(Self { classes, target })
    }
}
impl Function for CrossEntropy {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        assert_eq!(inps.len(), 1);
        let mut expected_shape = self.target.shape().to_vec();
        expected_shape.push(self.classes as usize);
        assert_eq!(inps[0].shape(), expected_shape);
        let mut out = Tensor::<f32>::zeros(self.target.shape());
        for ((mut r, o), t) in out
            .reshape_mut(&[0])
            .iter_mut()
            .zip(inps[0].reshape(&[0, self.classes as usize]).iter())
            .zip(self.target.reshape(&[0]).iter())
        {
            r.set(Tensor::scalar(-(o.get(t.scalar() as usize).scalar().ln())));
        }
        out
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        assert_eq!(inps.len(), 1);

        let mut result = Tensor::<f32>::zeros(inps[0].shape());
        for ((mut r, o), t) in result
            .reshape_mut(&[0, self.classes as usize])
            .iter_mut()
            .zip(inps[0].reshape(&[0, self.classes as usize]).iter())
            .zip(self.target.reshape(&[0]).iter())
        {
            for c in 0..self.classes as usize {
                r.get_mut(c)
                    .set(Tensor::scalar(if t.scalar() as usize == c {
                        -1. / o.get(c).scalar()
                    } else {
                        1. / (1. - o.get(c).scalar())
                    }));
            }
        }

        vec![&out_grad.unsqueeze(-1) * &result]
    }
}
