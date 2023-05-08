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
    fn run(&self, inp: &Tensor<f32>) -> Tensor<f32> {
        let mut expected_shape = self.target.shape().to_vec();
        expected_shape.push(self.classes as usize);
        assert_eq!(inp.shape(), expected_shape);
        let mut out = Tensor::<f32>::zeros(self.target.shape());
        for ((mut r, o), t) in out
            .reshape_mut(&[0])
            .iter_mut()
            .zip(inp.reshape(&[0, self.classes as usize]).iter())
            .zip(self.target.reshape(&[0]).iter())
        {
            r.set(Tensor::scalar(-(o.get(t.scalar() as usize).scalar().ln())));
        }
        out
    }
    fn grad(&self, inp: &Tensor<f32>, _out: &Tensor<f32>) -> Tensor<f32> {
        let mut result = Tensor::<f32>::zeros(inp.shape());
        for ((mut r, o), t) in result
            .reshape_mut(&[0, self.classes as usize])
            .iter_mut()
            .zip(inp.reshape(&[0, self.classes as usize]).iter())
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

        result
    }
}
