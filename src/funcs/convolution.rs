use super::{Function, Tensor, TensorOps};

pub struct Convolution {
    kernel_size: usize,
    padding: usize,
    in_chans: usize,
    out_chans: usize,
}
impl Convolution {
    pub fn new(
        kernel_size: usize,
        padding: usize,
        in_chans: usize,
        out_chans: usize,
    ) -> Box<dyn Function> {
        Box::new(Self {
            kernel_size,
            padding,
            in_chans,
            out_chans,
        })
    }
}

impl Function for Convolution {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].map(3, |t| {
            let height = t.shape()[0];
            let width = t.shape()[1];
            let chans = t.shape()[2];
            let mut data = Vec::<f32>::new();
            for ih in 0..height - self.kernel_size + 1 {
                for iw in 0..width - self.kernel_size + 1 {
                    for oh in ih..ih + self.kernel_size {
                        for ow in iw..iw + self.kernel_size {
                            data.extend(t.get(oh).get(ow).blob());
                        }
                    }
                }
            }
            let result = &Tensor::<f32>::raw(
                &[
                    (height - self.kernel_size + 1) * (width - self.kernel_size + 1),
                    chans * self.kernel_size * self.kernel_size,
                ],
                data,
            ) ^ inps[1];
            result
                .reshape(&[
                    height - self.kernel_size + 1,
                    width - self.kernel_size + 1,
                    self.out_chans,
                ])
                .into()
        })
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let inp0 = inps[0].map(3, |t| {
            let height = t.shape()[0];
            let width = t.shape()[1];
            let chans = t.shape()[2];
            let mut data = Vec::<f32>::new();
            for ih in 0..height - self.kernel_size + 1 {
                for iw in 0..width - self.kernel_size + 1 {
                    for oh in ih..ih + self.kernel_size {
                        for ow in iw..iw + self.kernel_size {
                            data.extend(t.get(oh).get(ow).blob());
                        }
                    }
                }
            }
            Tensor::<f32>::raw(
                &[
                    (height - self.kernel_size + 1) * (width - self.kernel_size + 1),
                    chans * self.kernel_size * self.kernel_size,
                ],
                data,
            )
        });
        let out_height = out.shape()[out.dim() - 3];
        let out_width = out.shape()[out.dim() - 2];
        let out_chans = out.shape()[out.dim() - 1];
        let grad = &out
            .reshape(&[0, out_height * out_width, out_chans])
            .transpose()
            ^ &inp0;
        vec![Tensor::zeros(inps[0].shape()), grad]
    }
}
