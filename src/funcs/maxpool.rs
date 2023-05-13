use super::{Function, Tensor, TensorOps};

pub struct MaxPool {
    kernel_size: usize,
}
impl MaxPool {
    pub fn new(kernel_size: usize) -> Box<dyn Function> {
        Box::new(Self { kernel_size })
    }
}

impl Function for MaxPool {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].map(2, |img| {
            let final_height = img.shape()[0] / self.kernel_size;
            let final_width = img.shape()[1] / self.kernel_size;
            let mut blob = Vec::<f32>::new();
            for h in 0..final_height {
                for w in 0..final_width {
                    let mut max = f32::MIN;
                    for i in 0..self.kernel_size {
                        for j in 0..self.kernel_size {
                            let v = img
                                .get(h * self.kernel_size + i)
                                .get(w * self.kernel_size + j)
                                .scalar();
                            if v > max {
                                max = v;
                            }
                        }
                    }
                    blob.push(max);
                }
            }
            Tensor::raw(&[final_height, final_width], blob)
        })
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        _out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let mut blob = Vec::new();
        for (img, out_grad) in inps[0].inners().iter().zip(out_grad.inners().iter()) {
            for (img_chans, out_chans) in img.inners().iter().zip(out_grad.inners().iter()) {
                let final_height = img_chans.shape()[0] / self.kernel_size;
                let final_width = img_chans.shape()[1] / self.kernel_size;
                for h in 0..final_height {
                    for w in 0..final_width {
                        let out = out_chans.get(h).get(w).scalar();
                        let mut grad = 0.;
                        let mut max_x = 0;
                        let mut max_y = 0;
                        let mut max = f32::MIN;
                        for i in 0..self.kernel_size {
                            for j in 0..self.kernel_size {
                                let v = img_chans
                                    .get(h * self.kernel_size + i)
                                    .get(w * self.kernel_size + j)
                                    .scalar();
                                if v > max {
                                    max = v;
                                    max_x = i;
                                    max_y = j;
                                    grad = out;
                                }
                            }
                        }
                        for i in 0..self.kernel_size {
                            for j in 0..self.kernel_size {
                                blob.push(if i == max_x && j == max_y { grad } else { 0. });
                            }
                        }
                    }
                }
            }
        }
        vec![Tensor::raw(inps[0].shape(), blob)]
    }
}
