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

pub fn apply_filter<T1: TensorOps<f32>, T2: TensorOps<f32>>(
    image: &T1,
    filter: &T2,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor<f32> {
    assert_eq!(image.dim(), 3);
    assert_eq!(filter.dim(), 3);
    let mut result = Tensor::scalar(0.);
    for (img, fil) in image.iter().zip(filter.iter()) {
        let c = conv(&img, &fil, padding, stride, dilation);
        result = &result + &c;
    }
    result
}

pub fn conv<T1: TensorOps<f32>, T2: TensorOps<f32>>(
    image: &T1,
    kernel: &T2,
    padding: usize,
    stride: usize,
    dilation: usize,
) -> Tensor<f32> {
    assert_eq!(image.dim(), 2);
    let input_height = image.shape()[0];
    let input_width = image.shape()[1];
    assert_eq!(kernel.dim(), 2);
    assert_eq!(kernel.shape()[0], kernel.shape()[1]);
    let kernel_size = kernel.shape()[0];
    let dilated_kernel_size = kernel_size + dilation * (kernel_size - 1);

    let exp_result_height = input_height + 2 * padding - dilated_kernel_size + stride;
    let exp_result_width = input_width + 2 * padding - dilated_kernel_size + stride;
    let result_height = exp_result_height / stride;
    let result_width = exp_result_width / stride;
    assert_eq!(result_height * stride, exp_result_height);
    assert_eq!(result_width * stride, exp_result_width);

    let mut data = Vec::<f32>::new();
    for ih in 0..result_height {
        for iw in 0..result_width {
            let start_h = (ih * stride) as isize - padding as isize;
            let start_w = (iw * stride) as isize - padding as isize;
            let mut sum = 0.;
            for (fh, oh) in (start_h..start_h + dilated_kernel_size as isize)
                .step_by(dilation + 1)
                .enumerate()
            {
                for (fw, ow) in (start_w..start_w + dilated_kernel_size as isize)
                    .step_by(dilation + 1)
                    .enumerate()
                {
                    let value = if oh < 0
                        || oh >= input_height as isize
                        || ow < 0
                        || ow >= input_width as isize
                    {
                        0.
                    } else {
                        image.get(oh as usize).get(ow as usize).scalar()
                    };
                    sum += value * kernel.get(fh).get(fw).scalar();
                }
            }
            data.push(sum);
        }
    }
    Tensor::raw(&[result_height, result_width], data)
}

impl Function for Convolution {
    fn run(&self, inps: &[&Tensor<f32>]) -> Tensor<f32> {
        inps[0].map(3, |t| {
            inps[1].map(3, |f| apply_filter(&t, &f, self.padding, 1, 0))
        })
    }
    fn grad(
        &self,
        inps: &[&Tensor<f32>],
        out: &Tensor<f32>,
        out_grad: &Tensor<f32>,
    ) -> Vec<Tensor<f32>> {
        let batch_size = out_grad.len();
        let filter_out_chans = inps[1].shape()[0];
        let filter_in_chans = inps[1].shape()[1];
        let filter_height = inps[1].shape()[2];
        let filter_width = inps[1].shape()[3];
        let mut grad_data = Vec::<f32>::new();
        for (imgs, fils) in inps[0].iter().zip(out_grad.iter()) {
            for (_, fil) in fils.iter().enumerate() {
                for (_, img) in imgs.iter().enumerate() {
                    grad_data.extend(conv(&img, &fil, self.padding, 1, 0).blob());
                }
            }
        }
        let fil_grad = Tensor::raw(
            &[
                batch_size,
                filter_out_chans,
                filter_in_chans,
                filter_height,
                filter_width,
            ],
            grad_data,
        );
        vec![Tensor::scalar(0.), fil_grad]
    }
}
