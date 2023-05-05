use super::*;

impl<V: TensorElement + std::ops::Add<Output = V>> Add for &Tensor<V> {
    type Output = Tensor<V>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        &self.view() + &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn add(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() + other
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn add(self, other: &Tensor<V>) -> Self::Output {
        self + &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Add<Output = V>> Add for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn add(self, other: &TensorView<V>) -> Self::Output {
        let shape = combine_shapes(self.shape(), other.shape());
        let mut result = Tensor::zeros(&shape);
        binary(self.shape(), other.shape(), |r_pos, a_pos, b_pos| {
            let mut a = self.view();
            for i in a_pos.iter() {
                a.zoom(*i);
            }
            let mut b = other.view();
            for i in b_pos.iter() {
                b.zoom(*i);
            }
            let mut r = result.view_mut();
            for i in r_pos.iter() {
                r.zoom(*i);
            }
            r.set(Tensor::scalar(a.scalar() + b.scalar()));
        });
        result
    }
}

impl<V: TensorElement + std::ops::Sub<Output = V>> Sub for &Tensor<V> {
    type Output = Tensor<V>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        &self.view() - &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn sub(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() - other
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn sub(self, other: &Tensor<V>) -> Self::Output {
        self - &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Sub<Output = V>> Sub for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn sub(self, other: &TensorView<V>) -> Self::Output {
        let shape = combine_shapes(self.shape(), other.shape());
        let mut result = Tensor::zeros(&shape);
        binary(self.shape(), other.shape(), |r_pos, a_pos, b_pos| {
            let mut a = self.view();
            for i in a_pos.iter() {
                a.zoom(*i);
            }
            let mut b = other.view();
            for i in b_pos.iter() {
                b.zoom(*i);
            }
            let mut r = result.view_mut();
            for i in r_pos.iter() {
                r.zoom(*i);
            }
            r.set(Tensor::scalar(a.scalar() - b.scalar()));
        });
        result
    }
}

impl<V: TensorElement + std::ops::Mul<Output = V>> Mul for &Tensor<V> {
    type Output = Tensor<V>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        &self.view() * &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&TensorView<'a, V>> for &Tensor<V> {
    type Output = Tensor<V>;
    fn mul(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() * other
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul<&Tensor<V>> for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn mul(self, other: &Tensor<V>) -> Self::Output {
        self * &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V>> Mul for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn mul(self, other: &TensorView<V>) -> Self::Output {
        let shape = combine_shapes(self.shape(), other.shape());
        let mut result = Tensor::zeros(&shape);
        binary(self.shape(), other.shape(), |r_pos, a_pos, b_pos| {
            let mut a = self.view();
            for i in a_pos.iter() {
                a.zoom(*i);
            }
            let mut b = other.view();
            for i in b_pos.iter() {
                b.zoom(*i);
            }
            let mut r = result.view_mut();
            for i in r_pos.iter() {
                r.zoom(*i);
            }
            r.set(Tensor::scalar(a.scalar() * b.scalar()));
        });
        result
    }
}
impl<V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>> BitXor
    for &Tensor<V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        &self.view() ^ &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>>
    BitXor<&TensorView<'a, V>> for &Tensor<V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &TensorView<'a, V>) -> Self::Output {
        &self.view() ^ other
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>>
    BitXor<&Tensor<V>> for &TensorView<'a, V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &Tensor<V>) -> Self::Output {
        self ^ &other.view()
    }
}
impl<'a, V: TensorElement + std::ops::Mul<Output = V> + std::ops::Add<Output = V>> BitXor
    for &TensorView<'a, V>
{
    type Output = Tensor<V>;
    fn bitxor(self, other: &TensorView<V>) -> Self::Output {
        let mat1_shape = self.shape()[self.dim() - 2..].to_vec();
        let mat2_shape = other.shape()[other.dim() - 2..].to_vec();
        let a_shape = self.shape()[..self.dim() - 2].to_vec();
        let b_shape = other.shape()[..other.dim() - 2].to_vec();
        let shape = combine_shapes(&a_shape, &b_shape);
        let mut full_shape = shape.clone();
        full_shape.extend(&[mat1_shape[0], mat2_shape[1]]);
        let mut result = Tensor::zeros(&full_shape);
        binary(&a_shape, &b_shape, |r_pos, a_pos, b_pos| {
            let mut a = self.view();
            for i in a_pos.iter() {
                a.zoom(*i);
            }
            let mut b = other.view();
            for i in b_pos.iter() {
                b.zoom(*i);
            }
            let mut r = result.view_mut();
            for i in r_pos.iter() {
                r.zoom(*i);
            }
            for i in 0..r.shape()[0] {
                for j in 0..r.shape()[1] {
                    let mut sum = Tensor::scalar(
                        <<V as std::ops::Mul>::Output as std::ops::Add>::Output::zero(),
                    );
                    for k in 0..a.shape()[1] {
                        sum = &sum + &(&a.get(i).get(k) * &b.get(k).get(j));
                    }
                    r.get_mut(i).get_mut(j).set(sum);
                }
            }
        });
        result
    }
}

impl<V: TensorElement + std::ops::Neg<Output = V>> Neg for &Tensor<V> {
    type Output = Tensor<V>;
    fn neg(self) -> Self::Output {
        -&self.view()
    }
}
impl<'a, V: TensorElement + std::ops::Neg<Output = V>> Neg for &TensorView<'a, V> {
    type Output = Tensor<V>;
    fn neg(self) -> Self::Output {
        self.mapf(|f| -f)
    }
}
