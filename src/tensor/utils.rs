use super::*;

pub fn reshape(size: usize, shape: &[usize]) -> Vec<usize> {
    let mut final_shape = shape.to_vec();
    if shape[0] == 0 && shape[1..].iter().all(|s| *s != 0) {
        let mul = shape[1..].iter().fold(1, |c, s| c * s);
        final_shape[0] = size / mul;
    } else if shape[shape.len() - 1] == 0 && shape[0..shape.len() - 1].iter().all(|s| *s != 0) {
        let mul = shape[..shape.len() - 1].iter().fold(1, |c, s| c * s);
        final_shape[shape.len() - 1] = size / mul;
    } else {
        assert!(shape.iter().all(|s| *s != 0));
    };
    final_shape
}

fn combine_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let shape_len = std::cmp::max(a.len(), b.len());
    let mut shape = Vec::new();
    for i in 0..shape_len {
        shape.insert(
            0,
            if i >= a.len() {
                b[b.len() - 1 - i]
            } else if i >= b.len() {
                a[a.len() - 1 - i]
            } else {
                let (a, b) = (a[a.len() - 1 - i], b[b.len() - 1 - i]);
                if a == b {
                    a
                } else if a == 1 {
                    b
                } else if b == 1 {
                    a
                } else {
                    panic!("Cannot be combined!")
                }
            },
        );
    }
    shape
}

pub fn combine_map<
    'a,
    V: TensorElement,
    W: TensorElement,
    X: TensorElement,
    T1: TensorOps<V>,
    T2: TensorOps<W>,
    F: Fn(&TensorView<'_, V>, &TensorView<'_, W>) -> Tensor<X> + Sync + Send,
>(
    t1: &T1,
    t2: &T2,
    dims: usize,
    f: F,
) -> Tensor<X> {
    fn calc_shape(pos: &[usize], shape: &[usize]) -> Vec<usize> {
        pos[pos.len() - shape.len()..]
            .iter()
            .zip(shape.iter())
            .map(|(p, s)| if *s == 1 { 0 } else { *p })
            .collect()
    }
    let mut shape = combine_shapes(
        &t1.shape()[..t1.dim() - dims],
        &t2.shape()[..t2.dim() - dims],
    );
    let works = shape.iter().fold(1, |a, b| a * b);
    let tensors = (0..works)
        .into_par_iter()
        .map(|mut i| {
            let mut result = vec![];
            for s in shape.iter().rev() {
                result.insert(0, i % s);
                i = i / s;
            }
            let t1_pos = calc_shape(&result, &t1.shape()[..t1.dim() - dims]);
            let t2_pos = calc_shape(&result, &t2.shape()[..t2.dim() - dims]);
            let mut t1_view = t1.view();
            for i in t1_pos.iter() {
                t1_view.zoom(*i);
            }
            let mut t2_view = t2.view();
            for i in t2_pos.iter() {
                t2_view.zoom(*i);
            }
            f(&t1_view, &t2_view)
        })
        .collect::<Vec<_>>();
    let t_shape = tensors.first().unwrap().shape().to_vec();
    assert!(tensors.iter().all(|t| t.shape() == t_shape));
    let data = tensors.into_iter().map(|t| t.blob).flatten().collect();
    shape.extend(t_shape);
    Tensor::raw(&shape, data)
}

pub fn shuffle_batch<
    R: Rng,
    V: TensorElement,
    W: TensorElement,
    T1: TensorOps<V>,
    T2: TensorOps<W>,
>(
    rng: &mut R,
    t1: &T1,
    t2: &T2,
) -> (Tensor<V>, Tensor<W>) {
    let mut batch = t1
        .inners()
        .into_iter()
        .zip(t2.inners().into_iter())
        .collect::<Vec<_>>();
    batch.shuffle(rng);
    let (t1_dat, t2_dat): (Vec<TensorView<'_, V>>, Vec<TensorView<'_, W>>) =
        batch.into_iter().unzip();
    (
        Tensor::raw(
            t1.shape(),
            t1_dat
                .into_iter()
                .map(|t| t.blob().to_vec())
                .flatten()
                .collect(),
        ),
        Tensor::raw(
            t2.shape(),
            t2_dat
                .into_iter()
                .map(|t| t.blob().to_vec())
                .flatten()
                .collect(),
        ),
    )
}
