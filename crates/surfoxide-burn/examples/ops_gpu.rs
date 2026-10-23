//! Temps des opérations élémentaires sur CUDA (diagnostic) :
//! `cargo run --release -p surfoxide-burn --features cuda --example ops_gpu`.
use burn::prelude::{Device, Int, Tensor, TensorData};
use burn::tensor::IndexingUpdateOp;
use std::time::Instant;

fn main() {
    let device = Device::cuda(0);
    let (n, f) = (121_608usize, 243_212usize);
    let v = Tensor::<2>::from_floats(
        TensorData::new((0..n * 3).map(|i| (i % 97) as f32).collect::<Vec<_>>(), [n, 3]),
        &device,
    );
    let idx = Tensor::<1, Int>::from_ints(
        TensorData::new((0..f).map(|i| ((i * 7919) % n) as i32).collect::<Vec<_>>(), [f]),
        &device,
    );
    let sync = |t: Tensor<2>| {
        let _ = t.sum().into_scalar::<f32>();
    };
    let time = |name: &str, op: &dyn Fn() -> Tensor<2>| {
        sync(op());
        let t = Instant::now();
        for _ in 0..20 {
            sync(op());
        }
        println!("{name:40} {:8.2} ms", t.elapsed().as_secs_f64() / 20.0 * 1e3);
    };
    time("somme (synchronisation seule)", &|| v.clone());
    time("select (gather) [F,3]", &|| v.clone().select(0, idx.clone()));
    let vals = v.clone().select(0, idx.clone());
    time("cross [F,3]", &|| vals.clone().cross(vals.clone() * 2.0, 1));
    time("select_assign Add [F,3] -> [N,3]", &|| {
        Tensor::<2>::zeros([n, 3], &device).select_assign(0, idx.clone(), vals.clone(), IndexingUpdateOp::Add)
    });
    time("select_assign Add [F,1] -> [N,1]", &|| {
        Tensor::<2>::zeros([n, 1], &device).select_assign(
            0,
            idx.clone(),
            vals.clone().narrow(1, 0, 1),
            IndexingUpdateOp::Add,
        )
    });
}
