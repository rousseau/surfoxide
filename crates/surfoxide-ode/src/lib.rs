//! `surfoxide-ode` : intégration d'une équation différentielle ordinaire `x′ = v(x, t)` sur des positions de
//! sommets (`[N, 3]`), en tenseurs Burn.
//!
//! C'est le cœur des réseaux de déformation à la CortexODE (Ma et al. 2022) : la surface se déplace dans un
//! champ de vitesse appris, le long de trajectoires qui ne se croisent pas tant que le champ est lisse et le
//! pas assez petit. Les schémas sont à pas fixe (Euler explicite, Runge–Kutta d'ordre 4) ; la rétropropagation
//! traverse simplement les pas (« discrétiser puis optimiser »), sans méthode adjointe : avec ~10 pas et
//! ~100 000 sommets, la mémoire de la 2080 Ti suffit.

use burn::prelude::Tensor;

/// Schéma d'intégration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scheme {
    /// Euler explicite, ordre 1 : une évaluation du champ par pas.
    Euler,
    /// Runge–Kutta classique, ordre 4 : quatre évaluations par pas.
    Rk4,
}

/// Un pas de `x` à `x + h v` (Euler) ou de RK4, depuis l'instant `t`.
pub fn step(v: &impl Fn(Tensor<2>, f64) -> Tensor<2>, x: Tensor<2>, t: f64, h: f64, scheme: Scheme) -> Tensor<2> {
    match scheme {
        Scheme::Euler => x.clone() + v(x, t) * h,
        Scheme::Rk4 => {
            let k1 = v(x.clone(), t);
            let k2 = v(x.clone() + k1.clone() * (h / 2.0), t + h / 2.0);
            let k3 = v(x.clone() + k2.clone() * (h / 2.0), t + h / 2.0);
            let k4 = v(x.clone() + k3.clone() * h, t + h);
            x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)
        }
    }
}

/// Positions aux instants `t0 + i (t1 − t0) / steps`, `i = 0..=steps` (la première est `x0`).
pub fn trajectory(
    v: &impl Fn(Tensor<2>, f64) -> Tensor<2>,
    x0: Tensor<2>,
    t0: f64,
    t1: f64,
    steps: usize,
    scheme: Scheme,
) -> Vec<Tensor<2>> {
    assert!(steps > 0, "au moins un pas");
    let h = (t1 - t0) / steps as f64;
    let mut out = Vec::with_capacity(steps + 1);
    out.push(x0);
    for i in 0..steps {
        let x = out.last().expect("non vide").clone();
        out.push(step(v, x, t0 + i as f64 * h, h, scheme));
    }
    out
}

/// Positions à l'instant `t1`.
pub fn integrate(
    v: &impl Fn(Tensor<2>, f64) -> Tensor<2>,
    x0: Tensor<2>,
    t0: f64,
    t1: f64,
    steps: usize,
    scheme: Scheme,
) -> Tensor<2> {
    trajectory(v, x0, t0, t1, steps, scheme)
        .pop()
        .expect("au moins un état")
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::{Device, TensorData};

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data().convert::<f32>().try_into_vec().unwrap()
    }

    /// Rotation d'axe z, de vitesse angulaire `w` : v = w (−y, x, 0).
    fn rotation(w: f64) -> impl Fn(Tensor<2>, f64) -> Tensor<2> {
        move |x: Tensor<2>, _t: f64| {
            let n = x.dims()[0];
            let xs = x.clone().narrow(1, 0, 1);
            let ys = x.clone().narrow(1, 1, 1);
            let zs = Tensor::<2>::zeros([n, 1], &x.device());
            Tensor::cat(vec![ys.neg(), xs, zs], 1) * w
        }
    }

    fn rotation_error(steps: usize, scheme: Scheme) -> f64 {
        let device = Device::flex();
        let x0 = Tensor::<2>::from_floats(TensorData::new(vec![1.0f32, 0.0, 0.5, 0.0, 2.0, -1.0], [2, 3]), &device);
        let w = 1.3;
        let x1 = to_vec(integrate(&rotation(w), x0, 0.0, 1.0, steps, scheme));
        let (c, s) = (w.cos(), w.sin());
        let exact = [c, s, 0.5, -2.0 * s, 2.0 * c, -1.0];
        x1.iter()
            .zip(exact)
            .map(|(a, b)| (*a as f64 - b).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn ordre_de_convergence_sur_une_rotation() {
        // Diviser le pas par 2 divise l'erreur par ~2 (Euler) et ~16 (RK4).
        let (e1, e2) = (rotation_error(20, Scheme::Euler), rotation_error(40, Scheme::Euler));
        let order_euler = (e1 / e2).log2();
        assert!((order_euler - 1.0).abs() < 0.2, "ordre d'Euler {order_euler}");
        let (r1, r2) = (rotation_error(4, Scheme::Rk4), rotation_error(8, Scheme::Rk4));
        let order_rk4 = (r1 / r2).log2();
        assert!(
            (order_rk4 - 4.0).abs() < 0.5,
            "ordre de RK4 {order_rk4} ({r1:e}, {r2:e})"
        );
        assert!(rotation_error(10, Scheme::Rk4) < 1e-5);
    }

    #[test]
    fn dilatation_exacte_et_trajectoire() {
        let device = Device::flex();
        let x0 = Tensor::<2>::from_floats(TensorData::new(vec![1.0f32, -2.0, 3.0], [1, 3]), &device);
        let traj = trajectory(&|x: Tensor<2>, _| x * 0.5, x0, 0.0, 2.0, 20, Scheme::Rk4);
        assert_eq!(traj.len(), 21);
        let x1 = to_vec(traj[20].clone());
        let e = 1.0f64.exp();
        for (a, b) in x1.iter().zip([e, -2.0 * e, 3.0 * e]) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} contre {b}");
        }
    }

    #[test]
    fn gradient_a_travers_l_integration() {
        // v = θ x : x(1) = x0 e^θ, d(Σ x(1))/dθ = Σ x0 · e^θ.
        let device = Device::flex().autodiff();
        let theta = Tensor::<2>::from_floats(TensorData::new(vec![0.3f32], [1, 1]), &device).require_grad();
        let x0 = Tensor::<2>::from_floats(TensorData::new(vec![1.0f32, 2.0, -0.5, 0.5, 0.0, 1.0], [2, 3]), &device);
        let th = theta.clone();
        let v = move |x: Tensor<2>, _t: f64| x * th.clone();
        let loss = integrate(&v, x0, 0.0, 1.0, 10, Scheme::Rk4).sum();
        let g = theta.grad(&loss.backward()).unwrap().into_scalar::<f32>() as f64;
        let expected = 4.0 * 0.3f64.exp();
        assert!((g - expected).abs() < 1e-4, "{g} contre {expected}");
    }

    #[test]
    fn surface_deformee_sans_auto_intersection() {
        // Icosphère dans un champ lisse (cisaillement tournant) : le flot est un difféomorphisme, la surface
        // déformée reste fermée et sans auto-intersection, et son volume se conserve (champ à divergence nulle).
        let device = Device::flex();
        let s = surfoxide_core::shapes::icosphere(3, 5.0);
        let x0 = surfoxide_burn::vertices_tensor(&s, &device);
        let v = |x: Tensor<2>, t: f64| {
            let n = x.dims()[0];
            let ys = x.clone().narrow(1, 1, 1);
            let zs = x.clone().narrow(1, 2, 1);
            let zero = Tensor::<2>::zeros([n, 1], &x.device());
            // v = (sin(y/3)·(1 + t), cos(z/4), 0) : chaque composante ne dépend pas de sa coordonnée.
            Tensor::cat(vec![(ys / 3.0).sin() * (1.0 + t), (zs / 4.0).cos(), zero], 1)
        };
        let x1 = integrate(&v, x0, 0.0, 2.0, 20, Scheme::Rk4);
        let moved = surfoxide_burn::mesh_with_vertices(&s, x1).unwrap();
        let r = surfoxide_check::check_surface(&moved);
        assert!(r.valid, "{r:?}");
        let ratio = moved.signed_volume() / s.signed_volume();
        assert!((ratio - 1.0).abs() < 1e-3, "volume × {ratio}");
    }

    /// Passe avant et rétropropagation à taille réelle sur GPU : blanche de `SURFOXIDE_TEST_SURFACE`, 10 pas
    /// RK4 dans un champ lu par échantillonnage trilinéaire d'un volume, pondéré par un paramètre appris.
    /// `cargo test --release -p surfoxide-ode --features cuda -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gpu"]
    fn passe_avant_et_arriere_sur_gpu() {
        use std::time::Instant;
        let Some(path) = std::env::var_os("SURFOXIDE_TEST_SURFACE") else {
            eprintln!("surface de test absente : SURFOXIDE_TEST_SURFACE");
            return;
        };
        let mesh = surfoxide_io::read_mesh(path).unwrap();
        let device = Device::cuda(0).autodiff();
        // Volume de 160³ voxels de 0,5 mm couvrant la surface : une fonction lisse quelconque.
        let lo = mesh
            .vertices()
            .iter()
            .fold([f64::MAX; 3], |a, p| std::array::from_fn(|k| a[k].min(p[k])));
        let mut aff = surfoxide_core::volume::IDENTITY;
        for (k, row) in aff.iter_mut().take(3).enumerate() {
            row[k] = 0.5;
            row[3] = lo[k] - 5.0;
        }
        let n = 160;
        let mut vol = surfoxide_core::Volume::filled([n, n, n], 0.0, aff);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    vol.set(i, j, k, ((i as f64) / 20.0).sin() + ((j + k) as f64 / 30.0).cos());
                }
            }
        }
        let field = surfoxide_burn::geometry::DeviceVolume::new(&vol, &device);
        let theta = Tensor::<2>::from_floats(TensorData::new(vec![0.1f32, -0.2, 0.05], [1, 3]), &device).require_grad();
        let x0 = surfoxide_burn::vertices_tensor(&mesh, &device);
        let th = theta.clone();
        let v = move |x: Tensor<2>, _t: f64| field.sample(&x) * th.clone();
        let t = Instant::now();
        let x1 = integrate(&v, x0, 0.0, 1.0, 10, Scheme::Rk4);
        let loss = (x1.clone() * x1).sum();
        let g = theta.grad(&loss.backward()).unwrap();
        let g: Vec<f32> = g.into_data().convert::<f32>().try_into_vec().unwrap();
        eprintln!(
            "{} sommets, 10 pas RK4 : passe avant + arrière {:.2} s ; gradient {g:?}",
            mesh.n_vertices(),
            t.elapsed().as_secs_f64()
        );
        assert!(g.iter().all(|x| x.is_finite()));
    }
}
