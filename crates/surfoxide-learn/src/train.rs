//! Entraînement du réseau de déformation et évaluation.
//!
//! Chaque pas prend un hémisphère et un tirage aléatoire de sommets : le champ de vitesse ne dépend que de la
//! position et des caractéristiques du volume, chaque sommet suit donc sa propre trajectoire, et quelques
//! milliers de sommets suffisent par pas (la rétropropagation à travers l'intégration tient en mémoire).

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use burn::module::Module;
use burn::optim::{AdamConfig, GradientsParams};
use burn::prelude::{Device, Int, Tensor, TensorData};
use surfoxide_burn::geometry::DeviceVolume;
use surfoxide_burn::vertices_tensor;

use crate::data::{load, read_index, Entry, Example};
use crate::model::DeformNet;

/// Réglages de l'entraînement.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub lr: f64,
    /// Sommets tirés par pas.
    pub vertices: usize,
    /// Pas RK4 de l'intégration.
    pub ode_steps: usize,
    /// Poids de la perte de distance signée (frontière des tissus) face à la perte sommet à sommet.
    pub w_sdf: f64,
    /// N'utiliser que les `limit` premiers exemples d'entraînement (et de validation) : surapprentissage contrôlé.
    pub limit: Option<usize>,
    /// Sommets tirés par exemple pour l'évaluation.
    pub eval_vertices: usize,
    pub seed: u64,
}

/// Générateur pseudo-aléatoire xorshift (tirages reproductibles sans dépendance).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

/// Tirage de `k` sommets parmi `n` (tous si `k ≥ n`).
fn sample_vertices(n: usize, k: usize, rng: &mut Rng, device: &Device) -> Tensor<1, Int> {
    let idx: Vec<i32> = if k >= n {
        (0..n as i32).collect()
    } else {
        (0..k).map(|_| rng.below(n) as i32).collect()
    };
    let len = idx.len();
    Tensor::from_ints(TensorData::new(idx, [len]), device)
}

/// Pertes d'un exemple sur un tirage de sommets : (perte totale, erreur quadratique moyenne aux sommets de la
/// piale en mm², distance signée quadratique moyenne en mm²).
fn losses(
    model: &DeformNet,
    ex: &Example,
    cfg: &TrainConfig,
    k: usize,
    rng: &mut Rng,
    device: &Device,
) -> (Tensor<1>, Tensor<1>, Tensor<1>) {
    let idx = sample_vertices(ex.white.n_vertices(), k, rng, device);
    let x0 = vertices_tensor(&ex.white, device).select(0, idx.clone());
    let target = vertices_tensor(&ex.pial, device).select(0, idx);
    let features = model.features(&ex.classes, device);
    let x1 = model.deform(&features, x0, cfg.ode_steps);
    let vert = (x1.clone() - target).powf_scalar(2.0).sum_dim(1).mean();
    let sdf = DeviceVolume::new(&ex.sdf, device).sample(&x1).powf_scalar(2.0).mean();
    (vert.clone() + sdf.clone() * cfg.w_sdf, vert, sdf)
}

fn scalar(t: Tensor<1>) -> f64 {
    t.into_scalar::<f32>() as f64
}

/// Évaluation sur des exemples : erreur moyenne aux sommets de la piale (mm), distance moyenne à la frontière
/// (mm), et erreur de la blanche non déformée (référence : la piale est à une épaisseur corticale).
pub fn evaluate(model: &DeformNet, entries: &[Entry], cfg: &TrainConfig, device: &Device) -> Result<[f64; 3], String> {
    let mut rng = Rng(cfg.seed ^ 0x9e37_79b9_7f4a_7c15);
    let (mut err, mut dist, mut base, mut n) = (0.0, 0.0, 0.0, 0.0);
    for e in entries {
        let ex = load(e)?;
        let idx = sample_vertices(ex.white.n_vertices(), cfg.eval_vertices, &mut rng, device);
        let x0 = vertices_tensor(&ex.white, device).select(0, idx.clone());
        let target = vertices_tensor(&ex.pial, device).select(0, idx);
        let features = model.features(&ex.classes, device);
        let x1 = model.deform(&features, x0.clone(), cfg.ode_steps).detach();
        let norm = |d: Tensor<2>| d.powf_scalar(2.0).sum_dim(1).sqrt().mean().reshape([1]);
        err += scalar(norm(x1.clone() - target.clone()));
        base += scalar(norm(x0 - target));
        dist += scalar(DeviceVolume::new(&ex.sdf, device).sample(&x1).abs().mean().reshape([1]));
        n += 1.0;
    }
    Ok([err / n, dist / n, base / n])
}

/// Entraîne un réseau sur le jeu de `root` et écrit dans `out` le journal (`train.tsv`) et les poids
/// (`model.mpk`, au meilleur score de validation).
pub fn train(root: &Path, out: &Path, cfg: &TrainConfig, device: &Device) -> Result<DeformNet, String> {
    std::fs::create_dir_all(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let index = read_index(root)?;
    let pick = |split: &str| -> Vec<Entry> {
        let v: Vec<Entry> = index.iter().filter(|e| e.split == split).cloned().collect();
        match cfg.limit {
            Some(n) => v.into_iter().take(n).collect(),
            None => v,
        }
    };
    let (train_set, val_set) = (pick("train"), pick("val"));
    // En surapprentissage contrôlé, la validation porte sur les exemples d'entraînement eux-mêmes.
    let val_set = if cfg.limit.is_some() {
        train_set.clone()
    } else {
        val_set
    };
    println!(
        "entraînement : {} exemples, validation : {}",
        train_set.len(),
        val_set.len()
    );
    let mut model = DeformNet::new(device);
    let mut optim = AdamConfig::new().init();
    let mut rng = Rng(cfg.seed.max(1));
    let mut log = std::fs::File::create(out.join("train.tsv")).map_err(|e| e.to_string())?;
    writeln!(
        log,
        "epoch\tloss\tvert_mse_mm2\tsdf_mse_mm2\tval_err_mm\tval_dist_mm\tval_base_mm\tseconds"
    )
    .ok();
    let mut best = f64::INFINITY;
    let t0 = Instant::now();
    for epoch in 1..=cfg.epochs {
        let mut order: Vec<usize> = (0..train_set.len()).collect();
        rng.shuffle(&mut order);
        let (mut l, mut v, mut s) = (0.0, 0.0, 0.0);
        for &i in &order {
            let ex = load(&train_set[i])?;
            let (loss, vert, sdf) = losses(&model, &ex, cfg, cfg.vertices, &mut rng, device);
            l += scalar(loss.clone());
            v += scalar(vert);
            s += scalar(sdf);
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
        let n = order.len() as f64;
        let [err, dist, base] = evaluate(&model, &val_set, cfg, device)?;
        let secs = t0.elapsed().as_secs_f64();
        println!(
            "époque {epoch} : perte {:.4} (sommets {:.4} mm², frontière {:.4} mm²) ; validation : erreur {err:.3} mm, distance à la frontière {dist:.3} mm (blanche : {base:.3} mm) ; {secs:.0} s",
            l / n,
            v / n,
            s / n
        );
        writeln!(
            log,
            "{epoch}\t{}\t{}\t{}\t{err}\t{dist}\t{base}\t{secs}",
            l / n,
            v / n,
            s / n
        )
        .ok();
        if err < best {
            best = err;
            model
                .clone()
                .save_file(out.join("model.mpk"))
                .map_err(|e| format!("{}: {e:?}", out.display()))?;
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::volume::IDENTITY;
    use surfoxide_core::Volume;

    /// Exemple synthétique : substance blanche = boule de rayon 4 mm, cortex jusqu'à 5 mm, blanche et piale
    /// sphériques correspondantes (même triangulation), voxels de 0,5 mm.
    fn sphere_example() -> Example {
        let n = 28;
        let mut aff = IDENTITY;
        for (k, row) in aff.iter_mut().take(3).enumerate() {
            row[k] = 0.5;
            row[3] = -0.5 * (n as f64 - 1.0) / 2.0;
        }
        let mut classes = Volume::filled([n, n, n], 0u8, aff);
        let mut sdf = Volume::filled([n, n, n], 0.0, aff);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = classes.voxel_to_world([i as f64, j as f64, k as f64]);
                    let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    classes.set(
                        i,
                        j,
                        k,
                        if r < 4.0 {
                            2
                        } else if r < 5.0 {
                            1
                        } else {
                            0
                        },
                    );
                    sdf.set(i, j, k, r - 5.0);
                }
            }
        }
        let white = icosphere(2, 4.0);
        let pial = icosphere(2, 5.0);
        Example {
            entry: Entry {
                id: "sphère".into(),
                split: "train".into(),
                ga_weeks: f64::NAN,
                prefix: "sphère".into(),
            },
            classes,
            sdf,
            white,
            pial,
        }
    }

    #[test]
    fn quelques_pas_reduisent_la_perte() {
        let device = Device::flex().autodiff();
        let ex = sphere_example();
        let cfg = TrainConfig {
            epochs: 1,
            lr: 3e-3,
            vertices: 256,
            ode_steps: 4,
            w_sdf: 1.0,
            limit: None,
            eval_vertices: 256,
            seed: 7,
        };
        let mut model = DeformNet::new(&device);
        let mut optim = AdamConfig::new().init();
        let mut rng = Rng(3);
        let first = scalar(losses(&model, &ex, &cfg, 256, &mut rng, &device).0);
        let mut last = first;
        for _ in 0..40 {
            let (loss, _, _) = losses(&model, &ex, &cfg, 256, &mut rng, &device);
            last = scalar(loss.clone());
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optim.step(cfg.lr, model, grads);
        }
        // La blanche est à 1 mm de la piale : perte initiale ~1 mm² (sommets) + ~1 mm² (frontière).
        assert!(last < 0.3 * first, "perte {first:.3} → {last:.3}");
    }
}
