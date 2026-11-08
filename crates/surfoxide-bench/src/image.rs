//! Étiquettes BOUNTI-19 et T2w synthétique à partir des compteurs de volume partiel.

use crate::rng::Rng;
use crate::voxel::{majority, Counts, Grid, CORTEX, CSF, SUB, WHITE};
use surfoxide_core::Hemisphere;

/// Labels BOUNTI-19 (blanche, cortex, LCR) d'un hémisphère.
pub fn bounti19_labels(hemi: Hemisphere) -> [i16; 3] {
    match hemi {
        Hemisphere::Left => [5, 3, 1],
        Hemisphere::Right => [6, 4, 2],
    }
}

/// Segmentation BOUNTI-19 : classe majoritaire de chaque voxel ([`majority`]).
pub fn labels(counts: &[Counts], hemi: Hemisphere) -> Vec<i16> {
    let l = bounti19_labels(hemi);
    counts.iter().map(|c| l[majority(c)]).collect()
}

/// Fractions de volume (blanche, cortex, LCR) de chaque voxel.
pub fn fractions(counts: &[Counts]) -> [Vec<f32>; 3] {
    let n = (SUB * SUB * SUB) as f32;
    [WHITE, CORTEX, CSF].map(|k| counts.iter().map(|c| c[k] as f32 / n).collect())
}

/// Paramètres de la T2w synthétique.
#[derive(Debug, Clone, PartialEq)]
pub struct T2wParams {
    /// Intensités des tissus purs.
    pub csf: f64,
    pub white: f64,
    pub cortex: f64,
    /// Écart-type du flou gaussien isotrope (mm) ; 0 : pas de flou.
    pub blur: f64,
    /// Rapport signal sur bruit : écart-type du bruit gaussien = intensité de la substance blanche / `snr` ;
    /// 0 : pas de bruit.
    pub snr: f64,
    /// Amplitude du biais multiplicatif lisse : facteur `exp(bias · b(x))`, `b` de moyenne quadratique ≈ 1 ;
    /// 0 : pas de biais.
    pub bias: f64,
}

impl Default for T2wParams {
    fn default() -> Self {
        Self {
            csf: 1.0,
            white: 0.75,
            cortex: 0.35,
            blur: 0.4,
            snr: 25.0,
            bias: 0.1,
        }
    }
}

/// Flou gaussien séparable d'écart-type `sigma` voxels, bords répliqués.
fn blur(data: &mut [f32], dims: [usize; 3], sigma: f64) {
    if sigma <= 0.0 {
        return;
    }
    let r = (3.0 * sigma).ceil() as isize;
    let mut k: Vec<f64> = (-r..=r)
        .map(|x| (-(x * x) as f64 / (2.0 * sigma * sigma)).exp())
        .collect();
    let s: f64 = k.iter().sum();
    k.iter_mut().for_each(|x| *x /= s);
    let strides = [1, dims[0], dims[0] * dims[1]];
    let mut line = Vec::new();
    for a in 0..3 {
        let n = dims[a];
        // Toutes les lignes parallèles à l'axe a : indice de départ de chacune.
        let starts: Vec<usize> = (0..data.len())
            .filter(|&i| (i / strides[a]).is_multiple_of(n))
            .collect();
        for start in starts {
            line.clear();
            line.extend((0..n).map(|t| data[start + t * strides[a]] as f64));
            for t in 0..n {
                let v: f64 = k
                    .iter()
                    .enumerate()
                    .map(|(q, w)| w * line[(t as isize + q as isize - r).clamp(0, n as isize - 1) as usize])
                    .sum();
                data[start + t * strides[a]] = v as f32;
            }
        }
    }
}

/// T2w synthétique : mélange des intensités pures par les fractions, flou, biais lisse, bruit gaussien.
/// `rng_bias` et `rng_noise` sont deux flux distincts.
pub fn t2w(grid: &Grid, counts: &[Counts], p: &T2wParams, rng_bias: &mut Rng, rng_noise: &mut Rng) -> Vec<f32> {
    let [fw, fc, fl] = fractions(counts);
    let mut img: Vec<f32> = (0..counts.len())
        .map(|i| (p.white * fw[i] as f64 + p.cortex * fc[i] as f64 + p.csf * fl[i] as f64) as f32)
        .collect();
    blur(&mut img, grid.dims, p.blur / grid.voxel);
    // Biais : trois ondes planes de directions et de phases aléatoires, de période la plus grande dimension.
    let waves: Vec<([f64; 3], f64)> = (0..3)
        .map(|_| {
            let d = [rng_bias.normal(), rng_bias.normal(), rng_bias.normal()];
            let n = d.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12);
            (d.map(|x| x / n), std::f64::consts::TAU * rng_bias.uniform())
        })
        .collect();
    let extent = grid.dims.iter().map(|&n| n as f64 * grid.voxel).fold(0.0, f64::max);
    let [nx, ny, _] = grid.dims;
    for (idx, v) in img.iter_mut().enumerate() {
        if p.bias != 0.0 {
            let q = [idx % nx, (idx / nx) % ny, idx / (nx * ny)].map(|i| i as f64 * grid.voxel);
            let b: f64 = waves
                .iter()
                .map(|(d, phi)| {
                    let s = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) / extent;
                    (std::f64::consts::TAU * s + phi).cos()
                })
                .sum::<f64>()
                * (2.0f64 / 3.0).sqrt();
            *v = (*v as f64 * (p.bias * b).exp()) as f32;
        }
        if p.snr > 0.0 {
            *v += (p.white / p.snr * rng_noise.normal()) as f32;
        }
    }
    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flou_conserve_la_somme_loin_des_bords() {
        let dims = [21, 21, 21];
        let mut d = vec![0f32; 21 * 21 * 21];
        d[10 + 21 * (10 + 21 * 10)] = 1.0;
        blur(&mut d, dims, 1.5);
        let s: f32 = d.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "somme {s}");
        // Symétrie.
        assert!((d[9 + 21 * (10 + 21 * 10)] - d[11 + 21 * (10 + 21 * 10)]).abs() < 1e-7);
    }
}
