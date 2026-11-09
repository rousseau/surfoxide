//! Profil T2w le long des normales d'une surface (plan qualité, Q0.4) : où est le bord de l'image par rapport à la
//! surface ?
//!
//! Pour chaque sommet : T2w interpolée (trilinéaire) le long de la normale sortante de −`half` à +`half` mm par pas
//! `step` ; dérivée lissée (profil convolué par la dérivée d'une gaussienne d'écart-type `sigma` mm, bords
//! répliqués) ; extremum du signe attendu (blanche : minimum, la substance blanche étant plus claire que le cortex en
//! T2 ; piale : maximum, le LCR étant plus clair que le cortex), affiné par une parabole sur trois échantillons.
//! Décalage signé = abscisse de l'extremum (+ vers l'extérieur) ; amplitude = |dérivée| à l'extremum ; bruit local =
//! écart-type robuste du profil brut (1,4826 × MAD des différences premières / √2) propagé au filtre dérivée ; bord net
//! si amplitude ≥ 2 × ce bruit.

use rayon::prelude::*;
use surfoxide_core::adjacency::vertex_normals;
use surfoxide_core::vec3;
use surfoxide_core::{TriMesh, Volume};

/// Surface profilée.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    White,
    Pial,
}

/// Réglages du profil.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileParams {
    pub half: f64,
    pub step: f64,
    pub sigma: f64,
}

impl Default for ProfileParams {
    fn default() -> Self {
        Self {
            half: 1.5,
            step: 0.05,
            sigma: 0.1,
        }
    }
}

/// Résultat par sommet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Edge {
    /// Décalage signé du bord (mm, + vers l'extérieur).
    pub offset: f64,
    /// |dérivée lissée| à l'extremum (intensité par mm).
    pub amplitude: f64,
    /// Bruit de la dérivée lissée (intensité par mm).
    pub noise: f64,
}

impl Edge {
    pub fn sharp(&self) -> bool {
        self.amplitude >= 2.0 * self.noise
    }
}

/// Filtre dérivée de gaussienne : `d[t] = Σ_j k[j + r] · s[t + j]`, `j` de −r à r, avec `k_j ∝ j · g(j)` normalisé
/// pour qu'une rampe de pente 1 par mm donne 1.
fn derivative_kernel(sigma: f64, step: f64) -> (Vec<f64>, isize) {
    let s = sigma / step;
    let r = (4.0 * s).ceil() as isize;
    let raw: Vec<f64> = (-r..=r)
        .map(|j| j as f64 * (-(j * j) as f64 / (2.0 * s * s)).exp())
        .collect();
    let ramp: f64 = (-r..=r).zip(&raw).map(|(j, k)| j as f64 * k).sum();
    (raw.iter().map(|k| k / (ramp * step)).collect(), r)
}

fn median(mut x: Vec<f64>) -> f64 {
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(f64::total_cmp);
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        0.5 * (x[n / 2 - 1] + x[n / 2])
    }
}

/// Bord le long d'un profil échantillonné (`samples[k]` à l'abscisse `−half + k · step`).
pub fn edge(samples: &[f64], kind: Kind, p: &ProfileParams) -> Edge {
    let (k, r) = derivative_kernel(p.sigma, p.step);
    let n = samples.len() as isize;
    let d: Vec<f64> = (0..n)
        .map(|t| {
            k.iter()
                .enumerate()
                .map(|(q, c)| c * samples[(t + q as isize - r).clamp(0, n - 1) as usize])
                .sum::<f64>()
        })
        .collect();
    let sign = match kind {
        Kind::White => -1.0,
        Kind::Pial => 1.0,
    };
    let (best, _) = d.iter().enumerate().fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
        if sign * v > bv {
            (i, sign * v)
        } else {
            (bi, bv)
        }
    });
    let mut x = best as f64;
    if best > 0 && best + 1 < d.len() {
        let (a, b, c) = (sign * d[best - 1], sign * d[best], sign * d[best + 1]);
        let den = a - 2.0 * b + c;
        if den < 0.0 {
            x += (0.5 * (a - c) / den).clamp(-0.5, 0.5);
        }
    }
    let diffs: Vec<f64> = samples.windows(2).map(|w| w[1] - w[0]).collect();
    let med = median(diffs.clone());
    let mad = median(diffs.iter().map(|v| (v - med).abs()).collect());
    let sigma_i = 1.4826 * mad / std::f64::consts::SQRT_2;
    let gain = k.iter().map(|c| c * c).sum::<f64>().sqrt();
    Edge {
        offset: -p.half + x * p.step,
        amplitude: (d[best]).abs(),
        noise: sigma_i * gain,
    }
}

/// Bords de tous les sommets de `mesh` dans l'image `image` ; `None` pour les sommets hors de `mask`.
pub fn profile(
    image: &Volume<f64>,
    mesh: &TriMesh,
    kind: Kind,
    mask: Option<&[bool]>,
    p: &ProfileParams,
) -> Vec<Option<Edge>> {
    let normals = vertex_normals(mesh);
    let inv = surfoxide_core::volume::invert_affine(image.affine()).expect("affine inversible");
    let n = (2.0 * p.half / p.step).round() as usize + 1;
    (0..mesh.n_vertices())
        .into_par_iter()
        .map(|i| {
            if mask.is_some_and(|m| !m[i]) {
                return None;
            }
            let (v, nrm) = (mesh.vertices()[i], normals[i]);
            let samples: Vec<f64> = (0..n)
                .map(|k| {
                    let t = -p.half + k as f64 * p.step;
                    let w = vec3::add(v, vec3::scale(nrm, t));
                    image.trilinear(surfoxide_core::volume::apply_affine(&inv, w))
                })
                .collect();
            Some(edge(&samples, kind, p))
        })
        .collect()
}

/// Résumé : médiane et écart interquartile du décalage, part des profils sans bord net, nombre de profils.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub median: f64,
    pub iqr: f64,
    pub not_sharp: f64,
    pub n: usize,
}

pub fn summary(edges: &[Option<Edge>]) -> Summary {
    let e: Vec<&Edge> = edges.iter().flatten().collect();
    let mut off: Vec<f64> = e.iter().map(|e| e.offset).collect();
    off.sort_by(f64::total_cmp);
    let q = |f: f64| {
        off.get(((off.len().max(1) - 1) as f64 * f).round() as usize)
            .copied()
            .unwrap_or(f64::NAN)
    };
    Summary {
        median: median(off.clone()),
        iqr: q(0.75) - q(0.25),
        not_sharp: e.iter().filter(|e| !e.sharp()).count() as f64 / e.len().max(1) as f64,
        n: e.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Marche lisse (erf) centrée en `x0`, de `lo` (intérieur) à `hi` (extérieur).
    fn step_profile(x0: f64, lo: f64, hi: f64, p: &ProfileParams) -> Vec<f64> {
        let n = (2.0 * p.half / p.step).round() as usize + 1;
        (0..n)
            .map(|k| {
                let x = -p.half + k as f64 * p.step;
                let t = 0.5 * (1.0 + ((x - x0) / 0.3).tanh());
                lo + (hi - lo) * t
            })
            .collect()
    }

    #[test]
    fn bord_retrouve_au_centieme_de_millimetre() {
        let p = ProfileParams::default();
        for x0 in [-0.4, 0.0, 0.123, 0.6] {
            let w = edge(&step_profile(x0, 0.75, 0.35, &p), Kind::White, &p);
            let q = edge(&step_profile(x0, 0.35, 1.0, &p), Kind::Pial, &p);
            assert!((w.offset - x0).abs() < 0.01, "blanche {x0} → {}", w.offset);
            assert!((q.offset - x0).abs() < 0.01, "piale {x0} → {}", q.offset);
            assert!(w.sharp() && q.sharp());
            // Amplitude : pente maximale de la marche, 0,4 / 0,6 par mm, un peu adoucie par le lissage de 0,1 mm.
            assert!(
                w.amplitude < 0.4 / 0.6 && w.amplitude > 0.85 * 0.4 / 0.6,
                "{}",
                w.amplitude
            );
        }
    }

    #[test]
    fn profil_plat_bruite_sans_bord_net() {
        let p = ProfileParams::default();
        let mut r = crate::rng::Rng::new(5);
        let flat: Vec<f64> = (0..61).map(|_| 0.5 + 0.05 * r.normal()).collect();
        let e = edge(&flat, Kind::White, &p);
        assert!(!e.sharp(), "{e:?}");
    }
}
