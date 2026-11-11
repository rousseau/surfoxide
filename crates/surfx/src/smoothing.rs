//! Lissage à attache aux données de la surface blanche (`--smoothing energy`, Q1.2).
//!
//! Minimise E(x) = w_d Σ A_i ρ(d(x_i)) + w_b Σ A_i |(Δx)_i|² :
//! - d est la distance signée (mm) aux faces de voxels de l'objet corrigé, échantillonnée en trilinéaire ;
//!   `signed_distance` vaut ±h aux centres des voxels voisins de la frontière (pente 2 à travers la face) :
//!   d = φ / 2 est la distance à la face pour une frontière plane ;
//! - ρ(d) = (|d| − r)² au-delà de r = 0,5 voxel, nulle en deçà (la surface reste libre dans la cellule) ;
//! - Δ est le laplacien cotangent à masse condensée (matrices S et B de `surfoxide-metrics`), A_i la masse du
//!   sommet : le second terme est une énergie de flexion (plaque mince), qui lisse sans rétrécir les crêtes
//!   comme le fait le laplacien.
//!
//! Descente par pas proximaux semi-implicites : le terme de flexion est traité implicitement (système
//! symétrique défini positif résolu par gradient conjugué), le terme d'attache explicitement. Chaque pas
//! passe par `safe_update` : un sommet dont le déplacement couperait la surface reste en place.
//!
//! Unités : w_d sans dimension (fixé à 1), w_b en mm⁴ (le rapport des deux termes), pas τ sans dimension.

use surfoxide_core::distance::signed_distance;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::volume::{apply_affine, invert_affine};
use surfoxide_core::{TriMesh, Volume};
use surfoxide_metrics::laplacian::fem_matrices;
use surfoxide_metrics::sparse::{conjugate_gradient, Csr};

/// Réglages du lissage à attache aux données.
#[derive(Debug, Clone, Copy)]
pub struct EnergyParams {
    /// Poids de l'attache aux données (la référence : 1).
    pub w_d: f64,
    /// Poids de la flexion (mm⁴).
    pub w_b: f64,
    /// Pas proximal (sans dimension ; 1/τ pondère l'inertie de chaque sommet).
    pub tau: f64,
    /// Nombre de pas.
    pub steps: usize,
    /// Demi-largeur de la zone sans pénalité (voxel).
    pub dead_zone: f64,
}

impl Default for EnergyParams {
    fn default() -> Self {
        Self {
            w_d: 1.0,
            w_b: 0.03,
            tau: 0.25,
            steps: 20,
            dead_zone: 0.5,
        }
    }
}

/// Bilan du lissage.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyReport {
    /// Sommets maintenus en place par `safe_update` (cumul sur les pas).
    pub held: usize,
    /// Itérations de gradient conjugué (cumul sur les pas et les trois coordonnées).
    pub cg_iterations: usize,
    /// Énergie d'attache et de flexion au départ et à l'arrivée : (attache, flexion).
    pub start: (f64, f64),
    pub end: (f64, f64),
}

/// Masse condensée (somme des lignes de la masse cohérente) et produit `S M⁻¹ S` de la flexion.
fn bending_matrices(mesh: &TriMesh) -> (Csr, Vec<f64>, Csr) {
    let (s, b) = fem_matrices(mesh);
    let ones = vec![1.0; mesh.n_vertices()];
    let mass = b.mul_vec(&ones);
    let sm = scaled_rows(&s, &mass.iter().map(|m| 1.0 / m).collect::<Vec<_>>());
    (s.clone(), mass, product(&s, &sm))
}

/// `diag(d) · a` (les lignes de `a` multipliées par `d`), en triplets.
fn scaled_rows(a: &Csr, d: &[f64]) -> Csr {
    let mut t = Vec::new();
    for (i, di) in d.iter().enumerate() {
        for (j, v) in a.row(i) {
            t.push((i as u32, j, di * v));
        }
    }
    Csr::from_triplets(a.n(), t)
}

/// Produit de deux matrices creuses `a · b`.
fn product(a: &Csr, b: &Csr) -> Csr {
    let mut t = Vec::new();
    for i in 0..a.n() {
        for (k, va) in a.row(i) {
            for (j, vb) in b.row(k as usize) {
                t.push((i as u32, j, va * vb));
            }
        }
    }
    Csr::from_triplets(a.n(), t)
}

/// Distance signée aux faces de voxels (mm) au point `p`.
fn face_distance(phi: &Volume<f64>, inv: &surfoxide_core::volume::Affine, p: Vec3) -> f64 {
    0.5 * phi.trilinear(apply_affine(inv, p))
}

/// Énergie d'attache Σ A_i ρ(d_i) (w_d = 1).
fn data_energy(mesh: &TriMesh, mass: &[f64], phi: &Volume<f64>, dead: f64) -> f64 {
    let inv = invert_affine(phi.affine()).expect("affine inversible");
    mesh.vertices()
        .iter()
        .zip(mass)
        .map(|(&p, &m)| {
            let e = (face_distance(phi, &inv, p).abs() - dead).max(0.0);
            m * e * e
        })
        .sum()
}

/// Énergie de flexion Σ A_i |(Δx)_i|² = xᵀ S M⁻¹ S x.
fn bending_energy(mesh: &TriMesh, k: &Csr) -> f64 {
    let v = mesh.vertices();
    (0..3)
        .map(|c| {
            let x: Vec<f64> = v.iter().map(|p| p[c]).collect();
            let kx = k.mul_vec(&x);
            x.iter().zip(&kx).map(|(a, b)| a * b).sum::<f64>()
        })
        .sum()
}

/// Lisse `mesh` (supposé sans auto-intersection) vers la frontière de l'objet dont `object_phi` est la
/// distance signée (`signed_distance(objet corrigé)`).
pub fn energy_smoothing(mesh: &mut TriMesh, object_phi: &Volume<f64>, p: &EnergyParams) -> EnergyReport {
    let inv = invert_affine(object_phi.affine()).expect("affine inversible");
    let spacing = object_phi.spacing().into_iter().fold(f64::INFINITY, f64::min);
    let dead = p.dead_zone * spacing;
    let eps = 0.25 * spacing;
    let mut report = EnergyReport::default();
    for step in 0..p.steps {
        let n = mesh.n_vertices();
        let (_, mass, k) = bending_matrices(mesh);
        if step == 0 {
            report.start = (data_energy(mesh, &mass, object_phi, dead), bending_energy(mesh, &k));
        }
        // Second membre de la correction δ : −2 w_b K x − w_d g, avec g_i = A_i ∇ρ(d_i) (∇ρ = 2 (|d| − r)₊ sgn d ∇d).
        let v = mesh.vertices().to_vec();
        let attach: Vec<Vec3> = v
            .iter()
            .zip(&mass)
            .map(|(&q, &m)| {
                let d = face_distance(object_phi, &inv, q);
                let e = (d.abs() - dead).max(0.0);
                if e == 0.0 {
                    return [0.0; 3];
                }
                let mut grad = [0.0; 3];
                for (a, gradient) in grad.iter_mut().enumerate() {
                    let (mut hi, mut lo) = (q, q);
                    hi[a] += eps;
                    lo[a] -= eps;
                    *gradient =
                        (face_distance(object_phi, &inv, hi) - face_distance(object_phi, &inv, lo)) / (2.0 * eps);
                }
                vec3::scale(grad, m * 2.0 * e * d.signum())
            })
            .collect();
        // Matrice du système : M / τ + 2 w_b K.
        let inertia = Csr::from_triplets(
            n,
            mass.iter()
                .enumerate()
                .map(|(i, &m)| (i as u32, i as u32, m / p.tau))
                .collect(),
        );
        let a = inertia.add_scaled(1.0, &k, 2.0 * p.w_b);
        let mut next = v.clone();
        for c in 0..3 {
            let x: Vec<f64> = v.iter().map(|q| q[c]).collect();
            let kx = k.mul_vec(&x);
            let b: Vec<f64> = (0..n).map(|i| -2.0 * p.w_b * kx[i] - p.w_d * attach[i][c]).collect();
            let sol = conjugate_gradient(&a, &b, 1e-8, 400);
            report.cg_iterations += sol.iterations;
            for (q, d) in next.iter_mut().zip(&sol.x) {
                q[c] += d;
            }
        }
        report.held += surfoxide_repair::safe_update(mesh, &next);
    }
    let (_, mass, k) = bending_matrices(mesh);
    report.end = (data_energy(mesh, &mass, object_phi, dead), bending_energy(mesh, &k));
    report
}

/// Distance signée de l'objet corrigé, à l'échelle de `energy_smoothing`.
pub fn object_distance(object: &Volume<bool>) -> Volume<f64> {
    signed_distance(object)
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::filter::gaussian_blur;
    use surfoxide_core::topology::topology_report;
    use surfoxide_core::volume::Affine;

    const ID: Affine = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    /// Surface de départ comme `white_surface` (flou 0,25 mm, marching cubes à 0,2 voxel).
    fn depart(mask: &Volume<bool>) -> TriMesh {
        surfoxide_extract::marching_cubes(&gaussian_blur(&signed_distance(mask), 0.25), 0.2)
    }

    fn boule(n: usize, r: f64) -> (Volume<bool>, usize) {
        let c = (n as f64 - 1.0) / 2.0;
        let mut data = vec![false; n * n * n];
        let mut count = 0;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let d = ((x as f64 - c).powi(2) + (y as f64 - c).powi(2) + (z as f64 - c).powi(2)).sqrt();
                    if d <= r {
                        data[(z * n + y) * n + x] = true;
                        count += 1;
                    }
                }
            }
        }
        (Volume::from_vec([n, n, n], data, ID), count)
    }

    /// Rayon moyen et écart type des rayons des sommets (centre de la grille).
    fn rayons(m: &TriMesh, n: usize) -> (f64, f64) {
        let c = (n as f64 - 1.0) / 2.0;
        let r: Vec<f64> = m
            .vertices()
            .iter()
            .map(|p| ((p[0] - c).powi(2) + (p[1] - c).powi(2) + (p[2] - c).powi(2)).sqrt())
            .collect();
        let mean = r.iter().sum::<f64>() / r.len() as f64;
        let var = r.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / r.len() as f64;
        (mean, var.sqrt())
    }

    #[test]
    fn la_boule_voxelisee_garde_son_rayon() {
        let n = 32;
        let (mask, count) = boule(n, 10.0);
        let r_eq = (3.0 * count as f64 / (4.0 * std::f64::consts::PI)).cbrt();
        let mut m = depart(&mask);
        let (r0, s0) = rayons(&m, n);
        let rep = energy_smoothing(&mut m, &signed_distance(&mask), &EnergyParams::default());
        let (r1, s1) = rayons(&m, n);
        let mut t = depart(&mask);
        surfoxide_extract::taubin(&mut t, 10, 0.5, -1.06 * 0.5);
        let (r2, s2) = rayons(&t, n);
        eprintln!(
            "r_eq {r_eq:.4} ; départ {r0:.4} ± {s0:.4} ; energy {r1:.4} ± {s1:.4} ; taubin {r2:.4} ± {s2:.4} ; {rep:?}"
        );
        assert!(topology_report(&m).is_sphere_like());
        assert!(m.signed_volume() > 0.0);
        assert!((r1 - r_eq).abs() <= 0.05, "rayon moyen {r1} contre {r_eq}");
        assert!(s1 < s0, "la surface doit être plus lisse : {s1} contre {s0}");
    }

    /// Rayon moyen des sommets autour de l'axe x (milieu de la tige, loin des extrémités).
    fn rayon_tige(m: &TriMesh, c: f64) -> f64 {
        let r: Vec<f64> = m
            .vertices()
            .iter()
            .filter(|p| (p[0] - c).abs() < 4.0)
            .map(|p| ((p[1] - c).powi(2) + (p[2] - c).powi(2)).sqrt())
            .collect();
        r.iter().sum::<f64>() / r.len() as f64
    }

    #[test]
    fn la_crete_mince_garde_sa_largeur() {
        // Tige de rayon 2 voxels (largeur 4) et de 22 voxels de long : la crête que Taubin amincit.
        let n = 30;
        let c = (n as f64 - 1.0) / 2.0;
        let mut data = vec![false; n * n * n];
        let mut section = 0;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let dedans =
                        (x as f64 - c).abs() <= 11.0 && ((y as f64 - c).powi(2) + (z as f64 - c).powi(2)).sqrt() <= 2.0;
                    data[(z * n + y) * n + x] = dedans;
                    section += usize::from(dedans && x == n / 2);
                }
            }
        }
        let r_eq = (section as f64 / std::f64::consts::PI).sqrt();
        let mask = Volume::from_vec([n, n, n], data, ID);
        let mut m = depart(&mask);
        let r0 = rayon_tige(&m, c);
        let rep = energy_smoothing(&mut m, &signed_distance(&mask), &EnergyParams::default());
        let r1 = rayon_tige(&m, c);
        let mut t = depart(&mask);
        surfoxide_extract::taubin(&mut t, 10, 0.5, -1.06 * 0.5);
        let r2 = rayon_tige(&t, c);
        eprintln!("tige : r_eq {r_eq:.3} ; départ {r0:.3} ; energy {r1:.3} ; taubin {r2:.3} ; {rep:?}");
        assert!(topology_report(&m).is_sphere_like());
        assert!((r1 - r_eq).abs() <= 0.1, "rayon de la tige {r1} contre {r_eq}");
    }
}
