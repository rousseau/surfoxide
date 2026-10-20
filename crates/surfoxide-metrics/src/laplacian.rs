//! Laplace–Beltrami par éléments finis P1 sur une surface triangulaire.
//!
//! Pour un champ `u` linéaire par face, la forme faible de `−Δu = f` s'écrit `S u = B f`, où :
//! - `S` est la matrice de raideur cotangente : `S_ij = −(cot α_ij + cot β_ij) / 2` pour une arête (i, j),
//!   α et β étant les angles opposés à l'arête, et `S_ii = −Σ_j S_ij` ; `S` est semi-définie positive ;
//! - `B` est la matrice de masse cohérente : `A/12` hors diagonale et `A/6` sur la diagonale, pour
//!   chaque face d'aire A.
//!
//! Mêmes conventions que `compute_mesh_laplacian(lap_type="fem")` du code de référence de la DPF*.

use surfoxide_core::{vec3, TriMesh};

use crate::sparse::Csr;

/// Matrices de raideur `S` et de masse `B` de la surface.
pub fn fem_matrices(mesh: &TriMesh) -> (Csr, Csr) {
    let v = mesh.vertices();
    let n = mesh.n_vertices();
    let mut s = Vec::with_capacity(12 * mesh.n_faces());
    let mut b = Vec::with_capacity(9 * mesh.n_faces());
    for f in mesh.faces() {
        let p = f.map(|i| v[i as usize]);
        let area = 0.5 * vec3::norm(vec3::triangle_normal(p[0], p[1], p[2]));
        for k in 0..3 {
            // Angle au sommet k, opposé à l'arête (k+1, k+2).
            let (i, j) = (f[(k + 1) % 3], f[(k + 2) % 3]);
            let e1 = vec3::sub(p[(k + 1) % 3], p[k]);
            let e2 = vec3::sub(p[(k + 2) % 3], p[k]);
            let cross = vec3::norm(vec3::cross(e1, e2));
            let w = if cross > 0.0 {
                0.5 * vec3::dot(e1, e2) / cross
            } else {
                0.0
            };
            s.extend([(i, j, -w), (j, i, -w), (i, i, w), (j, j, w)]);
            b.extend([(i, j, area / 12.0), (j, i, area / 12.0)]);
            b.push((f[k], f[k], area / 6.0));
        }
    }
    (Csr::from_triplets(n, s), Csr::from_triplets(n, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn constantes_dans_le_noyau_et_masse_egale_a_l_aire() {
        let m = icosphere(3, 10.0);
        let (s, b) = fem_matrices(&m);
        let ones = vec![1.0; m.n_vertices()];
        assert!(s.mul_vec(&ones).iter().all(|x| x.abs() < 1e-10));
        let total: f64 = b.mul_vec(&ones).iter().sum();
        assert!((total - m.area()).abs() < 1e-9 * m.area());
    }

    #[test]
    fn fonctions_propres_de_la_sphere() {
        // Sur la sphère de rayon R, la coordonnée z est fonction propre : −Δz = 2/R² z.
        let r = 10.0;
        let m = icosphere(5, r);
        let (s, b) = fem_matrices(&m);
        let z: Vec<f64> = m.vertices().iter().map(|p| p[2]).collect();
        let sz = s.mul_vec(&z);
        let bz = b.mul_vec(&z);
        let rayleigh =
            z.iter().zip(&sz).map(|(a, b)| a * b).sum::<f64>() / z.iter().zip(&bz).map(|(a, b)| a * b).sum::<f64>();
        assert!((rayleigh * r * r - 2.0).abs() < 0.01, "λ R² = {}", rayleigh * r * r);
    }
}
