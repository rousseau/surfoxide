//! Bruit de contour : déplacement des sommets le long de leur normale par un champ aléatoire lisse.
//!
//! Le champ part d'un bruit blanc gaussien par sommet, lissé par un pas implicite de l'équation de la chaleur en
//! éléments finis : `(B + t S) u = B w` (matrices de masse `B` et de raideur `S` de `surfoxide-metrics`), avec
//! `t = ℓ²` ; la longueur d'écran du noyau (`√t`) vaut donc la longueur de corrélation demandée ℓ. Le champ lissé
//! est ensuite recentré et ramené à une moyenne quadratique σ (moyennes pondérées par l'aire des sommets).

use surfoxide_core::adjacency::{vertex_areas, vertex_normals};
use surfoxide_core::{vec3, TriMesh};
use surfoxide_metrics::laplacian::fem_matrices;
use surfoxide_metrics::sparse::conjugate_gradient;

use crate::rng::Rng;

/// Champ lisse de moyenne nulle et de moyenne quadratique `sigma` (mm), de longueur de corrélation `length` (mm).
pub fn smooth_field(mesh: &TriMesh, sigma: f64, length: f64, rng: &mut Rng) -> Vec<f64> {
    let n = mesh.n_vertices();
    let white: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    if sigma == 0.0 {
        return vec![0.0; n];
    }
    let u = if length > 0.0 {
        let (s, b) = fem_matrices(mesh);
        let a = b.add_scaled(1.0, &s, length * length);
        conjugate_gradient(&a, &b.mul_vec(&white), 1e-10, 20 * n.max(100)).x
    } else {
        white
    };
    let area = vertex_areas(mesh);
    let total: f64 = area.iter().sum();
    let mean = u.iter().zip(&area).map(|(u, a)| u * a).sum::<f64>() / total;
    let rms = (u.iter().zip(&area).map(|(u, a)| (u - mean).powi(2) * a).sum::<f64>() / total).sqrt();
    u.iter().map(|u| (u - mean) * sigma / rms).collect()
}

/// Surface déplacée de `field[i]` mm le long de la normale de chaque sommet.
pub fn displace(mesh: &TriMesh, field: &[f64]) -> TriMesh {
    let normals = vertex_normals(mesh);
    let vertices = mesh
        .vertices()
        .iter()
        .zip(&normals)
        .zip(field)
        .map(|((p, n), d)| vec3::add(*p, vec3::scale(*n, *d)))
        .collect();
    TriMesh::new(vertices, mesh.faces().to_vec()).expect("même connectivité")
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn moyenne_nulle_amplitude_sigma_et_lissage() {
        let m = icosphere(4, 20.0);
        let area = vertex_areas(&m);
        let total: f64 = area.iter().sum();
        let rough = smooth_field(&m, 0.2, 0.0, &mut Rng::new(3));
        let smooth = smooth_field(&m, 0.2, 3.0, &mut Rng::new(3));
        for f in [&rough, &smooth] {
            let mean = f.iter().zip(&area).map(|(u, a)| u * a).sum::<f64>() / total;
            let rms = (f.iter().zip(&area).map(|(u, a)| u * u * a).sum::<f64>() / total).sqrt();
            assert!(mean.abs() < 1e-12 && (rms - 0.2).abs() < 1e-12);
        }
        // Écart moyen entre voisins : bien plus faible une fois lissé.
        let jump = |f: &[f64]| {
            let e = surfoxide_core::adjacency::edges(&m);
            e.iter()
                .map(|[a, b]| (f[*a as usize] - f[*b as usize]).abs())
                .sum::<f64>()
                / e.len() as f64
        };
        assert!(
            jump(&smooth) < 0.5 * jump(&rough),
            "{} contre {}",
            jump(&smooth),
            jump(&rough)
        );
    }
}
