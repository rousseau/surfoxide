//! Profondeur sulcale par la fonction potentiel de profondeur (DPF) et sa version invariante d'échelle.
//!
//! DPF (Boucher, Whitesides, Evans 2009) : solution de `(−Δ/2 + α) D = 2 (K − K̄)`, où K est la courbure
//! moyenne et K̄ sa moyenne pondérée par l'aire. C'est un filtre passe-bas de la courbure : α (mm⁻²) fixe
//! l'échelle d'intérêt, une grande valeur ne retenant que les plis fins. D est en mm, positive sur les
//! crêtes et négative au fond des sillons.
//!
//! DPF* (Dieudonné, Auzias, Lefèvre, « Scale-invariant brain morphometry: application to sulcal depth »,
//! Comput Biol Med 212, 2026) : avec la longueur caractéristique L de la surface, on résout la DPF avec
//! `α = α_ref / L²` puis on divise par L. Une homothétie de la surface laisse alors D* inchangée.
//!
//! Choix repris du code des auteurs (github.com/maximedieudonne/DPF-star) et confirmés par eux :
//! - le système discret est `(S/2 + α B) D = 2 B (K − K̄)`, S raideur et B masse éléments finis, K̄
//!   pondérée par la diagonale de B : c'est un choix de formalisation, l'équation 3 de l'article s'écrivant
//!   sans les facteurs 1/2 et 2 ;
//! - L est la racine cubique d'un volume, au choix celui de l'enveloppe convexe (par défaut, comme dans
//!   leur code) ou celui enfermé par la surface (comme dans le texte de l'article) : voir [`Length`] ;
//! - D* est sans dimension ; leur code la multiplie par 100 pour l'affichage, ce qui ne fait pas partie de
//!   la méthode et n'est pas repris ici ;
//! - le signe est conventionnel : avec des normales sortantes, la courbure est positive sur les parties
//!   convexes et D* positive sur les crêtes, négative au fond des sillons ; changer le signe de la courbure
//!   change seulement celui de D*.

use surfoxide_core::TriMesh;

use crate::hull::convex_hull_volume;
use crate::laplacian::fem_matrices;
use crate::sparse::conjugate_gradient;

/// Paramètre de référence retenu par les auteurs (sans dimension) après leur expérience n° 1.
pub const ALPHA_REF: f64 = 500.0;

/// Tolérance relative du gradient conjugué.
const TOLERANCE: f64 = 1e-10;

/// DPF de `mesh` (mm) pour la courbure moyenne `mean_curvature` (mm⁻¹) et le paramètre `alpha` (mm⁻²).
pub fn depth_potential(mesh: &TriMesh, mean_curvature: &[f64], alpha: f64) -> Vec<f64> {
    assert_eq!(mean_curvature.len(), mesh.n_vertices(), "une courbure par sommet");
    assert!(alpha > 0.0, "α doit être strictement positif");
    let (s, b) = fem_matrices(mesh);
    let d = b.diagonal();
    let k_bar = mean_curvature.iter().zip(&d).map(|(k, d)| k * d).sum::<f64>() / d.iter().sum::<f64>();
    let centered: Vec<f64> = mean_curvature.iter().map(|k| k - k_bar).collect();
    let rhs: Vec<f64> = b.mul_vec(&centered).into_iter().map(|x| 2.0 * x).collect();
    let a = s.add_scaled(0.5, &b, alpha);
    conjugate_gradient(&a, &rhs, TOLERANCE, 20 * mesh.n_vertices().max(100)).x
}

/// Volume dont la racine cubique donne la longueur caractéristique L.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Length {
    /// Volume de l'enveloppe convexe des sommets (code de référence).
    #[default]
    ConvexHull,
    /// Volume enfermé par la surface fermée (texte de l'article).
    Enclosed,
}

impl Length {
    /// Longueur caractéristique de `mesh`, en mm.
    pub fn of(self, mesh: &TriMesh) -> f64 {
        match self {
            Length::ConvexHull => convex_hull_volume(mesh.vertices()),
            Length::Enclosed => mesh.signed_volume().abs(),
        }
        .cbrt()
    }
}

/// Profondeur DPF* d'une surface.
#[derive(Debug, Clone)]
pub struct DpfStar {
    /// D*, sans dimension.
    pub depth: Vec<f64>,
    /// Longueur caractéristique L, en mm.
    pub characteristic_length: f64,
    /// Paramètre effectivement utilisé, α_ref / L² (mm⁻²).
    pub alpha: f64,
}

impl DpfStar {
    /// Version absolue DPF*_abs = L · D* (équation 14), en mm.
    pub fn absolute(&self) -> Vec<f64> {
        self.depth.iter().map(|d| d * self.characteristic_length).collect()
    }
}

/// DPF* de `mesh` (fermée, normales sortantes) pour la courbure moyenne `mean_curvature`, `alpha_ref` et
/// la longueur caractéristique `length`.
pub fn dpf_star(mesh: &TriMesh, mean_curvature: &[f64], alpha_ref: f64, length: Length) -> DpfStar {
    let l = length.of(mesh);
    let alpha = alpha_ref / (l * l);
    let depth = depth_potential(mesh, mean_curvature, alpha)
        .into_iter()
        .map(|d| d / l)
        .collect();
    DpfStar {
        depth,
        characteristic_length: l,
        alpha,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curvature::curvature;
    use surfoxide_core::shapes::{icosphere, torus};

    fn scaled(m: &TriMesh, s: f64) -> TriMesh {
        TriMesh::new(
            m.vertices().iter().map(|p| p.map(|x| x * s)).collect(),
            m.faces().to_vec(),
        )
        .unwrap()
    }

    /// Surface plissée : sphère dont le rayon est modulé par une harmonique (sillons et gyrus).
    fn bosselee(level: u32) -> TriMesh {
        let s = icosphere(level, 1.0);
        let v = s
            .vertices()
            .iter()
            .map(|p| {
                let r = 30.0 * (1.0 + 0.08 * (6.0 * p[0]).sin() * (5.0 * p[1]).cos() + 0.05 * (4.0 * p[2]).cos());
                p.map(|x| x * r)
            })
            .collect();
        TriMesh::new(v, s.faces().to_vec()).unwrap()
    }

    #[test]
    fn sphere_profondeur_nulle() {
        let m = icosphere(4, 20.0);
        let k = curvature(&m).mean;
        let d = dpf_star(&m, &k, ALPHA_REF, Length::ConvexHull);
        assert!(
            d.depth.iter().all(|x| x.abs() < 1e-4),
            "max {}",
            d.depth.iter().fold(0.0f64, |a, x| a.max(x.abs()))
        );
    }

    #[test]
    fn invariance_par_homothetie() {
        let m = bosselee(5);
        let reference = {
            let k = curvature(&m).mean;
            dpf_star(&m, &k, ALPHA_REF, Length::ConvexHull)
        };
        for s in [2.0, 3.0, 5.0, 0.5] {
            let ms = scaled(&m, s);
            let d = dpf_star(&ms, &curvature(&ms).mean, ALPHA_REF, Length::ConvexHull);
            assert!((d.characteristic_length / reference.characteristic_length - s).abs() < 1e-9);
            let err = d
                .depth
                .iter()
                .zip(&reference.depth)
                .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
            assert!(err < 1e-8, "écart {err} à l'échelle {s}");
            // La DPF brute, elle, n'est pas invariante.
            let raw = depth_potential(&ms, &curvature(&ms).mean, reference.alpha);
            let raw_ref = depth_potential(&m, &curvature(&m).mean, reference.alpha);
            let ratio = raw.iter().map(|x| x.abs()).sum::<f64>() / raw_ref.iter().map(|x| x.abs()).sum::<f64>();
            assert!((ratio - 1.0).abs() > 0.1, "DPF brute inchangée à l'échelle {s}");
        }
    }

    #[test]
    fn invariance_avec_le_volume_enferme() {
        let m = bosselee(4);
        let reference = dpf_star(&m, &curvature(&m).mean, ALPHA_REF, Length::Enclosed);
        let ms = scaled(&m, 3.0);
        let d = dpf_star(&ms, &curvature(&ms).mean, ALPHA_REF, Length::Enclosed);
        assert!((d.characteristic_length / reference.characteristic_length - 3.0).abs() < 1e-9);
        let err = d
            .depth
            .iter()
            .zip(&reference.depth)
            .fold(0.0f64, |a, (x, y)| a.max((x - y).abs()));
        assert!(err < 1e-8, "écart {err}");
        // Le volume enfermé est plus petit que celui de l'enveloppe : L plus courte.
        let hull = Length::ConvexHull.of(&m);
        assert!(reference.characteristic_length < hull);
    }

    #[test]
    fn signe_cretes_positives_sillons_negatifs() {
        // Tore : l'équateur extérieur est convexe (crête), l'intérieur concave (fond).
        let (big, small) = (10.0, 3.0);
        let m = torus(128, 64, big, small);
        let d = dpf_star(&m, &curvature(&m).mean, ALPHA_REF, Length::ConvexHull);
        for (i, p) in m.vertices().iter().enumerate() {
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if p[2].abs() < 1e-9 {
                if rho > big {
                    assert!(d.depth[i] > 0.0);
                } else {
                    assert!(d.depth[i] < 0.0);
                }
            }
        }
    }

    #[test]
    fn convergence_avec_le_raffinement() {
        // Même surface à deux résolutions : D* aux sommets communs (ceux du niveau grossier, conservés
        // par la subdivision de l'icosphère) doit converger.
        let coarse = bosselee(5);
        let fine = bosselee(6);
        let dc = dpf_star(&coarse, &curvature(&coarse).mean, ALPHA_REF, Length::ConvexHull).depth;
        let df = dpf_star(&fine, &curvature(&fine).mean, ALPHA_REF, Length::ConvexHull).depth;
        let range = df.iter().cloned().fold(f64::MIN, f64::max) - df.iter().cloned().fold(f64::MAX, f64::min);
        let err = dc.iter().zip(&df).map(|(a, b)| (a - b).abs()).sum::<f64>() / dc.len() as f64;
        assert!(err < 0.02 * range, "écart moyen {err} pour une étendue {range}");
    }
}
