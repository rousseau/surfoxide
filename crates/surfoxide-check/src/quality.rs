//! Qualité d'un maillage : angles des triangles, longueurs d'arête, valences.

use serde::Serialize;
use surfoxide_core::adjacency::{edges, vertex_neighbors};
use surfoxide_core::vec3;
use surfoxide_core::TriMesh;

#[derive(Debug, Clone, Serialize)]
pub struct QualityReport {
    /// Angle minimal moyen des triangles (degrés).
    pub mean_min_angle: f64,
    /// Fraction des triangles dont le plus petit angle est sous 20° et sous 10°.
    pub frac_min_angle_below_20: f64,
    pub frac_min_angle_below_10: f64,
    pub mean_edge_length: f64,
    /// Écart-type relatif des longueurs d'arête.
    pub edge_length_cv: f64,
    /// Fraction des sommets de valence 5, 6 ou 7.
    pub frac_valence_5_to_7: f64,
}

/// Plus petit angle (degrés) du triangle (abc).
pub fn min_angle(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ang = |p: [f64; 3], q: [f64; 3], r: [f64; 3]| {
        let (u, v) = (vec3::sub(q, p), vec3::sub(r, p));
        let c = vec3::dot(u, v) / (vec3::norm(u) * vec3::norm(v));
        c.clamp(-1.0, 1.0).acos().to_degrees()
    };
    ang(a, b, c).min(ang(b, c, a)).min(ang(c, a, b))
}

pub fn quality_report(mesh: &TriMesh) -> QualityReport {
    let v = mesh.vertices();
    let angles: Vec<f64> = mesh
        .faces()
        .iter()
        .map(|f| min_angle(v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]))
        .collect();
    let nf = angles.len().max(1) as f64;
    let lengths: Vec<f64> = edges(mesh)
        .iter()
        .map(|&[a, b]| vec3::dist(v[a as usize], v[b as usize]))
        .collect();
    let ne = lengths.len().max(1) as f64;
    let mean_len = lengths.iter().sum::<f64>() / ne;
    let var = lengths.iter().map(|l| (l - mean_len).powi(2)).sum::<f64>() / ne;
    let nb = vertex_neighbors(mesh);
    let regular = nb.iter().filter(|n| (5..=7).contains(&n.len())).count();
    QualityReport {
        mean_min_angle: angles.iter().sum::<f64>() / nf,
        frac_min_angle_below_20: angles.iter().filter(|&&a| a < 20.0).count() as f64 / nf,
        frac_min_angle_below_10: angles.iter().filter(|&&a| a < 10.0).count() as f64 / nf,
        mean_edge_length: mean_len,
        edge_length_cv: var.sqrt() / mean_len,
        frac_valence_5_to_7: regular as f64 / nb.len().max(1) as f64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn icosphere_de_bonne_qualite() {
        let q = quality_report(&icosphere(3, 1.0));
        assert!(q.mean_min_angle > 50.0);
        assert_eq!(q.frac_min_angle_below_20, 0.0);
        assert_eq!(q.frac_valence_5_to_7, 1.0);
        assert!((min_angle([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) - 45.0).abs() < 1e-9);
    }
}
