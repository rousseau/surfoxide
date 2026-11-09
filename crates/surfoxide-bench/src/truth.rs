//! Champs vrais d'un fantôme, portés par la blanche vraie.

use surfoxide_core::adjacency::vertex_normals;
use surfoxide_core::{vec3, TriMesh, VertexField};
use surfoxide_metrics::{curvature, dpf_star, Length};

/// Champs vrais et bilan de leur masque.
#[derive(Debug, Clone)]
pub struct Truth {
    /// `thickness` (mm), `curv` (courbure moyenne, mm⁻¹), `dpfstar` (D*, sans dimension), `mask` (1 : sommet mesuré,
    /// 0 : exclu), dans cet ordre.
    pub fields: Vec<VertexField>,
    /// Triangulation partagée par la blanche et la piale (mêmes faces) : épaisseur par sommets correspondants.
    pub shared: bool,
    /// Sommets de la face médiale.
    pub medial: usize,
    /// Sommets hors face médiale exclus parce que la piale y est à moins de `min_offset` le long de la normale de
    /// la blanche (triangulation partagée seulement).
    pub thin: usize,
}

/// Champs vrais de la blanche `white` : épaisseur (distance au sommet correspondant de la piale si la triangulation
/// est partagée, sinon au plus proche point de la piale), courbure moyenne, D* (α_ref et longueur caractéristique
/// par défaut de `surfoxide-metrics`, ceux de la chaîne), et masque des sommets mesurés : hors face médiale
/// (`medial`) et, si la triangulation est partagée, déplacement de la piale le long de la normale de la blanche
/// d'au moins `min_offset` mm.
pub fn truth(white: &TriMesh, pial: &TriMesh, medial: &[bool], min_offset: f64) -> Truth {
    assert_eq!(
        medial.len(),
        white.n_vertices(),
        "un indicateur de face médiale par sommet de la blanche"
    );
    let shared = white.n_vertices() == pial.n_vertices() && white.faces() == pial.faces();
    let (thickness, offset): (Vec<f64>, Vec<f64>) = if shared {
        let n = vertex_normals(white);
        white
            .vertices()
            .iter()
            .zip(pial.vertices())
            .zip(&n)
            .map(|((w, p), n)| {
                let d = vec3::sub(*p, *w);
                (vec3::norm(d), vec3::dot(d, *n))
            })
            .unzip()
    } else {
        let t = surfoxide_check::vertex_distances(white, pial);
        let o = vec![f64::INFINITY; t.len()];
        (t, o)
    };
    let thin = (0..medial.len())
        .filter(|&i| !medial[i] && offset[i] < min_offset)
        .count();
    let mask: Vec<f64> = (0..medial.len())
        .map(|i| f64::from(u8::from(!medial[i] && offset[i] >= min_offset)))
        .collect();
    let curv = curvature(white);
    let dpf = dpf_star(
        white,
        &curv.mean,
        surfoxide_metrics::depth::ALPHA_REF,
        Length::default(),
    );
    Truth {
        fields: vec![
            VertexField::new("thickness", thickness),
            VertexField::new("curv", curv.mean),
            VertexField::new("dpfstar", dpf.depth),
            VertexField::new("mask", mask),
        ],
        shared,
        medial: medial.iter().filter(|&&m| m).count(),
        thin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn epaisseur_de_deux_spheres_concentriques() {
        let w = icosphere(3, 10.0);
        let p = icosphere(3, 11.5);
        let mut medial = vec![false; w.n_vertices()];
        medial[0] = true;
        let t = truth(&w, &p, &medial, 0.1);
        assert!(t.shared);
        assert!(t.fields[0].values.iter().all(|x| (x - 1.5).abs() < 1e-9));
        assert_eq!((t.medial, t.thin), (1, 0));
        assert_eq!(t.fields[3].values.iter().sum::<f64>() as usize, w.n_vertices() - 1);
        // Triangulations différentes : plus proche point, un peu moins que 1,5 mm (sphère polyédrique).
        let p2 = icosphere(4, 11.5);
        let t2 = truth(&w, &p2, &medial, 0.1);
        assert!(!t2.shared);
        assert!(t2.fields[0].values.iter().all(|x| (x - 1.5).abs() < 0.05));
    }
}
