//! Surfaces de référence, aux propriétés connues, pour les tests et les démonstrations.

use std::collections::HashMap;

use crate::TriMesh;

/// Icosphère de rayon `radius` centrée à l'origine : icosaèdre subdivisé `level` fois (20 · 4^level faces),
/// normales sortantes.
pub fn icosphere(level: u32, radius: f64) -> TriMesh {
    let t = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut vertices: Vec<[f64; 3]> = vec![
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    let normalize = |p: [f64; 3]| {
        let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / n, p[1] / n, p[2] / n]
    };
    for v in &mut vertices {
        *v = normalize(*v);
    }
    for _ in 0..level {
        let mut midpoints: HashMap<(u32, u32), u32> = HashMap::new();
        let mut midpoint = |a: u32, b: u32, vertices: &mut Vec<[f64; 3]>| -> u32 {
            *midpoints.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let (p, q) = (vertices[a as usize], vertices[b as usize]);
                vertices.push(normalize([
                    (p[0] + q[0]) / 2.0,
                    (p[1] + q[1]) / 2.0,
                    (p[2] + q[2]) / 2.0,
                ]));
                (vertices.len() - 1) as u32
            })
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for &[a, b, c] in &faces {
            let ab = midpoint(a, b, &mut vertices);
            let bc = midpoint(b, c, &mut vertices);
            let ca = midpoint(c, a, &mut vertices);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    let vertices = vertices
        .into_iter()
        .map(|p| p.map(|x| x * radius))
        .collect();
    TriMesh::new(vertices, faces).expect("icosphère valide par construction")
}

/// Tore d'axe z (genre 1) : `n_major` × `n_minor` sommets, rayons `major` et `minor`, normales sortantes.
pub fn torus(n_major: u32, n_minor: u32, major: f64, minor: f64) -> TriMesh {
    use std::f64::consts::TAU;
    let mut vertices = Vec::with_capacity((n_major * n_minor) as usize);
    for i in 0..n_major {
        let u = TAU * i as f64 / n_major as f64;
        for j in 0..n_minor {
            let v = TAU * j as f64 / n_minor as f64;
            let r = major + minor * v.cos();
            vertices.push([r * u.cos(), r * u.sin(), minor * v.sin()]);
        }
    }
    let idx = |i: u32, j: u32| (i % n_major) * n_minor + (j % n_minor);
    let mut faces = Vec::with_capacity((2 * n_major * n_minor) as usize);
    for i in 0..n_major {
        for j in 0..n_minor {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            faces.push([a, b, c]);
            faces.push([a, c, d]);
        }
    }
    TriMesh::new(vertices, faces).expect("tore valide par construction")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icosphere_normales_sortantes_et_volume() {
        let s = icosphere(4, 2.0);
        assert_eq!(s.n_faces(), 20 * 4usize.pow(4));
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 8.0;
        let v = s.signed_volume();
        assert!(v > 0.0, "normales rentrantes");
        assert!(
            (v - exact).abs() / exact < 0.01,
            "volume {v} contre {exact}"
        );
    }

    #[test]
    fn tore_normales_sortantes_et_volume() {
        let t = torus(96, 48, 3.0, 1.0);
        let exact = 2.0 * std::f64::consts::PI.powi(2) * 3.0 * 1.0;
        let v = t.signed_volume();
        assert!(v > 0.0, "normales rentrantes");
        assert!(
            (v - exact).abs() / exact < 0.01,
            "volume {v} contre {exact}"
        );
    }
}
