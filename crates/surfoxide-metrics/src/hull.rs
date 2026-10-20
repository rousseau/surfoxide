//! Enveloppe convexe 3D par quickhull, avec le prédicat exact `orient3d` de Shewchuk.
//!
//! La visibilité d'une face depuis un point est décidée de façon exacte : l'ensemble des faces vues
//! est connexe et son horizon est un cycle simple, ce qui rend la construction robuste aux points
//! coplanaires des surfaces issues d'une grille de voxels. Seul le choix du point le plus éloigné
//! utilise l'arithmétique flottante, sans effet sur la validité du résultat.

use std::collections::HashMap;

use robust::{orient3d, Coord3D};
use surfoxide_core::vec3::{self, Vec3};

fn c3(p: Vec3) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Vrai si `q` est strictement à l'extérieur du plan de la face (a, b, c) orientée vers l'extérieur.
fn sees(p: &[Vec3], f: [u32; 3], q: u32) -> bool {
    orient3d(
        c3(p[f[0] as usize]),
        c3(p[f[1] as usize]),
        c3(p[f[2] as usize]),
        c3(p[q as usize]),
    ) < 0.0
}

struct Face {
    v: [u32; 3],
    alive: bool,
    outside: Vec<u32>,
}

/// Faces de l'enveloppe convexe de `points` (indices dans `points`, normales sortantes) ; `None` si les
/// points sont tous coplanaires.
pub fn convex_hull(points: &[Vec3]) -> Option<Vec<[u32; 3]>> {
    let n = points.len();
    if n < 4 {
        return None;
    }
    // Tétraèdre initial : extrême en x, puis le plus loin, puis le plus loin de la droite, puis du plan.
    let i0 = (0..n).min_by(|&a, &b| points[a][0].total_cmp(&points[b][0]))?;
    let far = |key: &dyn Fn(Vec3) -> f64| (0..n).max_by(|&a, &b| key(points[a]).total_cmp(&key(points[b])));
    let i1 = far(&|q| vec3::dist(q, points[i0]))?;
    let dir = vec3::normalize(vec3::sub(points[i1], points[i0]));
    let i2 = far(&|q| {
        let d = vec3::sub(q, points[i0]);
        vec3::norm(vec3::sub(d, vec3::scale(dir, vec3::dot(d, dir))))
    })?;
    let i3 = (0..n).max_by(|&a, &b| {
        let o = |q: usize| orient3d(c3(points[i0]), c3(points[i1]), c3(points[i2]), c3(points[q])).abs();
        o(a).total_cmp(&o(b))
    })?;
    let o = orient3d(c3(points[i0]), c3(points[i1]), c3(points[i2]), c3(points[i3]));
    if o == 0.0 {
        return None;
    }
    let s = [i0 as u32, i1 as u32, i2 as u32, i3 as u32];
    let mut faces: Vec<Face> = Vec::new();
    let mut edges: HashMap<(u32, u32), usize> = HashMap::new();
    for (f, opposite) in [([0, 1, 2], 3), ([0, 3, 1], 2), ([1, 3, 2], 0), ([2, 3, 0], 1)] {
        let mut v = f.map(|i| s[i]);
        // Le sommet opposé doit être du côté intérieur.
        if sees(points, v, s[opposite]) {
            v.swap(1, 2);
        }
        for e in 0..3 {
            edges.insert((v[e], v[(e + 1) % 3]), faces.len());
        }
        faces.push(Face {
            v,
            alive: true,
            outside: Vec::new(),
        });
    }
    for q in 0..n as u32 {
        if s.contains(&q) {
            continue;
        }
        if let Some(f) = faces.iter_mut().find(|f| sees(points, f.v, q)) {
            f.outside.push(q);
        }
    }
    let mut stack: Vec<usize> = (0..faces.len()).filter(|&f| !faces[f].outside.is_empty()).collect();
    while let Some(fi) = stack.pop() {
        if !faces[fi].alive || faces[fi].outside.is_empty() {
            continue;
        }
        let fv = faces[fi].v.map(|i| points[i as usize]);
        let normal = vec3::triangle_normal(fv[0], fv[1], fv[2]);
        let apex = *faces[fi]
            .outside
            .iter()
            .max_by(|&&a, &&b| {
                let h = |q: u32| vec3::dot(normal, vec3::sub(points[q as usize], fv[0]));
                h(a).total_cmp(&h(b))
            })
            .unwrap();
        // Faces vues depuis l'apex (connexes) et arêtes de l'horizon.
        let mut seen: HashMap<usize, bool> = HashMap::from([(fi, true)]);
        let mut visible = vec![fi];
        let mut horizon: Vec<(u32, u32)> = Vec::new();
        let mut k = 0;
        while k < visible.len() {
            let f = faces[visible[k]].v;
            for e in 0..3 {
                let (a, b) = (f[e], f[(e + 1) % 3]);
                let g = edges[&(b, a)];
                let vis = *seen.entry(g).or_insert_with(|| sees(points, faces[g].v, apex));
                if !vis {
                    horizon.push((a, b));
                } else if !visible.contains(&g) {
                    visible.push(g);
                }
            }
            k += 1;
        }
        let mut orphans: Vec<u32> = Vec::new();
        for &g in &visible {
            let f = &mut faces[g];
            f.alive = false;
            orphans.extend(f.outside.drain(..).filter(|&q| q != apex));
            for e in 0..3 {
                edges.remove(&(f.v[e], f.v[(e + 1) % 3]));
            }
        }
        let first = faces.len();
        for &(a, b) in &horizon {
            let v = [a, b, apex];
            for e in 0..3 {
                edges.insert((v[e], v[(e + 1) % 3]), faces.len());
            }
            faces.push(Face {
                v,
                alive: true,
                outside: Vec::new(),
            });
        }
        for q in orphans {
            if let Some(f) = faces[first..].iter_mut().find(|f| sees(points, f.v, q)) {
                f.outside.push(q);
            }
        }
        stack.extend((first..faces.len()).filter(|&f| !faces[f].outside.is_empty()));
    }
    Some(faces.into_iter().filter(|f| f.alive).map(|f| f.v).collect())
}

/// Volume de l'enveloppe convexe de `points` (0 si les points sont coplanaires).
pub fn convex_hull_volume(points: &[Vec3]) -> f64 {
    let Some(faces) = convex_hull(points) else { return 0.0 };
    // Tétraèdres depuis un point intérieur (le centroïde des sommets) pour limiter les erreurs d'arrondi.
    let c = vec3::scale(
        points.iter().fold([0.0; 3], |s, &p| vec3::add(s, p)),
        1.0 / points.len() as f64,
    );
    faces
        .iter()
        .map(|f| {
            let [a, b, d] = f.map(|i| vec3::sub(points[i as usize], c));
            vec3::dot(a, vec3::cross(b, d)) / 6.0
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::{icosphere, torus};
    use surfoxide_core::{topology_report, TriMesh};

    #[test]
    fn cube_avec_points_interieurs_et_coplanaires() {
        // Grille 5×5×5 : nombreux points coplanaires sur les faces, alignés sur les arêtes.
        let mut p = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    p.push([i as f64, j as f64, k as f64]);
                }
            }
        }
        let faces = convex_hull(&p).unwrap();
        assert!((convex_hull_volume(&p) - 64.0).abs() < 1e-9);
        let m = TriMesh::new(p.clone(), faces).unwrap();
        // Les sommets inutilisés n'interviennent pas dans les faces : on contrôle les arêtes.
        let r = topology_report(&m);
        assert_eq!(r.boundary_edges, 0);
        assert_eq!(r.non_manifold_edges, 0);
        assert_eq!(r.inconsistent_edges, 0);
        assert_eq!(r.euler_characteristic, 2);
    }

    #[test]
    fn sphere_et_tore() {
        let s = icosphere(4, 5.0);
        let v = convex_hull_volume(s.vertices());
        assert!((v - s.signed_volume()).abs() < 1e-9 * v, "{v}");
        // Enveloppe d'un tore (R, a) : volume du cylindre arrondi π(R+a)²·2a − corrections ;
        // on vérifie l'encadrement entre le tore et le cylindre englobant.
        let t = torus(64, 32, 10.0, 3.0);
        let vh = convex_hull_volume(t.vertices());
        let cyl = std::f64::consts::PI * 13.0f64.powi(2) * 6.0;
        assert!(vh > t.signed_volume() && vh < cyl, "{vh}");
    }
}
