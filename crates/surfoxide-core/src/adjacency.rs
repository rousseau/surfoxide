//! Relations d'adjacence et grandeurs différentielles élémentaires d'un maillage.

use std::collections::HashMap;

use crate::vec3::{self, Vec3};
use crate::TriMesh;

/// Arêtes non orientées d'un maillage, chacune donnée par (petit indice, grand indice).
pub fn edges(mesh: &TriMesh) -> Vec<[u32; 2]> {
    let mut seen: HashMap<(u32, u32), ()> = HashMap::with_capacity(mesh.n_faces() * 3 / 2);
    let mut out = Vec::with_capacity(mesh.n_faces() * 3 / 2);
    for f in mesh.faces() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if seen.insert(key, ()).is_none() {
                out.push([key.0, key.1]);
            }
        }
    }
    out
}

/// Voisins (par une arête) de chaque sommet, triés et sans doublon.
pub fn vertex_neighbors(mesh: &TriMesh) -> Vec<Vec<u32>> {
    let mut nb: Vec<Vec<u32>> = vec![Vec::new(); mesh.n_vertices()];
    for f in mesh.faces() {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            nb[a as usize].push(b);
            nb[b as usize].push(a);
        }
    }
    for l in &mut nb {
        l.sort_unstable();
        l.dedup();
    }
    nb
}

/// Faces incidentes à chaque sommet.
pub fn vertex_faces(mesh: &TriMesh) -> Vec<Vec<u32>> {
    let mut vf: Vec<Vec<u32>> = vec![Vec::new(); mesh.n_vertices()];
    for (i, f) in mesh.faces().iter().enumerate() {
        for &v in f {
            vf[v as usize].push(i as u32);
        }
    }
    vf
}

/// Normale unitaire de chaque face (vecteur nul pour une face d'aire nulle).
pub fn face_normals(mesh: &TriMesh) -> Vec<Vec3> {
    let v = mesh.vertices();
    mesh.faces()
        .iter()
        .map(|f| {
            vec3::normalize(vec3::triangle_normal(
                v[f[0] as usize],
                v[f[1] as usize],
                v[f[2] as usize],
            ))
        })
        .collect()
}

/// Aire de chaque face, en mm².
pub fn face_areas(mesh: &TriMesh) -> Vec<f64> {
    let v = mesh.vertices();
    mesh.faces()
        .iter()
        .map(|f| {
            0.5 * vec3::norm(vec3::triangle_normal(
                v[f[0] as usize],
                v[f[1] as usize],
                v[f[2] as usize],
            ))
        })
        .collect()
}

/// Normale unitaire de chaque sommet : moyenne des normales des faces incidentes pondérée par leur aire.
pub fn vertex_normals(mesh: &TriMesh) -> Vec<Vec3> {
    let v = mesh.vertices();
    let mut n = vec![[0.0; 3]; mesh.n_vertices()];
    for f in mesh.faces() {
        // La normale non normalisée porte déjà le poids de l'aire.
        let fn_ = vec3::triangle_normal(v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]);
        for &i in f {
            n[i as usize] = vec3::add(n[i as usize], fn_);
        }
    }
    n.into_iter().map(vec3::normalize).collect()
}

/// Aire associée à chaque sommet : un tiers de l'aire des faces incidentes (aire barycentrique).
pub fn vertex_areas(mesh: &TriMesh) -> Vec<f64> {
    let mut a = vec![0.0; mesh.n_vertices()];
    for (f, area) in mesh.faces().iter().zip(face_areas(mesh)) {
        for &i in f {
            a[i as usize] += area / 3.0;
        }
    }
    a
}

/// Longueur moyenne des arêtes.
pub fn mean_edge_length(mesh: &TriMesh) -> f64 {
    let e = edges(mesh);
    if e.is_empty() {
        return 0.0;
    }
    let v = mesh.vertices();
    e.iter()
        .map(|&[a, b]| vec3::dist(v[a as usize], v[b as usize]))
        .sum::<f64>()
        / e.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::icosphere;

    #[test]
    fn icosaedre_valence_cinq() {
        let s = icosphere(0, 1.0);
        assert_eq!(edges(&s).len(), 30);
        assert!(vertex_neighbors(&s).iter().all(|n| n.len() == 5));
        assert!(vertex_faces(&s).iter().all(|f| f.len() == 5));
    }

    #[test]
    fn normales_de_la_sphere_radiales() {
        let s = icosphere(3, 2.0);
        for (p, n) in s.vertices().iter().zip(vertex_normals(&s)) {
            let radial = vec3::normalize(*p);
            assert!(vec3::dot(radial, n) > 0.999);
        }
    }

    #[test]
    fn aires_des_sommets_somment_a_l_aire_totale() {
        let s = icosphere(2, 1.5);
        let total: f64 = vertex_areas(&s).iter().sum();
        assert!((total - s.area()).abs() < 1e-9);
    }
}
