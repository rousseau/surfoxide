//! Contrôles de topologie d'un maillage triangulaire.
//!
//! Ce sont les premiers éléments du contrat de validation (`CLAUDE.md`) : une surface livrée à gyroxide
//! doit être fermée, 2-variété, orientée de façon cohérente, d'un seul tenant et de genre 0.
//! Les auto-intersections et l'imbrication WM/piale, géométriques, viendront dans un module à part.

use std::collections::HashMap;

use crate::TriMesh;

/// Bilan topologique d'un maillage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyReport {
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    /// Sommets référencés par aucune face.
    pub isolated_vertices: usize,
    /// Arêtes portées par une seule face.
    pub boundary_edges: usize,
    /// Arêtes portées par plus de deux faces.
    pub non_manifold_edges: usize,
    /// Arêtes partagées par deux faces qui les parcourent dans le même sens (orientation incohérente).
    pub inconsistent_edges: usize,
    /// Sommets dont les faces incidentes forment plusieurs éventails (« sablier » en un sommet).
    pub non_manifold_vertices: usize,
    /// Composantes connexes (par les faces).
    pub components: usize,
    /// Caractéristique d'Euler V − E + F, calculée sur les sommets référencés.
    pub euler_characteristic: i64,
}

impl TopologyReport {
    /// Vrai si la surface est fermée, 2-variété (par les arêtes et par les sommets) et orientée de façon
    /// cohérente.
    pub fn is_closed_oriented_manifold(&self) -> bool {
        self.boundary_edges == 0
            && self.non_manifold_edges == 0
            && self.inconsistent_edges == 0
            && self.non_manifold_vertices == 0
    }

    /// Genre d'une surface fermée, orientée et d'un seul tenant : g = (2 − χ) / 2. `None` sinon.
    pub fn genus(&self) -> Option<i64> {
        if self.is_closed_oriented_manifold()
            && self.components == 1
            && (2 - self.euler_characteristic) % 2 == 0
        {
            Some((2 - self.euler_characteristic) / 2)
        } else {
            None
        }
    }

    /// Vrai si la surface est homéomorphe à une sphère : la condition topologique attendue par gyroxide.
    pub fn is_sphere_like(&self) -> bool {
        self.genus() == Some(0)
    }
}

/// Calcule le bilan topologique de `mesh`.
pub fn topology_report(mesh: &TriMesh) -> TopologyReport {
    let faces = mesh.faces();

    // Arête non orientée (petit indice, grand indice) -> nombre d'occurrences dans chaque sens.
    let mut edges: HashMap<(u32, u32), (u32, u32)> = HashMap::with_capacity(faces.len() * 3 / 2);
    for f in faces {
        for k in 0..3 {
            let (a, b) = (f[k], f[(k + 1) % 3]);
            let entry = edges.entry((a.min(b), a.max(b))).or_insert((0, 0));
            if a < b {
                entry.0 += 1;
            } else {
                entry.1 += 1;
            }
        }
    }

    let mut boundary_edges = 0;
    let mut non_manifold_edges = 0;
    let mut inconsistent_edges = 0;
    for &(forward, backward) in edges.values() {
        match forward + backward {
            1 => boundary_edges += 1,
            2 => {
                if forward != 1 {
                    inconsistent_edges += 1;
                }
            }
            _ => non_manifold_edges += 1,
        }
    }

    // Composantes connexes : union-find sur les sommets, reliés par les faces.
    let mut parent: Vec<u32> = (0..mesh.n_vertices() as u32).collect();
    fn find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    let mut used = vec![false; mesh.n_vertices()];
    for f in faces {
        for &v in f {
            used[v as usize] = true;
        }
        let r0 = find(&mut parent, f[0]);
        for &v in &f[1..] {
            let r = find(&mut parent, v);
            if r != r0 {
                parent[r as usize] = r0;
            }
        }
    }
    let n_used = used.iter().filter(|&&u| u).count();
    let components = (0..mesh.n_vertices() as u32)
        .filter(|&v| used[v as usize] && find(&mut parent, v) == v)
        .count();

    let non_manifold_vertices = count_non_manifold_vertices(mesh);

    TopologyReport {
        n_vertices: mesh.n_vertices(),
        n_edges: edges.len(),
        n_faces: faces.len(),
        isolated_vertices: mesh.n_vertices() - n_used,
        boundary_edges,
        non_manifold_edges,
        inconsistent_edges,
        non_manifold_vertices,
        components,
        euler_characteristic: n_used as i64 - edges.len() as i64 + faces.len() as i64,
    }
}

/// Compte les sommets dont les faces incidentes ne forment pas un seul éventail : on relie deux faces
/// incidentes au sommet si elles partagent une arête issue de ce sommet, et on compte les groupes.
fn count_non_manifold_vertices(mesh: &TriMesh) -> usize {
    let vf = crate::adjacency::vertex_faces(mesh);
    let faces = mesh.faces();
    let mut count = 0;
    let mut groups: Vec<usize> = Vec::new();
    for (v, incident) in vf.iter().enumerate() {
        if incident.len() < 2 {
            continue;
        }
        let v = v as u32;
        // Union-find local sur les faces incidentes.
        groups.clear();
        groups.extend(0..incident.len());
        fn root(g: &mut [usize], mut x: usize) -> usize {
            while g[x] != x {
                g[x] = g[g[x]];
                x = g[x];
            }
            x
        }
        // Les autres sommets de chaque face incidente : deux faces partageant l'un d'eux partagent une arête.
        let mut by_other: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        for (slot, &fi) in incident.iter().enumerate() {
            for &w in &faces[fi as usize] {
                if w == v {
                    continue;
                }
                match by_other.entry(w) {
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(slot);
                    }
                    std::collections::hash_map::Entry::Occupied(e) => {
                        let (a, b) = (root(&mut groups, *e.get()), root(&mut groups, slot));
                        groups[a] = b;
                    }
                }
            }
        }
        let n_groups = (0..incident.len())
            .filter(|&i| root(&mut groups, i) == i)
            .count();
        if n_groups > 1 {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes::{icosphere, torus};

    #[test]
    fn icosphere_est_une_sphere() {
        for level in 0..4 {
            let r = topology_report(&icosphere(level, 1.0));
            assert_eq!(r.euler_characteristic, 2, "niveau {level}");
            assert!(r.is_sphere_like(), "niveau {level} : {r:?}");
        }
    }

    #[test]
    fn tore_de_genre_un() {
        let r = topology_report(&torus(24, 12, 3.0, 1.0));
        assert_eq!(r.genus(), Some(1));
        assert!(!r.is_sphere_like());
    }

    #[test]
    fn surface_ouverte_detectee() {
        let mesh = TriMesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let r = topology_report(&mesh);
        assert_eq!(r.boundary_edges, 3);
        assert_eq!(r.genus(), None);
    }

    #[test]
    fn orientation_incoherente_detectee() {
        let sphere = icosphere(1, 1.0);
        let mut faces = sphere.faces().to_vec();
        faces[0].swap(1, 2);
        let mesh = TriMesh::new(sphere.vertices().to_vec(), faces).unwrap();
        let r = topology_report(&mesh);
        assert_eq!(r.inconsistent_edges, 3);
        assert!(!r.is_closed_oriented_manifold());
    }

    #[test]
    fn sablier_en_un_sommet_detecte() {
        // Deux tétraèdres qui ne partagent qu'un sommet : arêtes manifold, sommet non manifold.
        let tet = |o: [f64; 3], s: f64| {
            vec![
                o,
                [o[0] + s, o[1], o[2]],
                [o[0], o[1] + s, o[2]],
                [o[0], o[1], o[2] + s],
            ]
        };
        let mut vertices = tet([0.0; 3], 1.0);
        vertices.extend(tet([0.0; 3], -1.0).into_iter().skip(1));
        let t = |a: u32, b: u32, c: u32, d: u32| [[a, c, b], [a, b, d], [a, d, c], [b, c, d]];
        let mut faces = t(0, 1, 2, 3).to_vec();
        faces.extend(t(0, 4, 5, 6).iter().map(|f| [f[0], f[2], f[1]]));
        let r = topology_report(&TriMesh::new(vertices, faces).unwrap());
        assert_eq!(r.non_manifold_edges, 0);
        assert_eq!(r.non_manifold_vertices, 1);
        assert!(!r.is_closed_oriented_manifold());
    }

    #[test]
    fn deux_spheres_font_deux_composantes() {
        let a = icosphere(1, 1.0);
        let n = a.n_vertices() as u32;
        let mut vertices = a.vertices().to_vec();
        vertices.extend(a.vertices().iter().map(|p| [p[0] + 5.0, p[1], p[2]]));
        let mut faces = a.faces().to_vec();
        faces.extend(a.faces().iter().map(|f| f.map(|i| i + n)));
        let r = topology_report(&TriMesh::new(vertices, faces).unwrap());
        assert_eq!(r.components, 2);
        assert_eq!(r.genus(), None);
    }
}
