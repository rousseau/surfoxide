//! `surfoxide-repair` : suppression des auto-intersections en ne déplaçant que des sommets.
//!
//! La connectivité n'est jamais modifiée : la topologie (genre, orientation) est conservée par
//! construction, contrairement aux réparations qui suppriment et rebouchent des faces (MeshFix).
//!
//! - [`safe_update`] : applique de nouvelles positions à une surface saine, en ramenant à leur ancienne
//!   position les sommets des faces qui se coupent, jusqu'à ce qu'il n'y en ait plus. Sert de garde-fou
//!   aux flots (lissage, gonflement vers la piale).
//! - [`repair_intersections`] : lissage laplacien restreint au voisinage des faces fautives, élargi tant
//!   que des intersections subsistent.

use std::collections::HashSet;

use surfoxide_check::self_intersections;
use surfoxide_core::adjacency::vertex_neighbors;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Sommets des faces qui se coupent.
pub fn intersecting_vertices(mesh: &TriMesh) -> HashSet<u32> {
    let faces = mesh.faces();
    self_intersections(mesh)
        .into_iter()
        .flat_map(|(f, g)| faces[f as usize].into_iter().chain(faces[g as usize]))
        .collect()
}

/// Remplace les positions de `mesh` (supposé sans auto-intersection) par `new`, sauf pour les sommets
/// qu'il faut laisser en place pour ne créer aucune intersection. Rend le nombre de sommets maintenus.
pub fn safe_update(mesh: &mut TriMesh, new: &[Vec3]) -> usize {
    assert_eq!(new.len(), mesh.n_vertices());
    let old = mesh.vertices().to_vec();
    mesh.vertices_mut().copy_from_slice(new);
    let mut held: HashSet<u32> = HashSet::new();
    loop {
        let bad = intersecting_vertices(mesh);
        let fresh: Vec<u32> = bad.into_iter().filter(|v| !held.contains(v)).collect();
        if fresh.is_empty() {
            // Les intersections restantes ne concernent que des sommets déjà ramenés : impossible si
            // l'état initial était sain, puisqu'il suffit de tout ramener. On ramène tout le reste.
            if !intersecting_vertices(mesh).is_empty() {
                mesh.vertices_mut().copy_from_slice(&old);
                return mesh.n_vertices();
            }
            return held.len();
        }
        for v in fresh {
            mesh.vertices_mut()[v as usize] = old[v as usize];
            held.insert(v);
        }
    }
}

/// Bilan d'une réparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepairReport {
    pub initial: usize,
    pub remaining: usize,
    pub rounds: usize,
}

/// Lisse localement les régions qui se coupent jusqu'à disparition des intersections (au plus
/// `max_rounds` tours). Le voisinage lissé grandit d'un anneau tous les deux tours.
pub fn repair_intersections(mesh: &mut TriMesh, max_rounds: usize) -> RepairReport {
    let nb = vertex_neighbors(mesh);
    let initial = self_intersections(mesh).len();
    let mut remaining = initial;
    let mut rounds = 0;
    while remaining > 0 && rounds < max_rounds {
        let mut region: HashSet<u32> = intersecting_vertices(mesh);
        for _ in 0..1 + rounds / 2 {
            let ring: Vec<u32> = region
                .iter()
                .flat_map(|&v| nb[v as usize].iter().copied())
                .collect();
            region.extend(ring);
        }
        for _ in 0..3 {
            let v = mesh.vertices().to_vec();
            for &i in &region {
                let n = &nb[i as usize];
                let mean = n
                    .iter()
                    .fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
                let mean = vec3::scale(mean, 1.0 / n.len() as f64);
                mesh.vertices_mut()[i as usize] = vec3::lerp(v[i as usize], mean, 0.5);
            }
        }
        rounds += 1;
        remaining = self_intersections(mesh).len();
    }
    RepairReport {
        initial,
        remaining,
        rounds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::topology_report;

    /// Sphère dont un sommet est enfoncé à travers la surface.
    fn dented() -> TriMesh {
        let mut s = icosphere(3, 10.0);
        let p = s.vertices()[0];
        s.vertices_mut()[0] = vec3::scale(p, -1.1);
        s
    }

    #[test]
    fn mise_a_jour_sure() {
        let mut s = icosphere(3, 10.0);
        let mut new = s.vertices().to_vec();
        new[0] = vec3::scale(new[0], -1.3); // créerait des intersections
        new[5] = vec3::scale(new[5], 1.02); // inoffensif
        let held = safe_update(&mut s, &new);
        assert!(held >= 1);
        assert!(self_intersections(&s).is_empty());
        assert_eq!(s.vertices()[5], new[5]);
        assert_ne!(s.vertices()[0], new[0]);
    }

    #[test]
    fn reparation_locale() {
        let mut s = dented();
        let before = topology_report(&s);
        assert!(!self_intersections(&s).is_empty());
        let r = repair_intersections(&mut s, 30);
        assert!(r.initial > 0);
        assert_eq!(r.remaining, 0, "{r:?}");
        assert_eq!(topology_report(&s), before);
    }
}
