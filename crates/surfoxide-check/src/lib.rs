//! `surfoxide-check` : contrôles géométriques du contrat de surface.
//!
//! - auto-intersections (hiérarchie de boîtes [`bvh::Bvh`] et tests exacts de [`predicates`]) ;
//! - intersections entre deux surfaces (WM et piale) ;
//! - distances d'une surface à l'autre (épaisseur minimale), inclusion par l'indice d'enroulement ;
//! - qualité du maillage ([`quality`]) ;
//! - rapports [`SurfaceReport`] et [`PairReport`], sérialisables en JSON.
//!
//! Deux triangles qui partagent un sommet `v` et ne sont pas coplanaires se coupent ailleurs qu'en `v`
//! si et seulement si l'arête opposée à `v` de l'un rencontre l'autre : l'intersection est un segment issu
//! de `v`, dont l'autre extrémité est sur le bord de l'un des triangles, donc sur cette arête opposée ou
//! sur une arête issue de `v` contenue dans l'autre triangle, ce qui ramène au cas précédent.

use rayon::prelude::*;
use serde::Serialize;
use surfoxide_core::vec3::Vec3;
use surfoxide_core::{topology_report, TriMesh};

pub mod bvh;
pub mod distance;
pub mod predicates;
pub mod quality;

use bvh::Bvh;
use predicates::{is_degenerate, shared_edge_overlap, shared_vertex_intersect, triangles_intersect};

fn tri(mesh: &TriMesh, f: u32) -> [Vec3; 3] {
    let v = mesh.vertices();
    mesh.faces()[f as usize].map(|i| v[i as usize])
}

/// Vrai si les faces `f` et `g` du même maillage se coupent ailleurs que le long de leurs éléments communs.
fn faces_conflict(mesh: &TriMesh, f: u32, g: u32) -> bool {
    let (ff, fg) = (mesh.faces()[f as usize], mesh.faces()[g as usize]);
    let shared: Vec<u32> = ff.iter().copied().filter(|x| fg.contains(x)).collect();
    let v = mesh.vertices();
    let p = |i: u32| v[i as usize];
    match shared.len() {
        0 => triangles_intersect(tri(mesh, f), tri(mesh, g)),
        1 => {
            let s = shared[0];
            let others = |t: [u32; 3]| {
                let k = t.iter().position(|&x| x == s).unwrap();
                (t[(k + 1) % 3], t[(k + 2) % 3])
            };
            let ((a, b), (c, d)) = (others(ff), others(fg));
            shared_vertex_intersect(p(s), p(a), p(b), p(c), p(d))
        }
        2 => {
            let (a, b) = (shared[0], shared[1]);
            let c = *ff.iter().find(|x| !shared.contains(x)).unwrap();
            let d = *fg.iter().find(|x| !shared.contains(x)).unwrap();
            shared_edge_overlap(p(a), p(b), p(c), p(d))
        }
        _ => true, // face dupliquée
    }
}

/// Paires de faces (f < g) qui se coupent, triées.
pub fn self_intersections(mesh: &TriMesh) -> Vec<(u32, u32)> {
    let bvh = Bvh::new(mesh);
    (0..mesh.n_faces() as u32)
        .into_par_iter()
        .flat_map_iter(|f| {
            let mut out = Vec::new();
            let bbox = *bvh.face_box(f);
            bvh.query_box(&bbox, |g| {
                if g > f && faces_conflict(mesh, f, g) {
                    out.push((f, g));
                }
            });
            out
        })
        .collect()
}

/// Paires (face de `a`, face de `b`) qui se coupent ; les deux maillages n'ont aucun sommet commun.
pub fn cross_intersections(a: &TriMesh, b: &TriMesh) -> Vec<(u32, u32)> {
    let bvh = Bvh::new(b);
    (0..a.n_faces() as u32)
        .into_par_iter()
        .flat_map_iter(|f| {
            let mut out = Vec::new();
            let t = tri(a, f);
            let bbox = bvh::Aabb::of_points(&t);
            bvh.query_box(&bbox, |g| {
                if triangles_intersect(t, tri(b, g)) {
                    out.push((f, g));
                }
            });
            out
        })
        .collect()
}

/// Distance de chaque sommet de `from` à la surface `to`.
pub fn vertex_distances(from: &TriMesh, to: &TriMesh) -> Vec<f64> {
    let bvh = Bvh::new(to);
    from.vertices()
        .par_iter()
        .map(|&p| {
            bvh.nearest(p, |g| {
                let [a, b, c] = tri(to, g);
                distance::distance2_triangle(p, a, b, c)
            })
            .map_or(f64::INFINITY, |(_, d2)| d2.sqrt())
        })
        .collect()
}

/// Indice d'enroulement de la surface fermée `mesh` autour de `p` : 1 à l'intérieur (normales
/// sortantes), 0 à l'extérieur.
pub fn winding_number(mesh: &TriMesh, p: Vec3) -> f64 {
    let v = mesh.vertices();
    mesh.faces()
        .iter()
        .map(|f| distance::solid_angle(p, v[f[0] as usize], v[f[1] as usize], v[f[2] as usize]))
        .sum::<f64>()
        / (4.0 * std::f64::consts::PI)
}

/// Bilan d'une surface au regard du contrat gyroxide.
#[derive(Debug, Clone, Serialize)]
pub struct SurfaceReport {
    pub n_vertices: usize,
    pub n_faces: usize,
    pub euler_characteristic: i64,
    pub genus: Option<i64>,
    pub components: usize,
    pub boundary_edges: usize,
    pub non_manifold_edges: usize,
    pub non_manifold_vertices: usize,
    pub inconsistent_edges: usize,
    /// mm³ ; positif pour des normales sortantes.
    pub signed_volume: f64,
    /// mm².
    pub area: f64,
    pub degenerate_faces: usize,
    pub self_intersections: usize,
    /// Fermée, 2-variété, orientée vers l'extérieur, d'un seul tenant, de genre 0, sans auto-intersection
    /// ni face dégénérée.
    pub valid: bool,
}

/// Contrôle complet d'une surface.
pub fn check_surface(mesh: &TriMesh) -> SurfaceReport {
    let t = topology_report(mesh);
    let degenerate_faces = (0..mesh.n_faces() as u32)
        .filter(|&f| {
            let [a, b, c] = tri(mesh, f);
            is_degenerate(a, b, c)
        })
        .count();
    let self_intersections = self_intersections(mesh).len();
    let signed_volume = mesh.signed_volume();
    let valid = t.is_sphere_like() && signed_volume > 0.0 && self_intersections == 0 && degenerate_faces == 0;
    SurfaceReport {
        n_vertices: t.n_vertices,
        n_faces: t.n_faces,
        euler_characteristic: t.euler_characteristic,
        genus: t.genus(),
        components: t.components,
        boundary_edges: t.boundary_edges,
        non_manifold_edges: t.non_manifold_edges,
        non_manifold_vertices: t.non_manifold_vertices,
        inconsistent_edges: t.inconsistent_edges,
        signed_volume,
        area: mesh.area(),
        degenerate_faces,
        self_intersections,
        valid,
    }
}

/// Bilan d'un couple de surfaces imbriquées (WM dans la piale).
#[derive(Debug, Clone, Serialize)]
pub struct PairReport {
    /// Paires de faces (intérieure, extérieure) qui se coupent.
    pub intersections: usize,
    /// Un sommet de la surface intérieure est dans la surface extérieure.
    pub inner_inside_outer: bool,
    /// Distance minimale (mm) des sommets intérieurs à la surface extérieure.
    pub min_distance: f64,
    /// Distance moyenne (mm) des sommets intérieurs à la surface extérieure.
    pub mean_distance: f64,
    /// Épaisseur minimale exigée (mm).
    pub min_thickness: f64,
    /// Sans intersection, intérieure dans l'extérieure, distance minimale ≥ épaisseur exigée.
    pub valid: bool,
}

/// Contrôle de l'imbrication de `inner` dans `outer`.
pub fn check_pair(inner: &TriMesh, outer: &TriMesh, min_thickness: f64) -> PairReport {
    let intersections = cross_intersections(inner, outer).len();
    let inner_inside_outer = inner
        .vertices()
        .first()
        .is_some_and(|&p| winding_number(outer, p) > 0.5);
    let d = vertex_distances(inner, outer);
    let min_distance = d.iter().copied().fold(f64::INFINITY, f64::min);
    let mean_distance = d.iter().sum::<f64>() / d.len().max(1) as f64;
    PairReport {
        intersections,
        inner_inside_outer,
        min_distance,
        mean_distance,
        min_thickness,
        valid: intersections == 0 && inner_inside_outer && min_distance >= min_thickness,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::{icosphere, torus};
    use surfoxide_core::vec3;

    #[test]
    fn icosphere_valide() {
        let r = check_surface(&icosphere(4, 10.0));
        assert!(r.valid, "{r:?}");
        assert_eq!(r.self_intersections, 0);
    }

    #[test]
    fn tore_sans_intersection_mais_invalide() {
        let r = check_surface(&torus(48, 24, 3.0, 1.0));
        assert_eq!(r.self_intersections, 0);
        assert_eq!(r.genus, Some(1));
        assert!(!r.valid);
    }

    #[test]
    fn sphere_pliee_detectee() {
        let mut s = icosphere(3, 10.0);
        // On enfonce un sommet à travers la sphère : ses faces traversent le côté opposé.
        let p = s.vertices()[0];
        s.vertices_mut()[0] = vec3::scale(p, -1.3);
        let r = check_surface(&s);
        assert!(r.self_intersections > 0, "{r:?}");
        assert!(!r.valid);
    }

    #[test]
    fn spheres_concentriques() {
        let inner = icosphere(4, 10.0);
        let outer = icosphere(4, 12.0);
        let r = check_pair(&inner, &outer, 1.0);
        assert!(r.valid, "{r:?}");
        // Les sommets intérieurs sont à 2 mm de la sphère extérieure (aux facettes près).
        assert!(r.min_distance > 1.9 && r.min_distance <= 2.0 + 1e-9, "{r:?}");
        // Inversées : l'intérieure n'est pas dans l'extérieure.
        assert!(!check_pair(&outer, &inner, 1.0).valid);
        // Trop proches pour l'épaisseur exigée.
        assert!(!check_pair(&inner, &outer, 3.0).valid);
    }

    #[test]
    fn surfaces_qui_se_coupent() {
        let a = icosphere(3, 10.0);
        let mut b = icosphere(3, 10.0);
        for p in b.vertices_mut() {
            p[0] += 5.0;
        }
        assert!(!cross_intersections(&a, &b).is_empty());
    }

    #[test]
    fn indice_d_enroulement() {
        let s = icosphere(3, 5.0);
        assert!((winding_number(&s, [0.0, 0.0, 0.0]) - 1.0).abs() < 1e-9);
        assert!(winding_number(&s, [10.0, 0.0, 0.0]).abs() < 1e-9);
    }
}
