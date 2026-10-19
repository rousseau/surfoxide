//! `surfoxide-remesh` : remaillage isotrope de Botsch & Kobbelt (2004).
//!
//! À chaque itération : coupure des arêtes plus longues que 4/3 L, fusion de celles plus courtes que
//! 4/5 L (sous la *link condition*, qui préserve la topologie, et sans retourner de face), bascules qui
//! rapprochent les valences de 6, relaxation tangentielle, puis reprojection sur la surface d'origine.
//! La topologie (genre, orientation) est conservée par chaque opération.

use surfoxide_check::bvh::Bvh;
use surfoxide_check::distance::closest_point_triangle;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

pub mod halfedge;

use halfedge::{HalfEdgeMesh, NotClosedManifold};

/// Paramètres du remaillage.
#[derive(Debug, Clone, Copy)]
pub struct RemeshParams {
    /// Longueur d'arête visée (mm).
    pub target: f64,
    pub iterations: usize,
    /// Écart maximal (cosinus) entre les normales des deux faces d'une arête basculée.
    pub flip_min_cos: f64,
}

impl RemeshParams {
    pub fn new(target: f64) -> Self {
        Self {
            target,
            iterations: 5,
            flip_min_cos: 0.95,
        }
    }
}

fn split_long_edges(m: &mut HalfEdgeMesh, high: f64) {
    loop {
        let mut changed = false;
        for h in m.edges() {
            if m.face_alive[(h / 3) as usize] && m.edge_length(h) > high {
                m.split(h);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
}

fn collapse_short(m: &mut HalfEdgeMesh, low: f64, high: f64) {
    for h in m.edges() {
        if !m.face_alive[(h / 3) as usize] || m.edge_length(h) >= low {
            continue;
        }
        let (a, b) = (m.from(h), m.to(h));
        if !m.vertex_alive[a as usize] || !m.vertex_alive[b as usize] || !m.can_collapse(h) {
            continue;
        }
        let p = vec3::lerp(m.pos[a as usize], m.pos[b as usize], 0.5);
        // Pas d'arête trop longue autour du point de fusion.
        if m.neighbors(a)
            .into_iter()
            .chain(m.neighbors(b))
            .any(|v| v != a && v != b && vec3::dist(m.pos[v as usize], p) > high)
        {
            continue;
        }
        // Pas de face retournée ni dégénérée.
        let mut ok = true;
        for v in [a, b] {
            for e in m.outgoing(v) {
                let f = (e / 3) as usize;
                let t = m.faces[f];
                if t.contains(&a) && t.contains(&b) {
                    continue; // faces supprimées
                }
                let before = m.face_normal(f);
                let q = t.map(|w| if w == a || w == b { p } else { m.pos[w as usize] });
                let after = vec3::triangle_normal(q[0], q[1], q[2]);
                if vec3::dot(before, after) <= 0.1 * vec3::norm(before) * vec3::norm(after) || vec3::norm(after) < 1e-12
                {
                    ok = false;
                }
            }
        }
        if ok {
            m.collapse(h, p);
        }
    }
}

fn flip_edges(m: &mut HalfEdgeMesh, min_cos: f64) {
    for h in m.edges() {
        if !m.face_alive[(h / 3) as usize] {
            continue;
        }
        let g = m.twin(h);
        let (a, b) = (m.from(h), m.to(h));
        let c = m.to(m.next(h));
        let d = m.to(m.next(g));
        let (va, vb, vc, vd) = (m.valence(a), m.valence(b), m.valence(c), m.valence(d));
        if va <= 3 || vb <= 3 {
            continue;
        }
        let dev = |x: usize| (x as i64 - 6).pow(2);
        let before = dev(va) + dev(vb) + dev(vc) + dev(vd);
        let after = dev(va - 1) + dev(vb - 1) + dev(vc + 1) + dev(vd + 1);
        if after >= before {
            continue;
        }
        // Seulement entre faces presque coplanaires, et sans retourner les nouvelles faces.
        let (n1, n2) = (m.face_normal((h / 3) as usize), m.face_normal((g / 3) as usize));
        let cos = vec3::dot(vec3::normalize(n1), vec3::normalize(n2));
        if cos < min_cos {
            continue;
        }
        let p = |v: u32| m.pos[v as usize];
        let (m1, m2) = (
            vec3::triangle_normal(p(a), p(d), p(c)),
            vec3::triangle_normal(p(d), p(b), p(c)),
        );
        let avg = vec3::add(n1, n2);
        if vec3::dot(m1, avg) <= 0.0 || vec3::dot(m2, avg) <= 0.0 {
            continue;
        }
        m.flip(h);
    }
}

fn tangential_relaxation(m: &mut HalfEdgeMesh) {
    let verts = m.vertices();
    let new: Vec<(u32, Vec3)> = verts
        .iter()
        .map(|&v| {
            let nb = m.neighbors(v);
            let c = nb.iter().fold([0.0; 3], |acc, &w| vec3::add(acc, m.pos[w as usize]));
            let c = vec3::scale(c, 1.0 / nb.len() as f64);
            let n = vec3::normalize(m.vertex_normal(v));
            let d = vec3::sub(c, m.pos[v as usize]);
            // Composante tangentielle du déplacement vers le barycentre.
            let t = vec3::sub(d, vec3::scale(n, vec3::dot(d, n)));
            (v, vec3::add(m.pos[v as usize], t))
        })
        .collect();
    for (v, p) in new {
        m.pos[v as usize] = p;
    }
}

/// Reprojette chaque sommet sur la surface de référence, sauf si le point le plus proche est sur une face
/// d'orientation opposée (berge d'en face d'un sillon étroit).
fn project(m: &mut HalfEdgeMesh, reference: &TriMesh, bvh: &Bvh, ref_normals: &[Vec3]) {
    let rv = reference.vertices();
    let rf = reference.faces();
    for v in m.vertices() {
        let p = m.pos[v as usize];
        let Some((f, _)) = bvh.nearest(p, |f| {
            let t = rf[f as usize];
            let q = closest_point_triangle(p, rv[t[0] as usize], rv[t[1] as usize], rv[t[2] as usize]);
            let d = vec3::sub(p, q);
            vec3::dot(d, d)
        }) else {
            continue;
        };
        let n = m.vertex_normal(v);
        if vec3::dot(n, ref_normals[f as usize]) <= 0.0 {
            continue;
        }
        let t = rf[f as usize];
        m.pos[v as usize] = closest_point_triangle(p, rv[t[0] as usize], rv[t[1] as usize], rv[t[2] as usize]);
    }
}

/// Cosinus de l'angle entre les normales des deux faces de l'arête `h`.
fn dihedral_cos(m: &HalfEdgeMesh, h: u32) -> f64 {
    let (n1, n2) = (
        vec3::normalize(m.face_normal((h / 3) as usize)),
        vec3::normalize(m.face_normal((m.twin(h) / 3) as usize)),
    );
    vec3::dot(n1, n2)
}

/// Plus petit cosinus de l'angle dièdre sur les arêtes des faces autour de `v`.
fn local_min_cos(m: &HalfEdgeMesh, v: u32) -> f64 {
    m.outgoing(v)
        .into_iter()
        .flat_map(|e| [e, m.next(e)])
        .map(|e| dihedral_cos(m, e))
        .fold(f64::INFINITY, f64::min)
}

/// Déplie les plis : arêtes dont les deux faces ont des normales de cosinus inférieur à `max_cos` (surface
/// repliée en nageoire, triangles en aiguille). Pour chacune, la plus courte arête de ses deux faces est
/// fusionnée en son milieu, sous la link condition (topologie conservée) et sans créer de face d'aire
/// nulle ; contrairement au remaillage, une face peut changer d'orientation, puisque c'est le pli qu'on
/// défait. Rend la surface et le nombre de plis restants. Peut créer des intersections, à réparer ensuite.
pub fn unfold(mesh: &TriMesh, max_cos: f64, passes: usize) -> Result<(TriMesh, usize), NotClosedManifold> {
    let mut m = HalfEdgeMesh::from_mesh(mesh)?;
    let folded = |m: &HalfEdgeMesh| -> Vec<u32> {
        m.edges()
            .into_iter()
            .filter(|&h| m.face_alive[(h / 3) as usize] && dihedral_cos(m, h) < max_cos)
            .collect()
    };
    for _ in 0..passes {
        let edges = folded(&m);
        if edges.is_empty() {
            break;
        }
        for h in edges {
            if !m.face_alive[(h / 3) as usize] || dihedral_cos(&m, h) >= max_cos {
                continue;
            }
            let g = m.twin(h);
            // D'abord la bascule : deux triangles fins repliés l'un sur l'autre se déplient en reliant leurs
            // sommets opposés, s'ils ne sont pas déjà voisins.
            let (a, b) = (m.from(h), m.to(h));
            let (c, d) = (m.to(m.next(h)), m.to(m.next(g)));
            let p = |v: u32| m.pos[v as usize];
            let (n1, n2) = (
                vec3::triangle_normal(p(a), p(d), p(c)),
                vec3::triangle_normal(p(d), p(b), p(c)),
            );
            let unfolds = vec3::norm(n1) > 1e-10
                && vec3::norm(n2) > 1e-10
                && vec3::dot(vec3::normalize(n1), vec3::normalize(n2)) > max_cos.max(0.0);
            if unfolds && m.valence(a) > 3 && m.valence(b) > 3 && m.flip(h) {
                continue;
            }
            // Puis le déplacement d'un des quatre sommets au barycentre de ses voisins (une pointe de valence 3,
            // par exemple), retenu s'il défait le pli sans en créer autour.
            let mut moved = false;
            for v in [a, b, c, d] {
                let old = m.pos[v as usize];
                let nb = m.neighbors(v);
                let mean = nb.iter().fold([0.0; 3], |acc, &w| vec3::add(acc, m.pos[w as usize]));
                m.pos[v as usize] = vec3::scale(mean, 1.0 / nb.len() as f64);
                if dihedral_cos(&m, h) >= max_cos && local_min_cos(&m, v) >= max_cos {
                    moved = true;
                    break;
                }
                m.pos[v as usize] = old;
            }
            if moved {
                continue;
            }
            // Arêtes des deux faces, de la plus courte à la plus longue.
            // Les deux sens de chaque arête : la fusion supprime le sommet de départ, qui doit pouvoir être la
            // pointe à absorber.
            let mut cand: Vec<u32> = [h, m.next(h), m.prev(h), m.next(g), m.prev(g)]
                .into_iter()
                .flat_map(|e| [e, m.twin(e)])
                .collect();
            cand.sort_by(|&x, &y| {
                (m.valence(m.from(y)) == 3)
                    .cmp(&(m.valence(m.from(x)) == 3))
                    .then(m.edge_length(x).total_cmp(&m.edge_length(y)))
            });
            for e in cand {
                let (a, b) = (m.from(e), m.to(e));
                if !m.can_collapse(e) {
                    continue;
                }
                // Une pointe de valence 3 est absorbée par son voisin ; sinon fusion au milieu.
                let p = if m.valence(a) == 3 {
                    m.pos[b as usize]
                } else {
                    vec3::lerp(m.pos[a as usize], m.pos[b as usize], 0.5)
                };
                let degenerate = [a, b].iter().any(|&v| {
                    m.outgoing(v).into_iter().any(|o| {
                        let t = m.faces[(o / 3) as usize];
                        if t.contains(&a) && t.contains(&b) {
                            return false;
                        }
                        let q = t.map(|w| if w == a || w == b { p } else { m.pos[w as usize] });
                        vec3::norm(vec3::triangle_normal(q[0], q[1], q[2])) < 1e-10
                    })
                });
                if !degenerate {
                    m.collapse(e, p);
                    break;
                }
            }
        }
    }
    let remaining = folded(&m).len();
    Ok((m.to_mesh(), remaining))
}

/// Fusionne les arêtes plus courtes que `min_len` (sous la link condition, sans retourner de face) :
/// supprime les amas de micro-triangles sans changer la topologie. Peut créer des intersections, à
/// réparer ensuite.
pub fn collapse_short_edges(mesh: &TriMesh, min_len: f64) -> Result<TriMesh, NotClosedManifold> {
    let mut m = HalfEdgeMesh::from_mesh(mesh)?;
    for _ in 0..3 {
        collapse_short(&mut m, min_len, f64::INFINITY);
    }
    Ok(m.to_mesh())
}

/// Remaille `mesh` (surface fermée 2-variété) vers des arêtes de longueur `params.target`.
pub fn remesh(mesh: &TriMesh, params: &RemeshParams) -> Result<TriMesh, NotClosedManifold> {
    let mut m = HalfEdgeMesh::from_mesh(mesh)?;
    let bvh = Bvh::new(mesh);
    let ref_normals = surfoxide_core::adjacency::face_normals(mesh);
    let (low, high) = (0.8 * params.target, 4.0 / 3.0 * params.target);
    for _ in 0..params.iterations {
        split_long_edges(&mut m, high);
        collapse_short(&mut m, low, high);
        flip_edges(&mut m, params.flip_min_cos);
        tangential_relaxation(&mut m);
        project(&mut m, mesh, &bvh, &ref_normals);
    }
    Ok(m.to_mesh())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::adjacency::edges;
    use surfoxide_core::shapes::{icosphere, torus};
    use surfoxide_core::topology_report;

    fn edge_stats(m: &TriMesh) -> (f64, f64) {
        let v = m.vertices();
        let l: Vec<f64> = edges(m)
            .iter()
            .map(|&[a, b]| vec3::dist(v[a as usize], v[b as usize]))
            .collect();
        let mean = l.iter().sum::<f64>() / l.len() as f64;
        let in_band = l.iter().filter(|&&x| x > 0.5 * mean && x < 1.5 * mean).count();
        (mean, in_band as f64 / l.len() as f64)
    }

    #[test]
    fn sphere_remaillee() {
        let s = icosphere(5, 10.0); // arêtes d'environ 0,4 mm
        let r = remesh(&s, &RemeshParams::new(1.0)).unwrap();
        let rep = topology_report(&r);
        assert!(rep.is_sphere_like(), "{rep:?}");
        let (mean, band) = edge_stats(&r);
        assert!((mean - 1.0).abs() < 0.2, "longueur moyenne {mean}");
        assert!(band > 0.95, "{band}");
        // Les sommets restent sur la sphère.
        for p in r.vertices() {
            assert!((vec3::norm(*p) - 10.0).abs() < 0.05);
        }
        assert!((r.signed_volume() - s.signed_volume()).abs() / s.signed_volume() < 0.01);
    }

    #[test]
    fn pli_deplie() {
        // On replie la sphère : un sommet est tiré à travers son anneau, ses faces se retournent.
        let mut s = icosphere(3, 10.0);
        let p = s.vertices()[0];
        s.vertices_mut()[0] = vec3::scale(p, 0.9);
        let nb = surfoxide_core::adjacency::vertex_neighbors(&s);
        let q = s.vertices()[nb[0][0] as usize];
        s.vertices_mut()[nb[0][0] as usize] = vec3::lerp(q, s.vertices()[0], 1.6);
        let (r, remaining) = unfold(&s, -0.5, 5).unwrap();
        assert_eq!(remaining, 0);
        assert!(topology_report(&r).is_sphere_like());
    }

    #[test]
    fn tore_garde_son_genre() {
        let t = torus(120, 60, 6.0, 2.0);
        let r = remesh(&t, &RemeshParams::new(0.5)).unwrap();
        assert_eq!(topology_report(&r).genus(), Some(1));
        let (mean, _) = edge_stats(&r);
        assert!((mean - 0.5).abs() < 0.1, "longueur moyenne {mean}");
    }
}
