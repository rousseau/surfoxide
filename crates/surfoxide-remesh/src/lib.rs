//! `surfoxide-remesh` : remaillage isotrope de Botsch & Kobbelt (2004).
//!
//! À chaque itération : coupure des arêtes plus longues que 4/3 L, fusion de celles plus courtes que
//! 4/5 L (sous la *link condition*, qui préserve la topologie, et sans retourner de face), bascules qui
//! rapprochent les valences de 6, relaxation tangentielle, puis reprojection sur la surface d'origine.
//! La topologie (genre, orientation) est conservée par chaque opération.

use surfoxide_check::bvh::{Aabb, Bvh};
use surfoxide_check::distance::closest_point_triangle;
use surfoxide_check::{self_intersections_among, PointGrid};
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
            // Arêtes des deux faces, de la plus courte à la plus longue.
            let mut cand: Vec<u32> = [h, m.next(h), m.prev(h), m.next(g), m.prev(g)].to_vec();
            cand.sort_by(|&x, &y| m.edge_length(x).total_cmp(&m.edge_length(y)));
            for e in cand {
                let (a, b) = (m.from(e), m.to(e));
                if !m.can_collapse(e) {
                    continue;
                }
                let p = vec3::lerp(m.pos[a as usize], m.pos[b as usize], 0.5);
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

/// Score de cône du sommet `v` (cosinus de [`vec3::cone_axis`]) ; −1 si l'une de ses faces est dégénérée.
fn vertex_cone(m: &HalfEdgeMesh, v: u32) -> f64 {
    let mut normals = Vec::new();
    for h in m.outgoing(v) {
        let n = m.face_normal((h / 3) as usize);
        let len = vec3::norm(n);
        if len < 1e-12 {
            return -1.0;
        }
        normals.push(vec3::scale(n, 1.0 / len));
    }
    vec3::cone_axis(&normals).1
}

/// Plus petit score de cône parmi `vs`.
fn min_cone(m: &HalfEdgeMesh, vs: &[u32]) -> f64 {
    vs.iter().map(|&v| vertex_cone(m, v)).fold(f64::INFINITY, f64::min)
}

/// Nombre de paires de faces qui se coupent parmi celles des sommets `around`, contre toutes les faces
/// proches (sommets trouvés par `grid` dans la boîte de ces faces élargie de `pad`).
fn local_conflicts(m: &HalfEdgeMesh, grid: &PointGrid, pad: f64, around: &[u32]) -> usize {
    let faces_of = |v: u32| m.outgoing(v).into_iter().map(|h| (h / 3) as usize);
    let mut touched: Vec<usize> = around
        .iter()
        .filter(|&&v| m.vertex_alive[v as usize])
        .flat_map(|&v| faces_of(v))
        .collect();
    touched.sort_unstable();
    touched.dedup();
    let pts: Vec<Vec3> = touched
        .iter()
        .flat_map(|&f| m.faces[f].map(|v| m.pos[v as usize]))
        .collect();
    if pts.is_empty() {
        return 0;
    }
    let bbox = Aabb::of_points(&pts);
    let mut faces = touched.clone();
    grid.visit_box(vec3::sub(bbox.min, [pad; 3]), vec3::add(bbox.max, [pad; 3]), |v| {
        if m.vertex_alive[v as usize] {
            faces.extend(faces_of(v));
        }
    });
    faces.sort_unstable();
    faces.dedup();
    let mut map = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let local: Vec<[u32; 3]> = faces
        .iter()
        .map(|&f| {
            m.faces[f].map(|v| {
                *map.entry(v).or_insert_with(|| {
                    pos.push(m.pos[v as usize]);
                    pos.len() as u32 - 1
                })
            })
        })
        .collect();
    let mesh = TriMesh::new(pos, local).expect("faces vivantes");
    let ids: Vec<u32> = touched.iter().map(|f| faces.binary_search(f).unwrap() as u32).collect();
    self_intersections_among(&mesh, &ids).len()
}

/// Ouvre le cône de décalage des sommets « plissés » : ceux dont aucune direction ne fait un cosinus d'au
/// moins `min_cos` avec toutes les faces incidentes (somme des angles bien au-delà de 360°, faces incidentes
/// opposées), où tout décalage de la surface, si petit soit-il, croise ses propres faces. Les positions ne
/// changent pas : on bascule d'abord, une à une, les arêtes du sommet qui améliorent le plus petit score de
/// cône des quatre sommets concernés ; s'il reste plissé, on fusionne l'arête voisine (extrémité ou milieu) qui
/// relève le plus le pire score du voisinage (link condition : topologie conservée). Aucune opération
/// n'augmente le nombre d'auto-intersections de son voisinage. Rend la surface et le nombre de sommets
/// encore plissés.
pub fn open_cones(mesh: &TriMesh, min_cos: f64, passes: usize) -> Result<(TriMesh, usize), NotClosedManifold> {
    let mut m = HalfEdgeMesh::from_mesh(mesh)?;
    // Une fusion ne déplace un sommet que d'une arête au plus : la grille des positions initiales, élargie
    // de deux arêtes, retrouve toutes les faces proches.
    let pad = 2.0 * m.edges().into_iter().map(|h| m.edge_length(h)).fold(0.0, f64::max);
    let grid = PointGrid::new(&m.pos, pad.max(1e-6));
    let pleated = |m: &HalfEdgeMesh| -> Vec<u32> {
        m.vertices()
            .into_iter()
            .filter(|&v| vertex_cone(m, v) < min_cos)
            .collect()
    };
    for _ in 0..passes {
        let bad = pleated(&m);
        if bad.is_empty() {
            break;
        }
        for v in bad {
            if !m.vertex_alive[v as usize] || vertex_cone(&m, v) >= min_cos {
                continue;
            }
            // Bascules : on retient à chaque tour celle qui relève le plus le pire des quatre sommets.
            for _ in 0..m.valence(v) {
                let mut best: Option<(u32, f64)> = None;
                for h in m.outgoing(v) {
                    let g = m.twin(h);
                    let quad = [m.from(h), m.to(h), m.to(m.next(h)), m.to(m.next(g))];
                    if m.valence(quad[0]) <= 3 || m.valence(quad[1]) <= 3 {
                        continue;
                    }
                    let before = min_cone(&m, &quad);
                    let conflicts = local_conflicts(&m, &grid, pad, &quad);
                    let f1 = h / 3;
                    if !m.flip(h) {
                        continue;
                    }
                    let after = min_cone(&m, &quad);
                    let safe = after > before + 1e-9 && local_conflicts(&m, &grid, pad, &quad) <= conflicts;
                    // Retour : la nouvelle arête (d, c) est la deuxième demi-arête de la face (a, d, c).
                    let back = m.flip(3 * f1 + 1);
                    debug_assert!(back);
                    if safe && best.is_none_or(|(_, b)| after - before > b) {
                        best = Some((h, after - before));
                    }
                }
                match best {
                    Some((h, _)) => {
                        m.flip(h);
                    }
                    None => break,
                }
                if vertex_cone(&m, v) >= min_cos {
                    break;
                }
            }
            if vertex_cone(&m, v) >= min_cos {
                continue;
            }
            // Fusion d'une arête du sommet (le pli touche souvent deux ou trois sommets voisins) : chaque
            // extrémité et le milieu sont essayés, la fusion n'est retenue que si elle relève le pire score
            // du voisinage.
            let ring: Vec<u32> = std::iter::once(v)
                .chain(m.neighbors(v).into_iter().flat_map(|w| m.neighbors(w)))
                .collect();
            let worst_of = |m: &HalfEdgeMesh| {
                ring.iter()
                    .filter(|&&u| m.vertex_alive[u as usize])
                    .map(|&u| vertex_cone(m, u))
                    .fold(f64::INFINITY, f64::min)
            };
            let current = worst_of(&m);
            let conflicts = local_conflicts(&m, &grid, pad, &ring);
            let mut best: Option<(HalfEdgeMesh, f64)> = None;
            for w in m.neighbors(v) {
                for h in m.outgoing(w).into_iter().chain(m.outgoing(v)) {
                    if !m.can_collapse(h) {
                        continue;
                    }
                    let (a, b) = (m.pos[m.from(h) as usize], m.pos[m.to(h) as usize]);
                    for p in [a, b, vec3::lerp(a, b, 0.5)] {
                        let mut trial = m.clone();
                        trial.collapse(h, p);
                        let score = worst_of(&trial);
                        if score > current + 1e-9
                            && best.as_ref().is_none_or(|(_, s)| score > *s)
                            && local_conflicts(&trial, &grid, pad, &ring) <= conflicts
                        {
                            best = Some((trial, score));
                        }
                    }
                }
            }
            if let Some((trial, _)) = best {
                m = trial;
            }
        }
    }
    let remaining = pleated(&m).len();
    Ok((m.to_mesh(), remaining))
}

/// Plus courte arête parmi celles des sommets `vs` (vivants).
fn min_edge_around(m: &HalfEdgeMesh, vs: &[u32]) -> f64 {
    vs.iter()
        .filter(|&&v| m.vertex_alive[v as usize])
        .flat_map(|&v| m.outgoing(v))
        .map(|h| m.edge_length(h))
        .fold(f64::INFINITY, f64::min)
}

/// Fusionne `h` en `p` sur une copie, si la link condition le permet.
fn try_collapse(m: &HalfEdgeMesh, h: u32, p: Vec3) -> Option<HalfEdgeMesh> {
    if !m.face_alive[(h / 3) as usize] || !m.can_collapse(h) {
        return None;
    }
    let mut t = m.clone();
    t.collapse(h, p);
    Some(t)
}

/// Fusions de `h` possibles sur une copie : vers chaque extrémité et vers le milieu.
fn collapses(m: &HalfEdgeMesh, h: u32) -> Vec<HalfEdgeMesh> {
    let (a, b) = (m.pos[m.from(h) as usize], m.pos[m.to(h) as usize]);
    [vec3::lerp(a, b, 0.5), a, b]
        .into_iter()
        .filter_map(|p| try_collapse(m, h, p))
        .collect()
}

/// Demi-arête de `a` vers `b`, si elles sont voisines.
fn half_edge(m: &HalfEdgeMesh, a: u32, b: u32) -> Option<u32> {
    if !m.vertex_alive[a as usize] || !m.vertex_alive[b as usize] {
        return None;
    }
    m.outgoing(a).into_iter().find(|&h| m.to(h) == b)
}

/// Supprime les arêtes plus courtes que `min_len` que [`collapse_short_edges`] laisse : celles dont la fusion
/// directe est refusée par la link condition (les deux sommets ont un troisième voisin commun, ou un sommet
/// opposé de valence 3), souvent dans un amas de micro-triangles. Pour chaque arête courte, de la plus courte à
/// la plus longue, on essaie sa fusion (milieu ou extrémités), puis une opération préalable sur une arête
/// voisine (bascule, ou fusion qui supprime un sommet de valence 3 ou un voisin commun) suivie de sa fusion. Un
/// essai n'est retenu que s'il allonge la plus courte arête du voisinage, garde à chaque sommet du voisinage un
/// cône de décalage d'au moins `min(min_cos, score d'avant)` (aucune face retournée) et n'augmente pas le
/// nombre d'auto-intersections locales ; parmi les essais retenus, celui dont la plus courte arête locale est
/// la plus longue. Topologie conservée. Rend la surface et le nombre d'arêtes encore trop courtes ; une
/// surface sans arête courte est rendue telle quelle.
pub fn untangle_short_edges(
    mesh: &TriMesh,
    min_len: f64,
    min_cos: f64,
    passes: usize,
) -> Result<(TriMesh, usize), NotClosedManifold> {
    let short = |m: &HalfEdgeMesh| -> Vec<u32> {
        let mut s: Vec<(f64, u32)> = m
            .edges()
            .into_iter()
            .map(|h| (m.edge_length(h), h))
            .filter(|&(l, _)| l < min_len)
            .collect();
        s.sort_by(|x, y| x.0.total_cmp(&y.0));
        s.into_iter().map(|(_, h)| h).collect()
    };
    let mut m = HalfEdgeMesh::from_mesh(mesh)?;
    if short(&m).is_empty() {
        return Ok((mesh.clone(), 0));
    }
    let pad = 2.0 * m.edges().into_iter().map(|h| m.edge_length(h)).fold(0.0, f64::max);
    let grid = PointGrid::new(&m.pos, pad.max(1e-6));
    for _ in 0..passes {
        let todo = short(&m);
        if todo.is_empty() {
            break;
        }
        let mut changed = false;
        // Les indices de demi-arêtes changent avec les opérations : on suit chaque arête par ses sommets.
        let pairs: Vec<(u32, u32)> = todo.iter().map(|&h| (m.from(h), m.to(h))).collect();
        for (a, b) in pairs {
            let Some(h) = half_edge(&m, a, b) else { continue };
            if m.edge_length(h) >= min_len {
                continue;
            }
            // Voisinage : les deux sommets et leurs deux premiers anneaux.
            let mut ring: Vec<u32> = vec![a, b];
            for _ in 0..2 {
                let more: Vec<u32> = ring.iter().flat_map(|&v| m.neighbors(v)).collect();
                ring.extend(more);
                ring.sort_unstable();
                ring.dedup();
            }
            let edge_before = min_edge_around(&m, &ring);
            let cone_floor = min_cone(&m, &ring).min(min_cos);
            let conflicts = local_conflicts(&m, &grid, pad, &ring);
            let mut trials = collapses(&m, h);
            // Opérations préalables sur les arêtes des faces autour de a et b.
            let mut around: Vec<(u32, u32)> = [a, b]
                .iter()
                .flat_map(|&v| m.outgoing(v))
                .flat_map(|e| [e, m.next(e), m.prev(e)])
                .map(|e| {
                    let (x, y) = (m.from(e), m.to(e));
                    (x.min(y), x.max(y))
                })
                .filter(|&(x, y)| (x, y) != (a.min(b), a.max(b)))
                .collect();
            around.sort_unstable();
            around.dedup();
            for (x, y) in around {
                let Some(e) = half_edge(&m, x, y) else { continue };
                let mut first: Vec<HalfEdgeMesh> = Vec::new();
                let mut flipped = m.clone();
                if flipped.flip(e) {
                    first.push(flipped);
                }
                first.extend(collapses(&m, e));
                for t in first {
                    // Si a ou b a disparu dans la fusion préalable, l'arête est déjà supprimée.
                    match half_edge(&t, a, b) {
                        Some(g) => trials.extend(collapses(&t, g)),
                        None => trials.push(t),
                    }
                }
            }
            let mut best: Option<(HalfEdgeMesh, f64)> = None;
            for t in trials {
                let live: Vec<u32> = ring.iter().copied().filter(|&v| t.vertex_alive[v as usize]).collect();
                let edge = min_edge_around(&t, &live);
                if edge <= edge_before + 1e-12 || best.as_ref().is_some_and(|(_, e)| edge <= *e) {
                    continue;
                }
                if min_cone(&t, &live) < cone_floor || local_conflicts(&t, &grid, pad, &live) > conflicts {
                    continue;
                }
                best = Some((t, edge));
            }
            if let Some((t, _)) = best {
                m = t;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let remaining = short(&m).len();
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
    fn cone_ouvert_d_un_sommet_plisse() {
        // Sommet plissé : un voisin est tiré à travers le sommet, une partie de ses faces se retourne et aucune
        // direction ne s'éloigne de toutes.
        let mut s = icosphere(3, 10.0);
        let nb = surfoxide_core::adjacency::vertex_neighbors(&s);
        let q = s.vertices()[nb[0][0] as usize];
        s.vertices_mut()[nb[0][0] as usize] = vec3::lerp(q, s.vertices()[0], 1.6);
        let score = |m: &TriMesh, v: usize| {
            let vf = surfoxide_core::adjacency::vertex_faces(m);
            let fn_ = surfoxide_core::adjacency::face_normals(m);
            vec3::cone_axis(&vf[v].iter().map(|&f| fn_[f as usize]).collect::<Vec<_>>()).1
        };
        let n_bad = |m: &TriMesh| (0..m.n_vertices()).filter(|&v| score(m, v) < 0.05).count();
        assert!(n_bad(&s) > 0, "le cas d'essai doit contenir un sommet plissé");
        let (r, remaining) = open_cones(&s, 0.05, 5).unwrap();
        assert_eq!(remaining, 0);
        assert_eq!(n_bad(&r), 0);
        assert!(topology_report(&r).is_sphere_like());
    }

    #[test]
    fn arete_courte_bloquee_par_la_link_condition() {
        // Face (x, y, z) dont l'arête x–y est rapprochée à 0,01 mm, puis un sommet de valence 3 inséré dans
        // la face : x et y ont trois voisins communs (le sommet inséré, z et le sommet opposé), la fusion
        // directe de x–y est refusée.
        let s = icosphere(3, 10.0);
        let [x, y, z] = s.faces()[0];
        let mut v = s.vertices().to_vec();
        let (px, py) = (v[x as usize], v[y as usize]);
        v[y as usize] = vec3::add(px, vec3::scale(vec3::normalize(vec3::sub(py, px)), 0.01));
        let c = vec3::scale(
            vec3::add(vec3::add(v[x as usize], v[y as usize]), v[z as usize]),
            1.0 / 3.0,
        );
        let n = v.len() as u32;
        v.push(c);
        let mut f = s.faces().to_vec();
        f[0] = [x, y, n];
        f.extend([[y, z, n], [z, x, n]]);
        let m = TriMesh::new(v, f).unwrap();
        let min_edge = |m: &TriMesh| {
            let v = m.vertices();
            edges(m)
                .iter()
                .map(|&[a, b]| vec3::dist(v[a as usize], v[b as usize]))
                .fold(f64::INFINITY, f64::min)
        };
        let collapsed = collapse_short_edges(&m, 0.1).unwrap();
        assert!(
            min_edge(&collapsed) < 0.1,
            "le cas d'essai doit résister à la fusion directe"
        );
        let (r, remaining) = untangle_short_edges(&collapsed, 0.1, 0.05, 5).unwrap();
        assert_eq!(remaining, 0);
        assert!(min_edge(&r) >= 0.1, "{}", min_edge(&r));
        assert!(topology_report(&r).is_sphere_like());
        assert!(surfoxide_check::self_intersections(&r).is_empty());
        // Sans arête courte, la surface est rendue telle quelle.
        let (same, k) = untangle_short_edges(&s, 0.1, 0.05, 5).unwrap();
        assert_eq!(k, 0);
        assert_eq!(same.vertices(), s.vertices());
        assert_eq!(same.faces(), s.faces());
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
