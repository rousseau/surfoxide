//! Fusion couplée des arêtes courtes de la piale et de la surface blanche, qui partagent la même triangulation.
//!
//! La plus courte arête de la piale fixe le pas de temps de gyroxide : une arête de 0,05 mm double ou triple le
//! nombre de pas d'une simulation. La régularisation de la piale ne peut pas l'allonger quand ses sommets sont
//! retenus par la distance minimale à la blanche ou par le dégagement, et le démêlage de la blanche ne voit pas
//! les arêtes que seule la piale a raccourcies. Ici, chaque opération (bascule, fusion) est appliquée à la fois
//! aux deux surfaces, ce qui garde la correspondance sommet à sommet (et donc les lignes de courant).

use std::collections::HashMap;

use surfoxide_check::distance::distance2_triangle;
use surfoxide_check::{self_intersections_among, PointGrid};
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

use crate::halfedge::{HalfEdgeMesh, NotClosedManifold};
use crate::{half_edge, min_cone, min_edge_around};

/// Seuils de [`collapse_paired_short_edges`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedParams {
    /// Arêtes de la piale plus courtes que cette longueur (mm) : celles que l'on supprime.
    pub min_len: f64,
    /// Score de cône minimal (voir `open_cones`) exigé aux sommets du voisinage, sur chaque surface, s'il
    /// l'était déjà avant l'opération (aucune face retournée).
    pub min_cos: f64,
    /// Distance minimale (mm) entre la blanche et la piale, exigée si elle l'était avant l'opération.
    pub min_distance: f64,
    /// Dégagement minimal (mm) des sommets de la piale, exigé s'il l'était avant l'opération.
    pub min_clearance: f64,
    /// Nombre maximal de passes sur les arêtes courtes.
    pub passes: usize,
}

/// Résultat de [`collapse_paired_short_edges`].
#[derive(Debug, Clone)]
pub struct PairedCollapse {
    pub white: TriMesh,
    pub pial: TriMesh,
    /// Pour chaque sommet du résultat, les sommets d'origine fusionnés en lui (lui-même compris) : de quoi
    /// reporter les champs par sommet.
    pub sources: Vec<Vec<u32>>,
    /// Opérations retenues (chacune supprime au moins une arête courte de la piale).
    pub operations: usize,
    /// Arêtes de la piale encore plus courtes que `min_len`.
    pub remaining: usize,
}

/// Emplacement du sommet fusionné.
#[derive(Debug, Clone, Copy)]
enum At {
    Middle,
    From,
    To,
}

/// Opération sur une arête désignée par ses sommets (les indices de demi-arêtes changent).
#[derive(Debug, Clone, Copy)]
enum Op {
    Flip(u32, u32),
    Collapse(u32, u32, At),
}

/// Les deux surfaces, de même structure, et les sommets d'origine de chaque sommet.
#[derive(Clone)]
struct Pair {
    w: HalfEdgeMesh,
    p: HalfEdgeMesh,
    members: Vec<Vec<u32>>,
}

impl Pair {
    /// Applique `op` aux deux surfaces, sur une copie ; `None` si elle n'est pas permise.
    fn apply(&self, op: Op) -> Option<Pair> {
        let (Op::Flip(a, b) | Op::Collapse(a, b, _)) = op;
        let h = half_edge(&self.w, a, b)?;
        let mut t = self.clone();
        match op {
            Op::Flip(..) => {
                if !t.w.flip(h) {
                    return None;
                }
                let ok = t.p.flip(h);
                debug_assert!(ok, "même structure");
            }
            Op::Collapse(_, _, at) => {
                if !self.w.can_collapse(h) {
                    return None;
                }
                let place = |m: &HalfEdgeMesh| {
                    let (x, y) = (m.pos[a as usize], m.pos[b as usize]);
                    match at {
                        At::Middle => vec3::lerp(x, y, 0.5),
                        At::From => x,
                        At::To => y,
                    }
                };
                let (pw, pp) = (place(&self.w), place(&self.p));
                t.w.collapse(h, pw);
                t.p.collapse(h, pp);
                // a disparaît dans b.
                let moved = std::mem::take(&mut t.members[a as usize]);
                t.members[b as usize].extend(moved);
            }
        }
        Some(t)
    }
}

/// Faces vivantes des sommets vivants `vs`.
fn faces_of(m: &HalfEdgeMesh, vs: &[u32]) -> Vec<usize> {
    let mut f: Vec<usize> = vs
        .iter()
        .filter(|&&v| m.vertex_alive[v as usize])
        .flat_map(|&v| m.outgoing(v))
        .map(|h| (h / 3) as usize)
        .collect();
    f.sort_unstable();
    f.dedup();
    f
}

/// Mesures locales autour d'une opération, comparées avant et après.
struct Local {
    pial_edge: f64,
    white_edge: f64,
    cone: f64,
    conflicts: usize,
    gap: f64,
    clearance: f64,
}

/// Voisinage d'une opération : grille des positions d'origine des deux surfaces mises à la suite (blanche
/// puis piale : l'indice `i ≥ n` désigne le sommet `i − n`, le même sommet sur la piale), élargie de `pad`.
struct Neighborhood {
    grid: PointGrid,
    n: u32,
    pad: f64,
}

impl Neighborhood {
    /// Sommets vivants proches des sommets `ring` (ceux-ci compris), sur l'une ou l'autre surface.
    fn near(&self, pair: &Pair, ring: &[u32]) -> Vec<u32> {
        let pts: Vec<Vec3> = ring
            .iter()
            .filter(|&&v| pair.w.vertex_alive[v as usize])
            .flat_map(|&v| [pair.w.pos[v as usize], pair.p.pos[v as usize]])
            .collect();
        let mut out: Vec<u32> = ring.to_vec();
        if !pts.is_empty() {
            let bbox = surfoxide_check::bvh::Aabb::of_points(&pts);
            let n = self.n;
            self.grid.visit_box(
                vec3::sub(bbox.min, [self.pad; 3]),
                vec3::add(bbox.max, [self.pad; 3]),
                |v| out.push(v % n),
            );
        }
        out.sort_unstable();
        out.dedup();
        out.retain(|&v| pair.w.vertex_alive[v as usize]);
        out
    }

    fn measure(&self, pair: &Pair, ring: &[u32]) -> Local {
        let live: Vec<u32> = ring
            .iter()
            .copied()
            .filter(|&v| pair.w.vertex_alive[v as usize])
            .collect();
        let near = self.near(pair, &live);
        let (fw, fp) = (faces_of(&pair.w, &near), faces_of(&pair.p, &near));
        // Surface locale réunissant les faces proches des deux surfaces (sommets distincts) : ses
        // intersections sont celles de chaque surface et celles entre elles.
        let mut map = HashMap::new();
        let mut pos = Vec::new();
        let mut faces = Vec::new();
        for (side, (m, fs)) in [(&pair.w, &fw), (&pair.p, &fp)].into_iter().enumerate() {
            for &f in fs.iter() {
                faces.push(m.faces[f].map(|v| {
                    *map.entry((side, v)).or_insert_with(|| {
                        pos.push(m.pos[v as usize]);
                        pos.len() as u32 - 1
                    })
                }));
            }
        }
        let touched: Vec<u32> = {
            let (tw, tp) = (faces_of(&pair.w, &live), faces_of(&pair.p, &live));
            tw.iter()
                .map(|f| fw.binary_search(f).unwrap() as u32)
                .chain(tp.iter().map(|f| (fw.len() + fp.binary_search(f).unwrap()) as u32))
                .collect()
        };
        let conflicts = if faces.is_empty() {
            0
        } else {
            let mesh = TriMesh::new(pos, faces).expect("faces vivantes");
            self_intersections_among(&mesh, &touched).len()
        };
        let tri = |m: &HalfEdgeMesh, f: usize| m.faces[f].map(|v| m.pos[v as usize]);
        // Distance blanche–piale : sommets blancs proches contre faces de la piale proches, et l'inverse.
        let mut gap2 = f64::INFINITY;
        for &v in &near {
            let (pw, pp) = (pair.w.pos[v as usize], pair.p.pos[v as usize]);
            for &f in &fp {
                let [a, b, c] = tri(&pair.p, f);
                gap2 = gap2.min(distance2_triangle(pw, a, b, c));
            }
            for &f in &fw {
                let [a, b, c] = tri(&pair.w, f);
                gap2 = gap2.min(distance2_triangle(pp, a, b, c));
            }
        }
        // Dégagement des sommets de la piale du voisinage : distance aux faces proches non incidentes.
        let mut clear2 = f64::INFINITY;
        for &v in &live {
            let q = pair.p.pos[v as usize];
            for &f in &fp {
                if pair.p.faces[f].contains(&v) {
                    continue;
                }
                let [a, b, c] = tri(&pair.p, f);
                clear2 = clear2.min(distance2_triangle(q, a, b, c));
            }
        }
        Local {
            pial_edge: min_edge_around(&pair.p, &live),
            white_edge: min_edge_around(&pair.w, &live),
            cone: min_cone(&pair.w, &live).min(min_cone(&pair.p, &live)),
            conflicts,
            gap: gap2.sqrt(),
            clearance: clear2.sqrt(),
        }
    }
}

/// Supprime les arêtes de la piale plus courtes que `params.min_len` en opérant à la fois sur la piale et sur
/// la blanche (même triangulation, exigée). Pour chaque arête courte, de la plus courte à la plus longue, on
/// essaie sa fusion (milieu ou extrémités, au même emplacement relatif sur les deux surfaces), puis une
/// opération préalable sur une arête voisine (bascule, ou fusion qui lève la link condition) suivie de sa
/// fusion. Un essai n'est retenu que s'il allonge la plus courte arête locale de la piale, sans raccourcir celle
/// de la blanche sous `min_len`, sans abaisser sous `min(min_cos, avant)` le score de cône d'aucun sommet du
/// voisinage, sur l'une ou l'autre surface, sans ajouter d'intersection (de chaque surface ou entre elles), et
/// sans abaisser la distance blanche–piale ni le dégagement de la piale sous `min(seuil, avant)` ; parmi les
/// essais retenus, celui dont la plus courte arête locale de la piale est la plus longue. Une piale sans arête
/// courte est rendue telle quelle (ainsi que la blanche).
pub fn collapse_paired_short_edges(
    white: &TriMesh,
    pial: &TriMesh,
    params: &PairedParams,
) -> Result<PairedCollapse, NotClosedManifold> {
    assert_eq!(
        white.faces(),
        pial.faces(),
        "blanche et piale doivent avoir la même triangulation"
    );
    let n = white.n_vertices();
    let short = |m: &HalfEdgeMesh| -> Vec<(u32, u32)> {
        let mut s: Vec<(f64, u32)> = m
            .edges()
            .into_iter()
            .map(|h| (m.edge_length(h), h))
            .filter(|&(l, _)| l < params.min_len)
            .collect();
        s.sort_by(|x, y| x.0.total_cmp(&y.0));
        s.into_iter().map(|(_, h)| (m.from(h), m.to(h))).collect()
    };
    let unchanged = || PairedCollapse {
        white: white.clone(),
        pial: pial.clone(),
        sources: (0..n as u32).map(|v| vec![v]).collect(),
        operations: 0,
        remaining: 0,
    };
    let mut pair = Pair {
        w: HalfEdgeMesh::from_mesh(white)?,
        p: HalfEdgeMesh::from_mesh(pial)?,
        members: (0..n as u32).map(|v| vec![v]).collect(),
    };
    if short(&pair.p).is_empty() {
        return Ok(unchanged());
    }
    // Une opération ne déplace un sommet que d'une arête au plus : la grille des positions d'origine des deux
    // surfaces, élargie de deux arêtes, retrouve tout ce qui peut entrer en conflit.
    let longest = |m: &HalfEdgeMesh| m.edges().into_iter().map(|h| m.edge_length(h)).fold(0.0, f64::max);
    let pad = 2.0 * longest(&pair.w).max(longest(&pair.p)) + params.min_distance.max(params.min_clearance);
    let pts: Vec<Vec3> = white.vertices().iter().chain(pial.vertices()).copied().collect();
    let hood = Neighborhood {
        grid: PointGrid::new(&pts, pad.max(1e-6)),
        n: n as u32,
        pad,
    };
    let mut operations = 0;
    for _ in 0..params.passes {
        let todo = short(&pair.p);
        if todo.is_empty() {
            break;
        }
        let mut changed = false;
        for (a, b) in todo {
            let Some(h) = half_edge(&pair.p, a, b) else { continue };
            if pair.p.edge_length(h) >= params.min_len {
                continue;
            }
            let mut ring: Vec<u32> = vec![a, b];
            for _ in 0..2 {
                let more: Vec<u32> = ring.iter().flat_map(|&v| pair.p.neighbors(v)).collect();
                ring.extend(more);
                ring.sort_unstable();
                ring.dedup();
            }
            let before = hood.measure(&pair, &ring);
            let mut ops: Vec<Vec<Op>> = [At::Middle, At::From, At::To]
                .into_iter()
                .map(|at| vec![Op::Collapse(a, b, at)])
                .collect();
            let mut around: Vec<(u32, u32)> = [a, b]
                .iter()
                .flat_map(|&v| pair.p.outgoing(v))
                .flat_map(|e| [e, pair.p.next(e), pair.p.prev(e)])
                .map(|e| {
                    let (x, y) = (pair.p.from(e), pair.p.to(e));
                    (x.min(y), x.max(y))
                })
                .filter(|&(x, y)| (x, y) != (a.min(b), a.max(b)))
                .collect();
            around.sort_unstable();
            around.dedup();
            for (x, y) in around {
                let firsts = [
                    Op::Flip(x, y),
                    Op::Collapse(x, y, At::Middle),
                    Op::Collapse(x, y, At::From),
                    Op::Collapse(x, y, At::To),
                ];
                for first in firsts {
                    for at in [At::Middle, At::From, At::To] {
                        ops.push(vec![first, Op::Collapse(a, b, at)]);
                    }
                }
            }
            // Voisins communs de a et b en plus des deux sommets opposés (link condition violée, par exemple un
            // sommet de valence 3 posé sur l'arête) : deux opérations préalables peuvent être nécessaires, sur
            // les arêtes qui les relient à a ou b.
            let apex = [pair.p.to(pair.p.next(h)), pair.p.to(pair.p.next(pair.p.twin(h)))];
            let nb = pair.p.neighbors(b);
            let extra: Vec<u32> = pair
                .p
                .neighbors(a)
                .into_iter()
                .filter(|c| nb.contains(c) && !apex.contains(c))
                .collect();
            if !extra.is_empty() {
                let unlink: Vec<Op> = extra
                    .iter()
                    .flat_map(|&c| [(c, a), (c, b)])
                    .map(|(x, y)| (x.min(y), x.max(y)))
                    .flat_map(|(x, y)| {
                        [
                            Op::Flip(x, y),
                            Op::Collapse(x, y, At::Middle),
                            Op::Collapse(x, y, At::From),
                            Op::Collapse(x, y, At::To),
                        ]
                    })
                    .collect();
                for (i, &first) in unlink.iter().enumerate() {
                    for (j, &second) in unlink.iter().enumerate() {
                        if i != j {
                            for at in [At::Middle, At::From, At::To] {
                                ops.push(vec![first, second, Op::Collapse(a, b, at)]);
                            }
                        }
                    }
                }
            }
            let mut best: Option<(Pair, f64)> = None;
            // Diagnostic (SURFOXIDE_DEBUG) : refus par critère des essais qui allongent l'arête.
            let mut refused = [0usize; 6];
            let n_ops = ops.len();
            for seq in ops {
                let mut t = Some(pair.clone());
                for (k, &op) in seq.iter().enumerate() {
                    let cur = t.take().unwrap();
                    // Si la fusion préalable a déjà supprimé a ou b, l'arête courte a disparu avec elle.
                    let gone = k > 0 && half_edge(&cur.p, a, b).is_none();
                    t = if gone { Some(cur) } else { cur.apply(op) };
                    if t.is_none() {
                        break;
                    }
                }
                let Some(t) = t else { continue };
                let after = hood.measure(&t, &ring);
                if after.pial_edge <= before.pial_edge + 1e-12 {
                    refused[0] += 1;
                    continue;
                }
                if best.as_ref().is_some_and(|(_, e)| after.pial_edge <= *e) {
                    continue;
                }
                let tests = [
                    after.white_edge >= before.white_edge.min(params.min_len),
                    after.cone >= before.cone.min(params.min_cos),
                    after.conflicts <= before.conflicts,
                    after.gap >= before.gap.min(params.min_distance),
                    after.clearance >= before.clearance.min(params.min_clearance),
                ];
                for (k, &t) in tests.iter().enumerate() {
                    refused[k + 1] += usize::from(!t);
                }
                if tests.iter().all(|&t| t) {
                    let e = after.pial_edge;
                    best = Some((t, e));
                }
            }
            if best.is_none() && std::env::var_os("SURFOXIDE_DEBUG").is_some() {
                eprintln!(
                    "  fusion couplée : arête {a}–{b} ({:.4} mm) refusée ; essais {}, sans gain {}, refus blanche {} \
                     cône {} intersections {} distance {} dégagement {} (avant : blanche {:.4}, cône {:.3}, \
                     intersections {}, distance {:.4}, dégagement {:.4})",
                    pair.p.edge_length(h),
                    n_ops,
                    refused[0],
                    refused[1],
                    refused[2],
                    refused[3],
                    refused[4],
                    refused[5],
                    before.white_edge,
                    before.cone,
                    before.conflicts,
                    before.gap,
                    before.clearance
                );
            }
            if let Some((t, _)) = best {
                pair = t;
                operations += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    if operations == 0 {
        return Ok(PairedCollapse {
            remaining: short(&pair.p).len(),
            ..unchanged()
        });
    }
    let remaining = short(&pair.p).len();
    let sources = pair
        .members
        .iter()
        .zip(&pair.w.vertex_alive)
        .filter(|(_, &alive)| alive)
        .map(|(m, _)| {
            let mut m = m.clone();
            m.sort_unstable();
            m
        })
        .collect();
    Ok(PairedCollapse {
        white: pair.w.to_mesh(),
        pial: pair.p.to_mesh(),
        sources,
        operations,
        remaining,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    fn params(min_len: f64) -> PairedParams {
        PairedParams {
            min_len,
            min_cos: 0.05,
            min_distance: 0.03,
            min_clearance: 0.0,
            passes: 5,
        }
    }

    fn min_edge(m: &TriMesh) -> f64 {
        let v = m.vertices();
        m.faces()
            .iter()
            .flat_map(|f| (0..3).map(move |k| vec3::dist(v[f[k] as usize], v[f[(k + 1) % 3] as usize])))
            .fold(f64::INFINITY, f64::min)
    }

    /// Une arête de la piale raccourcie (la blanche est intacte) disparaît des deux surfaces, qui gardent la même
    /// triangulation, restent valides et imbriquées ; chaque sommet d'origine est reporté une fois.
    #[test]
    fn arete_courte_de_la_piale_fusionnee_sur_les_deux_surfaces() {
        let white = icosphere(3, 5.0);
        let mut pv: Vec<Vec3> = white.vertices().iter().map(|&p| vec3::scale(p, 1.3)).collect();
        let f = white.faces()[10];
        let (x, y) = (f[0] as usize, f[1] as usize);
        pv[x] = vec3::lerp(pv[x], pv[y], 0.95);
        let pial = TriMesh::new(pv, white.faces().to_vec()).unwrap();
        let min_len = 0.25 * min_edge(&white);
        assert!(min_edge(&pial) < min_len);
        let r = collapse_paired_short_edges(&white, &pial, &params(min_len)).unwrap();
        assert!(r.operations >= 1);
        assert_eq!(r.remaining, 0);
        assert_eq!(r.white.faces(), r.pial.faces());
        assert_eq!(r.white.n_vertices(), white.n_vertices() - r.operations);
        assert!(min_edge(&r.pial) >= min_len, "arête min {}", min_edge(&r.pial));
        let mut all: Vec<u32> = r.sources.iter().flatten().copied().collect();
        all.sort_unstable();
        assert_eq!(all, (0..white.n_vertices() as u32).collect::<Vec<_>>());
        assert!(surfoxide_check::check_surface(&r.white).valid);
        assert!(surfoxide_check::check_surface(&r.pial).valid);
        assert!(surfoxide_check::check_pair(&r.white, &r.pial, 0.02).valid);
    }

    /// Sans arête courte, les deux surfaces sont rendues telles quelles.
    #[test]
    fn sans_arete_courte_rien_ne_change() {
        let white = icosphere(2, 5.0);
        let pv: Vec<Vec3> = white.vertices().iter().map(|&p| vec3::scale(p, 1.3)).collect();
        let pial = TriMesh::new(pv, white.faces().to_vec()).unwrap();
        let r = collapse_paired_short_edges(&white, &pial, &params(0.1)).unwrap();
        assert_eq!(r.operations, 0);
        assert_eq!(r.white, white);
        assert_eq!(r.pial, pial);
    }
}
