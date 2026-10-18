//! Marching cubes à connexité cohérente (26, 6).
//!
//! Dans un cube, deux coins intérieurs sont toujours 26-adjacents : l'intérieur du cube est d'un seul
//! tenant. Les coins extérieurs ne sont reliés que par les arêtes du cube (6-connexité). La table des
//! 256 configurations est construite par programme à partir de cette règle :
//!
//! 1. sur chaque face, un segment par coin extérieur isolé ou par paire de coins extérieurs voisins ;
//!    une face ambiguë (coins intérieurs en diagonale) est coupée en deux segments qui isolent chacun
//!    un coin extérieur ;
//! 2. les segments, orientés de façon cohérente, forment des boucles fermées sur le bord du cube ;
//! 3. chaque composante 6-connexe des coins extérieurs est bordée par une boucle (on la ferme par un
//!    disque) ou par deux boucles (on les relie par un tube : les deux coins intérieurs opposés par la
//!    diagonale restent reliés).
//!
//! Les faces voisines étant découpées de la même façon dans les deux cubes qui les partagent, la surface
//! est fermée et 2-variété, et sa topologie est celle de l'objet digital en connexité (26, 6).

use std::collections::HashMap;
use std::sync::OnceLock;

use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::volume::affine_determinant;
use surfoxide_core::{TriMesh, Volume};

/// Coins du cube : bit `c` de l'indice, coordonnées `(c & 1, c >> 1 & 1, c >> 2 & 1)`.
fn corner(c: usize) -> [usize; 3] {
    [c & 1, (c >> 1) & 1, (c >> 2) & 1]
}

/// Les 12 arêtes du cube, par paires de coins.
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7], // selon x
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7], // selon y
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7], // selon z
];

/// Les 6 faces, coins dans l'ordre direct vu de l'extérieur du cube.
const FACES: [[usize; 4]; 6] = [
    [0, 4, 6, 2], // x = 0
    [1, 3, 7, 5], // x = 1
    [0, 1, 5, 4], // y = 0
    [2, 6, 7, 3], // y = 1
    [0, 2, 3, 1], // z = 0
    [4, 5, 7, 6], // z = 1
];

fn edge_of(a: usize, b: usize) -> usize {
    EDGES
        .iter()
        .position(|e| (e[0] == a && e[1] == b) || (e[0] == b && e[1] == a))
        .expect("arête du cube")
}

/// Morceau de surface dans un cube, en indices d'arêtes coupées.
#[derive(Debug, Clone)]
enum Patch {
    /// Boucle fermée par un disque.
    Disk(Vec<usize>),
    /// Deux boucles reliées par un tube.
    Tube(Vec<usize>, Vec<usize>),
}

/// Construit les morceaux de surface de la configuration `inside` (bit `c` : coin `c` intérieur).
fn build_case(inside: u8) -> Vec<Patch> {
    let is_in = |c: usize| inside >> c & 1 == 1;
    // 1. Segments orientés (arête de départ → arête d'arrivée) sur chaque face. Vu de l'extérieur du cube,
    //    en parcourant la face dans le sens direct, on va de l'arête où l'on entre dans un coin extérieur
    //    à celle où l'on en sort : l'intérieur reste du même côté de chaque segment.
    let mut next: HashMap<usize, usize> = HashMap::new();
    for face in FACES {
        for k in 0..4 {
            let c = face[k];
            if is_in(c) {
                continue;
            }
            let prev = face[(k + 3) % 4];
            let succ = face[(k + 1) % 4];
            // Entrée dans une suite de coins extérieurs : depuis un coin intérieur précédent.
            if !is_in(prev) {
                continue;
            }
            // Parcours de la suite de coins extérieurs consécutifs (au plus 3).
            let mut last = c;
            let mut m = k;
            while !is_in(face[(m + 1) % 4]) {
                m = (m + 1) % 4;
                last = face[m];
            }
            let exit_corner = face[(m + 1) % 4];
            let _ = succ;
            next.insert(edge_of(prev, c), edge_of(last, exit_corner));
        }
    }
    // 2. Boucles.
    let mut loops: Vec<Vec<usize>> = Vec::new();
    let mut used = [false; 12];
    let mut starts: Vec<usize> = next.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        if used[s] {
            continue;
        }
        let mut l = vec![s];
        used[s] = true;
        let mut e = next[&s];
        while e != s {
            used[e] = true;
            l.push(e);
            e = next[&e];
        }
        loops.push(l);
    }
    // 3. Composantes 6-connexes des coins extérieurs, et boucles qui les bordent.
    let mut comp = [usize::MAX; 8];
    let mut n_comp = 0;
    for c in 0..8 {
        if is_in(c) || comp[c] != usize::MAX {
            continue;
        }
        let mut stack = vec![c];
        comp[c] = n_comp;
        while let Some(x) = stack.pop() {
            for e in EDGES {
                let y = if e[0] == x {
                    e[1]
                } else if e[1] == x {
                    e[0]
                } else {
                    continue;
                };
                if !is_in(y) && comp[y] == usize::MAX {
                    comp[y] = n_comp;
                    stack.push(y);
                }
            }
        }
        n_comp += 1;
    }
    let loop_comp = |l: &Vec<usize>| {
        let [a, b] = EDGES[l[0]];
        comp[if is_in(a) { b } else { a }]
    };
    let mut patches = Vec::new();
    for k in 0..n_comp {
        let ls: Vec<&Vec<usize>> = loops.iter().filter(|l| loop_comp(l) == k).collect();
        match ls.as_slice() {
            [] => {}
            [l] => patches.push(Patch::Disk((*l).clone())),
            [a, b] => patches.push(Patch::Tube((*a).clone(), (*b).clone())),
            _ => unreachable!("au plus deux boucles par composante extérieure"),
        }
    }
    patches
}

fn table() -> &'static Vec<Vec<Patch>> {
    static T: OnceLock<Vec<Vec<Patch>>> = OnceLock::new();
    T.get_or_init(|| (0..=255u8).map(build_case).collect())
}

/// Ajoute des triangles fermant la boucle `l` (indices de sommets du maillage).
fn close_disk(l: &[u32], pos: &mut Vec<Vec3>, out: &mut Vec<[u32; 3]>) {
    match l.len() {
        3 => out.push([l[0], l[1], l[2]]),
        4 => {
            // Diagonale la plus courte.
            let d02 = vec3::dist(pos[l[0] as usize], pos[l[2] as usize]);
            let d13 = vec3::dist(pos[l[1] as usize], pos[l[3] as usize]);
            if d02 <= d13 {
                out.push([l[0], l[1], l[2]]);
                out.push([l[0], l[2], l[3]]);
            } else {
                out.push([l[1], l[2], l[3]]);
                out.push([l[1], l[3], l[0]]);
            }
        }
        n => {
            // Éventail autour du barycentre : évite les triangles qui se recouvrent sur les boucles longues.
            let c = l
                .iter()
                .fold([0.0; 3], |acc, &i| vec3::add(acc, pos[i as usize]));
            pos.push(vec3::scale(c, 1.0 / n as f64));
            let ci = (pos.len() - 1) as u32;
            for k in 0..n {
                out.push([l[k], l[(k + 1) % n], ci]);
            }
        }
    }
}

/// Relie deux boucles par un tube : `a` parcourue dans son sens, `b` à rebours, en choisissant à chaque
/// pas la diagonale la plus courte.
fn close_tube(a: &[u32], b: &[u32], pos: &[Vec3], out: &mut Vec<[u32; 3]>) {
    let (m, n) = (a.len(), b.len());
    let p = |i: u32| pos[i as usize];
    // Départ : la paire de sommets la plus proche.
    let (mut i0, mut j0, mut best) = (0, 0, f64::INFINITY);
    for (i, &x) in a.iter().enumerate() {
        for (j, &y) in b.iter().enumerate() {
            let d = vec3::dist(p(x), p(y));
            if d < best {
                (i0, j0, best) = (i, j, d);
            }
        }
    }
    let (mut ia, mut ib) = (0, 0); // pas effectués sur a et sur b
    while ia < m || ib < n {
        let ai = a[(i0 + ia) % m];
        let ai1 = a[(i0 + ia + 1) % m];
        let bj = b[(j0 + n - ib % n) % n];
        let bj1 = b[(j0 + 2 * n - ib - 1) % n];
        let advance_a = if ia == m {
            false
        } else if ib == n {
            true
        } else {
            vec3::dist(p(ai1), p(bj)) <= vec3::dist(p(ai), p(bj1))
        };
        if advance_a {
            out.push([ai, ai1, bj]);
            ia += 1;
        } else {
            out.push([bj1, bj, ai]);
            ib += 1;
        }
    }
}

/// Extrait la surface `φ = 0` séparant l'objet `φ < 0` du fond, en coordonnées monde.
///
/// Le volume doit être positif ou nul sur tout son bord (objet recadré avec une marge) pour que la
/// surface soit fermée. Les points sont placés par interpolation linéaire sur les arêtes, à au moins
/// `min_frac` de chaque extrémité (évite les triangles d'aire nulle). Les faces sont orientées vers
/// l'extérieur (`φ > 0`).
pub fn marching_cubes(phi: &Volume<f64>, min_frac: f64) -> TriMesh {
    let [nx, ny, nz] = phi.dims();
    let p = phi.data();
    let table = table();
    let mut vertex_of: HashMap<(usize, u8), u32> = HashMap::new();
    let mut pos: Vec<Vec3> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();
    let idx = |i: usize, j: usize, k: usize| i + nx * (j + ny * k);
    for k in 0..nz.saturating_sub(1) {
        for j in 0..ny.saturating_sub(1) {
            for i in 0..nx.saturating_sub(1) {
                let mut case = 0u8;
                let mut vals = [0.0; 8];
                for (c, v) in vals.iter_mut().enumerate() {
                    let [a, b, d] = corner(c);
                    *v = p[idx(i + a, j + b, k + d)];
                    if *v < 0.0 {
                        case |= 1 << c;
                    }
                }
                if case == 0 || case == 255 {
                    continue;
                }
                // Sommet (partagé entre cubes) de chaque arête coupée.
                let mut vid = |e: usize, pos: &mut Vec<Vec3>| -> u32 {
                    let [c0, c1] = EDGES[e];
                    let (o0, o1) = (corner(c0), corner(c1));
                    let base = idx(i + o0[0], j + o0[1], k + o0[2]);
                    let axis = (0..3).find(|&a| o0[a] != o1[a]).unwrap() as u8;
                    *vertex_of.entry((base, axis)).or_insert_with(|| {
                        let (f0, f1) = (vals[c0], vals[c1]);
                        let t = (f0 / (f0 - f1)).clamp(min_frac, 1.0 - min_frac);
                        let a = [(i + o0[0]) as f64, (j + o0[1]) as f64, (k + o0[2]) as f64];
                        let b = [(i + o1[0]) as f64, (j + o1[1]) as f64, (k + o1[2]) as f64];
                        pos.push(vec3::lerp(a, b, t));
                        (pos.len() - 1) as u32
                    })
                };
                for patch in &table[case as usize] {
                    match patch {
                        Patch::Disk(l) => {
                            let l: Vec<u32> = l.iter().map(|&e| vid(e, &mut pos)).collect();
                            close_disk(&l, &mut pos, &mut faces);
                        }
                        Patch::Tube(a, b) => {
                            let a: Vec<u32> = a.iter().map(|&e| vid(e, &mut pos)).collect();
                            let b: Vec<u32> = b.iter().map(|&e| vid(e, &mut pos)).collect();
                            close_tube(&a, &b, &pos, &mut faces);
                        }
                    }
                }
            }
        }
    }
    let world: Vec<Vec3> = pos.into_iter().map(|q| phi.voxel_to_world(q)).collect();
    let mut mesh = TriMesh::new(world, faces).expect("indices valides par construction");
    // Les tables orientent les faces dans le repère voxel ; une affine indirecte inverse l'orientation.
    if !ORIENTATION_OUTWARD ^ (affine_determinant(phi.affine()) < 0.0) {
        mesh.flip_orientation();
    }
    mesh
}

/// Les boucles sont orientées dans le sens direct vu de l'extérieur du cube autour des coins extérieurs :
/// les triangles obtenus regardent vers l'intérieur de l'objet dans le repère voxel (vérifié par les tests).
const ORIENTATION_OUTWARD: bool = false;

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::distance::signed_distance;
    use surfoxide_core::topology_report;
    use surfoxide_core::volume::IDENTITY;
    use surfoxide_topofix::{correct_topology, Strategy};

    fn rng(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    #[test]
    fn table_complete() {
        for case in 0..=255u8 {
            let patches = build_case(case);
            let n_in = case.count_ones();
            if n_in == 0 || n_in == 8 {
                assert!(patches.is_empty());
            } else {
                assert!(!patches.is_empty(), "cas {case:08b} sans surface");
            }
            // Chaque arête coupée apparaît dans exactement une boucle.
            let mut seen = [0; 12];
            for p in &patches {
                let ls: Vec<&Vec<usize>> = match p {
                    Patch::Disk(l) => vec![l],
                    Patch::Tube(a, b) => vec![a, b],
                };
                for l in ls {
                    for &e in l {
                        seen[e] += 1;
                    }
                }
            }
            for (e, &n) in seen.iter().enumerate() {
                let [a, b] = EDGES[e];
                let cut = (case >> a & 1) != (case >> b & 1);
                assert_eq!(n, cut as usize, "cas {case:08b}, arête {e}");
            }
        }
    }

    fn ball(n: usize, r: f64) -> Volume<bool> {
        let c = (n as f64 - 1.0) / 2.0;
        let mut v = Volume::filled([n, n, n], false, IDENTITY);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d2 =
                        (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
                    v.set(i, j, k, d2 < r * r);
                }
            }
        }
        v
    }

    #[test]
    fn sphere_orientee_et_volume() {
        let r = 10.0;
        let phi = signed_distance(&ball(28, r));
        let m = marching_cubes(&phi, 0.0);
        let rep = topology_report(&m);
        assert!(rep.is_sphere_like(), "{rep:?}");
        let v = m.signed_volume();
        let exact = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        assert!(v > 0.0, "normales rentrantes");
        assert!(
            (v - exact).abs() / exact < 0.03,
            "volume {v} contre {exact}"
        );
    }

    #[test]
    fn affine_indirecte_garde_les_normales_sortantes() {
        let b = ball(16, 5.0);
        let aff = [
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let b = Volume::from_vec(b.dims(), b.data().to_vec(), aff);
        let m = marching_cubes(&signed_distance(&b), 0.0);
        assert!(m.signed_volume() > 0.0);
    }

    /// Objets aléatoires corrigés en boules digitales : la surface doit être une sphère topologique,
    /// fermée, 2-variété et orientée, quelle que soit la configuration des cubes rencontrés.
    #[test]
    fn objets_aleatoires_donnent_des_spheres() {
        let mut seed = 0xDEADBEEFu64;
        for trial in 0..60 {
            let n = 10;
            let density = 3 + trial % 5; // sur 10
            let mut v = Volume::filled([n, n, n], 1.0, IDENTITY);
            for k in 1..n - 1 {
                for j in 1..n - 1 {
                    for i in 1..n - 1 {
                        if rng(&mut seed) % 10 < density as u64 {
                            v.set(i, j, k, -1.0);
                        }
                    }
                }
            }
            let c = correct_topology(&v, Strategy::GrowObject, 0.5).unwrap();
            let m = marching_cubes(&c.levelset, 0.0);
            let rep = topology_report(&m);
            assert!(rep.is_sphere_like(), "essai {trial} : {rep:?}");
            assert!(
                m.signed_volume() > 0.0,
                "essai {trial} : normales rentrantes"
            );
        }
    }

    /// Sans correction topologique, la topologie de la surface reste celle de l'objet digital (26, 6) :
    /// caractéristique d'Euler égale à celle du complexe cubique de l'objet.
    #[test]
    fn topologie_digitale_respectee() {
        let mut seed = 12345u64;
        for trial in 0..40 {
            let n = 9;
            let mut v = Volume::filled([n, n, n], 1.0, IDENTITY);
            for k in 1..n - 1 {
                for j in 1..n - 1 {
                    for i in 1..n - 1 {
                        if rng(&mut seed).is_multiple_of(2) {
                            v.set(i, j, k, -1.0);
                        }
                    }
                }
            }
            let m = marching_cubes(&v, 0.0);
            let rep = topology_report(&m);
            assert!(rep.is_closed_oriented_manifold(), "essai {trial} : {rep:?}");
            // χ(surface) = 2 χ(objet) pour le bord d'un solide.
            let obj = v.map(|&x| x < 0.0);
            let mut cells = std::collections::HashSet::new();
            for (idx, _) in obj.data().iter().enumerate().filter(|(_, &b)| b) {
                let [i, j, k] = obj.coords(idx);
                for c in 0..3 {
                    for b in 0..3 {
                        for a in 0..3 {
                            cells.insert((2 * i + a, 2 * j + b, 2 * k + c));
                        }
                    }
                }
            }
            let chi_obj: i64 = cells
                .iter()
                .map(|&(x, y, z)| {
                    if (x % 2 + y % 2 + z % 2) % 2 == 0 {
                        1
                    } else {
                        -1
                    }
                })
                .sum();
            assert_eq!(rep.euler_characteristic, 2 * chi_obj, "essai {trial}");
        }
    }
}
