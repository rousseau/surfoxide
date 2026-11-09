//! Voxelisation d'un couple blanche/piale : fractions de volume partiel par sous-échantillonnage.
//!
//! Chaque voxel est découpé en `SUB × SUB × SUB` sous-échantillons placés au centre de sous-cubes égaux. Un
//! sous-échantillon est dans une surface fermée si son indice d'enroulement y est positif ; il est blanc s'il est
//! dans la blanche, cortex s'il est dans la piale et hors de la blanche, LCR sinon. La blanche l'emporte donc là
//! où la piale passe sous elle.
//!
//! L'indice d'enroulement est calculé le long de rayons parallèles à l'axe x : pour une surface fermée orientée
//! (normales sortantes), il vaut la somme, sur les faces traversées par le rayon issu du point vers +x, du signe de
//! la composante x de leur normale. C'est la même quantité que [`surfoxide_check::winding_number`] (angle solide
//! total / 4π) hors des points de la surface elle-même, mais un rayon donne d'un coup les `SUB · nx` points de sa
//! ligne. Les rayons sont décalés d'une fraction irrationnelle infime de voxel pour ne jamais passer exactement par
//! une arête ou un sommet projetés.

use rayon::prelude::*;
use surfoxide_core::vec3::Vec3;
use surfoxide_core::{Affine, TriMesh};

/// Sous-échantillons par voxel et par axe.
pub const SUB: usize = 4;

/// Classes d'un sous-échantillon, dans l'ordre des compteurs de [`Counts`].
pub const WHITE: usize = 0;
pub const CORTEX: usize = 1;
pub const CSF: usize = 2;

/// Grille régulière alignée sur les axes du monde (RAS, mm), voxels cubiques.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    pub dims: [usize; 3],
    /// Centre du voxel (0, 0, 0), en mm.
    pub origin: Vec3,
    /// Côté d'un voxel, en mm.
    pub voxel: f64,
}

impl Grid {
    /// Plus petite grille dont les centres couvrent les boîtes englobantes de `meshes` agrandies de `margin` mm.
    pub fn around(meshes: &[&TriMesh], voxel: f64, margin: f64) -> Self {
        assert!(voxel > 0.0, "taille de voxel strictement positive attendue");
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for m in meshes {
            for p in m.vertices() {
                for a in 0..3 {
                    lo[a] = lo[a].min(p[a]);
                    hi[a] = hi[a].max(p[a]);
                }
            }
        }
        let origin = lo.map(|x| x - margin);
        let dims = [0, 1, 2].map(|a| ((hi[a] + margin - origin[a]) / voxel).ceil() as usize + 1);
        Self { dims, origin, voxel }
    }

    /// Affine voxel → monde.
    pub fn affine(&self) -> Affine {
        let v = self.voxel;
        let o = self.origin;
        [
            [v, 0.0, 0.0, o[0]],
            [0.0, v, 0.0, o[1]],
            [0.0, 0.0, v, o[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Coordonnée monde sur l'axe `a` du sous-échantillon `s` (`SUB` = centre du voxel) du voxel d'indice `i`.
    fn coord(&self, a: usize, i: usize, s: usize) -> f64 {
        let frac = if s == SUB {
            0.0
        } else {
            (s as f64 + 0.5) / SUB as f64 - 0.5
        };
        self.origin[a] + (i as f64 + frac) * self.voxel
    }
}

/// Compteurs d'un voxel : sous-échantillons blancs, cortex et LCR (somme `SUB³`), puis classe du centre du voxel.
pub type Counts = [u8; 4];

/// Triangles d'une surface rangés par cellule (y, z) de la grille, sommets projetés gardés en (x, y, z).
struct Bins {
    tris: Vec<[Vec3; 3]>,
    cells: Vec<Vec<u32>>,
}

impl Bins {
    fn new(mesh: &TriMesh, grid: &Grid) -> Self {
        let [_, ny, nz] = grid.dims;
        let v = mesh.vertices();
        let tris: Vec<[Vec3; 3]> = mesh.faces().iter().map(|f| f.map(|i| v[i as usize])).collect();
        let mut cells = vec![Vec::new(); ny * nz];
        // Cellule j de l'axe a : [origin + (j − ½) v, origin + (j + ½) v).
        let cell = |a: usize, x: f64, n: usize| {
            (((x - grid.origin[a]) / grid.voxel + 0.5).floor().max(0.0) as usize).min(n - 1)
        };
        for (t, p) in tris.iter().enumerate() {
            let (ylo, yhi) = (p[0][1].min(p[1][1]).min(p[2][1]), p[0][1].max(p[1][1]).max(p[2][1]));
            let (zlo, zhi) = (p[0][2].min(p[1][2]).min(p[2][2]), p[0][2].max(p[1][2]).max(p[2][2]));
            for k in cell(2, zlo, nz)..=cell(2, zhi, nz) {
                for j in cell(1, ylo, ny)..=cell(1, yhi, ny) {
                    cells[j + ny * k].push(t as u32);
                }
            }
        }
        Self { tris, cells }
    }

    /// Traversées du rayon (y, z) vers +x par les triangles de la cellule `cell` : (x, signe), triées par x.
    fn crossings(&self, cell: usize, y: f64, z: f64, out: &mut Vec<(f64, i32)>) {
        out.clear();
        for &t in &self.cells[cell] {
            let [a, b, c] = self.tris[t as usize];
            // Orientations en projection sur le plan (y, z) ; area2 = composante x de (b − a) × (c − a).
            let orient = |p: Vec3, q: Vec3, ry: f64, rz: f64| (q[1] - p[1]) * (rz - p[2]) - (q[2] - p[2]) * (ry - p[1]);
            let area2 = orient(a, b, c[1], c[2]);
            if area2 == 0.0 {
                continue;
            }
            let wa = orient(b, c, y, z);
            let wb = orient(c, a, y, z);
            let wc = orient(a, b, y, z);
            let inside = if area2 > 0.0 {
                wa > 0.0 && wb > 0.0 && wc > 0.0
            } else {
                wa < 0.0 && wb < 0.0 && wc < 0.0
            };
            if inside {
                let x = (wa * a[0] + wb * b[0] + wc * c[0]) / area2;
                out.push((x, if area2 > 0.0 { 1 } else { -1 }));
            }
        }
        out.sort_by(|p, q| p.0.total_cmp(&q.0));
    }
}

/// Décalage des rayons, en fraction de voxel : irrationnel et infime.
const JITTER: [f64; 2] = [1.414_213_562e-7, 1.732_050_808e-7];

/// Indice d'enroulement aux abscisses croissantes `xs` d'une ligne, à partir de ses traversées triées.
fn winding_along(xs: &[f64], crossings: &[(f64, i32)]) -> Vec<i32> {
    let mut w = vec![0; xs.len()];
    let mut acc = 0;
    let mut c = crossings.len();
    for (idx, &x) in xs.iter().enumerate().rev() {
        while c > 0 && crossings[c - 1].0 > x {
            c -= 1;
            acc += crossings[c].1;
        }
        w[idx] = acc;
    }
    w
}

/// Compteurs de chaque voxel de `grid` pour le couple (`white`, `pial`), rangés avec x le plus rapide.
pub fn counts(grid: &Grid, white: &TriMesh, pial: &TriMesh) -> Vec<Counts> {
    let [nx, ny, _] = grid.dims;
    let bins = [Bins::new(white, grid), Bins::new(pial, grid)];
    let mut out = vec![[0u8; 4]; grid.len()];
    out.par_chunks_mut(nx).enumerate().for_each(|(row, line)| {
        let (j, k) = (row % ny, row / ny);
        let mut cross = Vec::new();
        // Rayons : SUB × SUB sous-lignes, puis la ligne des centres (s = SUB).
        for sz in 0..=SUB {
            for sy in 0..=SUB {
                let centre = sy == SUB && sz == SUB;
                if (sy == SUB) != (sz == SUB) {
                    continue;
                }
                let y = grid.coord(1, j, sy) + JITTER[0] * grid.voxel;
                let z = grid.coord(2, k, sz) + JITTER[1] * grid.voxel;
                let subs: Vec<usize> = if centre { vec![SUB] } else { (0..SUB).collect() };
                let xs: Vec<f64> = (0..nx)
                    .flat_map(|i| subs.iter().map(move |&s| (i, s)))
                    .map(|(i, s)| grid.coord(0, i, s))
                    .collect();
                let w: [Vec<i32>; 2] = [0, 1].map(|b| {
                    bins[b].crossings(row, y, z, &mut cross);
                    winding_along(&xs, &cross)
                });
                for (idx, (ww, wp)) in w[0].iter().zip(&w[1]).enumerate() {
                    let class = if *ww > 0 {
                        WHITE
                    } else if *wp > 0 {
                        CORTEX
                    } else {
                        CSF
                    };
                    let voxel = &mut line[idx / subs.len()];
                    if centre {
                        voxel[3] = class as u8;
                    } else {
                        voxel[class] += 1;
                    }
                }
            }
        }
    });
    out
}

/// Classe majoritaire d'un voxel ; en cas d'égalité, celle du centre du voxel si elle en fait partie, sinon la
/// première dans l'ordre blanche, cortex, LCR.
pub fn majority(c: &Counts) -> usize {
    let best = c[..3].iter().copied().max().unwrap_or(0);
    let centre = c[3] as usize;
    if c[centre] == best {
        centre
    } else {
        (0..3).find(|&k| c[k] == best).unwrap_or(CSF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    fn total(c: &[Counts], class: usize, grid: &Grid) -> f64 {
        let n: u64 = c.iter().map(|v| v[class] as u64).sum();
        n as f64 / (SUB * SUB * SUB) as f64 * grid.voxel.powi(3)
    }

    #[test]
    fn indice_enroulement_identique_a_surfoxide_check() {
        let w = icosphere(2, 5.0);
        let p = icosphere(2, 8.0);
        let grid = Grid::around(&[&p], 1.3, 1.0);
        let c = counts(&grid, &w, &p);
        let [nx, ny, _] = grid.dims;
        for (idx, v) in c.iter().enumerate().step_by(7) {
            let (i, j, k) = (idx % nx, (idx / nx) % ny, idx / (nx * ny));
            let q = [0, 1, 2].map(|a| grid.coord(a, [i, j, k][a], SUB));
            let inside_w = surfoxide_check::winding_number(&w, q) > 0.5;
            let inside_p = surfoxide_check::winding_number(&p, q) > 0.5;
            let expected = if inside_w {
                WHITE
            } else if inside_p {
                CORTEX
            } else {
                CSF
            };
            assert_eq!(v[3] as usize, expected, "voxel {i} {j} {k}");
        }
    }

    #[test]
    fn volumes_des_fractions_egaux_aux_volumes_enfermes() {
        let w = icosphere(4, 6.0);
        let p = icosphere(4, 9.0);
        let grid = Grid::around(&[&p], 0.5, 1.0);
        let c = counts(&grid, &w, &p);
        let vw = total(&c, WHITE, &grid);
        let vc = total(&c, CORTEX, &grid);
        assert!(
            (vw / w.signed_volume() - 1.0).abs() < 2e-3,
            "blanche {vw} contre {}",
            w.signed_volume()
        );
        let vc_ref = p.signed_volume() - w.signed_volume();
        assert!((vc / vc_ref - 1.0).abs() < 2e-3, "cortex {vc} contre {vc_ref}");
        assert!(c
            .iter()
            .all(|v| v[..3].iter().map(|&x| x as usize).sum::<usize>() == SUB * SUB * SUB));
    }
}
