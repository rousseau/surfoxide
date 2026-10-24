//! Opérateurs géométriques sur des tenseurs Burn : normales, aires, laplacien éléments finis, échantillonnage
//! d'un volume aux sommets, DPF. Ce sont les briques des pertes et des champs de vitesse des réseaux de
//! déformation (P1) ; chacun reproduit, en `f32` et sur le device, un opérateur CPU déjà validé
//! (`surfoxide-core`, `surfoxide-metrics`), et tous sont différentiables par rapport aux positions.

use burn::prelude::{Device, Int, Tensor, TensorData};
use surfoxide_core::{Affine, TriMesh, Volume};

/// Connectivité d'un maillage sur le device : les trois colonnes de sommets des faces.
#[derive(Debug, Clone)]
pub struct Topology {
    pub n_vertices: usize,
    pub n_faces: usize,
    /// Indices des sommets `a`, `b`, `c` de chaque face (`[F]` chacun).
    pub corners: [Tensor<1, Int>; 3],
    /// Coins incidents à chaque sommet, format ELL : `[N · k_max]` indices dans la concaténation des valeurs
    /// des coins (`k · F + f` pour le coin `k` de la face `f`), complétés par `3F` (ligne nulle ajoutée).
    /// La somme par sommet se fait ainsi par une lecture indexée, bien plus rapide sur GPU qu'une
    /// accumulation indexée (`select_assign` avec addition, sérialisée).
    ell: Tensor<1, Int>,
    k_max: usize,
}

impl Topology {
    pub fn new(mesh: &TriMesh, device: &Device) -> Self {
        let corners = std::array::from_fn(|k| {
            let ids: Vec<i32> = mesh.faces().iter().map(|f| f[k] as i32).collect();
            Tensor::from_ints(TensorData::new(ids, [mesh.n_faces()]), device)
        });
        let (n, nf) = (mesh.n_vertices(), mesh.n_faces());
        let mut lists: Vec<Vec<i32>> = vec![Vec::new(); n];
        for (fi, f) in mesh.faces().iter().enumerate() {
            for (k, &v) in f.iter().enumerate() {
                lists[v as usize].push((k * nf + fi) as i32);
            }
        }
        let k_max = lists.iter().map(Vec::len).max().unwrap_or(0).max(1);
        let pad = (3 * nf) as i32;
        let ell: Vec<i32> = lists
            .into_iter()
            .flat_map(|l| {
                let len = l.len();
                l.into_iter().chain(std::iter::repeat_n(pad, k_max - len))
            })
            .collect();
        Topology {
            n_vertices: n,
            n_faces: nf,
            corners,
            ell: Tensor::from_ints(TensorData::new(ell, [n * k_max]), device),
            k_max,
        }
    }

    /// Positions des trois coins de chaque face (`[F, 3]` chacun).
    fn corner_positions(&self, v: &Tensor<2>) -> [Tensor<2>; 3] {
        std::array::from_fn(|k| v.clone().select(0, self.corners[k].clone()))
    }

    /// Somme, par sommet, de valeurs attachées aux coins des faces : `values[k]` (`[F, C]`) est ajouté au
    /// sommet `corners[k]`. Rend `[N, C]`.
    fn scatter_corners(&self, values: [Tensor<2>; 3]) -> Tensor<2> {
        let c = values[0].dims()[1];
        let zero = Tensor::<2>::zeros([1, c], &values[0].device());
        let [a, b, d] = values;
        let all = Tensor::cat(vec![a, b, d, zero], 0);
        all.select(0, self.ell.clone())
            .reshape([self.n_vertices, self.k_max, c])
            .sum_dim(1)
            .reshape([self.n_vertices, c])
    }
}

/// Produit scalaire ligne à ligne de deux tenseurs `[M, 3]` : `[M, 1]`.
fn dot(a: Tensor<2>, b: Tensor<2>) -> Tensor<2> {
    (a * b).sum_dim(1)
}

/// Normale non normalisée de chaque face, `(b − a) × (c − a)` (`[F, 3]`), de norme deux fois l'aire.
pub fn face_normals(v: &Tensor<2>, topo: &Topology) -> Tensor<2> {
    let [a, b, c] = topo.corner_positions(v);
    (b - a.clone()).cross(c - a, 1)
}

/// Aire de chaque face (`[F, 1]`).
pub fn face_areas(v: &Tensor<2>, topo: &Topology) -> Tensor<2> {
    let n = face_normals(v, topo);
    dot(n.clone(), n).sqrt() * 0.5
}

/// Normale unitaire de chaque sommet : moyenne des normales des faces incidentes pondérée par leur aire
/// (comme `surfoxide_core::adjacency::vertex_normals`).
pub fn vertex_normals(v: &Tensor<2>, topo: &Topology) -> Tensor<2> {
    let n = face_normals(v, topo);
    let sum = topo.scatter_corners([n.clone(), n.clone(), n]);
    let norm = dot(sum.clone(), sum.clone()).sqrt().clamp_min(1e-12);
    sum / norm
}

/// Poids cotangents des trois arêtes de chaque face : `w[k] = cot(angle au coin k) / 2`, pour l'arête opposée
/// au coin `k` (`[F, 1]` chacun).
fn cot_weights(v: &Tensor<2>, topo: &Topology) -> [Tensor<2>; 3] {
    let p = topo.corner_positions(v);
    std::array::from_fn(|k| {
        let e1 = p[(k + 1) % 3].clone() - p[k].clone();
        let e2 = p[(k + 2) % 3].clone() - p[k].clone();
        let cr = e1.clone().cross(e2.clone(), 1);
        let s = dot(cr.clone(), cr).sqrt().clamp_min(1e-12);
        dot(e1, e2) / s * 0.5
    })
}

/// Produit de la matrice de raideur cotangente `S` par un champ `u` (`[N, C]`) :
/// `(S u)_i = Σ_j w_ij (u_i − u_j)`, avec les conventions de `surfoxide_metrics::laplacian::fem_matrices`.
pub fn stiffness_apply(v: &Tensor<2>, topo: &Topology, u: &Tensor<2>) -> Tensor<2> {
    let w = cot_weights(v, topo);
    let uc: [Tensor<2>; 3] = std::array::from_fn(|k| u.clone().select(0, topo.corners[k].clone()));
    // Le coin i appartient aux arêtes opposées aux coins i+1 (vers i+2) et i+2 (vers i+1).
    let contrib: [Tensor<2>; 3] = std::array::from_fn(|i| {
        let (j, l) = ((i + 1) % 3, (i + 2) % 3);
        (uc[i].clone() - uc[l].clone()) * w[j].clone() + (uc[i].clone() - uc[j].clone()) * w[l].clone()
    });
    topo.scatter_corners(contrib)
}

/// Produit de la matrice de masse cohérente `B` par `u` (`[N, C]`) : `A/6` sur la diagonale, `A/12` hors
/// diagonale, pour chaque face d'aire `A`.
pub fn mass_apply(v: &Tensor<2>, topo: &Topology, u: &Tensor<2>) -> Tensor<2> {
    let a = face_areas(v, topo);
    let uc: [Tensor<2>; 3] = std::array::from_fn(|k| u.clone().select(0, topo.corners[k].clone()));
    let sum = uc[0].clone() + uc[1].clone() + uc[2].clone();
    // (B u)_i par face : A/12 · (2 u_i + u_j + u_k) = A/12 · (u_i + somme).
    topo.scatter_corners(std::array::from_fn(|k| {
        (uc[k].clone() + sum.clone()) * a.clone() / 12.0
    }))
}

/// Diagonale de la masse cohérente (`[N, 1]`) : un sixième de l'aire des faces incidentes.
pub fn mass_diagonal(v: &Tensor<2>, topo: &Topology) -> Tensor<2> {
    let a = face_areas(v, topo) / 6.0;
    topo.scatter_corners([a.clone(), a.clone(), a])
}

/// Volume sur le device, avec l'inverse de son affine pour passer des points monde aux coordonnées voxel.
#[derive(Debug, Clone)]
pub struct DeviceVolume {
    dims: [usize; 3],
    /// Valeurs à plat (`[nx · ny · nz, 1]`), indice `i + nx (j + ny k)`.
    data: Tensor<2>,
    /// Partie linéaire (`[3, 3]`, transposée pour `points · Lᵀ`) et translation (`[1, 3]`) de l'inverse.
    linear_t: Tensor<2>,
    translation: Tensor<2>,
}

impl DeviceVolume {
    pub fn new(volume: &Volume<f64>, device: &Device) -> Self {
        let inv: Affine = surfoxide_core::volume::invert_affine(volume.affine()).expect("affine inversible");
        let data: Vec<f32> = volume.data().iter().map(|&x| x as f32).collect();
        let lt: Vec<f32> = (0..3)
            .flat_map(|c| (0..3).map(move |r| (c, r)))
            .map(|(c, r)| inv[r][c] as f32)
            .collect();
        DeviceVolume {
            dims: volume.dims(),
            data: Tensor::from_floats(TensorData::new(data, [volume.len(), 1]), device),
            linear_t: Tensor::from_floats(TensorData::new(lt, [3, 3]), device),
            translation: Tensor::from_floats(
                TensorData::new((0..3).map(|r| inv[r][3] as f32).collect(), [1, 3]),
                device,
            ),
        }
    }

    /// Interpolation trilinéaire aux points monde `p` (`[M, 3]`), bords répliqués : `[M, 1]`.
    /// Différentiable par rapport aux points (les poids dépendent d'eux, pas les indices des coins).
    pub fn sample(&self, p: &Tensor<2>) -> Tensor<2> {
        let q = p.clone().matmul(self.linear_t.clone()) + self.translation.clone();
        let mut base = Vec::with_capacity(3);
        let mut frac = Vec::with_capacity(3);
        for a in 0..3 {
            let hi = self.dims[a] as f32 - 1.0;
            let x = q.clone().narrow(1, a, 1).clamp(0.0, hi);
            // Coin inférieur, au plus à l'avant-dernier voxel : le poids du coin supérieur vaut alors 1 au bord.
            let i = x.clone().detach().floor().clamp(0.0, (hi - 1.0).max(0.0));
            frac.push(x - i.clone());
            base.push(i.int());
        }
        let [nx, ny] = [self.dims[0] as i32, self.dims[1] as i32];
        let mut acc: Option<Tensor<2>> = None;
        for corner in 0..8 {
            let o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            let mut idx = base[0].clone() + o[0];
            idx = idx + (base[1].clone() + o[1]) * nx;
            idx = idx + (base[2].clone() + o[2]) * (nx * ny);
            let mut w: Option<Tensor<2>> = None;
            for a in 0..3 {
                let wa = if o[a] == 1 {
                    frac[a].clone()
                } else {
                    frac[a].clone().neg() + 1.0
                };
                w = Some(w.map_or(wa.clone(), |w| w * wa));
            }
            let m = idx.dims()[0];
            let val = self.data.clone().select(0, idx.reshape([m]));
            let term = val * w.expect("trois poids");
            acc = Some(acc.map_or(term.clone(), |a| a + term));
        }
        acc.expect("huit coins")
    }
}

/// Passage des points monde aux coordonnées voxel continues d'une grille (sur le device).
#[derive(Debug, Clone)]
pub struct GridMap {
    dims: [usize; 3],
    /// Partie linéaire transposée (`[3, 3]`) et translation (`[1, 3]`) de l'application monde → voxel.
    linear_t: Tensor<2>,
    translation: Tensor<2>,
}

impl GridMap {
    /// Grille de dimensions `dims` dont l'affine voxel → monde est `affine`, sous-échantillonnée d'un facteur
    /// `stride` (une convolution de pas 2, noyau 3, marge 1 : le voxel `j` est le voxel `stride · j` de la grille
    /// d'origine ; dimensions ⌈n / stride⌉).
    pub fn new(dims: [usize; 3], affine: &Affine, stride: usize, device: &Device) -> Self {
        let inv = surfoxide_core::volume::invert_affine(affine).expect("affine inversible");
        let s = stride as f64;
        let lt: Vec<f32> = (0..3)
            .flat_map(|c| (0..3).map(move |r| (c, r)))
            .map(|(c, r)| (inv[r][c] / s) as f32)
            .collect();
        GridMap {
            dims: dims.map(|n| n.div_ceil(stride)),
            linear_t: Tensor::from_floats(TensorData::new(lt, [3, 3]), device),
            translation: Tensor::from_floats(
                TensorData::new((0..3).map(|r| (inv[r][3] / s) as f32).collect(), [1, 3]),
                device,
            ),
        }
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    /// Interpolation trilinéaire des canaux `flat` (`[nx · ny · nz, C]`, indice `i + nx (j + ny k)`) aux points
    /// monde `p` (`[M, 3]`), bords répliqués : `[M, C]`. Différentiable par rapport aux points et aux valeurs.
    pub fn sample(&self, flat: &Tensor<2>, p: &Tensor<2>) -> Tensor<2> {
        let q = p.clone().matmul(self.linear_t.clone()) + self.translation.clone();
        let mut base = Vec::with_capacity(3);
        let mut frac = Vec::with_capacity(3);
        for a in 0..3 {
            let hi = self.dims[a] as f32 - 1.0;
            let x = q.clone().narrow(1, a, 1).clamp(0.0, hi);
            let i = x.clone().detach().floor().clamp(0.0, (hi - 1.0).max(0.0));
            frac.push(x - i.clone());
            base.push(i.int());
        }
        let [nx, ny] = [self.dims[0] as i32, self.dims[1] as i32];
        let m = p.dims()[0];
        let mut acc: Option<Tensor<2>> = None;
        for corner in 0..8 {
            let o = [corner & 1, (corner >> 1) & 1, (corner >> 2) & 1];
            let idx = (base[0].clone() + o[0]) + (base[1].clone() + o[1]) * nx + (base[2].clone() + o[2]) * (nx * ny);
            let mut w: Option<Tensor<2>> = None;
            for a in 0..3 {
                let wa = if o[a] == 1 {
                    frac[a].clone()
                } else {
                    frac[a].clone().neg() + 1.0
                };
                w = Some(w.map_or(wa.clone(), |w| w * wa));
            }
            let term = flat.clone().select(0, idx.reshape([m])) * w.expect("trois poids");
            acc = Some(acc.map_or(term.clone(), |a| a + term));
        }
        acc.expect("huit coins")
    }
}

/// Canaux d'un tenseur de convolution `[1, C, nz, ny, nx]` mis à plat en `[nx · ny · nz, C]` (indice
/// `i + nx (j + ny k)`, celui de [`Volume`]), pour [`GridMap::sample`].
pub fn flatten_channels(t: Tensor<5>) -> Tensor<2> {
    let [_, c, nz, ny, nx] = t.dims();
    t.reshape([c, nz * ny * nx]).transpose()
}

/// DPF (même système que `surfoxide_metrics::depth_potential`) : `(S/2 + α B) D = 2 B (K − K̄)`, résolu par
/// gradient conjugué préconditionné par la diagonale, sur le device. `k` : courbure moyenne `[N, 1]`.
pub fn depth_potential(
    v: &Tensor<2>,
    topo: &Topology,
    k: &Tensor<2>,
    alpha: f64,
    tol: f64,
    max_iter: usize,
) -> Tensor<2> {
    let v = v.clone().detach();
    let d = mass_diagonal(&v, topo);
    let k_bar = (k.clone() * d.clone()).sum() / d.clone().sum();
    let centered = k.clone() - k_bar.unsqueeze::<2>();
    let rhs = mass_apply(&v, topo, &centered) * 2.0;
    let apply = |x: &Tensor<2>| stiffness_apply(&v, topo, x) * 0.5 + mass_apply(&v, topo, x) * alpha;
    // Diagonale de A : S_ii/2 + α B_ii, avec S_ii = Σ des poids cotangents des arêtes incidentes.
    let w = cot_weights(&v, topo);
    let s_diag = topo.scatter_corners(std::array::from_fn(|k| w[(k + 1) % 3].clone() + w[(k + 2) % 3].clone()));
    let inv_diag = (s_diag * 0.5 + d * alpha).recip();
    let norm_b = rhs.clone().powf_scalar(2.0).sum().into_scalar::<f32>().sqrt() as f64;
    let mut x = rhs.zeros_like();
    if norm_b == 0.0 {
        return x;
    }
    let mut r = rhs;
    let mut z = r.clone() * inv_diag.clone();
    let mut p = z.clone();
    let mut rz = (r.clone() * z.clone()).sum();
    for _ in 0..max_iter {
        let ap = apply(&p);
        let step = rz.clone() / (p.clone() * ap.clone()).sum();
        x = x + p.clone() * step.clone().unsqueeze::<2>();
        r = r - ap * step.unsqueeze::<2>();
        let res = r.clone().powf_scalar(2.0).sum().into_scalar::<f32>().sqrt() as f64;
        if res / norm_b < tol {
            break;
        }
        z = r.clone() * inv_diag.clone();
        let rz_new = (r.clone() * z.clone()).sum();
        let beta = rz_new.clone() / rz;
        rz = rz_new;
        p = z.clone() + p * beta.unsqueeze::<2>();
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{vertices_tensor, TensorError};
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::volume::IDENTITY;

    fn to_vec(t: Tensor<2>) -> Vec<f32> {
        t.into_data()
            .convert::<f32>()
            .try_into_vec()
            .map_err(|e| TensorError::Read(format!("{e:?}")))
            .unwrap()
    }

    /// Sphère bosselée : courbures et cotangentes variées, sans symétrie qui masquerait une erreur d'indice.
    fn bumpy() -> TriMesh {
        let s = icosphere(3, 1.0);
        let v = s
            .vertices()
            .iter()
            .map(|p| {
                let r = 10.0 * (1.0 + 0.1 * (5.0 * p[0]).sin() * (4.0 * p[1]).cos() + 0.05 * (3.0 * p[2]).cos());
                p.map(|x| x * r)
            })
            .collect();
        TriMesh::new(v, s.faces().to_vec()).unwrap()
    }

    fn max_rel(a: &[f32], b: &[f64]) -> f64 {
        let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-12);
        a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - y).abs())) / scale
    }

    #[test]
    fn normales_et_aires_comme_sur_cpu() {
        let device = Device::flex();
        let m = bumpy();
        let (v, topo) = (vertices_tensor(&m, &device), Topology::new(&m, &device));
        let areas: Vec<f64> = surfoxide_core::adjacency::face_areas(&m);
        assert!(max_rel(&to_vec(face_areas(&v, &topo)), &areas) < 1e-5);
        let vn: Vec<f64> = surfoxide_core::adjacency::vertex_normals(&m)
            .into_iter()
            .flatten()
            .collect();
        assert!(max_rel(&to_vec(vertex_normals(&v, &topo)), &vn) < 1e-5);
    }

    #[test]
    fn raideur_et_masse_comme_sur_cpu() {
        let device = Device::flex();
        let m = bumpy();
        let (v, topo) = (vertices_tensor(&m, &device), Topology::new(&m, &device));
        let (s, b) = surfoxide_metrics::laplacian::fem_matrices(&m);
        // Champ test : une fonction lisse des positions.
        let u: Vec<f64> = m
            .vertices()
            .iter()
            .map(|p| (0.3 * p[0]).sin() + 0.1 * p[1] * p[2])
            .collect();
        let ut = Tensor::from_floats(
            TensorData::new(u.iter().map(|&x| x as f32).collect(), [u.len(), 1]),
            &device,
        );
        assert!(max_rel(&to_vec(stiffness_apply(&v, &topo, &ut)), &s.mul_vec(&u)) < 1e-4);
        assert!(max_rel(&to_vec(mass_apply(&v, &topo, &ut)), &b.mul_vec(&u)) < 1e-5);
        assert!(max_rel(&to_vec(mass_diagonal(&v, &topo)), &b.diagonal()) < 1e-5);
    }

    #[test]
    fn echantillonnage_trilineaire_comme_sur_cpu() {
        let device = Device::flex();
        let mut aff = IDENTITY;
        aff[0][0] = 0.5;
        aff[1][1] = 0.7;
        aff[2][2] = 0.4;
        aff[0][3] = -3.0;
        aff[1][3] = 2.0;
        let dims = [9, 7, 11];
        let mut vol = Volume::filled(dims, 0.0, aff);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    vol.set(i, j, k, ((i * 7 + j * 13 + k * 5) % 17) as f64 / 17.0);
                }
            }
        }
        // Points à l'intérieur et hors du volume (bords répliqués).
        let pts: Vec<[f64; 3]> = (0..200)
            .map(|n| {
                let t = n as f64;
                [-4.0 + (t * 0.37) % 6.0, 1.0 + (t * 0.53) % 6.5, -0.5 + (t * 0.29) % 5.0]
            })
            .collect();
        let inv = surfoxide_core::volume::invert_affine(&aff).unwrap();
        let cpu: Vec<f64> = pts
            .iter()
            .map(|p| vol.trilinear(surfoxide_core::volume::apply_affine(&inv, *p)))
            .collect();
        let pt = Tensor::from_floats(
            TensorData::new(pts.iter().flatten().map(|&x| x as f32).collect(), [pts.len(), 3]),
            &device,
        );
        let gpu = to_vec(DeviceVolume::new(&vol, &device).sample(&pt));
        assert!(max_rel(&gpu, &cpu) < 1e-5);
    }

    #[test]
    fn echantillonnage_de_canaux_et_sous_echantillonnage() {
        // Deux canaux affines en (i, j, k) sur une grille anisotrope : l'interpolation trilinéaire est exacte, y
        // compris sur une grille sous-échantillonnée d'un facteur 2 (voxel j = voxel 2j de l'original).
        let device = Device::flex();
        let mut aff = IDENTITY;
        aff[0][0] = 0.5;
        aff[1][1] = 0.5;
        aff[2][2] = 0.7;
        aff[0][3] = 10.0;
        let dims = [10, 8, 6];
        let f = |i: f64, j: f64, k: f64| [1.0 + i - 2.0 * j + 0.5 * k, 3.0 * k - i];
        for stride in [1usize, 2] {
            let map = GridMap::new(dims, &aff, stride, &device);
            let [nx, ny, nz] = map.dims();
            let mut data = Vec::new();
            for k in 0..nz {
                for j in 0..ny {
                    for i in 0..nx {
                        let v = f((stride * i) as f64, (stride * j) as f64, (stride * k) as f64);
                        data.extend(v.map(|x| x as f32));
                    }
                }
            }
            let flat = Tensor::<2>::from_floats(TensorData::new(data, [nx * ny * nz, 2]), &device);
            let pts = [[11.3, 1.2, 1.5], [12.0, 2.6, 2.1]];
            let p = Tensor::<2>::from_floats(
                TensorData::new(pts.iter().flatten().map(|&x| x as f32).collect(), [2, 3]),
                &device,
            );
            let got = to_vec(map.sample(&flat, &p));
            for (n, q) in pts.iter().enumerate() {
                let (i, j, k) = ((q[0] - 10.0) / 0.5, q[1] / 0.5, q[2] / 0.7);
                let e = f(i, j, k);
                for c in 0..2 {
                    assert!(
                        (got[2 * n + c] as f64 - e[c]).abs() < 1e-3,
                        "pas {stride} : {got:?} contre {e:?}"
                    );
                }
            }
        }
        // Mise à plat d'un tenseur de convolution : même indice que `Volume`.
        let t = Tensor::<1>::from_floats(
            TensorData::new((0..24).map(|x| x as f32).collect::<Vec<_>>(), [24]),
            &device,
        )
        .reshape([1, 1, 2, 3, 4]);
        let flat = to_vec(flatten_channels(t));
        // La valeur en (k, j, i) vaut i + 4 (j + 3 k) : elle doit se retrouver à cet indice.
        assert!(flat.iter().enumerate().all(|(n, &x)| x == n as f32), "{flat:?}");
    }

    #[test]
    fn dpf_comme_sur_cpu() {
        let device = Device::flex();
        let m = bumpy();
        let k = surfoxide_metrics::curvature(&m).mean;
        let alpha = 0.3;
        let cpu = surfoxide_metrics::depth_potential(&m, &k, alpha);
        let (v, topo) = (vertices_tensor(&m, &device), Topology::new(&m, &device));
        let kt = Tensor::from_floats(
            TensorData::new(k.iter().map(|&x| x as f32).collect(), [k.len(), 1]),
            &device,
        );
        let gpu = to_vec(depth_potential(&v, &topo, &kt, alpha, 1e-6, 2000));
        assert!(max_rel(&gpu, &cpu) < 1e-3, "écart relatif {}", max_rel(&gpu, &cpu));
    }

    #[test]
    fn gradients_de_l_aire_et_de_l_echantillonnage() {
        let device = Device::flex().autodiff();
        let m = icosphere(2, 5.0);
        let topo = Topology::new(&m, &device);
        // Aire totale : sa dérivée par rapport aux positions est le gradient de forme ; contrôle par différence
        // finie sur un sommet.
        let v = vertices_tensor(&m, &device).require_grad();
        let area = face_areas(&v, &topo).sum();
        let grads = area.backward();
        let g = to_vec(v.grad(&grads).unwrap());
        let h = 1e-3;
        let mut mp = m.clone();
        mp.vertices_mut()[7][0] += h;
        let mut mm = m.clone();
        mm.vertices_mut()[7][0] -= h;
        let fd = (mp.area() - mm.area()) / (2.0 * h);
        assert!(
            (g[7 * 3] as f64 - fd).abs() < 1e-2 * fd.abs().max(1.0),
            "{} contre {fd}",
            g[7 * 3]
        );
        // Échantillonnage d'un champ linéaire φ = 2x − y + 0,5z : gradient constant (2, −1, 0,5).
        let dims = [12, 12, 12];
        let mut vol = Volume::filled(dims, 0.0, IDENTITY);
        for k in 0..12 {
            for j in 0..12 {
                for i in 0..12 {
                    vol.set(i, j, k, 2.0 * i as f64 - j as f64 + 0.5 * k as f64);
                }
            }
        }
        let p = Tensor::<2>::from_floats(TensorData::new(vec![3.3f32, 4.6, 5.2, 7.9, 2.1, 6.4], [2, 3]), &device)
            .require_grad();
        let s = DeviceVolume::new(&vol, &device).sample(&p).sum();
        let gp = to_vec(p.grad(&s.backward()).unwrap());
        for (x, e) in gp.iter().zip([2.0, -1.0, 0.5, 2.0, -1.0, 0.5]) {
            assert!((*x as f64 - e).abs() < 1e-4, "{gp:?}");
        }
    }

    /// Temps sur GPU (CUDA) d'une blanche réelle de 36 SA, contre les opérateurs CPU : à lancer sur le PC Linux
    /// avec `cargo test --release -p surfoxide-burn --features cuda -- --ignored --nocapture`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "gpu"]
    fn temps_sur_gpu() {
        use std::time::Instant;
        let path = surfoxide_core_test_surface();
        let Some(m) = path else {
            eprintln!("surface de test absente : SURFOXIDE_TEST_SURFACE");
            return;
        };
        let device = Device::cuda(0);
        let (v, topo) = (vertices_tensor(&m, &device), Topology::new(&m, &device));
        let k = surfoxide_metrics::curvature(&m).mean;
        let kt = Tensor::from_floats(
            TensorData::new(k.iter().map(|&x| x as f32).collect(), [k.len(), 1]),
            &device,
        );
        let sync = |t: Tensor<2>| {
            let _ = t.sum().into_scalar::<f32>();
        };
        sync(stiffness_apply(&v, &topo, &v)); // échauffement (compilation des noyaux)
        let t = Instant::now();
        for _ in 0..100 {
            sync(stiffness_apply(&v, &topo, &v));
        }
        let lap_gpu = t.elapsed().as_secs_f64() / 100.0;
        let t = Instant::now();
        let (s, _) = surfoxide_metrics::laplacian::fem_matrices(&m);
        let x: Vec<f64> = m.vertices().iter().map(|p| p[0]).collect();
        let _ = s.mul_vec(&x);
        let lap_cpu = t.elapsed().as_secs_f64();
        let t = Instant::now();
        sync(depth_potential(&v, &topo, &kt, 0.2, 1e-6, 2000));
        let dpf_gpu = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = surfoxide_metrics::depth_potential(&m, &k, 0.2);
        let dpf_cpu = t.elapsed().as_secs_f64();
        eprintln!(
            "{} sommets : raideur·u {:.2} ms (GPU) contre assemblage + produit {:.0} ms (CPU) ; DPF {:.0} ms (GPU) contre {:.0} ms (CPU)",
            m.n_vertices(),
            lap_gpu * 1e3,
            lap_cpu * 1e3,
            dpf_gpu * 1e3,
            dpf_cpu * 1e3
        );
    }

    /// Surface de test réelle, donnée par `SURFOXIDE_TEST_SURFACE` (GIfTI).
    #[cfg(feature = "cuda")]
    fn surfoxide_core_test_surface() -> Option<TriMesh> {
        let p = std::env::var_os("SURFOXIDE_TEST_SURFACE")?;
        surfoxide_io::read_mesh(p).ok()
    }
}
