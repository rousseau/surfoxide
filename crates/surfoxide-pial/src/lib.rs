//! `surfoxide-pial` : surface piale sans apprentissage, par les lignes de courant du champ de Laplace
//! (Jones et al. 2000 ; Acosta et al. 2009).
//!
//! 1. [`sulcal_skeleton`] : dans les sillons enfouis, les deux berges de cortex se touchent sans LCR
//!    visible. La nappe médiane entre deux berges de substance blanche est repérée par le saut du plus
//!    proche voxel blanc entre voxels voisins, et traitée comme du LCR (comme l'ACE de CRUISE).
//! 2. [`solve_laplace`] : Δu = 0 sur le cortex, u = 0 sur la substance blanche, u = 1 ailleurs.
//! 3. [`Field::trace`] : depuis chaque sommet de la surface blanche, intégration RK4 de ∇u/‖∇u‖ jusqu'au
//!    niveau u = `stop`. Les lignes de courant d'un champ de gradient ne se croisent pas : l'application
//!    sommet blanc → sommet pial est bien définie et donne l'épaisseur corticale.
//! 4. [`pial_from_white`] : la piale part d'un léger décalage de la surface blanche et avance par pas le
//!    long des lignes de courant ; un pas qui créerait une intersection (avec elle-même ou avec la surface
//!    blanche) est retenu localement. La piale reste ainsi sans auto-intersection et contient strictement
//!    la surface blanche, avec la même connectivité.

use surfoxide_check::bvh::Bvh;
use surfoxide_check::distance::distance2_triangle;
use surfoxide_core::distance::feature_transform;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::volume::invert_affine;
use surfoxide_core::{Affine, TriMesh, VertexField, Volume};
use surfoxide_repair::{cone_direction, conflicting_vertices, safe_update_with};

/// Voxels du cortex sur la nappe médiane d'un sillon enfoui : le voxel blanc le plus proche diffère d'un
/// voisin à l'autre de plus de `max(3 h, ratio · d)` (h : plus grande taille de voxel, d : distance au
/// blanc).
pub fn sulcal_skeleton(white: &Volume<bool>, cortex: &Volume<bool>, ratio: f64) -> Volume<bool> {
    let (d2, feat) = feature_transform(white);
    let spacing = white.spacing();
    let h = spacing.into_iter().fold(0.0, f64::max);
    let dims = white.dims();
    let mut out = vec![false; white.len()];
    let world = |idx: u32| {
        let c = white.coords(idx as usize);
        [
            c[0] as f64 * spacing[0],
            c[1] as f64 * spacing[1],
            c[2] as f64 * spacing[2],
        ]
    };
    for idx in 0..white.len() {
        if !cortex.data()[idx] || white.data()[idx] {
            continue;
        }
        let f = feat.data()[idx];
        if f == u32::MAX {
            continue;
        }
        let d = d2.data()[idx].sqrt();
        let [i, j, k] = white.coords(idx);
        for (a, step) in [(0, 1usize), (1, dims[0]), (2, dims[0] * dims[1])] {
            let c = [i, j, k][a];
            if c + 1 >= dims[a] {
                continue;
            }
            let n = idx + step;
            if !cortex.data()[n] || white.data()[n] {
                continue;
            }
            let g = feat.data()[n];
            let jump = vec3::dist(world(f), world(g));
            let dn = d2.data()[n].sqrt();
            if jump > (3.0 * h).max(ratio * d.max(dn)) {
                out[idx] = true;
                out[n] = true;
            }
        }
    }
    white.with_data(out)
}

/// Résout Δu = 0 (surrelaxation rouge-noir, tailles de voxel prises en compte) sur `domain` privé de
/// `white`, avec u = 0 sur `white` et u = 1 ailleurs. S'arrête quand la plus grande variation d'un balayage
/// passe sous `tol`.
pub fn solve_laplace(
    white: &Volume<bool>,
    domain: &Volume<bool>,
    tol: f64,
    max_iter: usize,
) -> (Volume<f64>, usize) {
    let dims = white.dims();
    let s = white.spacing();
    let w = [
        1.0 / (s[0] * s[0]),
        1.0 / (s[1] * s[1]),
        1.0 / (s[2] * s[2]),
    ];
    let wsum = 2.0 * (w[0] + w[1] + w[2]);
    let stride = [1usize, dims[0], dims[0] * dims[1]];
    let mut u: Vec<f64> = (0..white.len())
        .map(|i| {
            if white.data()[i] {
                0.0
            } else if domain.data()[i] {
                0.5
            } else {
                1.0
            }
        })
        .collect();
    // Voxels actifs, hors du bord du volume (le bord reste fixé).
    let active: Vec<usize> = (0..white.len())
        .filter(|&i| {
            let c = white.coords(i);
            domain.data()[i] && !white.data()[i] && (0..3).all(|a| c[a] > 0 && c[a] + 1 < dims[a])
        })
        .collect();
    let (red, black): (Vec<usize>, Vec<usize>) = active.iter().partition(|&&i| {
        let c = white.coords(i);
        (c[0] + c[1] + c[2]).is_multiple_of(2)
    });
    let omega = 1.9;
    let mut iters = 0;
    while iters < max_iter {
        iters += 1;
        let mut max_delta: f64 = 0.0;
        for set in [&red, &black] {
            for &i in set.iter() {
                let mut acc = 0.0;
                for a in 0..3 {
                    acc += w[a] * (u[i + stride[a]] + u[i - stride[a]]);
                }
                let gs = acc / wsum;
                let new = u[i] + omega * (gs - u[i]);
                max_delta = max_delta.max((new - u[i]).abs());
                u[i] = new;
            }
        }
        if max_delta < tol {
            break;
        }
    }
    (white.with_data(u), iters)
}

/// Champ de Laplace prêt à l'intégration : valeurs, gradient (repère voxel) et géométrie.
pub struct Field {
    u: Volume<f64>,
    grad: [Vec<f64>; 3],
    inv: Affine,
    /// Transposée de l'inverse de la partie linéaire de l'affine : gradient voxel → gradient monde.
    to_world: [[f64; 3]; 3],
}

/// Paramètres de la piale.
#[derive(Debug, Clone, Copy)]
pub struct PialParams {
    /// Pas d'intégration (mm).
    pub step: f64,
    /// Longueur maximale d'une ligne de courant (mm).
    pub max_length: f64,
    /// Niveau de u où s'arrêtent les lignes de courant ; un peu sous 1, pour que deux berges d'un sillon
    /// enfoui ne se touchent pas.
    pub stop: f64,
    /// Décalage initial de la piale le long des lignes de courant (mm) : épaisseur minimale visée.
    pub min_offset: f64,
    /// Nombre de pas de progression de la piale.
    pub flow_steps: usize,
    /// Distance minimale (mm) qu'une face de la piale doit garder avec tout sommet de la surface blanche
    /// pendant la progression.
    pub min_distance: f64,
}

impl Default for PialParams {
    fn default() -> Self {
        Self {
            step: 0.05,
            max_length: 6.0,
            stop: 0.97,
            min_offset: 0.1,
            flow_steps: 10,
            min_distance: 0.03,
        }
    }
}

impl Field {
    pub fn new(u: Volume<f64>) -> Self {
        let dims = u.dims();
        let stride = [1usize, dims[0], dims[0] * dims[1]];
        let mut grad = [vec![0.0; u.len()], vec![0.0; u.len()], vec![0.0; u.len()]];
        for idx in 0..u.len() {
            let c = u.coords(idx);
            for (a, g) in grad.iter_mut().enumerate() {
                let lo = if c[a] > 0 { idx - stride[a] } else { idx };
                let hi = if c[a] + 1 < dims[a] {
                    idx + stride[a]
                } else {
                    idx
                };
                let span = ((hi - lo) / stride[a]) as f64;
                g[idx] = if span > 0.0 {
                    (u.data()[hi] - u.data()[lo]) / span
                } else {
                    0.0
                };
            }
        }
        let a = *u.affine();
        let inv = invert_affine(&a).expect("affine inversible");
        // ∇_monde = L^{-T} ∇_voxel, avec L^{-1} = partie linéaire de l'inverse.
        let to_world = std::array::from_fn(|r| std::array::from_fn(|c| inv[c][r]));
        Field {
            u,
            grad,
            inv,
            to_world,
        }
    }

    pub fn volume(&self) -> &Volume<f64> {
        &self.u
    }

    /// Interpolation trilinéaire de `data` au point voxel `p` (bords répliqués).
    fn trilinear(&self, data: &[f64], p: Vec3) -> f64 {
        let dims = self.u.dims();
        let mut base = [0usize; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            let x = p[a].clamp(0.0, (dims[a] - 1) as f64);
            let i = (x.floor() as usize).min(dims[a].saturating_sub(2));
            base[a] = i;
            t[a] = x - i as f64;
        }
        let mut acc = 0.0;
        for c in 0..8 {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let idx = (base[0] + o[0]).min(dims[0] - 1)
                + dims[0]
                    * ((base[1] + o[1]).min(dims[1] - 1)
                        + dims[1] * (base[2] + o[2]).min(dims[2] - 1));
            let wgt = (0..3)
                .map(|a| if o[a] == 1 { t[a] } else { 1.0 - t[a] })
                .product::<f64>();
            acc += wgt * data[idx];
        }
        acc
    }

    fn voxel(&self, p: Vec3) -> Vec3 {
        surfoxide_core::volume::apply_affine(&self.inv, p)
    }

    /// u au point monde `p`.
    pub fn value(&self, p: Vec3) -> f64 {
        self.trilinear(self.u.data(), self.voxel(p))
    }

    /// Gradient de u (repère monde, mm⁻¹) au point monde `p`.
    pub fn gradient(&self, p: Vec3) -> Vec3 {
        let q = self.voxel(p);
        let g = [0, 1, 2].map(|a| self.trilinear(&self.grad[a], q));
        std::array::from_fn(|r| (0..3).map(|c| self.to_world[r][c] * g[c]).sum())
    }

    /// Direction d'avancée : ∇u normalisé, ou `fallback` là où le gradient est négligeable (au cœur de la
    /// substance blanche).
    fn direction(&self, p: Vec3, fallback: Vec3) -> Vec3 {
        let g = self.gradient(p);
        if vec3::norm(g) > 1e-3 {
            vec3::normalize(g)
        } else {
            fallback
        }
    }

    /// Ligne de courant depuis `start` (normale sortante `normal`) jusqu'au niveau `params.stop`.
    pub fn trace(&self, start: Vec3, normal: Vec3, params: &PialParams) -> Vec<Vec3> {
        let h = params.step;
        let mut path = vec![start];
        let mut p = start;
        let mut length = 0.0;
        let mut dir = vec3::normalize(normal);
        let mut u = self.value(p);
        while u < params.stop && length < params.max_length {
            let k1 = self.direction(p, dir);
            let k2 = self.direction(vec3::add(p, vec3::scale(k1, h / 2.0)), k1);
            let k3 = self.direction(vec3::add(p, vec3::scale(k2, h / 2.0)), k2);
            let k4 = self.direction(vec3::add(p, vec3::scale(k3, h)), k3);
            let d = vec3::normalize(vec3::add(
                vec3::add(k1, vec3::scale(k2, 2.0)),
                vec3::add(vec3::scale(k3, 2.0), k4),
            ));
            let q = vec3::add(p, vec3::scale(d, h));
            let uq = self.value(q);
            if uq >= params.stop {
                // Dernier pas raccourci au niveau d'arrêt (interpolation linéaire de u).
                let t = if uq > u {
                    ((params.stop - u) / (uq - u)).clamp(0.0, 1.0)
                } else {
                    1.0
                };
                path.push(vec3::lerp(p, q, t));
                break;
            }
            path.push(q);
            p = q;
            u = uq;
            dir = d;
            length += h;
        }
        path
    }
}

/// Point situé à l'abscisse curviligne `s` sur `path` ; au-delà du bout, prolongement le long du dernier
/// segment (ou de `normal` si le chemin est réduit à un point).
pub fn point_at(path: &[Vec3], s: f64, normal: Vec3) -> Vec3 {
    let mut left = s;
    for w in path.windows(2) {
        let l = vec3::dist(w[0], w[1]);
        if left <= l && l > 0.0 {
            return vec3::lerp(w[0], w[1], left / l);
        }
        left -= l;
    }
    let last = *path.last().unwrap();
    let dir = if path.len() >= 2 {
        vec3::normalize(vec3::sub(last, path[path.len() - 2]))
    } else {
        vec3::normalize(normal)
    };
    vec3::add(last, vec3::scale(dir, left.max(0.0)))
}

fn path_length(path: &[Vec3]) -> f64 {
    path.windows(2).map(|w| vec3::dist(w[0], w[1])).sum()
}

/// Résultat de la construction de la piale.
pub struct Pial {
    pub mesh: TriMesh,
    /// Abscisse curviligne atteinte par chaque sommet (mm) : épaisseur corticale le long de la ligne de
    /// courant.
    pub thickness: VertexField,
    /// Sommets qui n'ont pas atteint le bout de leur ligne de courant.
    pub held: usize,
}

/// Erreur de construction de la piale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PialError {
    pub message: String,
    /// Sommets de la surface blanche qui bloquent (à détendre avant de réessayer).
    pub blocked: Vec<u32>,
}

impl std::fmt::Display for PialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for PialError {}

/// Sommets des faces de `surface` situées à moins de `eps` d'un sommet de `points`.
pub fn faces_near(surface: &TriMesh, points: &TriMesh, eps: f64) -> std::collections::HashSet<u32> {
    let bvh = Bvh::new(surface);
    let v = surface.vertices();
    let f = surface.faces();
    let mut out = std::collections::HashSet::new();
    for &p in points.vertices() {
        if let Some((g, d2)) = bvh.nearest(p, |g| {
            let t = f[g as usize];
            distance2_triangle(p, v[t[0] as usize], v[t[1] as usize], v[t[2] as usize])
        }) {
            if d2 < eps * eps {
                out.extend(f[g as usize]);
            }
        }
    }
    out
}

/// Construit la piale à partir de la surface blanche `white` (fermée, sans auto-intersection) et du champ.
pub fn pial_from_white(
    white: &TriMesh,
    field: &Field,
    params: &PialParams,
) -> Result<Pial, PialError> {
    // Normales prises dans le cône des normales des faces incidentes : au droit d'un pli, un décalage le
    // long de la normale moyenne traverserait une face voisine, aussi petit soit-il.
    let vf = surfoxide_core::adjacency::vertex_faces(white);
    let fnorm = surfoxide_core::adjacency::face_normals(white);
    let normals: Vec<Vec3> = surfoxide_core::adjacency::vertex_normals(white)
        .into_iter()
        .enumerate()
        .map(|(v, n)| {
            let fs: Vec<Vec3> = vf[v].iter().map(|&f| fnorm[f as usize]).collect();
            cone_direction(n, &fs, 0.05).unwrap_or(n)
        })
        .collect();
    let paths: Vec<Vec<Vec3>> = white
        .vertices()
        .iter()
        .zip(&normals)
        .map(|(&p, &n)| field.trace(p, n, params))
        .collect();
    let lengths: Vec<f64> = paths.iter().map(|p| path_length(p)).collect();
    let at = |i: usize, s: f64| point_at(&paths[i], s, normals[i]);

    // Décalage initial le long de la normale, réduit localement tant qu'il crée des intersections (entre
    // deux berges très proches, ou là où la ligne de courant part en biais).
    let mut offset = vec![params.min_offset; white.n_vertices()];
    let ring = surfoxide_core::adjacency::vertex_neighbors(white);
    let mut pial = white.clone();
    // Réduction du décalage des sommets en conflit et de leurs voisins : un sommet ramené seul contre la
    // surface blanche ferait toucher celle-ci par les faces qui le relient aux voisins restés au large.
    // S'il reste des conflits, les sommets en cause sont rendus à l'appelant, qui peut détendre localement
    // la surface blanche avant de réessayer.
    const HALVINGS: usize = 12;
    let debug = std::env::var_os("SURFOXIDE_DEBUG").is_some();
    for (i, p) in pial.vertices_mut().iter_mut().enumerate() {
        *p = vec3::add(white.vertices()[i], vec3::scale(normals[i], offset[i]));
    }
    let mut bad = conflicting_vertices(&pial, Some(white));
    let mut attempt = 0;
    while !bad.is_empty() && attempt < HALVINGS {
        let mut region: std::collections::HashSet<u32> = bad.clone();
        let rings = if attempt < HALVINGS {
            1
        } else {
            attempt - HALVINGS + 1
        };
        for _ in 0..rings {
            let more: Vec<u32> = region
                .iter()
                .flat_map(|&v| ring[v as usize].iter().copied())
                .collect();
            region.extend(more);
        }
        if attempt < HALVINGS {
            for &v in &region {
                offset[v as usize] /= 2.0;
                pial.vertices_mut()[v as usize] = vec3::add(
                    white.vertices()[v as usize],
                    vec3::scale(normals[v as usize], offset[v as usize]),
                );
            }
        } else {
            for _ in 0..20 {
                let pos = pial.vertices().to_vec();
                for &v in &region {
                    let n = &ring[v as usize];
                    let mean = n
                        .iter()
                        .fold([0.0; 3], |acc, &w| vec3::add(acc, pos[w as usize]));
                    pial.vertices_mut()[v as usize] = vec3::scale(mean, 1.0 / n.len() as f64);
                }
            }
        }
        bad = conflicting_vertices(&pial, Some(white));
        attempt += 1;
        if debug {
            eprintln!(
                "décalage initial, essai {attempt} : {} sommets en conflit",
                bad.len()
            );
        }
    }
    if !bad.is_empty() {
        let mut blocked: Vec<u32> = bad.iter().copied().collect();
        blocked.sort_unstable();
        return Err(PialError {
            message: format!(
                "décalage initial de la piale impossible sans intersection ({} sommets)",
                bad.len()
            ),
            blocked,
        });
    }
    // Progression par pas le long des lignes de courant.
    let mut reached = offset.clone();
    for k in 1..=params.flow_steps {
        let t = k as f64 / params.flow_steps as f64;
        let s: Vec<f64> = (0..white.n_vertices())
            .map(|i| (t * lengths[i]).max(offset[i]))
            .collect();
        let target: Vec<Vec3> = (0..white.n_vertices()).map(|i| at(i, s[i])).collect();
        // Un pas est retenu s'il ferait se couper la piale (avec elle-même ou avec la blanche), ou s'il
        // amènerait une face de la piale trop près d'un sommet blanc.
        safe_update_with(&mut pial, &target, |m| {
            let mut bad = conflicting_vertices(m, Some(white));
            bad.extend(faces_near(m, white, params.min_distance));
            bad
        });
        for i in 0..white.n_vertices() {
            if pial.vertices()[i] == target[i] {
                reached[i] = s[i];
            }
        }
    }
    let held = (0..white.n_vertices())
        .filter(|&i| reached[i] + 1e-9 < lengths[i].max(offset[i]))
        .count();
    Ok(Pial {
        mesh: pial,
        thickness: VertexField::new("thickness", reached),
        held,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;
    use surfoxide_core::volume::IDENTITY;

    /// Coquille sphérique : blanc pour r < r1, cortex pour r1 ≤ r < r2, voxels de `h` mm.
    fn shell(n: usize, h: f64, r1: f64, r2: f64) -> (Volume<bool>, Volume<bool>) {
        let mut aff = IDENTITY;
        let c = (n as f64 - 1.0) / 2.0 * h;
        for (a, row) in aff.iter_mut().take(3).enumerate() {
            row[a] = h;
            row[3] = -c;
        }
        let mut white = Volume::filled([n, n, n], false, aff);
        let mut cortex = Volume::filled([n, n, n], false, aff);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let p = white.voxel_to_world([i as f64, j as f64, k as f64]);
                    let r = vec3::norm(p);
                    white.set(i, j, k, r < r1);
                    cortex.set(i, j, k, r >= r1 && r < r2);
                }
            }
        }
        (white, cortex)
    }

    #[test]
    fn laplace_entre_spheres_concentriques() {
        let (r1, r2) = (5.0, 8.0);
        let (white, cortex) = shell(45, 0.4, r1, r2);
        let (u, iters) = solve_laplace(&white, &cortex, 1e-7, 5000);
        assert!(iters < 5000);
        // Solution exacte : u(r) = (1/r1 − 1/r) / (1/r1 − 1/r2) ; écart dû à la discrétisation des bords.
        let exact = |r: f64| (1.0 / r1 - 1.0 / r) / (1.0 / r1 - 1.0 / r2);
        let field = Field::new(u);
        for r in [5.8, 6.5, 7.2] {
            let got = field.value([r, 0.0, 0.0]);
            assert!(
                (got - exact(r)).abs() < 0.08,
                "u({r}) = {got}, attendu {}",
                exact(r)
            );
        }
    }

    #[test]
    fn lignes_de_courant_radiales_et_piale_valide() {
        let (white_vol, cortex) = shell(45, 0.4, 5.0, 8.0);
        let (u, _) = solve_laplace(&white_vol, &cortex, 1e-7, 5000);
        let field = Field::new(u);
        let white = icosphere(3, 5.0);
        let params = PialParams::default();
        // Ligne de courant radiale.
        let p = white.vertices()[7];
        let path = field.trace(p, vec3::normalize(p), &params);
        let end = *path.last().unwrap();
        let cos = vec3::dot(vec3::normalize(end), vec3::normalize(p));
        assert!(cos > 0.995, "ligne de courant non radiale (cos = {cos})");
        let pial = pial_from_white(&white, &field, &params).unwrap();
        let mean_r = pial
            .mesh
            .vertices()
            .iter()
            .map(|q| vec3::norm(*q))
            .sum::<f64>()
            / pial.mesh.n_vertices() as f64;
        assert!(
            (mean_r - 7.9).abs() < 0.3,
            "rayon moyen de la piale {mean_r}"
        );
        assert_eq!(pial.held, 0);
        let pair = surfoxide_check::check_pair(&white, &pial.mesh, 0.05);
        assert!(pair.valid, "{pair:?}");
        assert!(surfoxide_check::check_surface(&pial.mesh).valid);
    }

    /// Deux blocs de blanc séparés par une fente remplie de cortex (sillon enfoui) : la nappe médiane est
    /// détectée au milieu de la fente.
    #[test]
    fn nappe_mediane_d_un_sillon_enfoui() {
        let dims = [30, 12, 12];
        let mut white = Volume::filled(dims, false, IDENTITY);
        let mut cortex = Volume::filled(dims, false, IDENTITY);
        for k in 1..11 {
            for j in 1..11 {
                for i in 1..29 {
                    let wm = (2..=12).contains(&i) || (17..=27).contains(&i);
                    if wm && j < 8 {
                        white.set(i, j, k, true);
                    } else {
                        cortex.set(i, j, k, true);
                    }
                }
            }
        }
        let sk = sulcal_skeleton(&white, &cortex, 1.5);
        // Fente : i de 13 à 16 ; milieu entre 14 et 15.
        assert!(
            sk.get(14, 4, 5) || sk.get(15, 4, 5),
            "nappe médiane non détectée"
        );
        assert!(!sk.get(13, 4, 5) && !sk.get(16, 4, 5), "nappe trop épaisse");
        assert!(!sk.get(6, 9, 5), "faux positif sur une couronne");
    }
}
