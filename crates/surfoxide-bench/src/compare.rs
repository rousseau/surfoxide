//! Comparaison d'un couple blanche/piale estimé à sa vérité (plan qualité, Q0.3).
//!
//! - Distances : du sommet au plus proche point de l'autre surface, dans les deux sens ; distance symétrique moyenne
//!   (moyenne des deux moyennes) et HD95 (plus grand des deux 95ᵉ centiles).
//! - Champs : un champ estimé (épaisseur, courbure, D*) est transporté sur chaque sommet de la blanche vraie par le
//!   plus proche point de la blanche estimée, en interpolant les valeurs des trois sommets de sa face
//!   (coordonnées barycentriques) ; les statistiques portent sur les sommets du masque vrai.
//! - Taches et bassins de D* : composantes connexes de sommets de même signe (arêtes du maillage), d'aire égale au
//!   tiers de celle des faces incidentes ; bassins : D* < 0 et au moins `min_area` ; taches : moins de `min_area`.

use rayon::prelude::*;
use surfoxide_check::bvh::Bvh;
use surfoxide_check::distance::{closest_point_triangle, distance2_triangle};
use surfoxide_core::adjacency::{edges, vertex_areas};
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Distances entre deux surfaces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Distances {
    /// Moyenne de A → B, de B → A, et leur moyenne.
    pub mean_ab: f64,
    pub mean_ba: f64,
    pub mean: f64,
    /// Plus grand des deux 95ᵉ centiles.
    pub hd95: f64,
    /// Plus grande distance (Hausdorff).
    pub max: f64,
}

/// Valeur au rang `q` (0 à 1) d'une liste triée.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len().max(1) as f64
}

pub fn distances(a: &TriMesh, b: &TriMesh) -> Distances {
    let mut ab = surfoxide_check::vertex_distances(a, b);
    let mut ba = surfoxide_check::vertex_distances(b, a);
    ab.sort_by(f64::total_cmp);
    ba.sort_by(f64::total_cmp);
    let (mean_ab, mean_ba) = (mean(&ab), mean(&ba));
    Distances {
        mean_ab,
        mean_ba,
        mean: 0.5 * (mean_ab + mean_ba),
        hd95: quantile(&ab, 0.95).max(quantile(&ba, 0.95)),
        max: ab.last().copied().unwrap_or(0.0).max(ba.last().copied().unwrap_or(0.0)),
    }
}

/// Transport par plus proche point : pour chaque point de `targets`, face de `mesh` la plus proche et poids
/// barycentriques de ses trois sommets.
pub struct Transport {
    pub faces: Vec<u32>,
    pub weights: Vec<[f64; 3]>,
}

fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [f64; 3] {
    let (v0, v1, v2) = (vec3::sub(b, a), vec3::sub(c, a), vec3::sub(p, a));
    let (d00, d01, d11) = (vec3::dot(v0, v0), vec3::dot(v0, v1), vec3::dot(v1, v1));
    let (d20, d21) = (vec3::dot(v2, v0), vec3::dot(v2, v1));
    let den = d00 * d11 - d01 * d01;
    if den.abs() < 1e-300 {
        return [1.0, 0.0, 0.0];
    }
    let v = (d11 * d20 - d01 * d21) / den;
    let w = (d00 * d21 - d01 * d20) / den;
    [1.0 - v - w, v, w]
}

impl Transport {
    pub fn new(mesh: &TriMesh, targets: &[Vec3]) -> Self {
        let bvh = Bvh::new(mesh);
        let v = mesh.vertices();
        let tri = |f: u32| mesh.faces()[f as usize].map(|i| v[i as usize]);
        let (faces, weights) = targets
            .par_iter()
            .map(|&p| {
                let (f, _) = bvh
                    .nearest(p, |f| {
                        let [a, b, c] = tri(f);
                        distance2_triangle(p, a, b, c)
                    })
                    .expect("surface non vide");
                let [a, b, c] = tri(f);
                (f, barycentric(closest_point_triangle(p, a, b, c), a, b, c))
            })
            .unzip();
        Self { faces, weights }
    }

    /// Valeurs du champ `field` (par sommet de la surface source) aux points cibles.
    pub fn apply(&self, mesh: &TriMesh, field: &[f64]) -> Vec<f64> {
        self.faces
            .iter()
            .zip(&self.weights)
            .map(|(&f, w)| {
                let t = mesh.faces()[f as usize];
                (0..3).map(|k| w[k] * field[t[k] as usize]).sum()
            })
            .collect()
    }
}

/// Coefficient de corrélation de Pearson.
pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let (mx, my) = (mean(x), mean(y));
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx * syy).sqrt()
}

/// Rangs (1 à n), ex æquo au rang moyen.
fn ranks(x: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let mut r = vec![0.0; x.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && x[idx[j + 1]] == x[idx[i]] {
            j += 1;
        }
        let rank = 0.5 * (i + j) as f64 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = rank;
        }
        i = j + 1;
    }
    r
}

/// Coefficient de corrélation de Spearman (Pearson des rangs).
pub fn spearman(x: &[f64], y: &[f64]) -> f64 {
    pearson(&ranks(x), &ranks(y))
}

/// Écart d'un champ estimé (transporté) à son champ vrai.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FieldError {
    /// Moyenne de estimé − vrai.
    pub bias: f64,
    /// Médiane de |estimé − vrai|.
    pub median_abs: f64,
    pub pearson: f64,
    pub spearman: f64,
    /// Sommets comparés.
    pub n: usize,
}

pub fn field_error(estimated: &[f64], truth: &[f64], mask: Option<&[bool]>) -> FieldError {
    let keep: Vec<usize> = (0..truth.len())
        .filter(|&i| mask.is_none_or(|m| m[i]) && estimated[i].is_finite() && truth[i].is_finite())
        .collect();
    let e: Vec<f64> = keep.iter().map(|&i| estimated[i]).collect();
    let t: Vec<f64> = keep.iter().map(|&i| truth[i]).collect();
    let mut abs: Vec<f64> = e.iter().zip(&t).map(|(a, b)| (a - b).abs()).collect();
    abs.sort_by(f64::total_cmp);
    FieldError {
        bias: e.iter().zip(&t).map(|(a, b)| a - b).sum::<f64>() / keep.len().max(1) as f64,
        median_abs: quantile(&abs, 0.5),
        pearson: pearson(&e, &t),
        spearman: spearman(&e, &t),
        n: keep.len(),
    }
}

/// Taches et bassins de D* sur une surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Patches {
    /// Composantes D* < 0 d'au moins `min_area` mm².
    pub basins: usize,
    /// Composantes, de l'un ou l'autre signe, de moins de `min_area` mm².
    pub small: usize,
    /// Aire totale des petites taches (mm²).
    pub small_area: f64,
}

pub fn patches(mesh: &TriMesh, depth: &[f64], min_area: f64) -> Patches {
    let n = mesh.n_vertices();
    let neg: Vec<bool> = depth.iter().map(|&d| d < 0.0).collect();
    // Union-find sur les arêtes reliant deux sommets de même signe.
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    for [a, b] in edges(mesh) {
        if neg[a as usize] == neg[b as usize] {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra.max(rb) as usize] = ra.min(rb);
            }
        }
    }
    let area = vertex_areas(mesh);
    let mut comp_area = vec![0.0; n];
    for (i, a) in area.iter().enumerate() {
        let r = find(&mut parent, i as u32) as usize;
        comp_area[r] += a;
    }
    let mut p = Patches {
        basins: 0,
        small: 0,
        small_area: 0.0,
    };
    for i in 0..n {
        if find(&mut parent, i as u32) as usize != i {
            continue;
        }
        if comp_area[i] < min_area {
            p.small += 1;
            p.small_area += comp_area[i];
        } else if neg[i] {
            p.basins += 1;
        }
    }
    p
}

/// Indice de gyrification : aire / aire de l'enveloppe convexe des sommets.
pub fn gyrification(mesh: &TriMesh) -> f64 {
    mesh.area() / surfoxide_metrics::hull::convex_hull_area(mesh.vertices())
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    fn moved(m: &TriMesh, f: impl Fn(Vec3) -> Vec3) -> TriMesh {
        TriMesh::new(m.vertices().iter().map(|&p| f(p)).collect(), m.faces().to_vec()).unwrap()
    }

    #[test]
    fn surfaces_identiques_distances_nulles() {
        let s = icosphere(4, 10.0);
        let d = distances(&s, &s);
        assert_eq!((d.mean, d.hd95, d.max), (0.0, 0.0, 0.0));
        let c = surfoxide_metrics::curvature(&s).mean;
        let t = Transport::new(&s, s.vertices());
        let e = field_error(&t.apply(&s, &c), &c, None);
        assert!(e.bias.abs() < 1e-12 && e.median_abs < 1e-12);
    }

    #[test]
    fn sphere_decalee_de_0_1_mm() {
        let s = icosphere(5, 10.0);
        let out = moved(&s, |p| vec3::scale(p, 10.1 / 10.0));
        let d = distances(&out, &s);
        println!("sphère de rayon + 0,1 mm : {d:?}");
        assert!((d.mean - 0.1).abs() < 1e-3 && (d.hd95 - 0.1).abs() < 1e-3, "{d:?}");
        // Translation au sens strict : de 0 à 0,1 mm, moyenne 0,05.
        let tr = moved(&s, |p| vec3::add(p, [0.1, 0.0, 0.0]));
        let d = distances(&tr, &s);
        println!("sphère translatée de 0,1 mm : {d:?}");
        assert!((d.mean - 0.05).abs() < 1e-3 && (d.max - 0.1).abs() < 1e-3, "{d:?}");
    }

    #[test]
    fn rangs_et_correlations() {
        assert_eq!(ranks(&[3.0, 1.0, 3.0, 2.0]), vec![3.5, 1.0, 3.5, 2.0]);
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [1.0, 4.0, 9.0, 16.0];
        assert!((spearman(&x, &y) - 1.0).abs() < 1e-12);
        assert!(pearson(&x, &y) < 1.0);
    }

    #[test]
    fn bassins_d_une_sphere_bosselee() {
        let s = icosphere(4, 10.0);
        // Deux calottes négatives opposées et une petite tache.
        let d: Vec<f64> = s
            .vertices()
            .iter()
            .map(|p| if p[2].abs() > 8.0 { -1.0 } else { 1.0 })
            .collect();
        let p = patches(&s, &d, 5.0);
        assert_eq!(p.basins, 2);
        assert_eq!(p.small, 0);
    }
}
