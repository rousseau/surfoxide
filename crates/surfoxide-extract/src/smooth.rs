//! Lissage de Taubin (1995) : alternance d'un pas laplacien λ > 0 (lissage) et d'un pas μ < −λ
//! (regonflement), filtre passe-bas sans le rétrécissement du lissage laplacien seul. Seules les positions
//! changent : la topologie est conservée.

use surfoxide_core::adjacency::vertex_neighbors;
use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::volume::{apply_affine, invert_affine};
use surfoxide_core::{TriMesh, Volume};

/// `iterations` paires de pas (λ, μ) avec le laplacien uniforme (moyenne des voisins).
pub fn taubin(mesh: &mut TriMesh, iterations: usize, lambda: f64, mu: f64) {
    let nb = vertex_neighbors(mesh);
    let mut next = mesh.vertices().to_vec();
    for _ in 0..iterations {
        for factor in [lambda, mu] {
            let v = mesh.vertices();
            for (i, n) in nb.iter().enumerate() {
                if n.is_empty() {
                    continue;
                }
                let mean = n.iter().fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize]));
                let mean = vec3::scale(mean, 1.0 / n.len() as f64);
                next[i] = vec3::add(v[i], vec3::scale(vec3::sub(mean, v[i]), factor));
            }
            mesh.vertices_mut().copy_from_slice(&next);
        }
    }
}

/// Une paire de pas de Taubin (λ puis μ) contrainte à une bande autour de la frontière d'un masque :
/// positions proposées, sans contrôle d'intersection.
///
/// `phi` est la distance signée (mm) au masque dont la surface est extraite : les centres des voxels
/// voisins de la frontière y valent ±h. Un sommet ne peut pas sortir de la bande `|φ| ≤ band` (ni, s'il en
/// est déjà sorti, s'en éloigner) : le pas est raccourci par dichotomie. Avec `band` < h, aucun centre de
/// voxel ne change de côté, et le lissage efface les marches d'escalier sans s'écarter de la segmentation
/// (principe des « constrained elastic surface nets », Gibson 1998). `band_in` borne à part le côté
/// intérieur (φ ≥ −band_in) : une crête étroite de l'objet n'est pas amincie au-delà.
pub fn constrained_taubin(
    mesh: &TriMesh,
    phi: &Volume<f64>,
    band: f64,
    band_in: f64,
    lambda: f64,
    mu: f64,
) -> Vec<Vec3> {
    let inv = invert_affine(phi.affine()).expect("affine inversible");
    let at = |p: Vec3| phi.trilinear(apply_affine(&inv, p));
    let nb = vertex_neighbors(mesh);
    let mut v = mesh.vertices().to_vec();
    for factor in [lambda, mu] {
        let next: Vec<Vec3> = nb
            .iter()
            .enumerate()
            .map(|(i, n)| {
                if n.is_empty() {
                    return v[i];
                }
                let mean = vec3::scale(
                    n.iter().fold([0.0; 3], |acc, &j| vec3::add(acc, v[j as usize])),
                    1.0 / n.len() as f64,
                );
                let target = vec3::add(v[i], vec3::scale(vec3::sub(mean, v[i]), factor));
                let here = at(v[i]);
                let (lo, hi) = ((-band_in).min(here), band.max(here));
                let inside = |p: Vec3| (lo..=hi).contains(&at(p));
                if inside(target) {
                    return target;
                }
                // Plus grande fraction du pas qui reste dans la bande.
                let (mut lo, mut hi) = (0.0, 1.0);
                for _ in 0..8 {
                    let t = 0.5 * (lo + hi);
                    if inside(vec3::lerp(v[i], target, t)) {
                        lo = t;
                    } else {
                        hi = t;
                    }
                }
                vec3::lerp(v[i], target, lo)
            })
            .collect();
        v = keep_orientation(mesh, &v, next);
    }
    v
}

/// Annule le pas des sommets d'une face que le pas retournerait ou ferait tourner de plus de 60°, et des
/// deux faces d'une arête que le pas replierait (normales de cosinus < −0,35, soit un pli de plus de 110°,
/// sans l'être avant) : un pas raccourci par la bande à côté d'un voisin qui avance librement, ou une crête
/// étroite amincie par le laplacien, peut replier la surface. Répété jusqu'à ce qu'aucune face ne tourne trop
/// (les sommets annulés reprennent leur ancienne position).
fn keep_orientation(mesh: &TriMesh, old: &[Vec3], mut new: Vec<Vec3>) -> Vec<Vec3> {
    let normal = |v: &[Vec3], f: &[u32; 3]| {
        vec3::normalize(vec3::triangle_normal(
            v[f[0] as usize],
            v[f[1] as usize],
            v[f[2] as usize],
        ))
    };
    // Paires de faces partageant une arête.
    let mut by_edge: std::collections::HashMap<(u32, u32), Vec<usize>> = std::collections::HashMap::new();
    for (k, f) in mesh.faces().iter().enumerate() {
        for e in 0..3 {
            let (a, b) = (f[e], f[(e + 1) % 3]);
            by_edge.entry((a.min(b), a.max(b))).or_default().push(k);
        }
    }
    let pairs: Vec<(usize, usize)> = by_edge
        .values()
        .filter(|v| v.len() == 2)
        .map(|v| (v[0], v[1]))
        .collect();
    let faces = mesh.faces();
    let old_n: Vec<Vec3> = faces.iter().map(|f| normal(old, f)).collect();
    for _ in 0..20 {
        let new_n: Vec<Vec3> = faces.iter().map(|f| normal(&new, f)).collect();
        let mut bad: Vec<usize> = (0..faces.len())
            .filter(|&k| vec3::dot(old_n[k], new_n[k]) < 0.5)
            .collect();
        for &(p, q) in &pairs {
            if vec3::dot(new_n[p], new_n[q]) < -0.35 && vec3::dot(old_n[p], old_n[q]) >= -0.35 {
                bad.extend([p, q]);
            }
        }
        let mut changed = false;
        for k in bad {
            for &i in &faces[k] {
                if new[i as usize] != old[i as usize] {
                    new[i as usize] = old[i as usize];
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    new
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn sphere_bruitee_lissee_sans_retrecir() {
        let mut s = icosphere(4, 10.0);
        let v0 = s.signed_volume();
        // Bruit radial déterministe.
        for (i, p) in s.vertices_mut().iter_mut().enumerate() {
            let f = 1.0 + 0.03 * ((i * 7919 % 101) as f64 / 50.0 - 1.0);
            *p = vec3::scale(*p, f);
        }
        let rough = |m: &TriMesh| {
            m.vertices().iter().map(|p| (vec3::norm(*p) - 10.0).abs()).sum::<f64>() / m.n_vertices() as f64
        };
        let before = rough(&s);
        taubin(&mut s, 10, 0.5, -0.53);
        assert!(rough(&s) < before / 3.0, "{} contre {before}", rough(&s));
        assert!((s.signed_volume() - v0).abs() / v0 < 0.02);
    }

    /// Boule voxelisée : le lissage contraint efface les marches d'escalier (rayon presque constant) sans
    /// quitter la bande autour de la frontière du masque.
    #[test]
    fn lissage_contraint_d_une_boule_voxelisee() {
        use surfoxide_core::distance::signed_distance;
        use surfoxide_core::volume::IDENTITY;
        let (n, r) = (32usize, 10.3);
        let c = (n as f64 - 1.0) / 2.0;
        let mut mask = Volume::filled([n, n, n], false, IDENTITY);
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d = vec3::norm([i as f64 - c, j as f64 - c, k as f64 - c]);
                    mask.set(i, j, k, d < r);
                }
            }
        }
        let phi = signed_distance(&mask);
        let mut mesh = crate::marching_cubes(&phi, 0.2);
        let spread = |m: &TriMesh| {
            let radii: Vec<f64> = m
                .vertices()
                .iter()
                .map(|p| vec3::norm(vec3::sub(*p, [c, c, c])))
                .collect();
            let mean = radii.iter().sum::<f64>() / radii.len() as f64;
            (
                mean,
                (radii.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / radii.len() as f64).sqrt(),
            )
        };
        let (_, before) = spread(&mesh);
        for _ in 0..50 {
            let next = constrained_taubin(&mesh, &phi, 0.8, 0.8, 0.5, -0.53);
            mesh.vertices_mut().copy_from_slice(&next);
        }
        let (mean, after) = spread(&mesh);
        assert!(after < 0.5 * before, "écart-type du rayon {before:.3} → {after:.3}");
        assert!((mean - r).abs() < 0.3, "rayon moyen {mean}");
        let inv = invert_affine(phi.affine()).unwrap();
        for p in mesh.vertices() {
            assert!(phi.trilinear(apply_affine(&inv, *p)).abs() <= 0.8 + 1e-9);
        }
    }
}
