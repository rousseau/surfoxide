//! Courbures par estimation du tenseur de courbure sur chaque face (Rusinkiewicz 2004).
//!
//! Sur chaque face, la seconde forme fondamentale est ajustée aux moindres carrés sur la variation des
//! normales de sommets le long des trois arêtes ; elle est ensuite ramenée dans le repère tangent de
//! chaque sommet et moyennée avec les poids des aires de Voronoï mixtes (Meyer et al. 2002).
//!
//! Portage fidèle de `curvatures_and_derivatives` du code de référence de la DPF* (lui-même repris de
//! trimesh2) : mêmes normales de sommets (pondérées par `A / (|e|² |e'|²)`), mêmes aires de coin.
//! Convention de signe : avec des normales sortantes, la courbure est positive sur une partie convexe
//! (sphère de rayon R : `1/R`), donc positive sur les crêtes gyrales et négative au fond des sillons.

use surfoxide_core::vec3::{self, Vec3};
use surfoxide_core::TriMesh;

/// Courbures par sommet, en mm⁻¹.
#[derive(Debug, Clone)]
pub struct Curvature {
    /// Plus grande courbure principale.
    pub k_max: Vec<f64>,
    /// Plus petite courbure principale.
    pub k_min: Vec<f64>,
    /// Courbure moyenne `(k_max + k_min) / 2`.
    pub mean: Vec<f64>,
}

/// Rabat le repère (u, v) dans le plan de normale `nf` par la rotation minimale.
fn rotate_frame(u: Vec3, v: Vec3, nf: Vec3) -> (Vec3, Vec3) {
    let np = vec3::normalize(vec3::cross(u, v));
    let ndot = vec3::dot(nf, np);
    let (mut u, mut v) = (u, v);
    if ndot <= -1.0 {
        u = vec3::scale(u, -1.0);
        v = vec3::scale(v, -1.0);
        return (u, v);
    }
    let perp = vec3::sub(nf, vec3::scale(np, ndot));
    let dperp = vec3::scale(vec3::add(np, nf), 1.0 / (1.0 + ndot));
    let ru = vec3::sub(u, vec3::scale(dperp, vec3::dot(perp, u)));
    let rv = vec3::sub(v, vec3::scale(dperp, vec3::dot(perp, v)));
    (ru, rv)
}

/// Résout le système 3×3 symétrique `m x = r` (Cramer) ; `None` s'il est singulier.
fn solve3(m: [[f64; 3]; 3], r: Vec3) -> Option<Vec3> {
    let det = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(m);
    if d.abs() < 1e-300 {
        return None;
    }
    Some(std::array::from_fn(|c| {
        let mut mc = m;
        for (row, &x) in mc.iter_mut().zip(&r) {
            row[c] = x;
        }
        det(mc) / d
    }))
}

/// Courbures principales et moyenne de chaque sommet.
pub fn curvature(mesh: &TriMesh) -> Curvature {
    let v = mesh.vertices();
    let nv = mesh.n_vertices();
    let faces = mesh.faces();
    let pts = |f: &[u32; 3]| f.map(|i| v[i as usize]);

    // Normales de sommets, aires de coin (Voronoï mixte) et repère initial.
    let mut normals = vec![[0.0; 3]; nv];
    let mut corner = vec![[0.0; 3]; faces.len()];
    let mut vertex_area = vec![0.0; nv];
    let mut up = vec![[0.0; 3]; nv];
    let mut face_normal = Vec::with_capacity(faces.len());
    for (fi, f) in faces.iter().enumerate() {
        let p = pts(f);
        let e = [vec3::sub(p[2], p[1]), vec3::sub(p[0], p[2]), vec3::sub(p[1], p[0])];
        let l2 = e.map(|e| vec3::dot(e, e));
        let nf = vec3::normalize(vec3::triangle_normal(p[0], p[1], p[2]));
        face_normal.push(nf);
        let area = 0.5 * vec3::norm(vec3::triangle_normal(p[0], p[1], p[2]));
        for k in 0..3 {
            // Poids du sommet k : aire / (produit des carrés des deux arêtes incidentes).
            let w = area / (l2[(k + 1) % 3] * l2[(k + 2) % 3]);
            let i = f[k] as usize;
            normals[i] = vec3::add(normals[i], vec3::scale(nf, w));
        }
        let ew = [
            l2[0] * (l2[1] + l2[2] - l2[0]),
            l2[1] * (l2[2] + l2[0] - l2[1]),
            l2[2] * (l2[0] + l2[1] - l2[2]),
        ];
        let c = &mut corner[fi];
        if let Some(k) = (0..3).find(|&k| ew[k] <= 0.0) {
            // Triangle obtus en k : les deux autres coins reçoivent un quart de triangle.
            let (a, b) = ((k + 1) % 3, (k + 2) % 3);
            c[a] = -0.25 * l2[b] * area / vec3::dot(e[k], e[b]);
            c[b] = -0.25 * l2[a] * area / vec3::dot(e[k], e[a]);
            c[k] = area - c[a] - c[b];
        } else {
            let s = 0.5 * area / (ew[0] + ew[1] + ew[2]);
            for k in 0..3 {
                c[k] = s * (ew[(k + 1) % 3] + ew[(k + 2) % 3]);
            }
        }
        for k in 0..3 {
            vertex_area[f[k] as usize] += c[k];
        }
        up[f[0] as usize] = vec3::normalize(e[2]);
        up[f[1] as usize] = vec3::normalize(e[0]);
        up[f[2] as usize] = vec3::normalize(e[1]);
    }
    let normals: Vec<Vec3> = normals.into_iter().map(vec3::normalize).collect();
    let mut vp = vec![[0.0; 3]; nv];
    for i in 0..nv {
        up[i] = vec3::normalize(vec3::cross(up[i], normals[i]));
        vp[i] = vec3::cross(normals[i], up[i]);
    }

    // Tenseur par face, projeté et accumulé sur les sommets.
    let mut tensor = vec![[0.0; 3]; nv]; // (ku, kuv, kv) dans le repère (up, vp)
    for (fi, f) in faces.iter().enumerate() {
        let p = pts(f);
        let e = [vec3::sub(p[2], p[1]), vec3::sub(p[0], p[2]), vec3::sub(p[1], p[0])];
        let nf = face_normal[fi];
        let t = vec3::normalize(e[0]);
        let b = vec3::normalize(vec3::cross(nf, t));
        let n = f.map(|i| normals[i as usize]);
        // Variation de la normale le long de chaque arête : e_k va de f[k+1] à f[k+2].
        let dn = [vec3::sub(n[2], n[1]), vec3::sub(n[0], n[2]), vec3::sub(n[1], n[0])];
        let mut m = [[0.0; 3]; 3];
        let mut r = [0.0; 3];
        for k in 0..3 {
            let (et, eb) = (vec3::dot(e[k], t), vec3::dot(e[k], b));
            let (nt, nb) = (vec3::dot(dn[k], t), vec3::dot(dn[k], b));
            // Lignes [et, eb, 0] → nt et [0, et, eb] → nb ; équations normales.
            let rows = [([et, eb, 0.0], nt), ([0.0, et, eb], nb)];
            for (a, y) in rows {
                for i in 0..3 {
                    for j in 0..3 {
                        m[i][j] += a[i] * a[j];
                    }
                    r[i] += a[i] * y;
                }
            }
        }
        let Some([fku, fkuv, fkv]) = solve3(m, r) else { continue };
        for k in 0..3 {
            let i = f[k] as usize;
            if vertex_area[i] == 0.0 {
                continue;
            }
            let w = corner[fi][k] / vertex_area[i];
            let (ru, rv) = rotate_frame(up[i], vp[i], nf);
            let (u1, v1) = (vec3::dot(ru, t), vec3::dot(ru, b));
            let (u2, v2) = (vec3::dot(rv, t), vec3::dot(rv, b));
            let q = |a: [f64; 2], c: [f64; 2]| a[0] * (fku * c[0] + fkuv * c[1]) + a[1] * (fkuv * c[0] + fkv * c[1]);
            tensor[i][0] += w * q([u1, v1], [u1, v1]);
            tensor[i][1] += w * q([u1, v1], [u2, v2]);
            tensor[i][2] += w * q([u2, v2], [u2, v2]);
        }
    }

    let mut k_max = Vec::with_capacity(nv);
    let mut k_min = Vec::with_capacity(nv);
    for &[ku, kuv, kv] in &tensor {
        let half = 0.5 * (ku + kv);
        let rad = (0.25 * (ku - kv) * (ku - kv) + kuv * kuv).sqrt();
        k_max.push(half + rad);
        k_min.push(half - rad);
    }
    let mean = k_max.iter().zip(&k_min).map(|(a, b)| 0.5 * (a + b)).collect();
    Curvature { k_max, k_min, mean }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::{icosphere, torus};

    #[test]
    fn sphere_courbure_inverse_du_rayon() {
        let r = 12.0;
        let c = curvature(&icosphere(4, r));
        for i in 0..c.mean.len() {
            assert!((c.mean[i] * r - 1.0).abs() < 1e-2, "{}", c.mean[i] * r);
            assert!((c.k_max[i] - c.k_min[i]).abs() * r < 2e-2);
        }
    }

    #[test]
    fn tore_courbures_principales() {
        // Tore (R, a) : k = 1/a dans la direction du petit cercle ; sur l'équateur extérieur, l'autre
        // courbure vaut 1/(R + a), sur l'équateur intérieur −1/(R − a).
        let (big, small) = (10.0, 3.0);
        let m = torus(160, 80, big, small);
        let c = curvature(&m);
        for (i, p) in m.vertices().iter().enumerate() {
            let rho = (p[0] * p[0] + p[1] * p[1]).sqrt();
            if p[2].abs() < 1e-9 && (rho - (big + small)).abs() < 1e-6 {
                assert!((c.k_max[i] - 1.0 / small).abs() < 0.01, "k_max {}", c.k_max[i]);
                assert!((c.k_min[i] - 1.0 / (big + small)).abs() < 0.01, "k_min {}", c.k_min[i]);
            }
            if p[2].abs() < 1e-9 && (rho - (big - small)).abs() < 1e-6 {
                assert!((c.k_min[i] + 1.0 / (big - small)).abs() < 0.01, "k_min {}", c.k_min[i]);
            }
        }
    }
}
