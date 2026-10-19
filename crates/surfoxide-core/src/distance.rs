//! Transformée de distance euclidienne exacte (Felzenszwalb & Huttenlocher 2012) et fonction distance
//! signée d'un masque, en mm (tailles de voxel prises en compte).

use crate::Volume;

/// Distance au carré, sur une ligne, de chaque point au point de coût minimal : `min_q f(q) + (s (p − q))²`.
/// `arg[p]` reçoit le `q` qui réalise le minimum.
fn edt_1d(f: &[f64], s: f64, out: &mut [f64], arg: &mut [usize], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let s2 = s * s;
    let mut k = 0usize;
    // Premier point fini.
    let Some(first) = f.iter().position(|x| x.is_finite()) else {
        out.fill(f64::INFINITY);
        arg.iter_mut().enumerate().for_each(|(q, a)| *a = q);
        return;
    };
    v[0] = first;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in first + 1..n {
        if !f[q].is_finite() {
            continue;
        }
        loop {
            let p = v[k];
            // Abscisse de l'intersection des paraboles issues de q et de p.
            let sx = ((f[q] + s2 * (q * q) as f64) - (f[p] + s2 * (p * p) as f64)) / (2.0 * s2 * (q as f64 - p as f64));
            if sx <= z[k] {
                if k == 0 {
                    v[0] = q;
                    z[1] = f64::INFINITY;
                    break;
                }
                k -= 1;
            } else {
                k += 1;
                v[k] = q;
                z[k] = sx;
                z[k + 1] = f64::INFINITY;
                break;
            }
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let d = s * (q as f64 - v[k] as f64);
        *o = d * d + f[v[k]];
        arg[q] = v[k];
    }
}

/// Distance euclidienne au carré (mm²) de chaque voxel au voxel `true` le plus proche de `mask`
/// (infini si le masque est vide).
pub fn edt_squared(mask: &Volume<bool>) -> Volume<f64> {
    feature_transform(mask).0
}

/// Transformée de distance avec plus proches voisins : distance au carré (mm²) de chaque voxel au voxel
/// `true` le plus proche, et indice linéaire de ce voxel (`u32::MAX` si le masque est vide).
pub fn feature_transform(mask: &Volume<bool>) -> (Volume<f64>, Volume<u32>) {
    let dims = mask.dims();
    let spacing = mask.spacing();
    let mut d: Vec<f64> = mask
        .data()
        .iter()
        .map(|&b| if b { 0.0 } else { f64::INFINITY })
        .collect();
    let mut feat: Vec<u32> = (0..mask.len() as u32)
        .map(|i| if mask.data()[i as usize] { i } else { u32::MAX })
        .collect();
    let stride = [1, dims[0], dims[0] * dims[1]];
    let nmax = *dims.iter().max().unwrap_or(&0);
    let (mut f, mut out) = (vec![0.0; nmax], vec![0.0; nmax]);
    let (mut arg, mut line_feat) = (vec![0usize; nmax], vec![0u32; nmax]);
    let (mut v, mut z) = (vec![0usize; nmax], vec![0.0; nmax + 1]);
    for axis in 0..3 {
        let n = dims[axis];
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        for jb in 0..dims[b] {
            for ja in 0..dims[a] {
                let base = ja * stride[a] + jb * stride[b];
                for q in 0..n {
                    f[q] = d[base + q * stride[axis]];
                    line_feat[q] = feat[base + q * stride[axis]];
                }
                edt_1d(&f[..n], spacing[axis], &mut out[..n], &mut arg[..n], &mut v, &mut z);
                for q in 0..n {
                    d[base + q * stride[axis]] = out[q];
                    feat[base + q * stride[axis]] = line_feat[arg[q]];
                }
            }
        }
    }
    (mask.with_data(d), mask.with_data(feat))
}

/// Fonction distance signée (mm) : négative à l'intérieur (distance au fond le plus proche), positive à
/// l'extérieur (distance à l'objet le plus proche). Deux voxels voisins de part et d'autre de la frontière
/// valent −h et +h : le niveau 0 passe à mi-chemin.
pub fn signed_distance(mask: &Volume<bool>) -> Volume<f64> {
    let to_object = edt_squared(mask);
    let to_background = edt_squared(&mask.map(|&b| !b));
    let data = mask
        .data()
        .iter()
        .zip(to_object.data().iter().zip(to_background.data()))
        .map(
            |(&inside, (&dobj, &dbg))| {
                if inside {
                    -dbg.sqrt()
                } else {
                    dobj.sqrt()
                }
            },
        )
        .collect();
    mask.with_data(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::IDENTITY;

    fn brute(mask: &Volume<bool>) -> Vec<f64> {
        let s = mask.spacing();
        let pts: Vec<[usize; 3]> = (0..mask.len())
            .filter(|&i| mask.data()[i])
            .map(|i| mask.coords(i))
            .collect();
        (0..mask.len())
            .map(|i| {
                let c = mask.coords(i);
                pts.iter()
                    .map(|p| {
                        (0..3)
                            .map(|a| (s[a] * (c[a] as f64 - p[a] as f64)).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect()
    }

    #[test]
    fn egale_la_force_brute_en_anisotrope() {
        let aff = [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 1.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut m = Volume::filled([9, 7, 6], false, aff);
        // Quelques points dispersés, de façon déterministe.
        for (i, j, k) in [(1, 1, 1), (7, 2, 4), (3, 6, 2), (8, 6, 5), (4, 3, 3)] {
            m.set(i, j, k, true);
        }
        let d = edt_squared(&m);
        for (a, b) in d.data().iter().zip(brute(&m)) {
            assert!((a - b).abs() < 1e-9, "{a} contre {b}");
        }
    }

    #[test]
    fn plus_proches_voisins() {
        let mut m = Volume::filled([9, 7, 6], false, IDENTITY);
        for (i, j, k) in [(1, 1, 1), (7, 2, 4), (3, 6, 2)] {
            m.set(i, j, k, true);
        }
        let (d, feat) = feature_transform(&m);
        for idx in 0..m.len() {
            let f = feat.data()[idx] as usize;
            assert!(m.data()[f]);
            let (c, q) = (m.coords(idx), m.coords(f));
            let d2: f64 = (0..3).map(|a| (c[a] as f64 - q[a] as f64).powi(2)).sum();
            assert_eq!(d2, d.data()[idx]);
        }
    }

    #[test]
    fn distance_signee_d_un_cube() {
        let mut m = Volume::filled([12, 12, 12], false, IDENTITY);
        for k in 3..9 {
            for j in 3..9 {
                for i in 3..9 {
                    m.set(i, j, k, true);
                }
            }
        }
        let phi = signed_distance(&m);
        assert_eq!(phi.get(5, 5, 5), -3.0);
        assert_eq!(phi.get(3, 5, 5), -1.0);
        assert_eq!(phi.get(2, 5, 5), 1.0);
        assert_eq!(phi.get(0, 5, 5), 3.0);
    }
}
