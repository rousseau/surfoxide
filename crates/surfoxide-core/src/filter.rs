//! Filtres et opérations de grille sur les volumes : flou gaussien, extraction de sous-volume, boîte
//! englobante.

use crate::volume::apply_affine;
use crate::Volume;

/// Flou gaussien séparable d'écart-type `sigma_mm` (mm), tronqué à 3σ ; bords répliqués.
pub fn gaussian_blur(vol: &Volume<f64>, sigma_mm: f64) -> Volume<f64> {
    if sigma_mm <= 0.0 {
        return vol.clone();
    }
    let dims = vol.dims();
    let spacing = vol.spacing();
    let stride = [1, dims[0], dims[0] * dims[1]];
    let mut data = vol.data().to_vec();
    let mut line = Vec::new();
    for axis in 0..3 {
        let sigma = sigma_mm / spacing[axis];
        let radius = (3.0 * sigma).ceil() as isize;
        let kernel: Vec<f64> = (-radius..=radius)
            .map(|x| (-(x * x) as f64 / (2.0 * sigma * sigma)).exp())
            .collect();
        let norm: f64 = kernel.iter().sum();
        let n = dims[axis];
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        for jb in 0..dims[b] {
            for ja in 0..dims[a] {
                let base = ja * stride[a] + jb * stride[b];
                line.clear();
                line.extend((0..n).map(|q| data[base + q * stride[axis]]));
                for q in 0..n {
                    let mut acc = 0.0;
                    for (t, w) in kernel.iter().enumerate() {
                        let p =
                            (q as isize + t as isize - radius).clamp(0, n as isize - 1) as usize;
                        acc += w * line[p];
                    }
                    data[base + q * stride[axis]] = acc / norm;
                }
            }
        }
    }
    vol.with_data(data)
}

/// Boîte englobante (bornes inclusives) des voxels où `pred` est vrai ; `None` si aucun.
pub fn bounding_box<T>(
    vol: &Volume<T>,
    pred: impl Fn(&T) -> bool,
) -> Option<([usize; 3], [usize; 3])> {
    let mut lo = [usize::MAX; 3];
    let mut hi = [0usize; 3];
    let mut any = false;
    for (idx, v) in vol.data().iter().enumerate() {
        if pred(v) {
            any = true;
            let c = vol.coords(idx);
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
    }
    any.then_some((lo, hi))
}

/// Sous-volume de dimensions `dims` dont le voxel (0, 0, 0) est le voxel `origin` (éventuellement
/// négatif) de `vol` ; les voxels hors de `vol` valent `fill`. L'affine est décalée en conséquence : un
/// même point garde ses coordonnées monde.
pub fn extract<T: Copy>(
    vol: &Volume<T>,
    origin: [isize; 3],
    dims: [usize; 3],
    fill: T,
) -> Volume<T> {
    let mut data = Vec::with_capacity(dims[0] * dims[1] * dims[2]);
    for k in 0..dims[2] as isize {
        for j in 0..dims[1] as isize {
            for i in 0..dims[0] as isize {
                let (a, b, c) = (i + origin[0], j + origin[1], k + origin[2]);
                data.push(if vol.contains(a, b, c) {
                    vol.get(a as usize, b as usize, c as usize)
                } else {
                    fill
                });
            }
        }
    }
    let mut affine = *vol.affine();
    let shift = apply_affine(&affine, origin.map(|x| x as f64));
    for (r, row) in affine.iter_mut().take(3).enumerate() {
        row[3] = shift[r];
    }
    Volume::from_vec(dims, data, affine)
}

/// Recadre `vol` sur la boîte englobante des voxels où `pred` est vrai, élargie de `margin` voxels de
/// chaque côté (au-delà du volume, complété par `fill`). Rend aussi l'origine du sous-volume.
pub fn crop_to<T: Copy>(
    vol: &Volume<T>,
    pred: impl Fn(&T) -> bool,
    margin: usize,
    fill: T,
) -> Option<(Volume<T>, [isize; 3])> {
    let (lo, hi) = bounding_box(vol, pred)?;
    let m = margin as isize;
    let origin = lo.map(|x| x as isize - m);
    let dims = std::array::from_fn(|a| hi[a] - lo[a] + 1 + 2 * margin);
    Some((extract(vol, origin, dims, fill), origin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::IDENTITY;

    #[test]
    fn flou_conserve_la_somme_loin_des_bords() {
        let mut v = Volume::filled([21, 21, 21], 0.0, IDENTITY);
        v.set(10, 10, 10, 1.0);
        let b = gaussian_blur(&v, 1.5);
        let s: f64 = b.data().iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        assert!(b.get(10, 10, 10) < 0.1 && b.get(10, 10, 10) > b.get(11, 10, 10));
    }

    #[test]
    fn recadrage_conserve_les_coordonnees_monde() {
        let aff = [
            [0.5, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 0.5, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut v = Volume::filled([10, 10, 10], false, aff);
        v.set(4, 5, 6, true);
        v.set(6, 5, 6, true);
        let (c, origin) = crop_to(&v, |&b| b, 2, false).unwrap();
        assert_eq!(origin, [2, 3, 4]);
        assert_eq!(c.dims(), [7, 5, 5]);
        assert!(c.get(2, 2, 2));
        let p = v.voxel_to_world([4.0, 5.0, 6.0]);
        assert_eq!(c.voxel_to_world([2.0, 2.0, 2.0]), p);
        // Marge hors du volume : complétée.
        let (e, _) = crop_to(&v, |&b| b, 6, false).unwrap();
        assert_eq!(e.dims(), [15, 13, 13]);
    }
}
