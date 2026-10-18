//! Composantes connexes et remplissage des cavités dans un masque binaire 3D.

use std::collections::VecDeque;

use crate::Volume;

/// Connexité discrète : voisins par une face (6), une face ou une arête (18), ou aussi un sommet (26).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Connectivity {
    Six,
    Eighteen,
    TwentySix,
}

impl Connectivity {
    /// Décalages (di, dj, dk) des voisins.
    pub fn offsets(self) -> Vec<[isize; 3]> {
        let mut out = Vec::with_capacity(26);
        for dk in -1..=1isize {
            for dj in -1..=1isize {
                for di in -1..=1isize {
                    let n = di.abs() + dj.abs() + dk.abs();
                    let keep = match self {
                        Connectivity::Six => n == 1,
                        Connectivity::Eighteen => n == 1 || n == 2,
                        Connectivity::TwentySix => n >= 1,
                    };
                    if keep {
                        out.push([di, dj, dk]);
                    }
                }
            }
        }
        out
    }
}

/// Étiquette les composantes connexes de `mask` (0 = hors masque, 1..=n), triées par taille décroissante.
/// Rend le volume d'étiquettes et la taille de chaque composante (indice 0 : composante 1).
pub fn label_components(mask: &Volume<bool>, conn: Connectivity) -> (Volume<u32>, Vec<usize>) {
    let [nx, ny, nz] = mask.dims();
    let offsets = conn.offsets();
    let mut labels = vec![0u32; mask.len()];
    let mut sizes = Vec::new();
    let mut queue = VecDeque::new();
    for start in 0..mask.len() {
        if !mask.data()[start] || labels[start] != 0 {
            continue;
        }
        let id = sizes.len() as u32 + 1;
        labels[start] = id;
        queue.push_back(start);
        let mut size = 0;
        while let Some(idx) = queue.pop_front() {
            size += 1;
            let [i, j, k] = mask.coords(idx);
            for o in &offsets {
                let (a, b, c) = (i as isize + o[0], j as isize + o[1], k as isize + o[2]);
                if a < 0
                    || b < 0
                    || c < 0
                    || a >= nx as isize
                    || b >= ny as isize
                    || c >= nz as isize
                {
                    continue;
                }
                let n = mask.index(a as usize, b as usize, c as usize);
                if mask.data()[n] && labels[n] == 0 {
                    labels[n] = id;
                    queue.push_back(n);
                }
            }
        }
        sizes.push(size);
    }
    // Renumérotation par taille décroissante (à taille égale, ordre de découverte).
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]).then(a.cmp(&b)));
    let mut rank = vec![0u32; sizes.len() + 1];
    for (r, &old) in order.iter().enumerate() {
        rank[old + 1] = r as u32 + 1;
    }
    for l in &mut labels {
        *l = rank[*l as usize];
    }
    let sizes = order.iter().map(|&o| sizes[o]).collect();
    (mask.with_data(labels), sizes)
}

/// Plus grande composante connexe de `mask` (masque vide si `mask` est vide).
pub fn largest_component(mask: &Volume<bool>, conn: Connectivity) -> Volume<bool> {
    let (labels, _) = label_components(mask, conn);
    labels.map(|&l| l == 1)
}

/// Remplit les cavités : les composantes du fond, au sens de `background_conn`, qui ne touchent pas le
/// bord du volume.
pub fn fill_cavities(mask: &Volume<bool>, background_conn: Connectivity) -> Volume<bool> {
    let [nx, ny, nz] = mask.dims();
    let offsets = background_conn.offsets();
    // Parcours du fond à partir du bord du volume.
    let mut outside = vec![false; mask.len()];
    let mut queue = VecDeque::new();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let border =
                    i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1;
                let idx = mask.index(i, j, k);
                if border && !mask.data()[idx] {
                    outside[idx] = true;
                    queue.push_back(idx);
                }
            }
        }
    }
    while let Some(idx) = queue.pop_front() {
        let [i, j, k] = mask.coords(idx);
        for o in &offsets {
            let (a, b, c) = (i as isize + o[0], j as isize + o[1], k as isize + o[2]);
            if !mask.contains(a, b, c) {
                continue;
            }
            let n = mask.index(a as usize, b as usize, c as usize);
            if !mask.data()[n] && !outside[n] {
                outside[n] = true;
                queue.push_back(n);
            }
        }
    }
    mask.with_data(outside.into_iter().map(|o| !o).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::volume::IDENTITY;

    fn cube(dims: [usize; 3], lo: usize, hi: usize) -> Volume<bool> {
        let mut v = Volume::filled(dims, false, IDENTITY);
        for k in lo..hi {
            for j in lo..hi {
                for i in lo..hi {
                    v.set(i, j, k, true);
                }
            }
        }
        v
    }

    #[test]
    fn nombre_de_voisins() {
        assert_eq!(Connectivity::Six.offsets().len(), 6);
        assert_eq!(Connectivity::Eighteen.offsets().len(), 18);
        assert_eq!(Connectivity::TwentySix.offsets().len(), 26);
    }

    #[test]
    fn diagonale_selon_la_connexite() {
        let mut v = Volume::filled([4, 4, 4], false, IDENTITY);
        v.set(1, 1, 1, true);
        v.set(2, 2, 2, true);
        assert_eq!(label_components(&v, Connectivity::Six).1, vec![1, 1]);
        assert_eq!(label_components(&v, Connectivity::Eighteen).1, vec![1, 1]);
        assert_eq!(label_components(&v, Connectivity::TwentySix).1, vec![2]);
    }

    #[test]
    fn plus_grande_composante() {
        let mut v = cube([10, 10, 10], 1, 5);
        v.set(8, 8, 8, true);
        let l = largest_component(&v, Connectivity::Six);
        assert_eq!(l.data().iter().filter(|&&b| b).count(), 64);
        assert!(!l.get(8, 8, 8));
    }

    #[test]
    fn cavite_remplie() {
        let mut v = cube([8, 8, 8], 1, 6);
        v.set(3, 3, 3, false);
        let f = fill_cavities(&v, Connectivity::Six);
        assert!(f.get(3, 3, 3));
        assert_eq!(f.data().iter().filter(|&&b| b).count(), 125);
    }
}
