//! Volume 3D régulier, avec sa matrice affine voxel → monde (RAS, mm).

use crate::vec3::Vec3;

/// Matrice affine homogène 4×4, par lignes.
pub type Affine = [[f64; 4]; 4];

/// Matrice identité : voxels de 1 mm, origine au voxel (0, 0, 0).
pub const IDENTITY: Affine = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

/// Volume de dimensions `dims = [nx, ny, nz]`, rangé avec l'indice `i` (axe x) le plus rapide,
/// comme dans un fichier NIfTI.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume<T> {
    dims: [usize; 3],
    data: Vec<T>,
    affine: Affine,
}

impl<T> Volume<T> {
    /// Construit un volume ; panique si `data` n'a pas `nx · ny · nz` éléments.
    pub fn from_vec(dims: [usize; 3], data: Vec<T>, affine: Affine) -> Self {
        assert_eq!(
            data.len(),
            dims[0] * dims[1] * dims[2],
            "taille des données incompatible avec les dimensions {dims:?}"
        );
        Self { dims, data, affine }
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn affine(&self) -> &Affine {
        &self.affine
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<T> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Indice linéaire du voxel (i, j, k).
    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// Coordonnées (i, j, k) de l'indice linéaire `idx`.
    #[inline]
    pub fn coords(&self, idx: usize) -> [usize; 3] {
        let i = idx % self.dims[0];
        let r = idx / self.dims[0];
        [i, r % self.dims[1], r / self.dims[1]]
    }

    /// Vrai si (i, j, k), en entiers signés, est dans le volume.
    #[inline]
    pub fn contains(&self, i: isize, j: isize, k: isize) -> bool {
        i >= 0
            && j >= 0
            && k >= 0
            && (i as usize) < self.dims[0]
            && (j as usize) < self.dims[1]
            && (k as usize) < self.dims[2]
    }

    /// Nouveau volume de même géométrie, de valeurs `f(v)`.
    pub fn map<U>(&self, f: impl FnMut(&T) -> U) -> Volume<U> {
        Volume {
            dims: self.dims,
            data: self.data.iter().map(f).collect(),
            affine: self.affine,
        }
    }

    /// Nouveau volume de même géométrie, avec d'autres données.
    pub fn with_data<U>(&self, data: Vec<U>) -> Volume<U> {
        Volume::from_vec(self.dims, data, self.affine)
    }

    /// Point monde (mm) du voxel de coordonnées continues `p`.
    pub fn voxel_to_world(&self, p: Vec3) -> Vec3 {
        apply_affine(&self.affine, p)
    }

    /// Coordonnées voxel continues du point monde `p` ; `None` si l'affine n'est pas inversible.
    pub fn world_to_voxel(&self, p: Vec3) -> Option<Vec3> {
        invert_affine(&self.affine).map(|inv| apply_affine(&inv, p))
    }

    /// Taille des voxels (mm) le long des trois axes : normes des colonnes de l'affine.
    pub fn spacing(&self) -> Vec3 {
        let a = &self.affine;
        std::array::from_fn(|c| (a[0][c].powi(2) + a[1][c].powi(2) + a[2][c].powi(2)).sqrt())
    }
}

impl<T: Copy> Volume<T> {
    /// Volume constant.
    pub fn filled(dims: [usize; 3], value: T, affine: Affine) -> Self {
        Self::from_vec(dims, vec![value; dims[0] * dims[1] * dims[2]], affine)
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> T {
        self.data[self.index(i, j, k)]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, k: usize, v: T) {
        let idx = self.index(i, j, k);
        self.data[idx] = v;
    }
}

impl Volume<f64> {
    /// Interpolation trilinéaire au point de coordonnées voxel continues `q` (bords répliqués).
    pub fn trilinear(&self, q: Vec3) -> f64 {
        let dims = self.dims;
        let mut base = [0usize; 3];
        let mut t = [0.0; 3];
        for a in 0..3 {
            let x = q[a].clamp(0.0, (dims[a] - 1) as f64);
            let i = (x.floor() as usize).min(dims[a].saturating_sub(2));
            base[a] = i;
            t[a] = (x - i as f64).min(1.0);
        }
        let mut acc = 0.0;
        for c in 0..8 {
            let o = [c & 1, (c >> 1) & 1, (c >> 2) & 1];
            let idx = self.index(
                (base[0] + o[0]).min(dims[0] - 1),
                (base[1] + o[1]).min(dims[1] - 1),
                (base[2] + o[2]).min(dims[2] - 1),
            );
            let w: f64 = (0..3).map(|a| if o[a] == 1 { t[a] } else { 1.0 - t[a] }).product();
            acc += w * self.data[idx];
        }
        acc
    }
}

/// Applique l'affine `a` au point `p`.
pub fn apply_affine(a: &Affine, p: Vec3) -> Vec3 {
    std::array::from_fn(|r| a[r][0] * p[0] + a[r][1] * p[1] + a[r][2] * p[2] + a[r][3])
}

/// Inverse d'une affine (partie linéaire 3×3 inversée, translation recalculée).
pub fn invert_affine(a: &Affine) -> Option<Affine> {
    let m = |r: usize, c: usize| a[r][c];
    let det = affine_determinant(a);
    if det.abs() < 1e-12 {
        return None;
    }
    let mut inv = IDENTITY;
    for (r, row) in inv.iter_mut().take(3).enumerate() {
        for (c, x) in row.iter_mut().take(3).enumerate() {
            // Cofacteur transposé : inv[r][c] = cof(c, r) / det.
            let (r1, r2) = ((c + 1) % 3, (c + 2) % 3);
            let (c1, c2) = ((r + 1) % 3, (r + 2) % 3);
            *x = (m(r1, c1) * m(r2, c2) - m(r1, c2) * m(r2, c1)) / det;
        }
        row[3] = -(row[0] * a[0][3] + row[1] * a[1][3] + row[2] * a[2][3]);
    }
    Some(inv)
}

/// Déterminant de la partie linéaire : négatif si l'affine inverse l'orientation (repère indirect).
pub fn affine_determinant(a: &Affine) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exemple() -> Affine {
        [
            [-0.5, 0.0, 0.1, 30.0],
            [0.0, 0.5, 0.0, -20.0],
            [0.05, 0.0, 0.8, 10.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn aller_retour_voxel_monde() {
        let v = Volume::filled([4, 5, 6], 0u8, exemple());
        let p = [1.5, 2.25, 3.0];
        let q = v.world_to_voxel(v.voxel_to_world(p)).unwrap();
        for c in 0..3 {
            assert!((p[c] - q[c]).abs() < 1e-12);
        }
    }

    #[test]
    fn indices_et_coordonnees() {
        let v = Volume::filled([4, 5, 6], 0u8, IDENTITY);
        let idx = v.index(3, 2, 5);
        assert_eq!(v.coords(idx), [3, 2, 5]);
        assert_eq!(v.len(), 120);
    }

    #[test]
    fn interpolation_trilineaire_exacte_sur_une_fonction_affine() {
        let mut v = Volume::filled([4, 5, 6], 0.0, IDENTITY);
        for k in 0..6 {
            for j in 0..5 {
                for i in 0..4 {
                    v.set(i, j, k, 1.0 + 2.0 * i as f64 - j as f64 + 0.5 * k as f64);
                }
            }
        }
        let q = [1.25, 3.5, 4.75];
        assert!((v.trilinear(q) - (1.0 + 2.5 - 3.5 + 2.375)).abs() < 1e-12);
        // Hors du volume : valeur du bord.
        assert!((v.trilinear([-1.0, 0.0, 0.0]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn taille_des_voxels() {
        let v = Volume::filled([2, 2, 2], 0u8, exemple());
        let s = v.spacing();
        assert!((s[1] - 0.5).abs() < 1e-12);
        assert!(affine_determinant(&exemple()) < 0.0);
    }
}
