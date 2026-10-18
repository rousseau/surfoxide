//! Volumes NIfTI-1 (`.nii`, `.nii.gz`), par le crate `nifti`.
//!
//! L'affine voxel → monde est la sform si `sform_code > 0`, sinon la qform si `qform_code > 0`, sinon
//! la diagonale des tailles de voxel (convention NIfTI, méthode 1). L'en-tête d'origine est rendu avec
//! le volume pour écrire les volumes dérivés sur la même grille sans rien perdre.

use std::path::Path;

use ndarray::{Array3, IxDyn, ShapeBuilder};
use nifti::writer::WriterOptions;
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, ReaderOptions};
use surfoxide_core::volume::IDENTITY;
use surfoxide_core::{Affine, Volume};

use crate::{format_err, IoError, Result};

fn nifti_err(e: nifti::NiftiError) -> IoError {
    IoError::Format(format!("NIfTI : {e}"))
}

/// Affine voxel → monde d'un en-tête NIfTI.
pub fn header_affine(h: &NiftiHeader) -> Affine {
    if h.sform_code > 0 {
        let r = |row: [f32; 4]| row.map(f64::from);
        return [r(h.srow_x), r(h.srow_y), r(h.srow_z), [0.0, 0.0, 0.0, 1.0]];
    }
    let [_, dx, dy, dz, ..] = h.pixdim.map(f64::from);
    if h.qform_code > 0 {
        let (b, c, d) = (
            f64::from(h.quatern_b),
            f64::from(h.quatern_c),
            f64::from(h.quatern_d),
        );
        let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
        let qfac = if h.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
        let rot = [
            [
                a * a + b * b - c * c - d * d,
                2.0 * (b * c - a * d),
                2.0 * (b * d + a * c),
            ],
            [
                2.0 * (b * c + a * d),
                a * a + c * c - b * b - d * d,
                2.0 * (c * d - a * b),
            ],
            [
                2.0 * (b * d - a * c),
                2.0 * (c * d + a * b),
                a * a + d * d - c * c - b * b,
            ],
        ];
        let scale = [dx, dy, dz * qfac];
        let off = [h.quatern_x, h.quatern_y, h.quatern_z].map(f64::from);
        let mut m = IDENTITY;
        for r in 0..3 {
            for c in 0..3 {
                m[r][c] = rot[r][c] * scale[c];
            }
            m[r][3] = off[r];
        }
        return m;
    }
    [
        [dx, 0.0, 0.0, 0.0],
        [0.0, dy, 0.0, 0.0],
        [0.0, 0.0, dz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// Écrit `affine` dans la sform de l'en-tête (code 2, « aligned anat ») ; la qform n'est pas touchée.
pub fn set_header_affine(h: &mut NiftiHeader, affine: &Affine) {
    let r = |row: [f64; 4]| row.map(|x| x as f32);
    h.srow_x = r(affine[0]);
    h.srow_y = r(affine[1]);
    h.srow_z = r(affine[2]);
    if h.sform_code <= 0 {
        h.sform_code = 2;
    }
    for (c, d) in h.pixdim[1..4].iter_mut().enumerate() {
        *d = (affine[0][c].powi(2) + affine[1][c].powi(2) + affine[2][c].powi(2)).sqrt() as f32;
    }
}

/// Lit un volume 3D (les dimensions supplémentaires de taille 1 sont acceptées) converti en `T`, avec
/// son en-tête.
fn read_as<T>(path: &Path) -> Result<(Volume<T>, NiftiHeader)>
where
    T: nifti::DataElement + Copy,
{
    let obj = ReaderOptions::new().read_file(path).map_err(nifti_err)?;
    let header = obj.header().clone();
    let array = obj.into_volume().into_ndarray::<T>().map_err(nifti_err)?;
    let shape = array.shape().to_vec();
    if shape.len() < 3 || shape[3..].iter().any(|&n| n != 1) {
        return format_err(format!(
            "{} : volume 3D attendu, dimensions {shape:?}",
            path.display()
        ));
    }
    let dims = [shape[0], shape[1], shape[2]];
    let n = dims[0] * dims[1] * dims[2];
    // Le crate rend un tableau en ordre Fortran (x le plus rapide) : c'est déjà notre rangement.
    let data: Vec<T> = match array.as_slice_memory_order() {
        Some(s) if array.strides()[0] == 1 => s[..n].to_vec(),
        _ => {
            let mut idx = vec![0usize; shape.len()];
            let mut out = Vec::with_capacity(n);
            for k in 0..dims[2] {
                for j in 0..dims[1] {
                    for i in 0..dims[0] {
                        (idx[0], idx[1], idx[2]) = (i, j, k);
                        out.push(array[IxDyn(&idx)]);
                    }
                }
            }
            out
        }
    };
    Ok((Volume::from_vec(dims, data, header_affine(&header)), header))
}

/// Lit une segmentation : valeurs arrondies à l'entier le plus proche.
pub fn read_labels(path: impl AsRef<Path>) -> Result<(Volume<i32>, NiftiHeader)> {
    let (v, h) = read_as::<f32>(path.as_ref())?;
    Ok((v.map(|&x| x.round() as i32), h))
}

/// Lit un volume scalaire en `f32`.
pub fn read_scalar(path: impl AsRef<Path>) -> Result<(Volume<f32>, NiftiHeader)> {
    read_as::<f32>(path.as_ref())
}

/// Prépare l'en-tête d'un volume écrit : celui de référence (ou un en-tête vide) avec l'affine du volume.
fn output_header(vol_affine: &Affine, reference: Option<&NiftiHeader>) -> NiftiHeader {
    let mut header = reference.cloned().unwrap_or_default();
    // Valeurs écrites telles quelles : pas de mise à l'échelle.
    header.scl_slope = 1.0;
    header.scl_inter = 0.0;
    set_header_affine(&mut header, vol_affine);
    header
}

/// Les types écrits doivent satisfaire la contrainte `Pod` (bytemuck) du crate `nifti`, qui ne la
/// réexporte pas : on écrit une fonction par type concret.
macro_rules! writer {
    ($(#[$doc:meta])* $name:ident, $t:ty) => {
        $(#[$doc])*
        pub fn $name(
            path: impl AsRef<Path>,
            vol: &Volume<$t>,
            reference: Option<&NiftiHeader>,
        ) -> Result<()> {
            let [nx, ny, nz] = vol.dims();
            let array = Array3::from_shape_vec((nx, ny, nz).f(), vol.data().to_vec())
                .map_err(|e| IoError::Format(format!("NIfTI : {e}")))?;
            let header = output_header(vol.affine(), reference);
            WriterOptions::new(path.as_ref())
                .reference_header(&header)
                .write_nifti(&array)
                .map_err(nifti_err)
        }
    };
}

writer!(
    /// Écrit un masque ou une segmentation en `u8`.
    write_u8,
    u8
);
writer!(
    /// Écrit des labels en `i16`.
    write_i16,
    i16
);
writer!(
    /// Écrit un volume scalaire en `f32`.
    write_f32,
    f32
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::tmp;

    fn affine_exemple() -> Affine {
        [
            [-0.5, 0.0, 0.0, 40.0],
            [0.0, 0.5, 0.0, -30.0],
            [0.0, 0.0, 0.6, -20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    #[test]
    fn aller_retour_labels() {
        let dims = [5, 4, 3];
        let data: Vec<u8> = (0..60).map(|i| (i % 7) as u8).collect();
        let vol = Volume::from_vec(dims, data, affine_exemple());
        let p = tmp("labels.nii.gz");
        write_u8(&p, &vol, None).unwrap();
        let (back, h) = read_labels(&p).unwrap();
        assert_eq!(back.dims(), dims);
        // L'ordre des voxels est conservé : (i, j, k) → même valeur.
        assert_eq!(back.get(4, 1, 2) as u8, vol.get(4, 1, 2));
        assert_eq!(back.get(1, 3, 0) as u8, vol.get(1, 3, 0));
        // L'en-tête NIfTI stocke l'affine en f32.
        for (r, row) in back.affine().iter().enumerate() {
            for (c, x) in row.iter().enumerate() {
                assert!((x - affine_exemple()[r][c]).abs() < 1e-6);
            }
        }
        assert!(h.sform_code > 0);
    }

    #[test]
    fn affine_par_la_qform() {
        let mut h = NiftiHeader {
            qform_code: 1,
            sform_code: 0,
            pixdim: [1.0, 0.5, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0],
            quatern_x: 1.0,
            quatern_y: 2.0,
            quatern_z: 3.0,
            ..Default::default()
        };
        let a = header_affine(&h);
        assert_eq!(a[0][0], 0.5);
        assert_eq!(a[2][2], 0.6000000238418579);
        assert_eq!(a[1][3], 2.0);
        // qfac négatif : l'axe z est inversé.
        h.pixdim[0] = -1.0;
        assert!(header_affine(&h)[2][2] < 0.0);
    }

    /// Lecture d'une vraie segmentation BOUNTI : dimensions, labels attendus, affine non triviale.
    #[test]
    #[ignore = "données locales : ~/Exp/CRL-Cinema/data/raw/bounti"]
    fn lecture_bounti_reelle() {
        let home = std::env::var("HOME").unwrap();
        let p = format!(
            "{home}/Exp/CRL-Cinema/data/raw/bounti/sub-z7uv9/ses-01/anat/sub-z7uv9_ses-01_res-0p5mm_desc-bounti19_dseg.nii.gz"
        );
        let (v, _) = read_labels(p).unwrap();
        let max = *v.data().iter().max().unwrap();
        assert!((17..=19).contains(&max), "label maximal {max}");
        let s = v.spacing();
        assert!((s[0] - 0.5).abs() < 1e-3, "taille de voxel {s:?}");
    }
}
