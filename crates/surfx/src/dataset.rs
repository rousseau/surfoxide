//! Exemples d'entraînement des réseaux de déformation (P1) : pour un hémisphère, l'entrée du réseau (classes de
//! tissu sur le cadrage de l'hémisphère), la distance signée qui sert de perte faible, et le couple blanche–piale
//! (même triangulation) qui sert de supervision sommet à sommet.
//!
//! Distance signée (`f32`, mm) : à la frontière substance blanche et cortex, sillons enfouis ouverts, négative
//! dedans.
//!
//! Classes (`u8`), indépendantes de la table de labels : 0 fond et LCR, 1 cortex, 2 substance blanche fermée,
//! 3 nappe médiane des sillons enfouis (cortex où les deux berges se touchent, ouvert par la piale).

use std::path::Path;

use surfoxide_core::distance::signed_distance;
use surfoxide_core::Hemisphere;
use surfoxide_mask::LabelTable;

use crate::white::hemisphere_grids;

/// Écrit `<préfixe>_classes.nii.gz`, `<préfixe>_sdf.nii.gz`, `<préfixe>_white.surf.gii`, `<préfixe>_pial.surf.gii`.
pub fn run(
    seg: &Path,
    table: &LabelTable,
    hemi: Hemisphere,
    white: &Path,
    pial: &Path,
    out_prefix: &Path,
) -> Result<(), String> {
    let (labels, _) = surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    let grids = hemisphere_grids(&labels, table, hemi)?;
    let skeleton = surfoxide_pial::sulcal_skeleton(&grids.white, &grids.cortex, 1.5);
    let classes = grids.white.with_data(
        grids
            .white
            .data()
            .iter()
            .zip(grids.cortex.data().iter().zip(skeleton.data()))
            .map(|(&w, (&c, &k))| match (w, c, k) {
                (true, _, _) => 2u8,
                (false, true, true) => 3,
                (false, true, false) => 1,
                _ => 0,
            })
            .collect(),
    );
    // Frontière que la piale doit suivre : substance blanche et cortex, sillons enfouis ouverts. `signed_distance`
    // mesure la distance aux centres des voxels de l'autre côté (±h juste de part et d'autre) ; on retire h/2 pour
    // obtenir la distance aux faces des voxels, c'est-à-dire à la frontière elle-même (±h/2, nulle sur les faces).
    let tissue = classes.map(|&c| c == 1 || c == 2);
    let half = 0.5 * tissue.spacing().into_iter().fold(f64::INFINITY, f64::min);
    let sdf = signed_distance(&tissue).map(|&d| (d.signum() * (d.abs() - half)) as f32);
    if let Some(dir) = out_prefix.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let prefix = out_prefix.display();
    let path = format!("{prefix}_classes.nii.gz");
    surfoxide_io::nifti::write_u8(&path, &classes, None).map_err(|e| format!("{path}: {e}"))?;
    let path = format!("{prefix}_sdf.nii.gz");
    surfoxide_io::nifti::write_f32(&path, &sdf, None).map_err(|e| format!("{path}: {e}"))?;
    // Les surfaces doivent partager leur triangulation (supervision sommet à sommet).
    let (w, _) = surfoxide_io::gifti::read_surface(white).map_err(|e| format!("{}: {e}", white.display()))?;
    let (p, _) = surfoxide_io::gifti::read_surface(pial).map_err(|e| format!("{}: {e}", pial.display()))?;
    if w.faces() != p.faces() {
        return Err("blanche et piale n'ont pas la même triangulation".into());
    }
    for (src, name) in [(white, "white"), (pial, "pial")] {
        let path = format!("{prefix}_{name}.surf.gii");
        std::fs::copy(src, &path).map_err(|e| format!("{path}: {e}"))?;
    }
    let dims = classes.dims();
    println!(
        "{prefix} : cadrage {}×{}×{}, {} sommets, nappe médiane {} voxels",
        dims[0],
        dims[1],
        dims[2],
        w.n_vertices(),
        skeleton.data().iter().filter(|&&b| b).count()
    );
    Ok(())
}
