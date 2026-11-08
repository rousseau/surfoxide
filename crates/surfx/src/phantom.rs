//! `surfx phantom` : fantôme A (plan qualité, Q0.1) d'un couple blanche/piale vrai.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Args;
use surfoxide_bench::{PhantomParams, T2wParams};
use surfoxide_core::{Hemisphere, Volume};
use surfoxide_io::gifti::{read_fields, read_surface, write_fields, write_surface, SurfaceMeta};
use surfoxide_io::nifti::{write_f32, write_i16};

/// Arguments de `surfx phantom`.
#[derive(Args)]
pub struct PhantomArgs {
    /// Blanche vraie (GIfTI)
    #[arg(long)]
    white: PathBuf,
    /// Piale vraie (GIfTI)
    #[arg(long)]
    pial: PathBuf,
    /// Hémisphère (L ou R) : labels BOUNTI-19 et métadonnées GIfTI
    #[arg(long)]
    hemi: Hemisphere,
    /// Champ par sommet de la blanche (.shape.gii, premier tableau) valant exactement 0 sur la face médiale, exclue
    /// des champs vrais (pour l'atlas dHCP : son sulc)
    #[arg(long)]
    medial: Option<PathBuf>,
    /// Côté d'un voxel (mm)
    #[arg(long, default_value_t = 0.5)]
    voxel: f64,
    /// Marge autour des surfaces (mm)
    #[arg(long, default_value_t = 3.0)]
    margin: f64,
    /// Moyenne quadratique du bruit de contour (mm)
    #[arg(long, default_value_t = 0.0)]
    noise_sigma: f64,
    /// Longueur de corrélation du bruit de contour (mm)
    #[arg(long, default_value_t = 1.5)]
    noise_length: f64,
    /// Graine de tous les tirages
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Déplacement minimal de la piale le long de la normale de la blanche pour qu'un sommet soit mesuré (mm)
    #[arg(long, default_value_t = 0.1)]
    min_offset: f64,
    /// Intensité T2w du LCR
    #[arg(long, default_value_t = 1.0)]
    t2w_csf: f64,
    /// Intensité T2w de la substance blanche
    #[arg(long, default_value_t = 0.75)]
    t2w_white: f64,
    /// Intensité T2w du cortex
    #[arg(long, default_value_t = 0.35)]
    t2w_cortex: f64,
    /// Écart-type du flou gaussien de la T2w (mm ; 0 : sans flou)
    #[arg(long, default_value_t = 0.4)]
    blur: f64,
    /// Rapport signal sur bruit de la T2w, par rapport à la substance blanche (0 : sans bruit)
    #[arg(long, default_value_t = 25.0)]
    snr: f64,
    /// Amplitude du biais multiplicatif lisse de la T2w (0 : sans biais)
    #[arg(long, default_value_t = 0.1)]
    bias: f64,
    /// Préfixe des fichiers écrits : <préfixe>_dseg.nii.gz, <préfixe>_T2w.nii.gz,
    /// <préfixe>_label-{WM,GM,CSF}_probseg.nii.gz, <préfixe>_hemi-X_desc-truth_{white,pial}.surf.gii,
    /// <préfixe>_hemi-X_desc-truth_{thickness,curv,dpfstar,mask}.shape.gii, <préfixe>_phantom.json
    #[arg(long, short)]
    out: PathBuf,
}

fn err(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{} : {e}", path.display())
}

pub fn run(a: &PhantomArgs) -> Result<(), String> {
    let t0 = Instant::now();
    let (white, _) = read_surface(&a.white).map_err(|e| err(&a.white, e))?;
    let (pial, _) = read_surface(&a.pial).map_err(|e| err(&a.pial, e))?;
    let medial = match &a.medial {
        Some(path) => {
            let fields = read_fields(path).map_err(|e| err(path, e))?;
            let f = fields.first().ok_or_else(|| err(path, "aucun champ"))?;
            if f.values.len() != white.n_vertices() {
                return Err(err(
                    path,
                    format!("{} valeurs pour {} sommets", f.values.len(), white.n_vertices()),
                ));
            }
            f.values.iter().map(|&x| x == 0.0).collect()
        }
        None => vec![false; white.n_vertices()],
    };
    let params = PhantomParams {
        voxel: a.voxel,
        margin: a.margin,
        noise_sigma: a.noise_sigma,
        noise_length: a.noise_length,
        seed: a.seed,
        min_offset: a.min_offset,
        t2w: T2wParams {
            csf: a.t2w_csf,
            white: a.t2w_white,
            cortex: a.t2w_cortex,
            blur: a.blur,
            snr: a.snr,
            bias: a.bias,
        },
    };
    let ph = surfoxide_bench::phantom(&white, &pial, &medial, a.hemi, &params);
    let grid = &ph.grid;
    let affine = grid.affine();
    let prefix = a.out.display().to_string();
    if let Some(dir) = a.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| err(dir, e))?;
    }
    let mut written = Vec::new();
    let path = format!("{prefix}_dseg.nii.gz");
    write_i16(&path, &Volume::from_vec(grid.dims, ph.labels.clone(), affine), None)
        .map_err(|e| err(path.as_ref(), e))?;
    written.push(path);
    let path = format!("{prefix}_T2w.nii.gz");
    write_f32(&path, &Volume::from_vec(grid.dims, ph.t2w.clone(), affine), None).map_err(|e| err(path.as_ref(), e))?;
    written.push(path);
    for (name, f) in ["WM", "GM", "CSF"].iter().zip(&ph.fractions) {
        let path = format!("{prefix}_label-{name}_probseg.nii.gz");
        write_f32(&path, &Volume::from_vec(grid.dims, f.clone(), affine), None).map_err(|e| err(path.as_ref(), e))?;
        written.push(path);
    }
    let hemi = format!("{prefix}_hemi-{}_desc-truth", a.hemi.bids());
    for (name, mesh, secondary) in [("white", &white, "GrayWhite"), ("pial", &pial, "Pial")] {
        let mut meta = SurfaceMeta::hemisphere(a.hemi.is_left(), secondary);
        meta.source_affine = Some(affine);
        let path = format!("{hemi}_{name}.surf.gii");
        write_surface(&path, mesh, &meta).map_err(|e| err(path.as_ref(), e))?;
        written.push(path);
    }
    let meta = SurfaceMeta::hemisphere(a.hemi.is_left(), "GrayWhite");
    for f in &ph.truth.fields {
        let path = format!("{hemi}_{}.shape.gii", f.name);
        write_fields(&path, std::slice::from_ref(f), &meta).map_err(|e| err(path.as_ref(), e))?;
        written.push(path);
    }
    let v3 = grid.voxel.powi(3);
    let sum = |f: &[f32]| f.iter().map(|&x| x as f64).sum::<f64>() * v3;
    let report = serde_json::json!({
        "generator": format!("surfoxide {}", env!("CARGO_PKG_VERSION")),
        "white": a.white.display().to_string(),
        "pial": a.pial.display().to_string(),
        "medial": a.medial.as_ref().map(|p| p.display().to_string()),
        "hemi": a.hemi.bids(),
        "params": {
            "voxel": a.voxel, "margin": a.margin, "noise_sigma": a.noise_sigma, "noise_length": a.noise_length,
            "seed": a.seed, "min_offset": a.min_offset, "t2w_csf": a.t2w_csf, "t2w_white": a.t2w_white,
            "t2w_cortex": a.t2w_cortex, "blur": a.blur, "snr": a.snr, "bias": a.bias,
        },
        "grid": { "dims": grid.dims, "origin": grid.origin, "voxel": grid.voxel },
        "volumes_mm3": {
            "white_fractions": sum(&ph.fractions[0]),
            "cortex_fractions": sum(&ph.fractions[1]),
            "white_enclosed": white.signed_volume(),
            "pial_enclosed": pial.signed_volume(),
            "white_noisy_enclosed": ph.white_noisy.signed_volume(),
            "pial_noisy_enclosed": ph.pial_noisy.signed_volume(),
        },
        "truth": {
            "vertices": white.n_vertices(),
            "shared_triangulation": ph.truth.shared,
            "medial_vertices": ph.truth.medial,
            "thin_vertices": ph.truth.thin,
            "measured_fraction": ph.truth.fields[3].values.iter().sum::<f64>() / white.n_vertices() as f64,
        },
    });
    let path = format!("{prefix}_phantom.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report).expect("JSON") + "\n")
        .map_err(|e| err(path.as_ref(), e))?;
    written.push(path);
    println!(
        "fantôme {} : grille {:?} à {} mm, blanche {:.0} mm³ (enfermée {:.0}), {} sommets exclus (médiale {}, piale \
         à moins de {} mm {}) ; {:.1} s",
        a.hemi,
        grid.dims,
        grid.voxel,
        sum(&ph.fractions[0]),
        white.signed_volume(),
        ph.truth.medial + ph.truth.thin,
        ph.truth.medial,
        a.min_offset,
        ph.truth.thin,
        t0.elapsed().as_secs_f64()
    );
    for p in written {
        println!("écrit : {p}");
    }
    Ok(())
}
