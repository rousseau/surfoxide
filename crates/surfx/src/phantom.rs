//! `surfx phantom` : fantôme A (plan qualité, Q0.1) d'un couple blanche/piale vrai.

use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::{Args, ValueEnum};
use surfoxide_bench::analytic::{self, AnalyticParams, Config};
use surfoxide_bench::{PhantomParams, T2wParams};
use surfoxide_core::{Hemisphere, Volume};
use surfoxide_io::gifti::{read_fields, read_surface, write_fields, write_surface, SurfaceMeta};
use surfoxide_io::nifti::{write_f32, write_i16};

/// Configuration figée d'un fantôme B (`--analytic`).
#[derive(Clone, Copy, ValueEnum)]
pub enum AnalyticArg {
    /// Sillons larges, épaisseur constante
    Wide,
    /// Sillons étroits, épaisseur constante
    Narrow,
    /// Sillons enfouis (berges jointives), épaisseur constante
    Buried,
    /// Sillons larges, épaisseur variable
    Variable,
}

/// Arguments de `surfx phantom`.
#[derive(Args)]
pub struct PhantomArgs {
    /// Blanche vraie (GIfTI) ; fantôme A
    #[arg(long, required_unless_present = "analytic", requires = "pial")]
    white: Option<PathBuf>,
    /// Piale vraie (GIfTI) ; fantôme A
    #[arg(long, requires = "white")]
    pial: Option<PathBuf>,
    /// Fantôme B analytique (ellipsoïde creusé de sillons) dans cette configuration, à la place de --white et --pial
    #[arg(long, value_enum, conflicts_with_all = ["white", "pial", "medial"])]
    analytic: Option<AnalyticArg>,
    /// Demi-axes de l'ellipsoïde (mm, fantôme B)
    #[arg(long, value_delimiter = ',', num_args = 3, default_values_t = analytic::AXES_28)]
    axes: Vec<f64>,
    /// Profondeur des sillons (mm, fantôme B ; défaut de la configuration sinon)
    #[arg(long)]
    sulcus_depth: Option<f64>,
    /// Écart entre berges des sillons dans leur partie resserrée (mm, fantôme B ; défaut de la configuration sinon)
    #[arg(long)]
    sulcus_width: Option<f64>,
    /// Rotation des sillons autour de leur centre (degrés, fantôme B)
    #[arg(long, default_value_t = 0.0)]
    sulcus_angle: f64,
    /// Niveau de subdivision de l'icosphère (fantôme B)
    #[arg(long, default_value_t = 7)]
    level: u32,
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
    /// <préfixe>_hemi-X_desc-truth_{thickness,curv,dpfstar,mask}.shape.gii (fantôme B : aussi target, épaisseur
    /// voulue, et contact, 1 où le décalage est limité), <préfixe>_phantom.json
    #[arg(long, short)]
    out: PathBuf,
}

fn err(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{} : {e}", path.display())
}

pub fn run(a: &PhantomArgs) -> Result<(), String> {
    let t0 = Instant::now();
    let mut extra = serde_json::Value::Null;
    let mut extra_fields = Vec::new();
    let (white, pial) = match (&a.analytic, &a.white, &a.pial) {
        (Some(c), _, _) => {
            let config = match c {
                AnalyticArg::Wide => Config::Wide,
                AnalyticArg::Narrow => Config::Narrow,
                AnalyticArg::Buried => Config::Buried,
                AnalyticArg::Variable => Config::Variable,
            };
            let axes = [a.axes[0], a.axes[1], a.axes[2]];
            let mut p = AnalyticParams::config(config, axes, a.sulcus_depth, a.sulcus_width, a.sulcus_angle);
            p.level = a.level;
            let white = analytic::white_surface(&p);
            let pial = analytic::pial_surface(&white, &p);
            let contact = pial.contact.iter().filter(|&&c| c).count();
            // Écart entre la distance du sommet blanc au plus proche point de la piale et l'épaisseur voulue, hors
            // contact (pour information : l'épaisseur vraie est celle des sommets correspondants).
            let nearest = surfoxide_check::vertex_distances(&white, &pial.mesh);
            let mut gap: Vec<f64> = (0..nearest.len())
                .filter(|&i| !pial.contact[i])
                .map(|i| nearest[i] - pial.target[i])
                .collect();
            gap.sort_by(f64::total_cmp);
            let q = |f: f64| gap[((gap.len() - 1) as f64 * f).round() as usize];
            extra = serde_json::json!({
                "config": format!("{config:?}"),
                "axes": axes,
                "level": p.level,
                "profile_half_width": p.profile_half_width,
                "epsilon": p.epsilon,
                "thickness": format!("{:?}", p.thickness),
                "sulci": p.sulci.iter().map(|s| serde_json::json!({
                    "center": s.center, "direction": s.direction, "half_length": s.half_length,
                    "depth": s.depth, "width": s.width,
                })).collect::<Vec<_>>(),
                "contact_vertices": contact,
                "nearest_minus_target_off_contact": {
                    "min": q(0.0), "q01": q(0.01), "median": q(0.5), "max": q(1.0),
                },
            });
            extra_fields.push(surfoxide_core::VertexField::new("target", pial.target.clone()));
            extra_fields.push(surfoxide_core::VertexField::new(
                "contact",
                pial.contact.iter().map(|&c| f64::from(u8::from(c))).collect(),
            ));
            (white, pial.mesh)
        }
        (None, Some(w), Some(p)) => {
            let (white, _) = read_surface(w).map_err(|e| err(w, e))?;
            let (pial, _) = read_surface(p).map_err(|e| err(p, e))?;
            (white, pial)
        }
        _ => return Err("--white et --pial, ou --analytic, sont exigés".into()),
    };
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
    for f in ph.truth.fields.iter().chain(&extra_fields) {
        let path = format!("{hemi}_{}.shape.gii", f.name);
        write_fields(&path, std::slice::from_ref(f), &meta).map_err(|e| err(path.as_ref(), e))?;
        written.push(path);
    }
    let v3 = grid.voxel.powi(3);
    let sum = |f: &[f32]| f.iter().map(|&x| x as f64).sum::<f64>() * v3;
    let report = serde_json::json!({
        "generator": format!("surfoxide {}", env!("CARGO_PKG_VERSION")),
        "white": a.white.as_ref().map(|p| p.display().to_string()),
        "pial": a.pial.as_ref().map(|p| p.display().to_string()),
        "analytic": extra,
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
