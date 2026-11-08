//! `surfx profile` : position du bord T2w le long des normales d'une surface (plan qualité, Q0.4).

use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use serde_json::json;
use surfoxide_bench::profile::{self, Kind, ProfileParams};
use surfoxide_core::VertexField;
use surfoxide_io::gifti::{read_fields, write_fields, SurfaceMeta};

#[derive(Clone, Copy, ValueEnum)]
pub enum KindArg {
    /// Surface blanche : bord sombre vers l'extérieur (substance blanche plus claire que le cortex)
    White,
    /// Surface piale : bord clair vers l'extérieur (LCR plus clair que le cortex)
    Pial,
}

/// Arguments de `surfx profile`.
#[derive(Args)]
pub struct ProfileArgs {
    /// T2w (NIfTI), dans le même repère monde que la surface
    #[arg(long)]
    t2w: PathBuf,
    /// Surface profilée
    #[arg(long)]
    surface: PathBuf,
    /// Nature de la surface
    #[arg(long, value_enum)]
    kind: KindArg,
    /// Champ par sommet (.shape.gii, premier tableau) : seuls les sommets où il vaut plus de 0,5 sont profilés
    /// (masque cortical de la chaîne, ou masque vrai d'un fantôme)
    #[arg(long)]
    mask: Option<PathBuf>,
    /// Demi-longueur du profil (mm)
    #[arg(long, default_value_t = 1.5)]
    half: f64,
    /// Pas d'échantillonnage (mm)
    #[arg(long, default_value_t = 0.05)]
    step: f64,
    /// Écart-type du lissage de la dérivée (mm)
    #[arg(long, default_value_t = 0.1)]
    sigma: f64,
    /// Rapport JSON écrit dans ce fichier (sinon sur la sortie standard). Avec SURFOXIDE_DEBUG, champs par sommet
    /// (offset, amplitude, sharp) écrits dans <surface>_profile-<kind>.shape.gii
    #[arg(long)]
    json: Option<PathBuf>,
}

fn err(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{} : {e}", path.display())
}

pub fn run(a: &ProfileArgs) -> Result<(), String> {
    let (image, _) = surfoxide_io::nifti::read_scalar(&a.t2w).map_err(|e| err(&a.t2w, e))?;
    let image = image.map(|&x| x as f64);
    let mesh = surfoxide_io::read_mesh(&a.surface).map_err(|e| err(&a.surface, e))?;
    let mask: Option<Vec<bool>> = match &a.mask {
        Some(path) => {
            let f = read_fields(path).map_err(|e| err(path, e))?;
            let f = f.first().ok_or_else(|| err(path, "aucun champ"))?;
            if f.values.len() != mesh.n_vertices() {
                return Err(err(
                    path,
                    format!("{} valeurs pour {} sommets", f.values.len(), mesh.n_vertices()),
                ));
            }
            Some(f.values.iter().map(|&x| x > 0.5).collect())
        }
        None => None,
    };
    let kind = match a.kind {
        KindArg::White => Kind::White,
        KindArg::Pial => Kind::Pial,
    };
    let p = ProfileParams {
        half: a.half,
        step: a.step,
        sigma: a.sigma,
    };
    let edges = profile::profile(&image, &mesh, kind, mask.as_deref(), &p);
    let s = profile::summary(&edges);
    let report = json!({
        "surface": a.surface.display().to_string(),
        "t2w": a.t2w.display().to_string(),
        "kind": match kind { Kind::White => "white", Kind::Pial => "pial" },
        "params": { "half": a.half, "step": a.step, "sigma": a.sigma },
        "n": s.n,
        "median_offset": s.median,
        "iqr_offset": s.iqr,
        "not_sharp_fraction": s.not_sharp,
    });
    let text = serde_json::to_string_pretty(&report).expect("JSON") + "\n";
    match &a.json {
        Some(path) => std::fs::write(path, text).map_err(|e| err(path, e))?,
        None => print!("{text}"),
    }
    if std::env::var_os("SURFOXIDE_DEBUG").is_some() {
        let stem = a.surface.display().to_string();
        let stem = stem.strip_suffix(".surf.gii").unwrap_or(&stem);
        let path = format!(
            "{stem}_profile-{}.shape.gii",
            report["kind"].as_str().unwrap_or("surface")
        );
        let get = |f: &dyn Fn(&profile::Edge) -> f64| edges.iter().map(|e| e.as_ref().map_or(f64::NAN, f)).collect();
        let fields = [
            VertexField::new("offset", get(&|e| e.offset)),
            VertexField::new("amplitude", get(&|e| e.amplitude)),
            VertexField::new("sharp", get(&|e| f64::from(u8::from(e.sharp())))),
        ];
        write_fields(&path, &fields, &SurfaceMeta::default()).map_err(|e| err(path.as_ref(), e))?;
        eprintln!("écrit : {path}");
    }
    Ok(())
}
