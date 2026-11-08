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
    /// Courbe d'étalonnage (TSV de scripts/profile_calibration.py : week, kind, …, offset) : avec --ga, le rapport
    /// donne aussi le décalage médian corrigé du biais des fantômes au même âge
    #[arg(long, requires = "ga")]
    calibration: Option<PathBuf>,
    /// Âge gestationnel de la session (SA), pour --calibration
    #[arg(long, requires = "calibration")]
    ga: Option<f64>,
    /// Rapport JSON écrit dans ce fichier (sinon sur la sortie standard). Avec SURFOXIDE_DEBUG, champs par sommet
    /// (offset, amplitude, sharp) écrits dans <surface>_profile-<kind>.shape.gii
    #[arg(long)]
    json: Option<PathBuf>,
}

fn err(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{} : {e}", path.display())
}

/// Biais de la courbe d'étalonnage `path` pour la surface `kind` à l'âge `ga` : interpolation linéaire entre
/// semaines, bornée à la première et à la dernière.
fn calibration_bias(path: &Path, kind: &str, ga: f64) -> Result<f64, String> {
    let text = std::fs::read_to_string(path).map_err(|e| err(path, e))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| err(path, "fichier vide"))?
        .split('\t')
        .collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|h| *h == name)
            .ok_or_else(|| err(path, format!("colonne {name} absente")))
    };
    let (cw, ck, co) = (col("week")?, col("kind")?, col("offset")?);
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for l in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = l.split('\t').collect();
        if f.get(ck) == Some(&kind) {
            let parse = |i: usize| -> Result<f64, String> {
                f.get(i)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| err(path, format!("ligne illisible : {l}")))
            };
            pts.push((parse(cw)?, parse(co)?));
        }
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (first, last) = match (pts.first(), pts.last()) {
        (Some(f), Some(l)) => (*f, *l),
        _ => return Err(err(path, format!("aucune ligne pour {kind}"))),
    };
    if ga <= first.0 {
        return Ok(first.1);
    }
    if ga >= last.0 {
        return Ok(last.1);
    }
    let i = pts.iter().position(|p| p.0 >= ga).expect("borné");
    let (a, b) = (pts[i - 1], pts[i]);
    Ok(a.1 + (b.1 - a.1) * (ga - a.0) / (b.0 - a.0))
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
    let mut report = report;
    if let (Some(path), Some(ga)) = (&a.calibration, a.ga) {
        let kind = report["kind"].as_str().unwrap_or("white").to_string();
        let bias = calibration_bias(path, &kind, ga)?;
        report["calibration"] = json!({ "file": path.display().to_string(), "ga": ga, "bias": bias });
        report["calibrated_median_offset"] = json!(s.median - bias);
    }
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
