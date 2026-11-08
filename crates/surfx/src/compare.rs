//! `surfx compare` : couple blanche/piale estimé contre sa vérité (plan qualité, Q0.3).

use std::path::{Path, PathBuf};

use clap::Args;
use serde_json::{json, Value};
use surfoxide_bench::compare::{self, Distances, FieldError, Patches, Transport};
use surfoxide_core::TriMesh;
use surfoxide_io::gifti::read_fields;

/// Arguments de `surfx compare`.
#[derive(Args)]
pub struct CompareArgs {
    /// Blanche estimée
    #[arg(long)]
    white: PathBuf,
    /// Piale estimée
    #[arg(long)]
    pial: PathBuf,
    /// Blanche vraie
    #[arg(long)]
    truth_white: PathBuf,
    /// Piale vraie
    #[arg(long)]
    truth_pial: PathBuf,
    /// Épaisseur estimée, par sommet de la blanche estimée (.shape.gii)
    #[arg(long)]
    thickness: Option<PathBuf>,
    /// Champs vrais par sommet de la blanche vraie (.shape.gii), reconnus par leur nom : thickness, curv, dpfstar,
    /// mask (1 : sommet mesuré). Courbure et D* vraies sont recalculées sur la blanche vraie si absentes.
    #[arg(long, num_args = 1..)]
    truth_fields: Vec<PathBuf>,
    /// Aire séparant petites taches et bassins de D* (mm²)
    #[arg(long, default_value_t = 5.0)]
    min_area: f64,
    /// Rapport JSON écrit dans ce fichier (sinon sur la sortie standard)
    #[arg(long)]
    json: Option<PathBuf>,
}

fn err(path: &Path, e: impl std::fmt::Display) -> String {
    format!("{} : {e}", path.display())
}

fn mesh(path: &Path) -> Result<TriMesh, String> {
    surfoxide_io::read_mesh(path).map_err(|e| err(path, e))
}

fn distances(d: Distances) -> Value {
    json!({ "mean": d.mean, "mean_est_to_truth": d.mean_ab, "mean_truth_to_est": d.mean_ba, "hd95": d.hd95,
            "max": d.max })
}

fn field(e: FieldError) -> Value {
    json!({ "bias": e.bias, "median_abs": e.median_abs, "pearson": e.pearson, "spearman": e.spearman, "n": e.n })
}

fn patches(p: Patches) -> Value {
    json!({ "basins": p.basins, "small": p.small, "small_area": p.small_area })
}

fn relative(est: f64, truth: f64) -> Value {
    json!({ "estimated": est, "truth": truth, "relative_error": (est - truth) / truth })
}

pub fn run(a: &CompareArgs) -> Result<(), String> {
    let (white, pial) = (mesh(&a.white)?, mesh(&a.pial)?);
    let (twhite, tpial) = (mesh(&a.truth_white)?, mesh(&a.truth_pial)?);
    let mut truth = std::collections::HashMap::new();
    for path in &a.truth_fields {
        for f in read_fields(path).map_err(|e| err(path, e))? {
            if f.values.len() != twhite.n_vertices() {
                return Err(err(
                    path,
                    format!("{} valeurs pour {} sommets", f.values.len(), twhite.n_vertices()),
                ));
            }
            truth.insert(f.name.clone(), f.values);
        }
    }
    let mask: Option<Vec<bool>> = truth.get("mask").map(|m| m.iter().map(|&x| x > 0.5).collect());
    let alpha = surfoxide_metrics::depth::ALPHA_REF;
    let length = surfoxide_metrics::Length::default();
    let est_curv = surfoxide_metrics::curvature(&white).mean;
    let est_dpf = surfoxide_metrics::dpf_star(&white, &est_curv, alpha, length).depth;
    let true_curv = match truth.get("curv") {
        Some(c) => c.clone(),
        None => surfoxide_metrics::curvature(&twhite).mean,
    };
    let true_dpf = match truth.get("dpfstar") {
        Some(d) => d.clone(),
        None => surfoxide_metrics::dpf_star(&twhite, &true_curv, alpha, length).depth,
    };
    let transport = Transport::new(&white, twhite.vertices());
    let m = mask.as_deref();
    let mut report = json!({
        "white": distances(compare::distances(&white, &twhite)),
        "pial": distances(compare::distances(&pial, &tpial)),
        "curv": field(compare::field_error(&transport.apply(&white, &est_curv), &true_curv, m)),
        "dpfstar": field(compare::field_error(&transport.apply(&white, &est_dpf), &true_dpf, m)),
        "area": {
            "white": relative(white.area(), twhite.area()),
            "pial": relative(pial.area(), tpial.area()),
        },
        "gi": {
            "white": relative(compare::gyrification(&white), compare::gyrification(&twhite)),
            "pial": relative(compare::gyrification(&pial), compare::gyrification(&tpial)),
        },
        "patches": {
            "min_area": a.min_area,
            "estimated": patches(compare::patches(&white, &est_dpf, a.min_area)),
            "truth": patches(compare::patches(&twhite, &true_dpf, a.min_area)),
        },
        "mask_vertices": mask.as_ref().map(|m| m.iter().filter(|&&x| x).count()),
        "truth_vertices": twhite.n_vertices(),
    });
    if let (Some(path), Some(t)) = (&a.thickness, truth.get("thickness")) {
        let fields = read_fields(path).map_err(|e| err(path, e))?;
        let est = &fields.first().ok_or_else(|| err(path, "aucun champ"))?.values;
        if est.len() != white.n_vertices() {
            return Err(err(
                path,
                format!("{} valeurs pour {} sommets", est.len(), white.n_vertices()),
            ));
        }
        report["thickness"] = field(compare::field_error(&transport.apply(&white, est), t, m));
    }
    let text = serde_json::to_string_pretty(&report).expect("JSON") + "\n";
    match &a.json {
        Some(path) => std::fs::write(path, text).map_err(|e| err(path, e))?,
        None => print!("{text}"),
    }
    Ok(())
}
