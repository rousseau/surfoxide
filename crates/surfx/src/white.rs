//! Surface blanche d'un hémisphère : segmentation → masque → distance signée → correction topologique →
//! marching cubes → lissage de Taubin (pas à pas, sans créer d'intersection) → remaillage isotrope →
//! réparation locale des intersections restantes → contrôle du contrat.

use std::path::Path;
use std::time::Instant;

use surfoxide_core::distance::signed_distance;
use surfoxide_core::filter::{crop_to, gaussian_blur};
use surfoxide_core::{Hemisphere, TriMesh, Volume};
use surfoxide_io::gifti::SurfaceMeta;
use surfoxide_mask::LabelTable;
use surfoxide_topofix::{correct_topology, Strategy};

/// Paramètres de l'extraction.
#[derive(Debug, Clone, clap::Args)]
pub struct WhiteParams {
    /// Écart-type (mm) du flou de la distance signée avant extraction
    #[arg(long, default_value_t = 0.25)]
    pub sigma: f64,
    /// Nombre de paires de pas de Taubin
    #[arg(long, default_value_t = 10)]
    pub taubin: usize,
    /// Pas λ de Taubin (le pas μ vaut −1,06 λ)
    #[arg(long, default_value_t = 0.5)]
    pub lambda: f64,
    /// Longueur d'arête visée par le remaillage isotrope (mm) ; 0 : pas de remaillage
    #[arg(long, default_value_t = 0.5)]
    pub edge: f64,
}

/// Résultat de l'extraction : surface et bilan.
pub struct WhiteSurface {
    pub mesh: TriMesh,
    /// Sommets maintenus en place pendant le lissage (cumul sur les pas).
    pub held: usize,
    /// Réparation après remaillage.
    pub repair: surfoxide_repair::RepairReport,
    pub added: usize,
    pub removed: usize,
    pub strategy: Strategy,
}

/// Extrait la surface blanche de `hemi` à partir des labels.
pub fn white_surface(
    labels: &Volume<i32>,
    table: &LabelTable,
    hemi: Hemisphere,
    params: &WhiteParams,
) -> Result<WhiteSurface, String> {
    let mask = surfoxide_mask::white_mask(labels, table, hemi);
    let (mask, _) = crop_to(&mask, |&b| b, 4, false)
        .ok_or_else(|| format!("hémisphère {hemi} : masque vide"))?;
    let phi = gaussian_blur(&signed_distance(&mask), params.sigma);
    let min_spacing = mask.spacing().into_iter().fold(f64::INFINITY, f64::min);
    let c =
        correct_topology(&phi, Strategy::Best, 0.05 * min_spacing).map_err(|e| e.to_string())?;
    let mut mesh = surfoxide_extract::marching_cubes(&c.levelset, 0.02);
    // Lissage pas à pas : un pas qui ferait se couper la surface est annulé localement.
    let mut held = 0;
    for _ in 0..params.taubin {
        let mut trial = mesh.clone();
        surfoxide_extract::taubin(&mut trial, 1, params.lambda, -1.06 * params.lambda);
        held += surfoxide_repair::safe_update(&mut mesh, trial.vertices());
    }
    if params.edge > 0.0 {
        mesh = surfoxide_remesh::remesh(&mesh, &surfoxide_remesh::RemeshParams::new(params.edge))
            .map_err(|_| "remaillage : surface non variété".to_string())?;
    }
    let repair = surfoxide_repair::repair_intersections(&mut mesh, 30);
    Ok(WhiteSurface {
        mesh,
        held,
        repair,
        added: c.added,
        removed: c.removed,
        strategy: c.strategy,
    })
}

pub fn run(
    seg: &Path,
    table: &LabelTable,
    hemis: &[Hemisphere],
    params: &WhiteParams,
    out_prefix: &Path,
) -> Result<(), String> {
    let (labels, _) =
        surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    for &hemi in hemis {
        let t = Instant::now();
        let w = white_surface(&labels, table, hemi, params)?;
        let report = surfoxide_check::check_surface(&w.mesh);
        println!(
            "hémisphère {hemi} : {:?} (+{} −{} voxels) ; lissage : {} sommets retenus ; réparation : {} → {} intersections ; V={} F={} genre={} volume={:.1} mm³ aire={:.1} mm² ; contrat {} ; {:.1} s",
            w.strategy,
            w.added,
            w.removed,
            w.held,
            w.repair.initial,
            w.repair.remaining,
            report.n_vertices,
            report.n_faces,
            report.genus.map_or("—".into(), |g| g.to_string()),
            report.signed_volume,
            report.area,
            if report.valid { "respecté" } else { "NON RESPECTÉ" },
            t.elapsed().as_secs_f64()
        );
        let mut meta = SurfaceMeta::hemisphere(hemi.is_left(), "GrayWhite");
        meta.source_affine = Some(*labels.affine());
        if let Some(name) = seg.file_name().and_then(|n| n.to_str()) {
            meta.other.insert("SourceImage".into(), name.into());
        }
        meta.other.insert(
            "Generator".into(),
            format!("surfoxide {}", env!("CARGO_PKG_VERSION")),
        );
        let path = format!(
            "{}_hemi-{}_white.surf.gii",
            out_prefix.display(),
            hemi.bids()
        );
        surfoxide_io::gifti::write_surface(&path, &w.mesh, &meta)
            .map_err(|e| format!("{path}: {e}"))?;
        println!("écrit : {path}");
    }
    Ok(())
}
