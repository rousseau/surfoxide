//! Surface blanche d'un hémisphère : segmentation → masque → distance signée → correction topologique →
//! marching cubes → lissage de Taubin.

use std::path::Path;
use std::time::Instant;

use surfoxide_core::distance::signed_distance;
use surfoxide_core::filter::{crop_to, gaussian_blur};
use surfoxide_core::{topology_report, Hemisphere, TriMesh, Volume};
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
}

/// Résultat de l'extraction : surface et bilan.
pub struct WhiteSurface {
    pub mesh: TriMesh,
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
    surfoxide_extract::taubin(
        &mut mesh,
        params.taubin,
        params.lambda,
        -1.06 * params.lambda,
    );
    Ok(WhiteSurface {
        mesh,
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
        let r = topology_report(&w.mesh);
        println!(
            "hémisphère {hemi} : {:?} (+{} −{} voxels) ; V={} F={} χ={} genre={} volume={:.1} mm³ aire={:.1} mm² ; {:.1} s",
            w.strategy,
            w.added,
            w.removed,
            r.n_vertices,
            r.n_faces,
            r.euler_characteristic,
            r.genus().map_or("—".into(), |g| g.to_string()),
            w.mesh.signed_volume(),
            w.mesh.area(),
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
