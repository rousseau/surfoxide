//! Surfaces d'un hémisphère.
//!
//! Blanche : masque → distance signée → correction topologique → marching cubes → lissage de Taubin (pas à
//! pas, sans créer d'intersection) → remaillage isotrope → réparation locale des intersections restantes.
//!
//! Piale : champ de Laplace sur le cortex (sillons enfouis ouverts par leur nappe médiane) → lignes de
//! courant depuis la surface blanche → progression de la piale sans intersection.

use std::path::Path;
use std::time::Instant;

use surfoxide_core::distance::signed_distance;
use surfoxide_core::filter::{crop_to, extract, gaussian_blur};
use surfoxide_core::{Hemisphere, TriMesh, Volume};
use surfoxide_io::gifti::SurfaceMeta;
use surfoxide_mask::LabelTable;
use surfoxide_pial::{pial_from_white, solve_laplace, sulcal_skeleton, Field, PialParams};
use surfoxide_topofix::{correct_topology, Strategy};

/// Paramètres de la surface blanche.
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

/// Paramètres de la piale.
#[derive(Debug, Clone, clap::Args)]
pub struct PialArgs {
    /// Décalage minimal de la piale par rapport à la surface blanche (mm)
    #[arg(long, default_value_t = 0.1)]
    pub min_offset: f64,
    /// Niveau du champ de Laplace où s'arrêtent les lignes de courant
    #[arg(long, default_value_t = 0.97)]
    pub stop: f64,
    /// Ne pas ouvrir les sillons enfouis par leur nappe médiane
    #[arg(long)]
    pub no_skeleton: bool,
}

/// Grilles recadrées d'un hémisphère : masque blanc fermé et cortex.
pub struct HemisphereGrids {
    pub white: Volume<bool>,
    pub cortex: Volume<bool>,
}

pub fn hemisphere_grids(
    labels: &Volume<i32>,
    table: &LabelTable,
    hemi: Hemisphere,
) -> Result<HemisphereGrids, String> {
    let (white, cortex) = surfoxide_mask::hemisphere_masks(labels, table, hemi);
    let both = white.with_data(
        white
            .data()
            .iter()
            .zip(cortex.data())
            .map(|(&a, &b)| a || b)
            .collect(),
    );
    let (_, origin) = crop_to(&both, |&b| b, 4, false)
        .ok_or_else(|| format!("hémisphère {hemi} : masque vide"))?;
    let (bbox, _) = crop_to(&both, |&b| b, 4, false).unwrap();
    let dims = bbox.dims();
    Ok(HemisphereGrids {
        white: extract(&white, origin, dims, false),
        cortex: extract(&cortex, origin, dims, false),
    })
}

/// Surface blanche et bilan de sa construction.
pub struct WhiteSurface {
    pub mesh: TriMesh,
    /// Objet blanc après correction topologique (même grille que le masque).
    pub object: Volume<bool>,
    pub added: usize,
    pub removed: usize,
    pub strategy: Strategy,
    /// Sommets maintenus en place pendant le lissage (cumul sur les pas).
    pub held: usize,
    /// Réparation après remaillage.
    pub repair: surfoxide_repair::RepairReport,
}

/// Surface blanche de genre 0 du masque `mask` (recadré avec une marge).
pub fn white_surface(mask: &Volume<bool>, params: &WhiteParams) -> Result<WhiteSurface, String> {
    let phi = gaussian_blur(&signed_distance(mask), params.sigma);
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
    // Plis aigus : gênent le décalage de la piale et trahissent un artefact local.
    surfoxide_repair::relax_sharp_vertices(&mut mesh, 0.1, 20);
    Ok(WhiteSurface {
        mesh,
        object: c.object,
        added: c.added,
        removed: c.removed,
        strategy: c.strategy,
        held,
        repair,
    })
}

/// Piale et champ d'épaisseur, à partir de la surface blanche.
pub fn pial_surface(
    grids: &HemisphereGrids,
    white: &WhiteSurface,
    args: &PialArgs,
) -> Result<(surfoxide_pial::Pial, usize, usize), String> {
    // L'objet blanc corrigé prime sur le cortex.
    let cortex = grids.cortex.with_data(
        grids
            .cortex
            .data()
            .iter()
            .zip(white.object.data())
            .map(|(&c, &w)| c && !w)
            .collect(),
    );
    let skeleton = if args.no_skeleton {
        cortex.map(|_| false)
    } else {
        sulcal_skeleton(&white.object, &cortex, 1.5)
    };
    let n_skeleton = skeleton.data().iter().filter(|&&b| b).count();
    let domain = cortex.with_data(
        cortex
            .data()
            .iter()
            .zip(skeleton.data())
            .map(|(&c, &s)| c && !s)
            .collect(),
    );
    let (u, iters) = solve_laplace(&white.object, &domain, 1e-6, 5000);
    let field = Field::new(u);
    let params = PialParams {
        min_offset: args.min_offset,
        stop: args.stop,
        ..PialParams::default()
    };
    let pial = pial_from_white(&white.mesh, &field, &params).map_err(|e| e.to_string())?;
    Ok((pial, n_skeleton, iters))
}

fn meta(hemi: Hemisphere, secondary: &str, labels: &Volume<i32>, seg: &Path) -> SurfaceMeta {
    let mut meta = SurfaceMeta::hemisphere(hemi.is_left(), secondary);
    meta.source_affine = Some(*labels.affine());
    if let Some(name) = seg.file_name().and_then(|n| n.to_str()) {
        meta.other.insert("SourceImage".into(), name.into());
    }
    meta.other.insert(
        "Generator".into(),
        format!("surfoxide {}", env!("CARGO_PKG_VERSION")),
    );
    meta
}

fn write_json(path: &str, value: &serde_json::Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("{path}: {e}"))
}

/// `with_pial` : construit aussi la piale et l'épaisseur.
pub fn run(
    seg: &Path,
    table: &LabelTable,
    hemis: &[Hemisphere],
    params: &WhiteParams,
    pial_args: Option<&PialArgs>,
    min_thickness: f64,
    out_prefix: &Path,
) -> Result<bool, String> {
    let (labels, _) =
        surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    let mut all_valid = true;
    for &hemi in hemis {
        let t = Instant::now();
        let grids = hemisphere_grids(&labels, table, hemi)?;
        let w = white_surface(&grids.white, params)?;
        let white_report = surfoxide_check::check_surface(&w.mesh);
        println!(
            "hémisphère {hemi}, blanche : {:?} (+{} −{} voxels) ; lissage : {} sommets retenus ; réparation : {} → {} intersections ; V={} genre={} volume={:.1} mm³ aire={:.1} mm² ; contrat {} ; {:.1} s",
            w.strategy,
            w.added,
            w.removed,
            w.held,
            w.repair.initial,
            w.repair.remaining,
            white_report.n_vertices,
            white_report.genus.map_or("—".into(), |g| g.to_string()),
            white_report.signed_volume,
            white_report.area,
            if white_report.valid { "respecté" } else { "NON RESPECTÉ" },
            t.elapsed().as_secs_f64()
        );
        let prefix = format!("{}_hemi-{}", out_prefix.display(), hemi.bids());
        let path = format!("{prefix}_white.surf.gii");
        surfoxide_io::gifti::write_surface(&path, &w.mesh, &meta(hemi, "GrayWhite", &labels, seg))
            .map_err(|e| format!("{path}: {e}"))?;
        let mut report = serde_json::json!({ "white": white_report });
        let mut valid = white_report.valid;
        if let Some(args) = pial_args {
            let t = Instant::now();
            let (pial, n_skeleton, iters) = pial_surface(&grids, &w, args)?;
            let pial_report = surfoxide_check::check_surface(&pial.mesh);
            let pair = surfoxide_check::check_pair(&w.mesh, &pial.mesh, min_thickness);
            let th = &pial.thickness.values;
            let mean_th = th.iter().sum::<f64>() / th.len() as f64;
            println!(
                "hémisphère {hemi}, piale : nappe des sillons enfouis {n_skeleton} voxels ; Laplace {iters} itérations ; {} sommets retenus ; épaisseur moyenne {mean_th:.2} mm ; distance minimale {:.3} mm ; intersections WM/piale {} ; contrat {} ; {:.1} s",
                pial.held,
                pair.min_distance,
                pair.intersections,
                if pial_report.valid && pair.valid { "respecté" } else { "NON RESPECTÉ" },
                t.elapsed().as_secs_f64()
            );
            valid &= pial_report.valid && pair.valid;
            let path = format!("{prefix}_pial.surf.gii");
            surfoxide_io::gifti::write_surface(
                &path,
                &pial.mesh,
                &meta(hemi, "Pial", &labels, seg),
            )
            .map_err(|e| format!("{path}: {e}"))?;
            let path = format!("{prefix}_thickness.shape.gii");
            surfoxide_io::gifti::write_fields(
                &path,
                std::slice::from_ref(&pial.thickness),
                &meta(hemi, "GrayWhite", &labels, seg),
            )
            .map_err(|e| format!("{path}: {e}"))?;
            report["pial"] = serde_json::to_value(pial_report).map_err(|e| e.to_string())?;
            report["pair"] = serde_json::to_value(pair).map_err(|e| e.to_string())?;
            report["pial_held_vertices"] = pial.held.into();
            report["mean_thickness"] = mean_th.into();
        }
        report["valid"] = valid.into();
        write_json(&format!("{prefix}_report.json"), &report)?;
        println!("écrit : {prefix}_*");
        all_valid &= valid;
    }
    Ok(all_valid)
}
