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

pub fn hemisphere_grids(labels: &Volume<i32>, table: &LabelTable, hemi: Hemisphere) -> Result<HemisphereGrids, String> {
    let (white, cortex) = surfoxide_mask::hemisphere_masks(labels, table, hemi);
    let both = white.with_data(white.data().iter().zip(cortex.data()).map(|(&a, &b)| a || b).collect());
    let (_, origin) = crop_to(&both, |&b| b, 4, false).ok_or_else(|| format!("hémisphère {hemi} : masque vide"))?;
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
    /// Remaillage abandonné (réparation impossible) : surface lissée non remaillée.
    pub remesh_dropped: bool,
}

/// Surface blanche de genre 0 du masque `mask` (recadré avec une marge).
pub fn white_surface(mask: &Volume<bool>, params: &WhiteParams) -> Result<WhiteSurface, String> {
    let phi = gaussian_blur(&signed_distance(mask), params.sigma);
    let min_spacing = mask.spacing().into_iter().fold(f64::INFINITY, f64::min);
    let c = correct_topology(&phi, Strategy::Best, 0.05 * min_spacing).map_err(|e| e.to_string())?;
    // Points à au moins 0,2 voxel des centres : une lame ou un sillon d'un voxel garde une épaisseur
    // d'au moins 0,2 voxel, où la piale peut passer.
    let mut mesh = surfoxide_extract::marching_cubes(&c.levelset, 0.2);
    debug_topology("marching cubes", &mesh);
    // Lissage pas à pas : un pas qui ferait se couper la surface est annulé localement.
    let mut held = 0;
    for _ in 0..params.taubin {
        let mut trial = mesh.clone();
        surfoxide_extract::taubin(&mut trial, 1, params.lambda, -1.06 * params.lambda);
        held += surfoxide_repair::safe_update(&mut mesh, trial.vertices());
    }
    debug_topology("Taubin", &mesh);
    let smoothed = mesh.clone();
    let mut remesh_failed = false;
    if params.edge > 0.0 {
        match surfoxide_remesh::remesh(&mesh, &surfoxide_remesh::RemeshParams::new(params.edge)) {
            Ok(m) => mesh = m,
            Err(_) => remesh_failed = true,
        }
    }
    let mut repair = surfoxide_repair::repair_intersections(&mut mesh, 30);
    // Le contrat d'abord : si le remaillage ou la réparation échoue, on garde la surface lissée (saine)
    // non remaillée.
    let remesh_dropped = remesh_failed || repair.remaining > 0;
    if remesh_dropped {
        mesh = smoothed;
        repair.remaining = 0;
    }
    // Amas de micro-triangles, puis plis aigus : gênent le décalage de la piale et trahissent un artefact
    // local.
    clean_short_edges(&mut mesh, 0.1 * params.edge.max(0.5));
    unfold(&mut mesh);
    debug_topology("micro-arêtes et plis", &mesh);
    surfoxide_repair::relax_sharp_vertices(&mut mesh, 0.1, 20);
    Ok(WhiteSurface {
        mesh,
        object: c.object,
        added: c.added,
        removed: c.removed,
        strategy: c.strategy,
        held,
        repair,
        remesh_dropped,
    })
}

/// Avec `SURFOXIDE_DEBUG`, affiche le bilan topologique après une étape.
fn debug_topology(step: &str, mesh: &TriMesh) {
    if std::env::var_os("SURFOXIDE_DEBUG").is_some() {
        let r = surfoxide_core::topology_report(mesh);
        eprintln!(
            "  après {step} : V={} χ={} composantes={} bords={} arêtes non variété={} sommets non variété={} orientation incohérente={}",
            r.n_vertices,
            r.euler_characteristic,
            r.components,
            r.boundary_edges,
            r.non_manifold_edges,
            r.non_manifold_vertices,
            r.inconsistent_edges
        );
    }
}

/// Déplie les plis (faces voisines de normales opposées) ; ne garde le résultat que s'il reste (après
/// réparation) sans intersection.
fn unfold(mesh: &mut TriMesh) {
    if let Ok((mut m, remaining)) = surfoxide_remesh::unfold(mesh, -0.5, 5) {
        let repair = surfoxide_repair::repair_intersections(&mut m, 30);
        if std::env::var_os("SURFOXIDE_DEBUG").is_some() {
            eprintln!(
                "  dépliage : {remaining} plis restants ; réparation {} → {} intersections",
                repair.initial, repair.remaining
            );
        }
        if repair.remaining == 0 {
            *mesh = m;
        }
    }
}

/// Fusionne les arêtes plus courtes que `min_len` ; ne garde le résultat que s'il reste (après
/// réparation) sans intersection.
fn clean_short_edges(mesh: &mut TriMesh, min_len: f64) {
    if let Ok(mut m) = surfoxide_remesh::collapse_short_edges(mesh, min_len) {
        if surfoxide_repair::repair_intersections(&mut m, 30).remaining == 0 {
            *mesh = m;
        }
    }
}

/// Piale et champ d'épaisseur, à partir de la surface blanche.
pub fn pial_surface(
    grids: &HemisphereGrids,
    white: &mut WhiteSurface,
    args: &PialArgs,
    min_thickness: f64,
) -> Result<(surfoxide_pial::Pial, usize, usize, surfoxide_pial::RegularizeReport), String> {
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
        // Marge de 50 % sur le seuil du contrat.
        min_distance: 1.5 * min_thickness,
        // Peau répulsive de l'auto-contact de gyroxide (0,2 × arête moyenne), avec une marge de 25 %.
        min_clearance: 0.25 * surfoxide_core::adjacency::mean_edge_length(&white.mesh),
        ..PialParams::default()
    };
    // Si quelques sommets bloquent le départ de la piale, on détend localement la surface blanche
    // (sans intersection) et on réessaie.
    let mut attempt = 0;
    let mut pial = loop {
        match pial_from_white(&white.mesh, &field, &params) {
            Ok(p) => break p,
            Err(e) if attempt < 10 && !e.blocked.is_empty() => {
                if std::env::var_os("SURFOXIDE_DEBUG").is_some() {
                    eprintln!("piale : essai {attempt}, {} sommets bloquants", e.blocked.len());
                }
                match attempt {
                    // Détente sans créer d'intersection.
                    0 | 1 => surfoxide_repair::relax_region(
                        &mut white.mesh,
                        &e.blocked,
                        2 + attempt,
                        5 * (1 + attempt),
                        true,
                    ),
                    // Amas de micro-triangles : la fusion renumérote les sommets, les indices bloquants
                    // ne valent plus ; on réessaie tel quel.
                    2 => {
                        clean_short_edges(&mut white.mesh, 0.05);
                        unfold(&mut white.mesh);
                    }
                    // Lissage libre suivi d'une réparation (crête en lame de couteau).
                    3..=5 => {
                        surfoxide_repair::relax_region(&mut white.mesh, &e.blocked, 2 + attempt, 5 * attempt, false)
                    }
                    // En dernier recours, on recule localement la blanche pour faire de la place.
                    _ => shrink_locally(&mut white.mesh, &e.blocked, attempt - 4),
                }
                attempt += 1;
            }
            Err(e) => {
                // Diagnostic : surface blanche et sommets bloquants au moment de l'échec.
                if let Some(dir) = std::env::var_os("SURFOXIDE_DEBUG_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    let _ = surfoxide_io::gifti::write_surface(
                        dir.join("blocked_white.surf.gii"),
                        &white.mesh,
                        &SurfaceMeta::default(),
                    );
                    let _ = std::fs::write(
                        dir.join("blocked.json"),
                        serde_json::to_string(&e.blocked).unwrap_or_default(),
                    );
                }
                return Err(e.to_string());
            }
        }
    };
    // Triangles de la piale : arêtes d'au moins un quart de l'arête moyenne de la blanche, angles d'au
    // moins 10°, pour que la tétraédrisation (gyroxide) ne crée pas d'éléments dégénérés.
    let edge = surfoxide_core::adjacency::mean_edge_length(&white.mesh);
    let reg = surfoxide_pial::regularize(
        &mut pial.mesh,
        &white.mesh,
        &surfoxide_pial::RegularizeParams {
            min_edge: 0.25 * edge,
            min_angle: 10.0,
            min_distance: params.min_distance,
            min_clearance: params.min_clearance,
            rings: 2,
            rounds: 50,
        },
    );
    // Distance minimale du contrat : les sommets blancs restés trop près de la piale reculent un peu.
    surfoxide_pial::enforce_min_distance(&mut white.mesh, &pial.mesh, 1.5 * min_thickness, 5);
    Ok((pial, n_skeleton, iters, reg))
}

/// Recule localement la surface blanche autour des sommets bloquants, de 0,05 mm par anneau.
fn shrink_locally(mesh: &mut TriMesh, blocked: &[u32], rings: usize) {
    surfoxide_pial::shrink_white(mesh, blocked, rings, 0.05 * rings as f64, None);
}

fn meta(hemi: Hemisphere, secondary: &str, labels: &Volume<i32>, seg: &Path) -> SurfaceMeta {
    let mut meta = SurfaceMeta::hemisphere(hemi.is_left(), secondary);
    meta.source_affine = Some(*labels.affine());
    if let Some(name) = seg.file_name().and_then(|n| n.to_str()) {
        meta.other.insert("SourceImage".into(), name.into());
    }
    meta.other
        .insert("Generator".into(), format!("surfoxide {}", env!("CARGO_PKG_VERSION")));
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
    let (labels, _) = surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    let mut all_valid = true;
    let mut qc = Vec::new();
    for &hemi in hemis {
        let t = Instant::now();
        let grids = hemisphere_grids(&labels, table, hemi)?;
        let mut w = white_surface(&grids.white, params)?;
        // La piale peut détendre localement la surface blanche : on la construit avant d'écrire celle-ci.
        // Un échec de la piale n'interrompt pas la session : la blanche est écrite, le rapport le note.
        let mut pial_error = None;
        let pial = match pial_args {
            Some(args) => {
                let t = Instant::now();
                match pial_surface(&grids, &mut w, args, min_thickness) {
                    Ok(p) => Some((p, t.elapsed().as_secs_f64())),
                    Err(e) => {
                        println!("hémisphère {hemi}, piale : erreur : {e}");
                        pial_error = Some(e);
                        None
                    }
                }
            }
            None => None,
        };
        let white_report = surfoxide_check::check_surface(&w.mesh);
        println!(
            "hémisphère {hemi}, blanche : {:?} (+{} −{} voxels) ; lissage : {} sommets retenus ; réparation : {} → {} intersections{} ; V={} genre={} volume={:.1} mm³ aire={:.1} mm² ; contrat {} ; {:.1} s",
            w.strategy,
            w.added,
            w.removed,
            w.held,
            w.repair.initial,
            w.repair.remaining,
            if w.remesh_dropped { " (remaillage abandonné)" } else { "" },
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
        let mut report = serde_json::json!({ "white": white_report, "remesh_dropped": w.remesh_dropped });
        let mut valid = white_report.valid && pial_error.is_none();
        if let Some(e) = &pial_error {
            report["pial_error"] = e.clone().into();
        }
        // Sommets corticaux de la surface blanche (0 sur la face médiale : ventricules, noyaux gris).
        let cortical = surfoxide_mask::cortical_vertices(&w.mesh, &grids.cortex, 1.0, 3);
        let n_cortical = cortical.iter().filter(|&&c| c).count();
        let path = format!("{prefix}_cortex.shape.gii");
        surfoxide_io::gifti::write_fields(
            &path,
            &[surfoxide_core::VertexField::new(
                "cortex",
                cortical.iter().map(|&c| c as u8 as f64).collect(),
            )],
            &meta(hemi, "GrayWhite", &labels, seg),
        )
        .map_err(|e| format!("{path}: {e}"))?;
        report["cortex_fraction"] = (n_cortical as f64 / cortical.len().max(1) as f64).into();
        // Courbure moyenne et profondeur sulcale DPF* de la blanche (surface de référence de la DPF*).
        let curv = surfoxide_metrics::curvature(&w.mesh);
        let dpf = surfoxide_metrics::dpf_star(
            &w.mesh,
            &curv.mean,
            surfoxide_metrics::depth::ALPHA_REF,
            surfoxide_metrics::Length::ConvexHull,
        );
        // Champs du contrat de guioxide : courbure moyenne et profondeur sulcale (DPF*), sur la blanche.
        let mut curv_meta = meta(hemi, "GrayWhite", &labels, seg);
        curv_meta.other.insert(
            "Description".into(),
            "Courbure moyenne (mm^-1) de la surface blanche, tenseur par face (Rusinkiewicz 2004) ; positive sur les parties convexes".into(),
        );
        let mut sulc_meta = meta(hemi, "GrayWhite", &labels, seg);
        sulc_meta.other.insert(
            "Description".into(),
            format!(
                "Profondeur sulcale DPF* (Dieudonné, Auzias, Lefèvre, Comput Biol Med 2026) de la surface blanche, sans dimension ; positive sur les crêtes ; alpha_ref = {}, L = (volume de l'enveloppe convexe)^(1/3) = {:.3} mm",
                surfoxide_metrics::depth::ALPHA_REF,
                dpf.characteristic_length
            ),
        );
        sulc_meta.other.insert("SulcalDepthMethod".into(), "DPF*".into());
        for (name, values, m) in [("curv", curv.mean, &curv_meta), ("sulc", dpf.depth.clone(), &sulc_meta)] {
            let path = format!("{prefix}_{name}.shape.gii");
            surfoxide_io::gifti::write_fields(&path, &[surfoxide_core::VertexField::new(name, values)], m)
                .map_err(|e| format!("{path}: {e}"))?;
        }
        qc.push(surface_qc(
            &format!("hemi-{}_white", hemi.bids()),
            &w.mesh,
            &white_report,
            &grids.white,
        ));
        report["characteristic_length"] = dpf.characteristic_length.into();
        if let Some(((pial, n_skeleton, iters, reg), secs)) = pial {
            let pial_report = surfoxide_check::check_surface(&pial.mesh);
            // Référence de la piale : substance blanche et cortex, sillons enfouis ouverts par leur nappe
            // médiane (la piale y entre ; même frontière que `surfx eval --boundary pial-opened`).
            let skeleton = surfoxide_pial::sulcal_skeleton(&grids.white, &grids.cortex, 1.5);
            let tissue = grids.white.with_data(
                grids
                    .white
                    .data()
                    .iter()
                    .zip(grids.cortex.data().iter().zip(skeleton.data()))
                    .map(|(&w, (&c, &k))| w || (c && !k))
                    .collect(),
            );
            qc.push(surface_qc(
                &format!("hemi-{}_pial", hemi.bids()),
                &pial.mesh,
                &pial_report,
                &tissue,
            ));
            let pair = surfoxide_check::check_pair(&w.mesh, &pial.mesh, min_thickness);
            // Épaisseur moyenne sur les seuls sommets corticaux.
            let th = &pial.thickness.values;
            let mean_th = th
                .iter()
                .zip(&cortical)
                .filter(|(_, &c)| c)
                .map(|(t, _)| t)
                .sum::<f64>()
                / n_cortical.max(1) as f64;
            println!(
                "hémisphère {hemi}, piale : nappe des sillons enfouis {n_skeleton} voxels ; Laplace {iters} itérations ; {} sommets retenus (manque moyen {:.2} mm) ; épaisseur moyenne (cortex, {:.0} % des sommets) {mean_th:.2} mm ; distance minimale {:.3} mm ; intersections WM/piale {} ; contrat {} ; {:.1} s",
                pial.held,
                pial.mean_missing,
                100.0 * n_cortical as f64 / cortical.len().max(1) as f64,
                pair.min_distance,
                pair.intersections,
                if pial_report.valid && pair.valid { "respecté" } else { "NON RESPECTÉ" },
                secs
            );
            println!(
                "hémisphère {hemi}, piale : régularisation, arêtes courtes {} → {}, triangles aigus {} → {}, arête min {:.3} → {:.3} mm, angle min {:.1}° → {:.1}°",
                reg.short_edges.0,
                reg.short_edges.1,
                reg.sharp_triangles.0,
                reg.sharp_triangles.1,
                reg.min_edge.0,
                reg.min_edge.1,
                reg.min_angle.0,
                reg.min_angle.1
            );
            valid &= pial_report.valid && pair.valid;
            let path = format!("{prefix}_pial.surf.gii");
            surfoxide_io::gifti::write_surface(&path, &pial.mesh, &meta(hemi, "Pial", &labels, seg))
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
            report["pial_mean_missing"] = pial.mean_missing.into();
            report["pial_short_edges"] = reg.short_edges.1.into();
            report["pial_sharp_triangles"] = reg.sharp_triangles.1.into();
            report["pial_min_edge"] = reg.min_edge.1.into();
            report["pial_min_angle"] = reg.min_angle.1.into();
            if std::env::var_os("SURFOXIDE_DEBUG").is_some() {
                let path = format!("{prefix}_missing.shape.gii");
                surfoxide_io::gifti::write_fields(&path, std::slice::from_ref(&pial.missing), &SurfaceMeta::default())
                    .map_err(|e| format!("{path}: {e}"))?;
            }
            report["mean_thickness"] = mean_th.into();
        }
        report["valid"] = valid.into();
        write_json(&format!("{prefix}_report.json"), &report)?;
        println!("écrit : {prefix}_*");
        all_valid &= valid;
    }
    let path = format!("{}_desc-surfqc.tsv", out_prefix.display());
    surfoxide_io::surfqc::write_surfqc(&path, &qc).map_err(|e| format!("{path}: {e}"))?;
    println!("écrit : {path}");
    Ok(all_valid)
}

/// Ligne du contrôle des maillages (`desc-surfqc.tsv`) : topologie, auto-intersections, aire, indice de
/// gyrification (aire / aire de l'enveloppe convexe) et distance moyenne à la frontière de `mask`.
fn surface_qc(
    name: &str,
    mesh: &TriMesh,
    report: &surfoxide_check::SurfaceReport,
    mask: &Volume<bool>,
) -> surfoxide_io::surfqc::SurfQc {
    let hull = surfoxide_metrics::hull::convex_hull_area(mesh.vertices());
    let eval = crate::eval::evaluate(mesh, mask);
    surfoxide_io::surfqc::SurfQc {
        surface: name.into(),
        n_vertices: Some(report.n_vertices),
        n_faces: Some(report.n_faces),
        euler: Some(report.euler_characteristic),
        self_intersections: Some(report.self_intersections),
        area_mm2: Some(report.area),
        gi: (hull > 0.0).then(|| report.area / hull),
        mean_dist_to_seg_mm: Some(eval.surface_to_boundary.mean),
    }
}
