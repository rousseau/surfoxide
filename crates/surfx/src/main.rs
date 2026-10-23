//! `surfx` : point d'entrée CLI unique de surfoxide, sur le modèle de `medx` pour medoxide.
//!
//! Sous-commandes :
//! - `device` : vérifie qu'un backend Burn calcule sur la machine (CPU, ou GPU via `cuda` / `wgpu`) ;
//!   sert à valider l'installation, notamment CUDA sous WSL.
//! - `selftest` : construit des surfaces de référence et affiche leur bilan topologique.
//! - `convert` : convertit une surface d'un format à l'autre (GIfTI, VTP, PLY, STL, FreeSurfer).
//! - `mask` : masque blanc fermé d'un hémisphère à partir d'une segmentation (multi-)BOUNTI.
//! - `white` : surface blanche (WM/GM) de genre 0 de chaque hémisphère, en GIfTI.
//! - `run` : surfaces blanche et piale de chaque hémisphère, épaisseur et rapport de contrat ; avec `--bids`,
//!   lit la segmentation de `medx bounti --bids` et écrit les dérivées `surfx-surfaces` (contrat de guioxide).
//! - `eval` : distance d'une surface à la frontière de la segmentation, qualité du maillage, contrat.
//! - `check` : contrôle du contrat d'une surface (topologie, auto-intersections) ou d'un couple WM/piale.
//! - `dataset` : exemple d'entraînement d'un hémisphère pour les réseaux de déformation (P1).
//! - `depth` : courbure moyenne et profondeur sulcale DPF* (invariante d'échelle) d'une surface.

mod dataset;
mod eval;
mod white;

use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use burn::prelude::{Device, Tensor, TensorData};
use clap::{Parser, Subcommand, ValueEnum};
use surfoxide_core::{shapes, topology_report, Hemisphere, TriMesh};
use surfoxide_mask::LabelTable;

#[derive(Parser)]
#[command(
    name = "surfx",
    version,
    about = "Boîte à outils surfoxide pour les surfaces corticales"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// Longueur caractéristique de la DPF*, telle qu'écrite sur la ligne de commande (`--length`).
#[derive(Clone, Copy, ValueEnum)]
enum LengthArg {
    /// Volume de l'enveloppe convexe (code de référence)
    Hull,
    /// Volume enfermé par la surface (texte de l'article)
    Volume,
}

/// Backend de calcul, tel qu'écrit sur la ligne de commande (`--backend`).
#[derive(Clone, Copy, ValueEnum)]
enum BackendArg {
    /// CPU, par le backend `flex` de Burn (Rust pur, déterministe)
    Cpu,
    /// GPU NVIDIA par CUDA (compiler avec `--features cuda`)
    Cuda,
    /// GPU par wgpu : Vulkan, Metal ou DX12 (compiler avec `--features wgpu`)
    Wgpu,
}

#[derive(Subcommand)]
enum Command {
    /// Vérifie qu'un backend Burn fonctionne : produit matriciel sur le device choisi, comparé au CPU
    Device {
        #[arg(long, value_enum, default_value_t = BackendArg::Cpu)]
        backend: BackendArg,
        /// Taille des matrices carrées multipliées
        #[arg(long, default_value_t = 1024)]
        size: usize,
    },
    /// Construit une icosphère et un tore et affiche leur bilan topologique
    Selftest,
    /// Convertit une surface ; formats déduits des extensions (.gii, .vtp, .ply, .stl, sinon FreeSurfer)
    Convert {
        /// Surface lue
        input: PathBuf,
        /// Surface écrite
        output: PathBuf,
    },
    /// Surface blanche de genre 0 de chaque hémisphère (GIfTI) à partir d'une segmentation
    White {
        /// Segmentation NIfTI
        seg: PathBuf,
        /// Table de labels : bounti19 ou multibounti43
        #[arg(long, default_value = "bounti19")]
        labels: LabelTable,
        /// Hémisphère (L ou R) ; les deux par défaut
        #[arg(long)]
        hemi: Option<Hemisphere>,
        #[command(flatten)]
        params: white::WhiteParams,
        /// Préfixe des fichiers écrits : <préfixe>_hemi-L_white.surf.gii
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Surfaces blanche et piale de chaque hémisphère, épaisseur corticale et rapport de contrat
    Run {
        /// Segmentation NIfTI (hors BIDS)
        #[arg(required_unless_present = "bids", conflicts_with = "bids")]
        seg: Option<PathBuf>,
        /// Table de labels : bounti19 ou multibounti43
        #[arg(long, default_value = "bounti19")]
        labels: LabelTable,
        /// Hémisphère (L ou R) ; les deux par défaut
        #[arg(long)]
        hemi: Option<Hemisphere>,
        #[command(flatten)]
        params: white::WhiteParams,
        #[command(flatten)]
        pial: white::PialArgs,
        /// Distance minimale exigée entre surface blanche et piale (mm)
        #[arg(long, default_value_t = 0.02)]
        min_thickness: f64,
        /// Préfixe des fichiers écrits : <préfixe>_hemi-L_{white,pial}.surf.gii, … (hors BIDS)
        #[arg(long, short, required_unless_present = "bids", conflicts_with = "bids")]
        out: Option<PathBuf>,
        /// Racine BIDS : entrée derivatives/medx-bounti/…/sub-X[_ses-Y]_desc-bounti_dseg.nii.gz, sorties dans
        /// derivatives/surfx-surfaces/sub-X[/ses-Y]/anat/ (le parent de la racine si elle s'appelle sourcedata/)
        #[arg(long, requires = "participant")]
        bids: Option<PathBuf>,
        /// Identifiant du participant, sans préfixe `sub-` (avec --bids)
        #[arg(long, requires = "bids")]
        participant: Option<String>,
        /// Identifiant de la session, sans préfixe `ses-` (avec --bids)
        #[arg(long, requires = "bids")]
        session: Option<String>,
        /// Racine de l'étude à utiliser à la place de celle déduite de --bids : les dérivées vont dans
        /// <dossier>/derivatives/ (avec --bids)
        #[arg(long, requires = "bids")]
        output_dir: Option<PathBuf>,
    },
    /// Évalue une surface contre sa segmentation : distances à la frontière, qualité, contrat (JSON)
    Eval {
        /// Surface évaluée
        surface: PathBuf,
        /// Segmentation NIfTI
        seg: PathBuf,
        /// Table de labels : bounti19 ou multibounti43
        #[arg(long, default_value = "bounti19")]
        labels: LabelTable,
        /// Hémisphère : L ou R
        #[arg(long)]
        hemi: Hemisphere,
        /// Frontière de référence
        #[arg(long, value_enum, default_value = "white")]
        boundary: eval::Boundary,
        /// Rapport JSON écrit dans ce fichier (sinon sur la sortie standard)
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Contrôle le contrat d'une surface, et de son imbrication dans une autre ; rapport JSON
    Check {
        /// Surface contrôlée (la WM pour un couple)
        surface: PathBuf,
        /// Surface qui doit la contenir (piale)
        #[arg(long)]
        outer: Option<PathBuf>,
        /// Épaisseur minimale exigée entre les deux surfaces (mm)
        #[arg(long, default_value_t = 0.1)]
        min_thickness: f64,
        /// Rapport JSON écrit dans ce fichier (sinon sur la sortie standard)
        #[arg(long)]
        json: Option<PathBuf>,
    },
    /// Exemple d'entraînement d'un hémisphère (P1) : classes de tissu, distance signée, blanche et piale
    Dataset {
        /// Segmentation NIfTI
        seg: PathBuf,
        /// Table de labels : bounti19 ou multibounti43
        #[arg(long, default_value = "bounti19")]
        labels: LabelTable,
        /// Hémisphère : L ou R
        #[arg(long)]
        hemi: Hemisphere,
        /// Surface blanche (GIfTI)
        #[arg(long)]
        white: PathBuf,
        /// Surface piale de même triangulation (GIfTI)
        #[arg(long)]
        pial: PathBuf,
        /// Préfixe des fichiers écrits : <préfixe>_{classes,sdf}.nii.gz, <préfixe>_{white,pial}.surf.gii
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Courbure moyenne et profondeur sulcale DPF* d'une surface fermée
    Depth {
        /// Surface (GIfTI, VTP, PLY, STL, FreeSurfer)
        surface: PathBuf,
        /// Paramètre d'échelle α_ref de la DPF* (sans dimension ; 500 dans l'article de référence)
        #[arg(long, default_value_t = surfoxide_metrics::depth::ALPHA_REF)]
        alpha: f64,
        /// Volume définissant la longueur caractéristique L = V^(1/3) : enveloppe convexe ou volume enfermé
        #[arg(long, value_enum, default_value = "hull")]
        length: LengthArg,
        /// Préfixe des fichiers écrits : <préfixe>_curv.shape.gii, <préfixe>_dpfstar.shape.gii
        #[arg(long, short)]
        out: PathBuf,
    },
    /// Masque blanc fermé d'un hémisphère (NIfTI u8) à partir d'une segmentation
    Mask {
        /// Segmentation NIfTI
        seg: PathBuf,
        /// Table de labels : bounti19 ou multibounti43
        #[arg(long, default_value = "bounti19")]
        labels: LabelTable,
        /// Hémisphère : L ou R
        #[arg(long)]
        hemi: Hemisphere,
        /// Masque écrit
        #[arg(long, short)]
        out: PathBuf,
    },
}

fn device_for(backend: BackendArg) -> Result<Device, String> {
    match backend {
        BackendArg::Cpu => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        BackendArg::Cuda => Ok(Device::cuda(0)),
        #[cfg(not(feature = "cuda"))]
        BackendArg::Cuda => Err("surfx compilé sans CUDA : relancer avec `cargo run --release --features cuda`".into()),
        #[cfg(feature = "wgpu")]
        BackendArg::Wgpu => Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(not(feature = "wgpu"))]
        BackendArg::Wgpu => Err("surfx compilé sans wgpu : relancer avec `cargo run --release --features wgpu`".into()),
    }
}

/// Produit `a · b` sur `device` ; rend le résultat (lu sur l'hôte) et la durée du second passage (le premier compile les noyaux).
fn timed_matmul(a: &[f32], b: &[f32], n: usize, device: &Device) -> (Vec<f32>, f64) {
    let ta = Tensor::<2>::from_floats(TensorData::new(a.to_vec(), [n, n]), device);
    let tb = Tensor::<2>::from_floats(TensorData::new(b.to_vec(), [n, n]), device);
    let _ = ta.clone().matmul(tb.clone()).into_data(); // échauffement
    let start = Instant::now();
    let c = ta
        .matmul(tb)
        .into_data()
        .try_to_vec::<f32>()
        .expect("lecture du résultat");
    (c, start.elapsed().as_secs_f64())
}

fn run_device(backend: BackendArg, n: usize) -> Result<(), String> {
    let device = device_for(backend)?;
    // Matrices déterministes, sans dépendance à un générateur aléatoire.
    let a: Vec<f32> = (0..n * n).map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5).collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    println!("device : {device:?}");
    let (c, secs) = timed_matmul(&a, &b, n, &device);
    let gflops = 2.0 * (n as f64).powi(3) / secs / 1e9;
    println!("produit {n}×{n} : {:.1} ms ({gflops:.1} GFLOP/s)", secs * 1e3);

    if !matches!(backend, BackendArg::Cpu) {
        let (reference, _) = timed_matmul(&a, &b, n, &Device::flex());
        let max_err = c
            .iter()
            .zip(&reference)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max);
        println!("écart maximal au CPU : {max_err:.2e}");
        if max_err > 1e-2 {
            return Err(format!("résultat incohérent avec le CPU (écart {max_err})"));
        }
    }
    println!("OK");
    Ok(())
}

fn print_report(name: &str, mesh: &TriMesh) {
    let r = topology_report(mesh);
    println!(
        "{name:<16} V={:<6} E={:<6} F={:<6} χ={:<3} composantes={} bords={} genre={} volume={:.3} aire={:.3}",
        r.n_vertices,
        r.n_edges,
        r.n_faces,
        r.euler_characteristic,
        r.components,
        r.boundary_edges,
        r.genus().map_or("—".to_string(), |g| g.to_string()),
        mesh.signed_volume(),
        mesh.area(),
    );
}

fn run_convert(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    let mesh = surfoxide_io::read_mesh(input).map_err(|e| format!("{}: {e}", input.display()))?;
    print_report(&input.display().to_string(), &mesh);
    surfoxide_io::write_mesh(output, &mesh).map_err(|e| format!("{}: {e}", output.display()))?;
    println!("écrit : {}", output.display());
    Ok(())
}

/// Mode BIDS de `surfx run` : segmentation BOUNTI de la session, préfixe des sorties dans
/// `derivatives/surfx-surfaces/sub-X[/ses-Y]/anat/`, et `dataset_description.json` du pipeline.
fn bids_target(
    root: &std::path::Path,
    output_dir: Option<&std::path::Path>,
    participant: &str,
    session: Option<&str>,
) -> Result<(PathBuf, PathBuf), String> {
    use surfoxide_io::bids;
    let seg = bids::bounti_segmentation(root, output_dir, participant, session).map_err(|e| e.to_string())?;
    let dir = bids::derivative_dir(root, output_dir, bids::PIPELINE_SURFACES, participant, session)
        .map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&dir.anat).map_err(|e| format!("{}: {e}", dir.anat.display()))?;
    let generator = bids::Generator {
        name: "surfx run".into(),
        version: env!("CARGO_PKG_VERSION").into(),
        commit: option_env!("SURFX_COMMIT").map(str::to_string),
        command_line: Some(command_line()),
    };
    bids::write_derivative_description(&dir.pipeline, bids::PIPELINE_SURFACES, &generator)
        .map_err(|e| format!("{}: {e}", dir.pipeline.display()))?;
    println!("segmentation : {}\nsorties : {}", seg.display(), dir.anat.display());
    Ok((seg, dir.file_prefix()))
}

/// Ligne de commande telle que lancée (arguments contenant un espace entre apostrophes).
fn command_line() -> String {
    std::env::args()
        .map(|a| {
            if a.contains(char::is_whitespace) {
                format!("'{a}'")
            } else {
                a
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn run_depth(
    surface: &PathBuf,
    alpha: f64,
    length: surfoxide_metrics::Length,
    out: &std::path::Path,
) -> Result<(), String> {
    use surfoxide_core::VertexField;
    use surfoxide_io::gifti::{write_fields, SurfaceMeta};
    let mesh = surfoxide_io::read_mesh(surface).map_err(|e| format!("{}: {e}", surface.display()))?;
    let t0 = Instant::now();
    let curv = surfoxide_metrics::curvature(&mesh);
    let dpf = surfoxide_metrics::dpf_star(&mesh, &curv.mean, alpha, length);
    println!(
        "{} : {} sommets, L = {:.2} mm, α = {:.4} mm⁻², DPF* dans [{:.3}, {:.3}] ({:.1} s)",
        surface.display(),
        mesh.n_vertices(),
        dpf.characteristic_length,
        dpf.alpha,
        dpf.depth.iter().cloned().fold(f64::MAX, f64::min),
        dpf.depth.iter().cloned().fold(f64::MIN, f64::max),
        t0.elapsed().as_secs_f64()
    );
    let prefix = out.display();
    for (name, values) in [("curv", curv.mean), ("dpfstar", dpf.depth)] {
        let path = format!("{prefix}_{name}.shape.gii");
        write_fields(&path, &[VertexField::new(name, values)], &SurfaceMeta::default())
            .map_err(|e| format!("{path}: {e}"))?;
        println!("écrit : {path}");
    }
    Ok(())
}

fn run_mask(seg: &PathBuf, table: &LabelTable, hemi: Hemisphere, out: &PathBuf) -> Result<(), String> {
    let (labels, header) = surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    let mask = surfoxide_mask::white_mask(&labels, table, hemi);
    let n = mask.data().iter().filter(|&&b| b).count();
    let s = mask.spacing();
    println!(
        "hémisphère {hemi} ({table}) : {n} voxels, {:.1} mL",
        n as f64 * s[0] * s[1] * s[2] / 1000.0
    );
    let mask = mask.map(|&b| b as u8);
    surfoxide_io::nifti::write_u8(out, &mask, Some(&header)).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("écrit : {}", out.display());
    Ok(())
}

fn run_check(
    surface: &PathBuf,
    outer: Option<&PathBuf>,
    min_thickness: f64,
    json: Option<&PathBuf>,
) -> Result<(), String> {
    let read = |p: &PathBuf| surfoxide_io::read_mesh(p).map_err(|e| format!("{}: {e}", p.display()));
    let mesh = read(surface)?;
    let mut report = serde_json::json!({ "surface": surfoxide_check::check_surface(&mesh) });
    let mut valid = report["surface"]["valid"].as_bool() == Some(true);
    if let Some(o) = outer {
        let outer_mesh = read(o)?;
        let outer_report = surfoxide_check::check_surface(&outer_mesh);
        let pair = surfoxide_check::check_pair(&mesh, &outer_mesh, min_thickness);
        valid &= outer_report.valid && pair.valid;
        report["outer"] = serde_json::to_value(outer_report).map_err(|e| e.to_string())?;
        report["pair"] = serde_json::to_value(pair).map_err(|e| e.to_string())?;
    }
    report["valid"] = valid.into();
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    match json {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display()))?,
        None => println!("{text}"),
    }
    if valid {
        Ok(())
    } else {
        Err("contrat non respecté".into())
    }
}

/// Code de sortie d'un pipeline : échec si le contrat n'est pas respecté.
fn contract(valid: bool) -> Result<(), String> {
    if valid {
        Ok(())
    } else {
        Err("contrat non respecté (voir les rapports JSON)".into())
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Device { backend, size } => run_device(backend, size),
        Command::Selftest => {
            print_report("icosphère (4)", &shapes::icosphere(4, 1.0));
            print_report("tore (48×24)", &shapes::torus(48, 24, 3.0, 1.0));
            Ok(())
        }
        Command::Convert { input, output } => run_convert(&input, &output),
        Command::Mask { seg, labels, hemi, out } => run_mask(&seg, &labels, hemi, &out),
        Command::Dataset {
            seg,
            labels,
            hemi,
            white,
            pial,
            out,
        } => dataset::run(&seg, &labels, hemi, &white, &pial, &out),
        Command::Depth {
            surface,
            alpha,
            length,
            out,
        } => {
            let length = match length {
                LengthArg::Hull => surfoxide_metrics::Length::ConvexHull,
                LengthArg::Volume => surfoxide_metrics::Length::Enclosed,
            };
            run_depth(&surface, alpha, length, &out)
        }
        Command::Eval {
            surface,
            seg,
            labels,
            hemi,
            boundary,
            json,
        } => eval::run(&surface, &seg, &labels, hemi, boundary, json.as_deref()),
        Command::Check {
            surface,
            outer,
            min_thickness,
            json,
        } => run_check(&surface, outer.as_ref(), min_thickness, json.as_ref()),
        Command::White {
            seg,
            labels,
            hemi,
            params,
            out,
        } => {
            let hemis = hemi.map_or(Hemisphere::BOTH.to_vec(), |h| vec![h]);
            white::run(&seg, &labels, &hemis, &params, None, 0.0, &out).and_then(contract)
        }
        Command::Run {
            seg,
            labels,
            hemi,
            params,
            pial,
            min_thickness,
            out,
            bids,
            participant,
            session,
            output_dir,
        } => {
            let hemis = hemi.map_or(Hemisphere::BOTH.to_vec(), |h| vec![h]);
            let target = match (&bids, &participant) {
                (Some(root), Some(p)) => bids_target(root, output_dir.as_deref(), p, session.as_deref()),
                _ => Ok((seg.expect("exigé par clap"), out.expect("exigé par clap"))),
            };
            target.and_then(|(seg, out)| {
                white::run(&seg, &labels, &hemis, &params, Some(&pial), min_thickness, &out).and_then(contract)
            })
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("erreur : {e}");
            ExitCode::FAILURE
        }
    }
}
