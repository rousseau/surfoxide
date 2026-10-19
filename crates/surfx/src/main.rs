//! `surfx` : point d'entrée CLI unique de surfoxide, sur le modèle de `medx` pour medoxide.
//!
//! Sous-commandes :
//! - `device` : vérifie qu'un backend Burn calcule sur la machine (CPU, ou GPU via `cuda` / `wgpu`) ;
//!   sert à valider l'installation, notamment CUDA sous WSL.
//! - `selftest` : construit des surfaces de référence et affiche leur bilan topologique.
//! - `convert` : convertit une surface d'un format à l'autre (GIfTI, VTP, PLY, STL, FreeSurfer).
//! - `mask` : masque blanc fermé d'un hémisphère à partir d'une segmentation (multi-)BOUNTI.
//! - `white` : surface blanche (WM/GM) de genre 0 de chaque hémisphère, en GIfTI.
//! - `run` : surfaces blanche et piale de chaque hémisphère, épaisseur et rapport de contrat.
//! - `eval` : distance d'une surface à la frontière de la segmentation, qualité du maillage, contrat.
//! - `check` : contrôle du contrat d'une surface (topologie, auto-intersections) ou d'un couple WM/piale.

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
        #[command(flatten)]
        pial: white::PialArgs,
        /// Distance minimale exigée entre surface blanche et piale (mm)
        #[arg(long, default_value_t = 0.02)]
        min_thickness: f64,
        /// Préfixe des fichiers écrits : <préfixe>_hemi-L_{white,pial}.surf.gii, …
        #[arg(long, short)]
        out: PathBuf,
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
        } => {
            let hemis = hemi.map_or(Hemisphere::BOTH.to_vec(), |h| vec![h]);
            white::run(&seg, &labels, &hemis, &params, Some(&pial), min_thickness, &out).and_then(contract)
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
