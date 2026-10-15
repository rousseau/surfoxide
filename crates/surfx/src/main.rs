//! `surfx` : point d'entrée CLI unique de surfoxide, sur le modèle de `medx` pour medoxide.
//!
//! Sous-commandes :
//! - `device` : vérifie qu'un backend Burn calcule sur la machine (CPU, ou GPU via `cuda` / `wgpu`) ;
//!   sert à valider l'installation, notamment CUDA sous WSL.
//! - `selftest` : construit des surfaces de référence et affiche leur bilan topologique.

use std::process::ExitCode;
use std::time::Instant;

use burn::prelude::{Device, Tensor, TensorData};
use clap::{Parser, Subcommand, ValueEnum};
use surfoxide_core::{shapes, topology_report, TriMesh};

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
}

fn device_for(backend: BackendArg) -> Result<Device, String> {
    match backend {
        BackendArg::Cpu => Ok(Device::flex()),
        #[cfg(feature = "cuda")]
        BackendArg::Cuda => Ok(Device::cuda(0)),
        #[cfg(not(feature = "cuda"))]
        BackendArg::Cuda => Err(
            "surfx compilé sans CUDA : relancer avec `cargo run --release --features cuda`".into(),
        ),
        #[cfg(feature = "wgpu")]
        BackendArg::Wgpu => Ok(Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)),
        #[cfg(not(feature = "wgpu"))]
        BackendArg::Wgpu => Err(
            "surfx compilé sans wgpu : relancer avec `cargo run --release --features wgpu`".into(),
        ),
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
    let a: Vec<f32> = (0..n * n)
        .map(|i| ((i * 7919) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    let b: Vec<f32> = (0..n * n)
        .map(|i| ((i * 104729) % 1000) as f32 / 1000.0 - 0.5)
        .collect();

    println!("device : {device:?}");
    let (c, secs) = timed_matmul(&a, &b, n, &device);
    let gflops = 2.0 * (n as f64).powi(3) / secs / 1e9;
    println!(
        "produit {n}×{n} : {:.1} ms ({gflops:.1} GFLOP/s)",
        secs * 1e3
    );

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

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Device { backend, size } => run_device(backend, size),
        Command::Selftest => {
            print_report("icosphère (4)", &shapes::icosphere(4, 1.0));
            print_report("tore (48×24)", &shapes::torus(48, 24, 3.0, 1.0));
            Ok(())
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
