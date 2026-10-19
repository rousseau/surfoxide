//! Évaluation d'une surface contre la segmentation dont elle est issue : distance à la frontière du
//! masque (dans les deux sens), qualité du maillage, contrat.

use std::path::Path;

use serde::Serialize;
use surfoxide_check::bvh::Bvh;
use surfoxide_check::distance::distance2_triangle;
use surfoxide_check::quality::{quality_report, QualityReport};
use surfoxide_check::SurfaceReport;
use surfoxide_core::filter::crop_to;
use surfoxide_core::vec3::Vec3;
use surfoxide_core::{Hemisphere, TriMesh, Volume};
use surfoxide_mask::{LabelTable, Side, Tissue};

/// Frontière de référence.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Boundary {
    /// Frontière du masque blanc fermé (surface WM/GM)
    White,
    /// Frontière du masque blanc ∪ cortex (surface GM/LCR)
    Pial,
    /// Frontière du masque blanc ∪ cortex privé de la nappe médiane des sillons enfouis : la piale doit
    /// entrer dans les sillons dont les berges de cortex se touchent
    PialOpened,
    /// Frontière du tissu blanc latéralisé brut (sans structures médianes ni remplissage) : référence
    /// neutre pour comparer des méthodes dont les masques diffèrent
    Tissue,
}

/// Statistiques d'une distribution de distances (mm).
#[derive(Debug, Clone, Serialize)]
pub struct DistanceStats {
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

fn stats(mut d: Vec<f64>) -> DistanceStats {
    d.sort_by(f64::total_cmp);
    let q = |p: f64| d[((d.len() - 1) as f64 * p).round() as usize];
    DistanceStats {
        mean: d.iter().sum::<f64>() / d.len() as f64,
        median: q(0.5),
        p95: q(0.95),
        max: *d.last().unwrap(),
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalReport {
    /// Distance des sommets de la surface à la frontière du masque voxelique.
    pub surface_to_boundary: DistanceStats,
    /// Distance des centres des faces de voxels de la frontière à la surface.
    pub boundary_to_surface: DistanceStats,
    /// 95ᵉ centile symétrique (max des deux 95ᵉ centiles).
    pub hausdorff95: f64,
    pub quality: QualityReport,
    pub contract: SurfaceReport,
}

/// Faces de voxels séparant le masque du fond (frontière exacte du masque voxelique), chacune coupée en
/// deux triangles ; rend aussi leurs centres.
fn boundary_faces(mask: &Volume<bool>) -> (TriMesh, Vec<Vec3>) {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut centers = Vec::new();
    for idx in 0..mask.len() {
        if !mask.data()[idx] {
            continue;
        }
        let c = mask.coords(idx);
        for a in 0..3 {
            for s in [-1isize, 1] {
                let mut n = c.map(|x| x as isize);
                n[a] += s;
                let outside =
                    !mask.contains(n[0], n[1], n[2]) || !mask.get(n[0] as usize, n[1] as usize, n[2] as usize);
                if !outside {
                    continue;
                }
                let mut center = c.map(|x| x as f64);
                center[a] += 0.5 * s as f64;
                centers.push(mask.voxel_to_world(center));
                let (u, w) = ((a + 1) % 3, (a + 2) % 3);
                let base = vertices.len() as u32;
                for (du, dw) in [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)] {
                    let mut q = center;
                    q[u] += du;
                    q[w] += dw;
                    vertices.push(mask.voxel_to_world(q));
                }
                faces.push([base, base + 1, base + 2]);
                faces.push([base, base + 2, base + 3]);
            }
        }
    }
    (TriMesh::new(vertices, faces).expect("indices valides"), centers)
}

/// Distance de chaque point de `points` à la surface `mesh`.
fn to_surface(points: &[Vec3], mesh: &TriMesh) -> Vec<f64> {
    use rayon::prelude::*;
    let bvh = Bvh::new(mesh);
    let v = mesh.vertices();
    let f = mesh.faces();
    points
        .par_iter()
        .map(|&p| {
            bvh.nearest(p, |g| {
                let t = f[g as usize];
                distance2_triangle(p, v[t[0] as usize], v[t[1] as usize], v[t[2] as usize])
            })
            .map_or(f64::INFINITY, |(_, d2)| d2.sqrt())
        })
        .collect()
}

pub fn evaluate(mesh: &TriMesh, mask: &Volume<bool>) -> EvalReport {
    let (mask, _) = crop_to(mask, |&b| b, 2, false).expect("masque non vide");
    let (boundary, centers) = boundary_faces(&mask);
    let s2b = stats(to_surface(mesh.vertices(), &boundary));
    let b2s = stats(to_surface(&centers, mesh));
    EvalReport {
        hausdorff95: s2b.p95.max(b2s.p95),
        surface_to_boundary: s2b,
        boundary_to_surface: b2s,
        quality: quality_report(mesh),
        contract: surfoxide_check::check_surface(mesh),
    }
}

pub fn run(
    surface: &Path,
    seg: &Path,
    table: &LabelTable,
    hemi: Hemisphere,
    boundary: Boundary,
    json: Option<&Path>,
) -> Result<(), String> {
    let mesh = surfoxide_io::read_mesh(surface).map_err(|e| format!("{}: {e}", surface.display()))?;
    let (labels, _) = surfoxide_io::nifti::read_labels(seg).map_err(|e| format!("{}: {e}", seg.display()))?;
    let (white, cortex) = surfoxide_mask::hemisphere_masks(&labels, table, hemi);
    let mask = match boundary {
        Boundary::White => white,
        Boundary::Tissue => labels.map(|&l| {
            matches!(
                (table.get(l), hemi),
                (Some((Tissue::White, Side::Left)), Hemisphere::Left)
                    | (Some((Tissue::White, Side::Right)), Hemisphere::Right)
            )
        }),
        Boundary::PialOpened => {
            let skeleton = surfoxide_pial::sulcal_skeleton(&white, &cortex, 1.5);
            white.with_data(
                white
                    .data()
                    .iter()
                    .zip(cortex.data().iter().zip(skeleton.data()))
                    .map(|(&w, (&c, &s))| w || (c && !s))
                    .collect(),
            )
        }
        Boundary::Pial => white.with_data(white.data().iter().zip(cortex.data()).map(|(&a, &b)| a || b).collect()),
    };
    let report = evaluate(&mesh, &mask);
    let text = serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?;
    match json {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{}: {e}", p.display()))?,
        None => println!("{text}"),
    }
    Ok(())
}
