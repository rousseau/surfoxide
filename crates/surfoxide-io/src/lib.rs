//! `surfoxide-io` : lecture et écriture des formats de fichiers.
//!
//! - [`nifti`] : volumes (segmentations, cartes) avec leur affine.
//! - [`gifti`] : surfaces (`.surf.gii`) et champs par sommet (`.shape.gii`, `.func.gii`), format de sortie
//!   principal.
//! - [`vtk`] : `.vtp` (PolyData XML) pour ParaView.
//! - [`ply`], [`stl`] : formats d'échange généraux.
//! - [`freesurfer`] : surfaces et fichiers « curv » de FreeSurfer.
//!
//! - [`bids`] : chemins des dérivées BIDS (`surfx run --bids`) et `dataset_description.json`.
//! - [`data`] : emplacement des données réelles rapatriées (`scripts/fetch-data.sh`).
//!
//! [`read_mesh`] et [`write_mesh`] choisissent le format d'après l'extension.

use std::fmt;
use std::path::Path;

use surfoxide_core::TriMesh;

pub mod bids;
pub mod data;
pub mod freesurfer;
pub mod gifti;
pub mod nifti;
pub mod ply;
pub mod stl;
pub mod vtk;

/// Erreur d'entrée-sortie ou de format.
#[derive(Debug)]
pub enum IoError {
    /// Erreur du système de fichiers.
    Io(std::io::Error),
    /// Fichier mal formé ou variante non prise en charge.
    Format(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "{e}"),
            IoError::Format(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for IoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            IoError::Io(e) => Some(e),
            IoError::Format(_) => None,
        }
    }
}

impl From<std::io::Error> for IoError {
    fn from(e: std::io::Error) -> Self {
        IoError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, IoError>;

pub(crate) fn format_err<T>(msg: impl Into<String>) -> Result<T> {
    Err(IoError::Format(msg.into()))
}

/// Format de maillage, déduit du nom de fichier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    Gifti,
    Vtp,
    Ply,
    Stl,
    FreeSurfer,
}

impl MeshFormat {
    /// `.gii`, `.vtp`, `.ply`, `.stl` ; tout autre nom (`lh.white`, `rh.pial`, `.surf`…) est lu comme une
    /// surface FreeSurfer.
    pub fn from_path(path: &Path) -> Self {
        let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        match ext.as_deref() {
            Some("gii") => MeshFormat::Gifti,
            Some("vtp") => MeshFormat::Vtp,
            Some("ply") => MeshFormat::Ply,
            Some("stl") => MeshFormat::Stl,
            _ => MeshFormat::FreeSurfer,
        }
    }
}

/// Lit un maillage, au format déduit de l'extension.
pub fn read_mesh(path: impl AsRef<Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    match MeshFormat::from_path(path) {
        MeshFormat::Gifti => gifti::read_surface(path).map(|(m, _)| m),
        MeshFormat::Vtp => vtk::read_vtp(path).map(|(m, _)| m),
        MeshFormat::Ply => ply::read_ply(path),
        MeshFormat::Stl => stl::read_stl(path),
        MeshFormat::FreeSurfer => freesurfer::read_surf(path),
    }
}

/// Écrit un maillage, au format déduit de l'extension.
pub fn write_mesh(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<()> {
    let path = path.as_ref();
    match MeshFormat::from_path(path) {
        MeshFormat::Gifti => gifti::write_surface(path, mesh, &gifti::SurfaceMeta::default()),
        MeshFormat::Vtp => vtk::write_vtp(path, mesh, &[]),
        MeshFormat::Ply => ply::write_ply(path, mesh),
        MeshFormat::Stl => stl::write_stl(path, mesh),
        MeshFormat::FreeSurfer => freesurfer::write_surf(path, mesh),
    }
}

/// Construit un maillage lu dans un fichier, en traduisant l'erreur de validation.
pub(crate) fn build_mesh(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Result<TriMesh> {
    TriMesh::new(vertices, faces).map_err(|e| IoError::Format(format!("maillage invalide : {e}")))
}

#[cfg(test)]
pub(crate) mod testutil {
    use std::path::PathBuf;

    /// Chemin de fichier temporaire unique pour un test.
    pub fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("surfoxide-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Vérifie que deux maillages ont la même connectivité et des positions égales à `tol` près.
    pub fn assert_same(a: &surfoxide_core::TriMesh, b: &surfoxide_core::TriMesh, tol: f64) {
        assert_eq!(a.faces(), b.faces());
        assert_eq!(a.n_vertices(), b.n_vertices());
        for (p, q) in a.vertices().iter().zip(b.vertices()) {
            for c in 0..3 {
                assert!((p[c] - q[c]).abs() <= tol, "{p:?} contre {q:?}");
            }
        }
    }
}
