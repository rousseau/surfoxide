//! `surfoxide-metrics` : mesures corticales sur une surface triangulaire.
//!
//! - [`laplacian`] : opérateur de Laplace–Beltrami par éléments finis P1 (raideur cotangente, masse cohérente).
//! - [`curvature`] : courbures principales et moyenne (tenseur par face, Rusinkiewicz 2004).
//! - [`hull`] : enveloppe convexe 3D (quickhull, prédicats exacts) et son volume.
//! - [`depth`] : profondeur sulcale DPF (Boucher et al. 2009) et DPF* invariante d'échelle
//!   (Dieudonné, Auzias, Lefèvre, Comput Biol Med 2026).
//! - [`sparse`] : matrice creuse symétrique et gradient conjugué préconditionné.

pub mod curvature;
pub mod depth;
pub mod hull;
pub mod laplacian;
pub mod sparse;

pub use curvature::{curvature, Curvature};
pub use depth::{depth_potential, dpf_star, DpfStar};
