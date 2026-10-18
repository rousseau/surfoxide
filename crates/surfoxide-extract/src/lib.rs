//! `surfoxide-extract` : extraction de surface et lissage.
//!
//! - [`marching_cubes`] : marching cubes à connexité cohérente (26, 6), dont la surface a exactement la
//!   topologie digitale de l'objet `φ < 0` (Han, Xu & Prince 2003).
//! - [`smooth`] : lissage de Taubin λ/μ, sans rétrécissement.

pub mod marching_cubes;
pub mod smooth;

pub use marching_cubes::marching_cubes;
pub use smooth::taubin;
