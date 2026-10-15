//! `surfoxide-core` : briques communes de surfoxide, sans dépendance vers Burn ni vers un format de fichier.
//!
//! - [`TriMesh`] : maillage triangulaire (sommets en mm dans le repère monde, faces indexées).
//! - [`topology`] : contrôles de topologie (variété, bords, orientation, composantes, genre),
//!   première brique du contrat de validation demandé par gyroxide.
//! - [`shapes`] : surfaces de référence (icosphère, tore) pour les tests.

pub mod mesh;
pub mod shapes;
pub mod topology;

pub use mesh::{MeshError, TriMesh};
pub use topology::{topology_report, TopologyReport};
