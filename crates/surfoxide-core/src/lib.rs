//! `surfoxide-core` : briques communes de surfoxide, sans dépendance vers Burn ni vers un format de fichier.
//!
//! - [`TriMesh`] : maillage triangulaire (sommets en mm dans le repère monde, faces indexées).
//! - [`Volume`] : volume régulier avec son affine voxel → monde.
//! - [`VertexField`] : champ scalaire par sommet.
//! - [`adjacency`] : arêtes, voisinages, normales, aires.
//! - [`components`] : composantes connexes et cavités d'un masque 3D.
//! - [`distance`] : transformée de distance euclidienne, distance signée.
//! - [`filter`] : flou gaussien, recadrage de volume.
//! - [`Hemisphere`] : gauche ou droit.
//! - [`topology`] : contrôles de topologie (variété, bords, orientation, composantes, genre),
//!   première brique du contrat de validation demandé par gyroxide.
//! - [`shapes`] : surfaces de référence (icosphère, tore) pour les tests.
//! - [`vec3`] : opérations sur les vecteurs 3D.

pub mod adjacency;
pub mod components;
pub mod distance;
pub mod field;
pub mod filter;
pub mod hemisphere;
pub mod mesh;
pub mod shapes;
pub mod topology;
pub mod vec3;
pub mod volume;

pub use components::Connectivity;
pub use field::VertexField;
pub use hemisphere::Hemisphere;
pub use mesh::{MeshError, TriMesh};
pub use topology::{topology_report, TopologyReport};
pub use volume::{Affine, Volume};
