//! Champs scalaires portés par les sommets d'un maillage (épaisseur, courbure, profondeur sulcale…).

use std::fmt;

use crate::TriMesh;

/// Champ scalaire nommé, une valeur par sommet.
#[derive(Debug, Clone, PartialEq)]
pub struct VertexField {
    pub name: String,
    pub values: Vec<f64>,
}

/// Le champ n'a pas autant de valeurs que le maillage a de sommets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSizeError {
    pub name: String,
    pub n_values: usize,
    pub n_vertices: usize,
}

impl fmt::Display for FieldSizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "champ « {} » : {} valeurs pour {} sommets",
            self.name, self.n_values, self.n_vertices
        )
    }
}

impl std::error::Error for FieldSizeError {}

impl VertexField {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    /// Vérifie que le champ a une valeur par sommet de `mesh`.
    pub fn check(&self, mesh: &TriMesh) -> Result<(), FieldSizeError> {
        if self.values.len() == mesh.n_vertices() {
            Ok(())
        } else {
            Err(FieldSizeError {
                name: self.name.clone(),
                n_values: self.values.len(),
                n_vertices: mesh.n_vertices(),
            })
        }
    }
}
