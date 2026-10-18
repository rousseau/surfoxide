//! Maillage triangulaire indexé.

use std::fmt;

/// Maillage triangulaire : une liste de sommets et une liste de faces qui les indexent.
///
/// Convention (voir `CLAUDE.md`, « contrat de données ») : coordonnées dans le repère monde RAS, en mm ;
/// faces orientées dans le sens trigonométrique vu de l'extérieur, donc normales sortantes.
#[derive(Debug, Clone, PartialEq)]
pub struct TriMesh {
    vertices: Vec<[f64; 3]>,
    faces: Vec<[u32; 3]>,
}

/// Erreurs de construction d'un maillage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeshError {
    /// Une face référence un sommet qui n'existe pas.
    IndexOutOfRange {
        face: usize,
        index: u32,
        n_vertices: usize,
    },
    /// Une face répète un sommet (triangle dégénéré au sens combinatoire).
    RepeatedVertex { face: usize },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::IndexOutOfRange {
                face,
                index,
                n_vertices,
            } => {
                write!(
                    f,
                    "face {face} : indice de sommet {index} hors limites ({n_vertices} sommets)"
                )
            }
            MeshError::RepeatedVertex { face } => write!(f, "face {face} : un sommet est répété"),
        }
    }
}

impl std::error::Error for MeshError {}

impl TriMesh {
    /// Construit un maillage en vérifiant que chaque face référence trois sommets distincts et existants.
    pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> Result<Self, MeshError> {
        let n = vertices.len();
        for (i, f) in faces.iter().enumerate() {
            for &v in f {
                if v as usize >= n {
                    return Err(MeshError::IndexOutOfRange {
                        face: i,
                        index: v,
                        n_vertices: n,
                    });
                }
            }
            if f[0] == f[1] || f[1] == f[2] || f[0] == f[2] {
                return Err(MeshError::RepeatedVertex { face: i });
            }
        }
        Ok(Self { vertices, faces })
    }

    pub fn vertices(&self) -> &[[f64; 3]] {
        &self.vertices
    }

    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Accès en écriture aux positions : déplacer des sommets ne change pas la topologie.
    pub fn vertices_mut(&mut self) -> &mut [[f64; 3]] {
        &mut self.vertices
    }

    /// Démonte le maillage en (sommets, faces).
    pub fn into_parts(self) -> (Vec<[f64; 3]>, Vec<[u32; 3]>) {
        (self.vertices, self.faces)
    }

    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Volume signé enfermé par la surface (théorème de la divergence), en mm³.
    ///
    /// Positif pour une surface fermée aux normales sortantes, négatif si l'orientation est inversée.
    /// N'a de sens que pour une surface fermée.
    pub fn signed_volume(&self) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|i| self.vertices[i as usize]);
                // a · (b × c) / 6
                let cross = [
                    b[1] * c[2] - b[2] * c[1],
                    b[2] * c[0] - b[0] * c[2],
                    b[0] * c[1] - b[1] * c[0],
                ];
                (a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]) / 6.0
            })
            .sum()
    }

    /// Aire totale de la surface, en mm².
    pub fn area(&self) -> f64 {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|i| self.vertices[i as usize]);
                let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    u[1] * v[2] - u[2] * v[1],
                    u[2] * v[0] - u[0] * v[2],
                    u[0] * v[1] - u[1] * v[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum()
    }

    /// Inverse l'orientation de toutes les faces.
    pub fn flip_orientation(&mut self) {
        for f in &mut self.faces {
            f.swap(1, 2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuse_un_indice_hors_limites() {
        let err = TriMesh::new(vec![[0.0; 3]; 3], vec![[0, 1, 3]]).unwrap_err();
        assert_eq!(
            err,
            MeshError::IndexOutOfRange {
                face: 0,
                index: 3,
                n_vertices: 3
            }
        );
    }

    #[test]
    fn refuse_un_sommet_repete() {
        let err = TriMesh::new(vec![[0.0; 3]; 3], vec![[0, 1, 1]]).unwrap_err();
        assert_eq!(err, MeshError::RepeatedVertex { face: 0 });
    }
}
