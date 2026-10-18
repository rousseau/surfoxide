//! `surfoxide-burn` : passage des maillages et champs de `surfoxide-core` aux tenseurs Burn, et retour.
//!
//! Les positions passent en `f32` (précision des calculs sur GPU) ; les faces en entiers.
//! Le retour vers un [`TriMesh`] garde la connectivité d'origine : seules les positions changent, ce qui
//! est le cas des déformations (gonflement, réseaux de déformation, intégrateur ODE).

use std::fmt;

use burn::prelude::{Device, Int, Tensor, TensorData};
use surfoxide_core::{TriMesh, VertexField};

/// Erreur de relecture d'un tenseur.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    /// Forme inattendue (attendue, trouvée).
    Shape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// Lecture des données du tenseur impossible.
    Read(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::Shape { expected, found } => {
                write!(f, "forme du tenseur {found:?}, attendue {expected:?}")
            }
            TensorError::Read(e) => write!(f, "lecture du tenseur impossible : {e}"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Positions des sommets, tenseur `[N, 3]` en `f32`.
pub fn vertices_tensor(mesh: &TriMesh, device: &Device) -> Tensor<2> {
    let data: Vec<f32> = mesh
        .vertices()
        .iter()
        .flatten()
        .map(|&x| x as f32)
        .collect();
    Tensor::from_floats(TensorData::new(data, [mesh.n_vertices(), 3]), device)
}

/// Faces, tenseur entier `[F, 3]`.
pub fn faces_tensor(mesh: &TriMesh, device: &Device) -> Tensor<2, Int> {
    let data: Vec<i32> = mesh.faces().iter().flatten().map(|&i| i as i32).collect();
    Tensor::from_ints(TensorData::new(data, [mesh.n_faces(), 3]), device)
}

/// Champ par sommet, tenseur `[N]` en `f32`.
pub fn field_tensor(field: &VertexField, device: &Device) -> Tensor<1> {
    let data: Vec<f32> = field.values.iter().map(|&x| x as f32).collect();
    Tensor::from_floats(TensorData::new(data, [field.values.len()]), device)
}

/// Copie de `mesh` dont les positions sont lues dans `vertices` (`[N, 3]`).
pub fn mesh_with_vertices(mesh: &TriMesh, vertices: Tensor<2>) -> Result<TriMesh, TensorError> {
    let shape = vertices.dims().to_vec();
    if shape != [mesh.n_vertices(), 3] {
        return Err(TensorError::Shape {
            expected: vec![mesh.n_vertices(), 3],
            found: shape,
        });
    }
    let flat: Vec<f32> = vertices
        .into_data()
        .convert::<f32>()
        .try_into_vec()
        .map_err(|e| TensorError::Read(format!("{e:?}")))?;
    let mut out = mesh.clone();
    for (p, q) in out.vertices_mut().iter_mut().zip(flat.as_chunks::<3>().0) {
        *p = [q[0] as f64, q[1] as f64, q[2] as f64];
    }
    Ok(out)
}

/// Champ par sommet relu depuis un tenseur `[N]`.
pub fn field_from_tensor(name: &str, values: Tensor<1>) -> Result<VertexField, TensorError> {
    let flat: Vec<f32> = values
        .into_data()
        .convert::<f32>()
        .try_into_vec()
        .map_err(|e| TensorError::Read(format!("{e:?}")))?;
    Ok(VertexField::new(
        name,
        flat.into_iter().map(f64::from).collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn aller_retour_maillage() {
        let device = Device::flex();
        let s = icosphere(2, 3.0);
        let v = vertices_tensor(&s, &device);
        let f = faces_tensor(&s, &device);
        assert_eq!(v.dims(), [s.n_vertices(), 3]);
        assert_eq!(f.dims(), [s.n_faces(), 3]);
        // Homothétie de rapport 2 sur le device, puis retour : le volume est multiplié par 8.
        let s2 = mesh_with_vertices(&s, v * 2.0).unwrap();
        let ratio = s2.signed_volume() / s.signed_volume();
        assert!((ratio - 8.0).abs() < 1e-4, "rapport {ratio}");
        let faces: Vec<i32> = f.into_data().convert::<i32>().try_into_vec().unwrap();
        assert_eq!(faces[..3], s.faces()[0].map(|i| i as i32));
    }

    #[test]
    fn aller_retour_champ() {
        let device = Device::flex();
        let field = VertexField::new("epaisseur", vec![1.0, 2.5, 3.25]);
        let back = field_from_tensor("epaisseur", field_tensor(&field, &device)).unwrap();
        assert_eq!(back, field);
    }

    #[test]
    fn forme_incorrecte_refusee() {
        let device = Device::flex();
        let s = icosphere(1, 1.0);
        let t = Tensor::<2>::zeros([3, 3], &device);
        assert!(matches!(
            mesh_with_vertices(&s, t),
            Err(TensorError::Shape { .. })
        ));
    }
}
