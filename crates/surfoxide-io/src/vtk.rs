//! VTK PolyData XML (`.vtp`), pour la visualisation dans ParaView.
//!
//! Écriture en ASCII (positions en `Float64`, faces en `Int64`, champs par sommet en `PointData`).
//! Lecture limitée aux fichiers ASCII non compressés, comme ceux écrits ici.

use std::fmt::Write as _;
use std::path::Path;

use quick_xml::events::Event;
use quick_xml::Reader;
use surfoxide_core::{TriMesh, VertexField};

use crate::{build_mesh, format_err, IoError, Result};

/// Écrit `mesh` et ses champs par sommet.
pub fn write_vtp(path: impl AsRef<Path>, mesh: &TriMesh, fields: &[VertexField]) -> Result<()> {
    for f in fields {
        f.check(mesh)
            .map_err(|e| IoError::Format(format!("VTP : {e}")))?;
    }
    let mut s = String::new();
    let w = |s: &mut String, t: &str| s.push_str(t);
    w(&mut s, "<?xml version=\"1.0\"?>\n<VTKFile type=\"PolyData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n<PolyData>\n");
    let _ = writeln!(
        s,
        "<Piece NumberOfPoints=\"{}\" NumberOfPolys=\"{}\">",
        mesh.n_vertices(),
        mesh.n_faces()
    );
    if !fields.is_empty() {
        let _ = writeln!(s, "<PointData Scalars=\"{}\">", fields[0].name);
        for f in fields {
            let _ = writeln!(
                s,
                "<DataArray type=\"Float64\" Name=\"{}\" format=\"ascii\">",
                f.name
            );
            for v in &f.values {
                let _ = writeln!(s, "{v}");
            }
            w(&mut s, "</DataArray>\n");
        }
        w(&mut s, "</PointData>\n");
    }
    w(
        &mut s,
        "<Points>\n<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n",
    );
    for p in mesh.vertices() {
        let _ = writeln!(s, "{} {} {}", p[0], p[1], p[2]);
    }
    w(&mut s, "</DataArray>\n</Points>\n<Polys>\n<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
    for f in mesh.faces() {
        let _ = writeln!(s, "{} {} {}", f[0], f[1], f[2]);
    }
    w(
        &mut s,
        "</DataArray>\n<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n",
    );
    for i in 1..=mesh.n_faces() {
        let _ = writeln!(s, "{}", 3 * i);
    }
    w(
        &mut s,
        "</DataArray>\n</Polys>\n</Piece>\n</PolyData>\n</VTKFile>\n",
    );
    std::fs::write(path, s)?;
    Ok(())
}

/// Lit un `.vtp` ASCII : maillage triangulaire et champs scalaires par sommet.
pub fn read_vtp(path: impl AsRef<Path>) -> Result<(TriMesh, Vec<VertexField>)> {
    let path = path.as_ref();
    let xml = std::fs::read_to_string(path)?;
    let mut reader = Reader::from_str(&xml);
    let mut section = String::new();
    let mut current: Option<(String, String)> = None; // (nom, section) du DataArray en cours
    let mut text = String::new();
    let (mut points, mut conn, mut offsets) = (Vec::new(), Vec::new(), Vec::new());
    let mut fields = Vec::new();
    loop {
        match reader
            .read_event()
            .map_err(|e| IoError::Format(format!("VTP : XML invalide : {e}")))?
        {
            Event::Start(e) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if name == "DataArray" {
                    let mut arr_name = String::new();
                    for a in e.attributes().flatten() {
                        let v = String::from_utf8_lossy(&a.value).into_owned();
                        match a.key.as_ref() {
                            b"Name" => arr_name = v,
                            b"format" if v != "ascii" => {
                                return format_err(format!(
                                    "{} : seul le format ascii est lu (trouvé « {v} »)",
                                    path.display()
                                ))
                            }
                            _ => {}
                        }
                    }
                    current = Some((arr_name, section.clone()));
                    text.clear();
                } else {
                    section = name;
                }
            }
            Event::Text(t) => text.push_str(&t.decode().unwrap_or_default()),
            Event::End(e) => {
                if e.name().as_ref() == b"DataArray" {
                    let (name, sec) = current.take().unwrap_or_default();
                    let vals: Vec<f64> = text
                        .split_ascii_whitespace()
                        .map(|t| t.parse::<f64>())
                        .collect::<std::result::Result<_, _>>()
                        .map_err(|_| IoError::Format("VTP : nombre invalide".into()))?;
                    match (sec.as_str(), name.as_str()) {
                        ("Points", _) => points = vals,
                        ("Polys", "connectivity") => conn = vals,
                        ("Polys", "offsets") => offsets = vals,
                        ("PointData", _) => fields.push(VertexField::new(name, vals)),
                        _ => {}
                    }
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }
    // Vérifie que tous les polygones sont des triangles.
    let mut prev = 0usize;
    for &o in &offsets {
        if o as usize - prev != 3 {
            return format_err(format!("{} : polygone non triangulaire", path.display()));
        }
        prev = o as usize;
    }
    let vertices = points
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let faces = conn
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    Ok((build_mesh(vertices, faces)?, fields))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_same, tmp};
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn aller_retour_vtp() {
        let s = icosphere(2, 10.0);
        let f = VertexField::new("x", s.vertices().iter().map(|p| p[0]).collect());
        let p = tmp("s.vtp");
        write_vtp(&p, &s, std::slice::from_ref(&f)).unwrap();
        let (back, fields) = read_vtp(&p).unwrap();
        assert_same(&s, &back, 0.0);
        assert_eq!(fields, vec![f]);
    }
}
