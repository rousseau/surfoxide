//! Surfaces FreeSurfer (`lh.white`, `rh.pial`…) et fichiers de valeurs par sommet (`lh.thickness`,
//! `lh.curv`…), au format binaire grand-boutiste.
//!
//! Les coordonnées d'une surface FreeSurfer sont en « tkregister RAS ». Si le fichier porte ses
//! informations de volume (`cras = …`), la lecture ajoute `cras` pour revenir au repère scanner (RAS),
//! comme le veut le contrat de surfoxide. L'écriture enregistre les coordonnées telles quelles, sans
//! informations de volume (donc `cras` nul).

use std::io::Write;
use std::path::Path;

use surfoxide_core::{TriMesh, VertexField};

use crate::{build_mesh, format_err, Result};

const TRIANGLE_MAGIC: [u8; 3] = [0xFF, 0xFF, 0xFE];
const CURV_MAGIC: [u8; 3] = [0xFF, 0xFF, 0xFF];

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.b.len() {
            return format_err("FreeSurfer : fin de fichier prématurée");
        }
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_be_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Lit une surface triangulaire FreeSurfer.
pub fn read_surf(path: impl AsRef<Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(&TRIANGLE_MAGIC) {
        return format_err(format!("{} : pas une surface FreeSurfer triangulaire", path.display()));
    }
    // Commentaire « created by … » terminé par deux retours à la ligne.
    let start = bytes[3..]
        .windows(2)
        .position(|w| w == b"\n\n")
        .map(|p| p + 5)
        .ok_or_else(|| crate::IoError::Format("FreeSurfer : commentaire non terminé".into()))?;
    let mut c = Cursor { b: &bytes, pos: start };
    let nv = c.i32()? as usize;
    let nf = c.i32()? as usize;
    let mut vertices = Vec::with_capacity(nv);
    for _ in 0..nv {
        vertices.push([c.f32()? as f64, c.f32()? as f64, c.f32()? as f64]);
    }
    let mut faces = Vec::with_capacity(nf);
    for _ in 0..nf {
        faces.push([c.i32()? as u32, c.i32()? as u32, c.i32()? as u32]);
    }
    // Informations de volume éventuelles : lignes « clé = valeur » après les faces.
    let tail = String::from_utf8_lossy(&bytes[c.pos..]);
    if let Some(line) = tail.lines().find(|l| l.trim_start().starts_with("cras")) {
        let cras: Vec<f64> = line
            .split('=')
            .nth(1)
            .unwrap_or("")
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if cras.len() == 3 {
            for p in &mut vertices {
                for k in 0..3 {
                    p[k] += cras[k];
                }
            }
        }
    }
    build_mesh(vertices, faces)
}

/// Écrit une surface triangulaire FreeSurfer.
pub fn write_surf(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&TRIANGLE_MAGIC)?;
    w.write_all(b"created by surfoxide\n\n")?;
    w.write_all(&(mesh.n_vertices() as i32).to_be_bytes())?;
    w.write_all(&(mesh.n_faces() as i32).to_be_bytes())?;
    for p in mesh.vertices() {
        for x in p {
            w.write_all(&(*x as f32).to_be_bytes())?;
        }
    }
    for f in mesh.faces() {
        for &i in f {
            w.write_all(&(i as i32).to_be_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

/// Lit un fichier de valeurs par sommet (format « curv » récent).
pub fn read_curv(path: impl AsRef<Path>) -> Result<VertexField> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    if !bytes.starts_with(&CURV_MAGIC) {
        return format_err(format!(
            "{} : pas un fichier curv FreeSurfer (format récent)",
            path.display()
        ));
    }
    let mut c = Cursor { b: &bytes, pos: 3 };
    let nv = c.i32()? as usize;
    let _nf = c.i32()?;
    let per_vertex = c.i32()?;
    if per_vertex != 1 {
        return format_err(format!(
            "{} : {per_vertex} valeurs par sommet (1 attendue)",
            path.display()
        ));
    }
    let values = (0..nv).map(|_| c.f32().map(f64::from)).collect::<Result<_>>()?;
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.rsplit('.').next())
        .unwrap_or("curv");
    Ok(VertexField::new(name, values))
}

/// Écrit un fichier de valeurs par sommet ; `n_faces` est celui de la surface associée.
pub fn write_curv(path: impl AsRef<Path>, field: &VertexField, n_faces: usize) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    w.write_all(&CURV_MAGIC)?;
    w.write_all(&(field.values.len() as i32).to_be_bytes())?;
    w.write_all(&(n_faces as i32).to_be_bytes())?;
    w.write_all(&1i32.to_be_bytes())?;
    for v in &field.values {
        w.write_all(&(*v as f32).to_be_bytes())?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_same, tmp};
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn aller_retour_surface() {
        let s = icosphere(2, 10.0);
        let p = tmp("lh.white");
        write_surf(&p, &s).unwrap();
        assert_same(&s, &read_surf(&p).unwrap(), 1e-5);
    }

    #[test]
    fn decalage_cras_applique() {
        let s = icosphere(0, 1.0);
        let p = tmp("rh.pial");
        write_surf(&p, &s).unwrap();
        let mut f = std::fs::OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(&[0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 20]).unwrap();
        f.write_all(b"valid = 1  # volume info valid\ncras   = 1.0 -2.0 3.0\n")
            .unwrap();
        drop(f);
        let back = read_surf(&p).unwrap();
        let (a, b) = (s.vertices()[0], back.vertices()[0]);
        assert!((b[0] - a[0] - 1.0).abs() < 1e-6 && (b[1] - a[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn aller_retour_curv() {
        let f = VertexField::new("thickness", vec![1.5, 2.0, 2.5]);
        let p = tmp("lh.thickness");
        write_curv(&p, &f, 2).unwrap();
        assert_eq!(read_curv(&p).unwrap(), f);
    }
}
