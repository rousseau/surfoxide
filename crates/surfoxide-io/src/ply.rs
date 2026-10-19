//! PLY (Stanford) : écriture binaire petit-boutiste (positions en `double`), lecture ASCII et binaire.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

use surfoxide_core::TriMesh;

use crate::{build_mesh, format_err, IoError, Result};

/// Écrit `mesh` en PLY binaire petit-boutiste.
pub fn write_ply(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    write!(
        w,
        "ply\nformat binary_little_endian 1.0\ncomment surfoxide\nelement vertex {}\nproperty double x\nproperty double y\nproperty double z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n",
        mesh.n_vertices(),
        mesh.n_faces()
    )?;
    for p in mesh.vertices() {
        for x in p {
            w.write_all(&x.to_le_bytes())?;
        }
    }
    for f in mesh.faces() {
        w.write_all(&[3u8])?;
        for &i in f {
            w.write_all(&(i as i32).to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Encoding {
    Ascii,
    LittleEndian,
    BigEndian,
}

/// Taille en octets d'un type scalaire PLY.
fn type_size(t: &str) -> Result<usize> {
    Ok(match t {
        "char" | "uchar" | "int8" | "uint8" => 1,
        "short" | "ushort" | "int16" | "uint16" => 2,
        "int" | "uint" | "float" | "int32" | "uint32" | "float32" => 4,
        "double" | "float64" => 8,
        other => return format_err(format!("PLY : type « {other} » inconnu")),
    })
}

fn read_scalar(r: &mut impl Read, t: &str, enc: Encoding) -> Result<f64> {
    let mut b = [0u8; 8];
    let n = type_size(t)?;
    r.read_exact(&mut b[..n])?;
    macro_rules! conv {
        ($ty:ty, $n:expr) => {{
            let a: [u8; $n] = b[..$n].try_into().unwrap();
            (if enc == Encoding::BigEndian {
                <$ty>::from_be_bytes(a)
            } else {
                <$ty>::from_le_bytes(a)
            }) as f64
        }};
    }
    Ok(match t {
        "char" | "int8" => b[0] as i8 as f64,
        "uchar" | "uint8" => b[0] as f64,
        "short" | "int16" => conv!(i16, 2),
        "ushort" | "uint16" => conv!(u16, 2),
        "int" | "int32" => conv!(i32, 4),
        "uint" | "uint32" => conv!(u32, 4),
        "float" | "float32" => conv!(f32, 4),
        _ => conv!(f64, 8),
    })
}

/// Propriété d'un élément : scalaire (type) ou liste (type du compte, type des valeurs).
#[derive(Debug, Clone)]
enum Prop {
    Scalar { name: String, ty: String },
    List { name: String, count: String, item: String },
}

#[derive(Debug, Clone)]
struct Element {
    name: String,
    count: usize,
    props: Vec<Prop>,
}

/// Lit un PLY dont les faces sont des triangles (propriété `vertex_indices` ou `vertex_index`).
pub fn read_ply(path: impl AsRef<Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let mut r = BufReader::new(std::fs::File::open(path)?);
    let mut line = String::new();
    r.read_line(&mut line)?;
    if line.trim() != "ply" {
        return format_err(format!("{} : en-tête PLY absent", path.display()));
    }
    let mut enc = Encoding::Ascii;
    let mut elements: Vec<Element> = Vec::new();
    loop {
        line.clear();
        if r.read_line(&mut line)? == 0 {
            return format_err("PLY : fin de fichier dans l'en-tête");
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        match t.as_slice() {
            ["format", "ascii", ..] => enc = Encoding::Ascii,
            ["format", "binary_little_endian", ..] => enc = Encoding::LittleEndian,
            ["format", "binary_big_endian", ..] => enc = Encoding::BigEndian,
            ["element", name, count] => elements.push(Element {
                name: name.to_string(),
                count: count
                    .parse()
                    .map_err(|_| IoError::Format("PLY : nombre d'éléments invalide".into()))?,
                props: Vec::new(),
            }),
            ["property", "list", count, item, name] => {
                if let Some(e) = elements.last_mut() {
                    e.props.push(Prop::List {
                        name: name.to_string(),
                        count: count.to_string(),
                        item: item.to_string(),
                    });
                }
            }
            ["property", ty, name] => {
                if let Some(e) = elements.last_mut() {
                    e.props.push(Prop::Scalar {
                        name: name.to_string(),
                        ty: ty.to_string(),
                    });
                }
            }
            ["end_header"] => break,
            _ => {}
        }
    }

    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    let mut ascii_tokens: Vec<String> = Vec::new();
    if enc == Encoding::Ascii {
        let mut rest = String::new();
        r.read_to_string(&mut rest)?;
        ascii_tokens = rest.split_whitespace().rev().map(String::from).collect();
    }
    let mut next = |ty: &str, r: &mut BufReader<std::fs::File>| -> Result<f64> {
        if enc == Encoding::Ascii {
            ascii_tokens
                .pop()
                .ok_or_else(|| IoError::Format("PLY : fin de fichier prématurée".into()))?
                .parse::<f64>()
                .map_err(|_| IoError::Format("PLY : nombre invalide".into()))
        } else {
            read_scalar(r, ty, enc)
        }
    };
    for e in &elements {
        for _ in 0..e.count {
            let mut p = [0.0; 3];
            let mut face: Option<Vec<u32>> = None;
            for prop in &e.props {
                match prop {
                    Prop::Scalar { name, ty } => {
                        let v = next(ty, &mut r)?;
                        match name.as_str() {
                            "x" => p[0] = v,
                            "y" => p[1] = v,
                            "z" => p[2] = v,
                            _ => {}
                        }
                    }
                    Prop::List { name, count, item } => {
                        let n = next(count, &mut r)? as usize;
                        let vals: Vec<u32> = (0..n)
                            .map(|_| next(item, &mut r).map(|v| v as u32))
                            .collect::<Result<_>>()?;
                        if name == "vertex_indices" || name == "vertex_index" {
                            face = Some(vals);
                        }
                    }
                }
            }
            match e.name.as_str() {
                "vertex" => vertices.push(p),
                "face" => {
                    let f = face.ok_or_else(|| IoError::Format("PLY : face sans vertex_indices".into()))?;
                    if f.len() != 3 {
                        return format_err(format!(
                            "{} : face à {} sommets (triangles attendus)",
                            path.display(),
                            f.len()
                        ));
                    }
                    faces.push([f[0], f[1], f[2]]);
                }
                _ => {}
            }
        }
    }
    build_mesh(vertices, faces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_same, tmp};
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn aller_retour_binaire() {
        let s = icosphere(2, 10.0);
        let p = tmp("s.ply");
        write_ply(&p, &s).unwrap();
        assert_same(&s, &read_ply(&p).unwrap(), 0.0);
    }

    #[test]
    fn lecture_ascii() {
        let p = tmp("a.ply");
        std::fs::write(
            &p,
            "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 255\n1 0 0 255\n0 1 0 255\n3 0 1 2\n",
        )
        .unwrap();
        let m = read_ply(&p).unwrap();
        assert_eq!(m.faces(), &[[0, 1, 2]]);
        assert_eq!(m.vertices()[1], [1.0, 0.0, 0.0]);
    }
}
