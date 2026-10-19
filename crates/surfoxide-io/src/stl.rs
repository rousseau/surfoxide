//! STL : écriture binaire, lecture binaire et ASCII.
//!
//! STL stocke chaque triangle indépendamment : à la lecture, les sommets de positions identiques (au bit
//! près, en `f32`) sont fusionnés pour retrouver un maillage indexé.

use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use surfoxide_core::{vec3, TriMesh};

use crate::{build_mesh, format_err, Result};

/// Écrit `mesh` en STL binaire (positions en `f32`).
pub fn write_stl(path: impl AsRef<Path>, mesh: &TriMesh) -> Result<()> {
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut header = [0u8; 80];
    header[..9].copy_from_slice(b"surfoxide");
    w.write_all(&header)?;
    w.write_all(&(mesh.n_faces() as u32).to_le_bytes())?;
    let v = mesh.vertices();
    for f in mesh.faces() {
        let [a, b, c] = f.map(|i| v[i as usize]);
        let n = vec3::normalize(vec3::triangle_normal(a, b, c));
        for p in [n, a, b, c] {
            for x in p {
                w.write_all(&(x as f32).to_le_bytes())?;
            }
        }
        w.write_all(&[0u8; 2])?;
    }
    w.flush()?;
    Ok(())
}

/// Lit un STL binaire ou ASCII et fusionne les sommets confondus.
pub fn read_stl(path: impl AsRef<Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path)?;
    let mut triangles: Vec<[[f32; 3]; 3]> = Vec::new();
    let binary_len = |n: usize| 84 + 50 * n;
    let is_binary = bytes.len() >= 84 && {
        let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        bytes.len() == binary_len(n)
    };
    if is_binary {
        let n = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        for t in 0..n {
            let base = 84 + 50 * t + 12; // saute la normale
            let f = |k: usize| f32::from_le_bytes(bytes[base + 4 * k..base + 4 * k + 4].try_into().unwrap());
            triangles.push(std::array::from_fn(|v| std::array::from_fn(|c| f(3 * v + c))));
        }
    } else {
        let text = String::from_utf8_lossy(&bytes);
        if !text.trim_start().starts_with("solid") {
            return format_err(format!("{} : ni STL binaire ni STL ASCII", path.display()));
        }
        let mut current: Vec<[f32; 3]> = Vec::new();
        for line in text.lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.first() == Some(&"vertex") && t.len() == 4 {
                let p: Vec<f32> = t[1..].iter().filter_map(|s| s.parse().ok()).collect();
                if p.len() != 3 {
                    return format_err(format!("{} : sommet invalide", path.display()));
                }
                current.push([p[0], p[1], p[2]]);
                if current.len() == 3 {
                    triangles.push([current[0], current[1], current[2]]);
                    current.clear();
                }
            }
        }
    }
    let mut index: HashMap<[u32; 3], u32> = HashMap::new();
    let mut vertices = Vec::new();
    let mut faces = Vec::with_capacity(triangles.len());
    for t in &triangles {
        let f = t.map(|p| {
            *index.entry(p.map(f32::to_bits)).or_insert_with(|| {
                vertices.push(p.map(f64::from));
                (vertices.len() - 1) as u32
            })
        });
        faces.push(f);
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
        let p = tmp("s.stl");
        write_stl(&p, &s).unwrap();
        let back = read_stl(&p).unwrap();
        // La numérotation des sommets suit l'ordre d'apparition dans les faces : on compare la topologie
        // et la géométrie globale.
        assert_eq!(back.n_vertices(), s.n_vertices());
        assert_eq!(back.n_faces(), s.n_faces());
        assert!((back.signed_volume() - s.signed_volume()).abs() < 1e-3);
        let r = surfoxide_core::topology_report(&back);
        assert!(r.is_sphere_like());
        let _ = assert_same; // ordre des sommets non conservé en STL
    }

    #[test]
    fn lecture_ascii() {
        let p = tmp("a.stl");
        std::fs::write(
            &p,
            "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n",
        )
        .unwrap();
        let m = read_stl(&p).unwrap();
        assert_eq!(m.n_faces(), 1);
        assert_eq!(m.n_vertices(), 3);
    }
}
