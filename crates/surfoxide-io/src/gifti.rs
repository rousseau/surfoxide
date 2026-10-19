//! GIfTI (`.surf.gii`, `.shape.gii`, `.func.gii`).
//!
//! Lecture : encodages ASCII, Base64Binary et GZipBase64Binary (zlib ou gzip), types FLOAT32, FLOAT64,
//! INT32, UINT32, INT16, UINT8, ordres ligne et colonne, petit et grand boutisme. Les fichiers de données
//! externes (`ExternalFileName`) ne sont pas pris en charge.
//!
//! Écriture : GZipBase64Binary (zlib, comme nibabel et Connectome Workbench), petit boutisme, ordre ligne.
//! Les coordonnées sont celles du repère monde (RAS, mm) : la matrice `CoordinateSystemTransformMatrix`
//! est l'identité, de NIFTI_XFORM_SCANNER_ANAT vers lui-même. L'affine de l'image source est conservée
//! dans les métadonnées du fichier (`SourceAffine`, seize nombres par lignes).

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use flate2::read::{GzDecoder, ZlibDecoder};
use flate2::write::ZlibEncoder;
use flate2::Compression;
use quick_xml::events::Event;
use quick_xml::Reader;
use surfoxide_core::{Affine, TriMesh, VertexField};

use crate::{build_mesh, format_err, IoError, Result};

/// Métadonnées d'une surface GIfTI.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SurfaceMeta {
    /// `AnatomicalStructurePrimary` : `CortexLeft`, `CortexRight`…
    pub primary: Option<String>,
    /// `AnatomicalStructureSecondary` : `GrayWhite`, `Pial`, `MidThickness`…
    pub secondary: Option<String>,
    /// `GeometricType` : `Anatomical`, `Inflated`, `Spherical`…
    pub geometric_type: Option<String>,
    /// Affine voxel → monde de l'image dont la surface est issue.
    pub source_affine: Option<Affine>,
    /// Autres métadonnées du fichier (nom, valeur).
    pub other: BTreeMap<String, String>,
}

impl SurfaceMeta {
    /// Métadonnées d'une surface d'hémisphère : `left` vrai pour la gauche ; `secondary` comme `GrayWhite`.
    pub fn hemisphere(left: bool, secondary: &str) -> Self {
        Self {
            primary: Some(if left { "CortexLeft" } else { "CortexRight" }.into()),
            secondary: Some(secondary.into()),
            geometric_type: Some("Anatomical".into()),
            ..Default::default()
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Lecture

/// Un `DataArray` décodé.
#[derive(Debug, Clone)]
struct DataArray {
    intent: String,
    dims: Vec<usize>,
    meta: Vec<(String, String)>,
    /// Valeurs en ordre ligne (dernier indice le plus rapide).
    values: Vec<f64>,
}

#[derive(Debug, Default)]
struct Document {
    meta: Vec<(String, String)>,
    arrays: Vec<DataArray>,
}

fn attr_map(e: &quick_xml::events::BytesStart) -> Result<BTreeMap<String, String>> {
    let mut m = BTreeMap::new();
    for a in e.attributes() {
        let a = a.map_err(|e| IoError::Format(format!("GIfTI : attribut illisible : {e}")))?;
        let key = String::from_utf8_lossy(a.key.as_ref()).into_owned();
        let value = a
            .unescape_value()
            .map_err(|e| IoError::Format(format!("GIfTI : attribut illisible : {e}")))?
            .into_owned();
        m.insert(key, value);
    }
    Ok(m)
}

fn parse_document(xml: &str) -> Result<Document> {
    let mut reader = Reader::from_str(xml);
    let mut doc = Document::default();
    let mut stack: Vec<String> = Vec::new();
    let mut text = String::new();
    let mut attrs: BTreeMap<String, String> = BTreeMap::new();
    let mut da_meta: Vec<(String, String)> = Vec::new();
    let (mut md_name, mut md_value) = (String::new(), String::new());
    loop {
        let ev = reader
            .read_event()
            .map_err(|e| IoError::Format(format!("GIfTI : XML invalide : {e}")))?;
        match ev {
            Event::Start(e) => {
                let name = String::from_utf8_lossy(e.name().as_ref()).into_owned();
                if name == "DataArray" {
                    attrs = attr_map(&e)?;
                    da_meta.clear();
                }
                text.clear();
                stack.push(name);
            }
            Event::Empty(e) => {
                // <Data/> vide ou <DataArray .../> sans contenu.
                if e.name().as_ref() == b"DataArray" {
                    return format_err("GIfTI : DataArray sans données");
                }
            }
            Event::Text(t) => {
                let s = t
                    .decode()
                    .map_err(|e| IoError::Format(format!("GIfTI : texte illisible : {e}")))?;
                text.push_str(&s);
            }
            Event::CData(t) => {
                let s = t
                    .decode()
                    .map_err(|e| IoError::Format(format!("GIfTI : texte illisible : {e}")))?;
                text.push_str(&s);
            }
            Event::GeneralRef(r) => {
                let ent = String::from_utf8_lossy(r.as_ref()).into_owned();
                text.push_str(match ent.as_str() {
                    "amp" => "&",
                    "lt" => "<",
                    "gt" => ">",
                    "quot" => "\"",
                    "apos" => "'",
                    _ => "",
                });
            }
            Event::End(_) => {
                let name = stack.pop().unwrap_or_default();
                let in_da = stack.iter().any(|s| s == "DataArray");
                match name.as_str() {
                    "Name" => md_name = text.trim().to_string(),
                    "Value" => md_value = text.trim().to_string(),
                    "MD" => {
                        let pair = (std::mem::take(&mut md_name), std::mem::take(&mut md_value));
                        if in_da {
                            da_meta.push(pair);
                        } else {
                            doc.meta.push(pair);
                        }
                    }
                    "Data" => {
                        let values = decode_data(&attrs, &text)?;
                        let dims = array_dims(&attrs)?;
                        let n: usize = dims.iter().product();
                        if values.len() != n {
                            return format_err(format!(
                                "GIfTI : {} valeurs pour des dimensions {dims:?}",
                                values.len()
                            ));
                        }
                        let values = if attrs.get("ArrayIndexingOrder").map(String::as_str) == Some("ColumnMajorOrder")
                            && dims.len() == 2
                        {
                            transpose(&values, dims[0], dims[1])
                        } else {
                            values
                        };
                        doc.arrays.push(DataArray {
                            intent: attrs.get("Intent").cloned().unwrap_or_default(),
                            dims,
                            meta: da_meta.clone(),
                            values,
                        });
                    }
                    _ => {}
                }
                text.clear();
            }
            Event::Eof => break,
            _ => {}
        }
    }
    Ok(doc)
}

fn array_dims(attrs: &BTreeMap<String, String>) -> Result<Vec<usize>> {
    let nd: usize = attrs
        .get("Dimensionality")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| IoError::Format("GIfTI : Dimensionality manquant".into()))?;
    (0..nd)
        .map(|i| {
            attrs
                .get(&format!("Dim{i}"))
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| IoError::Format(format!("GIfTI : Dim{i} manquant")))
        })
        .collect()
}

/// Passe des valeurs en ordre colonne (`[r, c]` rangé avec `r` le plus rapide) à l'ordre ligne.
fn transpose(v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = v[c * rows + r];
        }
    }
    out
}

fn decode_data(attrs: &BTreeMap<String, String>, text: &str) -> Result<Vec<f64>> {
    if attrs.get("ExternalFileName").is_some_and(|s| !s.trim().is_empty()) {
        return format_err("GIfTI : fichier de données externe non pris en charge");
    }
    let dtype = attrs.get("DataType").map(String::as_str).unwrap_or("");
    let encoding = attrs.get("Encoding").map(String::as_str).unwrap_or("ASCII");
    if encoding == "ASCII" {
        return text
            .split_ascii_whitespace()
            .map(|t| {
                t.parse::<f64>()
                    .map_err(|_| IoError::Format(format!("GIfTI : nombre ASCII invalide « {t} »")))
            })
            .collect();
    }
    let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    let raw = BASE64
        .decode(compact.as_bytes())
        .map_err(|e| IoError::Format(format!("GIfTI : base64 invalide : {e}")))?;
    let bytes = match encoding {
        "Base64Binary" => raw,
        "GZipBase64Binary" => {
            let mut out = Vec::new();
            // La norme dit « gzip », mais la plupart des outils écrivent un flux zlib.
            if raw.starts_with(&[0x1f, 0x8b]) {
                GzDecoder::new(&raw[..]).read_to_end(&mut out)?;
            } else {
                ZlibDecoder::new(&raw[..]).read_to_end(&mut out)?;
            }
            out
        }
        other => return format_err(format!("GIfTI : encodage « {other} » non pris en charge")),
    };
    let big = attrs.get("Endian").map(String::as_str) == Some("BigEndian");
    macro_rules! conv {
        ($t:ty, $n:expr) => {
            bytes
                .chunks_exact($n)
                .map(|c| {
                    let a: [u8; $n] = c.try_into().unwrap();
                    (if big {
                        <$t>::from_be_bytes(a)
                    } else {
                        <$t>::from_le_bytes(a)
                    }) as f64
                })
                .collect()
        };
    }
    Ok(match dtype {
        "NIFTI_TYPE_FLOAT32" => conv!(f32, 4),
        "NIFTI_TYPE_FLOAT64" => conv!(f64, 8),
        "NIFTI_TYPE_INT32" => conv!(i32, 4),
        "NIFTI_TYPE_UINT32" => conv!(u32, 4),
        "NIFTI_TYPE_INT16" => conv!(i16, 2),
        "NIFTI_TYPE_UINT8" => bytes.iter().map(|&b| b as f64).collect(),
        other => return format_err(format!("GIfTI : type « {other} » non pris en charge")),
    })
}

fn read_document(path: &Path) -> Result<Document> {
    let xml = std::fs::read_to_string(path)?;
    parse_document(&xml).map_err(|e| match e {
        IoError::Format(m) => IoError::Format(format!("{} : {m}", path.display())),
        other => other,
    })
}

fn parse_affine(s: &str) -> Option<Affine> {
    let v: Vec<f64> = s.split_ascii_whitespace().filter_map(|t| t.parse().ok()).collect();
    (v.len() == 16).then(|| std::array::from_fn(|r| std::array::from_fn(|c| v[4 * r + c])))
}

/// Lit une surface : le premier tableau POINTSET et le premier tableau TRIANGLE.
pub fn read_surface(path: impl AsRef<Path>) -> Result<(TriMesh, SurfaceMeta)> {
    let path = path.as_ref();
    let doc = read_document(path)?;
    let points = doc
        .arrays
        .iter()
        .find(|a| a.intent == "NIFTI_INTENT_POINTSET")
        .ok_or_else(|| IoError::Format(format!("{} : pas de POINTSET", path.display())))?;
    let tris = doc
        .arrays
        .iter()
        .find(|a| a.intent == "NIFTI_INTENT_TRIANGLE")
        .ok_or_else(|| IoError::Format(format!("{} : pas de TRIANGLE", path.display())))?;
    if points.dims.get(1) != Some(&3) || tris.dims.get(1) != Some(&3) {
        return format_err(format!("{} : tableaux de largeur 3 attendus", path.display()));
    }
    let vertices = points
        .values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let faces = tris
        .values
        .as_chunks::<3>()
        .0
        .iter()
        .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32])
        .collect();
    let mesh = build_mesh(vertices, faces)?;

    let get = |key: &str| {
        points
            .meta
            .iter()
            .chain(&doc.meta)
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    };
    let mut meta = SurfaceMeta {
        primary: get("AnatomicalStructurePrimary"),
        secondary: get("AnatomicalStructureSecondary"),
        geometric_type: get("GeometricType"),
        source_affine: get("SourceAffine").as_deref().and_then(parse_affine),
        other: BTreeMap::new(),
    };
    for (k, v) in &doc.meta {
        if !matches!(
            k.as_str(),
            "AnatomicalStructurePrimary" | "AnatomicalStructureSecondary" | "GeometricType" | "SourceAffine"
        ) {
            meta.other.insert(k.clone(), v.clone());
        }
    }
    Ok((mesh, meta))
}

/// Lit les champs par sommet : tous les tableaux à une dimension (ou `[N, 1]`) qui ne sont ni des
/// sommets ni des faces. Le nom vient de la métadonnée `Name` du tableau, à défaut `#i`.
pub fn read_fields(path: impl AsRef<Path>) -> Result<Vec<VertexField>> {
    let doc = read_document(path.as_ref())?;
    Ok(doc
        .arrays
        .into_iter()
        .enumerate()
        .filter(|(_, a)| {
            a.intent != "NIFTI_INTENT_POINTSET"
                && a.intent != "NIFTI_INTENT_TRIANGLE"
                && (a.dims.len() == 1 || a.dims.get(1) == Some(&1))
        })
        .map(|(i, a)| {
            let name = a
                .meta
                .iter()
                .find(|(k, _)| k == "Name")
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| format!("#{i}"));
            VertexField::new(name, a.values)
        })
        .collect())
}

// ------------------------------------------------------------------------------------------------
// Écriture

fn escape(s: &str) -> String {
    // Le contenu va dans une section CDATA : seule la séquence de fin doit être coupée.
    s.replace("]]>", "]]]]><![CDATA[>")
}

fn write_meta(out: &mut String, indent: &str, meta: &[(String, String)]) {
    if meta.is_empty() {
        out.push_str(&format!("{indent}<MetaData/>\n"));
        return;
    }
    out.push_str(&format!("{indent}<MetaData>\n"));
    for (k, v) in meta {
        out.push_str(&format!(
            "{indent}   <MD>\n{indent}      <Name><![CDATA[{}]]></Name>\n{indent}      <Value><![CDATA[{}]]></Value>\n{indent}   </MD>\n",
            escape(k),
            escape(v)
        ));
    }
    out.push_str(&format!("{indent}</MetaData>\n"));
}

fn encode(bytes: &[u8]) -> Result<String> {
    let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
    z.write_all(bytes)?;
    Ok(BASE64.encode(z.finish()?))
}

enum Payload<'a> {
    F32(&'a [f32]),
    I32(&'a [i32]),
}

fn write_array(
    out: &mut String,
    intent: &str,
    dims: &[usize],
    meta: &[(String, String)],
    payload: Payload,
    world_transform: bool,
) -> Result<()> {
    let (dtype, bytes): (&str, Vec<u8>) = match payload {
        Payload::F32(v) => ("NIFTI_TYPE_FLOAT32", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
        Payload::I32(v) => ("NIFTI_TYPE_INT32", v.iter().flat_map(|x| x.to_le_bytes()).collect()),
    };
    let dim_attrs: String = dims
        .iter()
        .enumerate()
        .map(|(i, d)| format!(" Dim{i}=\"{d}\""))
        .collect();
    out.push_str(&format!(
        "   <DataArray Intent=\"{intent}\" DataType=\"{dtype}\" ArrayIndexingOrder=\"RowMajorOrder\" Dimensionality=\"{}\"{dim_attrs} Encoding=\"GZipBase64Binary\" Endian=\"LittleEndian\" ExternalFileName=\"\" ExternalFileOffset=\"0\">\n",
        dims.len()
    ));
    write_meta(out, "      ", meta);
    if world_transform {
        out.push_str(
            "      <CoordinateSystemTransformMatrix>\n         <DataSpace><![CDATA[NIFTI_XFORM_SCANNER_ANAT]]></DataSpace>\n         <TransformedSpace><![CDATA[NIFTI_XFORM_SCANNER_ANAT]]></TransformedSpace>\n         <MatrixData>1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</MatrixData>\n      </CoordinateSystemTransformMatrix>\n",
        );
    }
    out.push_str(&format!("      <Data>{}</Data>\n", encode(&bytes)?));
    out.push_str("   </DataArray>\n");
    Ok(())
}

fn header(n_arrays: usize, meta: &[(String, String)]) -> String {
    let mut out = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!DOCTYPE GIFTI SYSTEM \"http://www.nitrc.org/frs/download.php/115/gifti.dtd\">\n<GIFTI Version=\"1.0\" NumberOfDataArrays=\"{n_arrays}\">\n"
    );
    write_meta(&mut out, "   ", meta);
    out.push_str("   <LabelTable/>\n");
    out
}

fn file_meta(meta: &SurfaceMeta) -> Vec<(String, String)> {
    let mut m: Vec<(String, String)> = Vec::new();
    if let Some(a) = &meta.source_affine {
        let s: Vec<String> = a.iter().flatten().map(|x| format!("{x}")).collect();
        m.push(("SourceAffine".into(), s.join(" ")));
    }
    m.extend(meta.other.iter().map(|(k, v)| (k.clone(), v.clone())));
    m
}

/// Écrit une surface (sommets en `f32`, faces en `i32`).
pub fn write_surface(path: impl AsRef<Path>, mesh: &TriMesh, meta: &SurfaceMeta) -> Result<()> {
    let mut point_meta: Vec<(String, String)> = Vec::new();
    for (k, v) in [
        ("AnatomicalStructurePrimary", &meta.primary),
        ("AnatomicalStructureSecondary", &meta.secondary),
        ("GeometricType", &meta.geometric_type),
    ] {
        if let Some(v) = v {
            point_meta.push((k.into(), v.clone()));
        }
    }
    let mut out = header(2, &file_meta(meta));
    let coords: Vec<f32> = mesh.vertices().iter().flatten().map(|&x| x as f32).collect();
    write_array(
        &mut out,
        "NIFTI_INTENT_POINTSET",
        &[mesh.n_vertices(), 3],
        &point_meta,
        Payload::F32(&coords),
        true,
    )?;
    let tris: Vec<i32> = mesh.faces().iter().flatten().map(|&i| i as i32).collect();
    write_array(
        &mut out,
        "NIFTI_INTENT_TRIANGLE",
        &[mesh.n_faces(), 3],
        &[],
        Payload::I32(&tris),
        false,
    )?;
    out.push_str("</GIFTI>\n");
    std::fs::write(path, out)?;
    Ok(())
}

/// Écrit des champs par sommet (`.shape.gii`), un tableau NIFTI_INTENT_SHAPE par champ.
pub fn write_fields(path: impl AsRef<Path>, fields: &[VertexField], meta: &SurfaceMeta) -> Result<()> {
    let mut out = header(fields.len(), &file_meta(meta));
    for f in fields {
        let mut m = vec![("Name".to_string(), f.name.clone())];
        if let Some(p) = &meta.primary {
            m.push(("AnatomicalStructurePrimary".into(), p.clone()));
        }
        let v: Vec<f32> = f.values.iter().map(|&x| x as f32).collect();
        write_array(&mut out, "NIFTI_INTENT_SHAPE", &[v.len()], &m, Payload::F32(&v), false)?;
    }
    out.push_str("</GIFTI>\n");
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{assert_same, tmp};
    use surfoxide_core::shapes::icosphere;

    #[test]
    fn aller_retour_surface_et_metadonnees() {
        let s = icosphere(2, 30.0);
        let mut meta = SurfaceMeta::hemisphere(true, "GrayWhite");
        meta.source_affine = Some([
            [-0.5, 0.0, 0.0, 40.0],
            [0.0, 0.5, 0.0, -30.0],
            [0.0, 0.0, 0.5, -20.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        meta.other.insert("SourceImage".into(), "sub-01_dseg.nii.gz".into());
        let p = tmp("s.surf.gii");
        write_surface(&p, &s, &meta).unwrap();
        let (back, m) = read_surface(&p).unwrap();
        assert_same(&s, &back, 1e-5);
        assert_eq!(m, meta);
    }

    #[test]
    fn aller_retour_champs() {
        let fields = vec![
            VertexField::new("thickness", vec![1.0, 2.0, 3.5]),
            VertexField::new("sulc", vec![-1.0, 0.0, 0.25]),
        ];
        let p = tmp("f.shape.gii");
        write_fields(&p, &fields, &SurfaceMeta::default()).unwrap();
        assert_eq!(read_fields(&p).unwrap(), fields);
    }

    #[test]
    fn lecture_ascii_ordre_colonne() {
        let xml = r#"<?xml version="1.0"?>
<GIFTI Version="1.0" NumberOfDataArrays="2">
 <DataArray Intent="NIFTI_INTENT_POINTSET" DataType="NIFTI_TYPE_FLOAT32" ArrayIndexingOrder="ColumnMajorOrder"
   Dimensionality="2" Dim0="3" Dim1="3" Encoding="ASCII" Endian="LittleEndian" ExternalFileName="">
  <Data>0 1 0  0 0 1  0 0 0</Data>
 </DataArray>
 <DataArray Intent="NIFTI_INTENT_TRIANGLE" DataType="NIFTI_TYPE_INT32" ArrayIndexingOrder="RowMajorOrder"
   Dimensionality="2" Dim0="1" Dim1="3" Encoding="ASCII" Endian="LittleEndian" ExternalFileName="">
  <Data>0 1 2</Data>
 </DataArray>
</GIFTI>"#;
        let p = tmp("ascii.surf.gii");
        std::fs::write(&p, xml).unwrap();
        let (m, _) = read_surface(&p).unwrap();
        assert_eq!(m.vertices(), &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    }

    /// Surfaces de l'atlas fœtal dHCP (écrites par gifticlib et Workbench).
    #[test]
    #[ignore = "données locales : ~/Exp/gyroxide/data/atlas_dhcp"]
    fn lecture_atlas_dhcp() {
        let home = std::env::var("HOME").unwrap();
        let dir = format!("{home}/Exp/gyroxide/data/atlas_dhcp");
        let (wm, meta) = read_surface(format!("{dir}/fetal.week28.left.wm.surf.gii")).unwrap();
        assert_eq!(wm.n_vertices(), 32492);
        assert_eq!(meta.primary.as_deref(), Some("CortexLeft"));
        let r = surfoxide_core::topology_report(&wm);
        assert!(r.boundary_edges == 0, "{r:?}");
        let sulc = read_fields(format!("{dir}/fetal.week28.left.sulc.shape.gii")).unwrap();
        assert_eq!(sulc[0].values.len(), 32492);
    }
}
