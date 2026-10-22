//! Contrôle des maillages d'une session : `<préfixe>_desc-surfqc.tsv`, une ligne par surface (contrat de
//! données de guioxide, `docs/donnees.md`, section 3).
//!
//! Colonnes : `surface` (`hemi-L_white`…), `n_vertices`, `n_faces`, `euler` (caractéristique d'Euler, 2 pour
//! une sphère), `self_intersections` (paires de faces qui se coupent), `area_mm2`, `gi` (indice de
//! gyrification : aire de la surface / aire de son enveloppe convexe), `mean_dist_to_seg_mm` (distance
//! moyenne des sommets à la frontière de la segmentation). Séparateur tabulation, décimal `.`, `n/a` pour une
//! valeur absente ; un `.json` voisin décrit les colonnes.

use std::path::Path;

use crate::{IoError, Result};

/// En-tête du TSV, dans l'ordre du contrat.
pub const COLUMNS: [&str; 8] = [
    "surface",
    "n_vertices",
    "n_faces",
    "euler",
    "self_intersections",
    "area_mm2",
    "gi",
    "mean_dist_to_seg_mm",
];

/// Une ligne du contrôle : une surface. Les grandeurs absentes valent `None` (`n/a`).
#[derive(Debug, Clone, PartialEq)]
pub struct SurfQc {
    /// Nom de la surface : `hemi-{L,R}_{white,pial}`.
    pub surface: String,
    pub n_vertices: Option<usize>,
    pub n_faces: Option<usize>,
    pub euler: Option<i64>,
    pub self_intersections: Option<usize>,
    pub area_mm2: Option<f64>,
    pub gi: Option<f64>,
    pub mean_dist_to_seg_mm: Option<f64>,
}

fn cell<T: ToString>(v: &Option<T>) -> String {
    v.as_ref().map_or("n/a".to_string(), T::to_string)
}

fn cell_f(v: Option<f64>, digits: usize) -> String {
    v.filter(|x| x.is_finite())
        .map_or("n/a".to_string(), |x| format!("{x:.digits$}"))
}

/// Écrit le TSV et son `.json` voisin (même nom, extension `.json`).
pub fn write_surfqc(path: impl AsRef<Path>, rows: &[SurfQc]) -> Result<()> {
    let path = path.as_ref();
    let mut text = COLUMNS.join("\t") + "\n";
    for r in rows {
        let line = [
            r.surface.clone(),
            cell(&r.n_vertices),
            cell(&r.n_faces),
            cell(&r.euler),
            cell(&r.self_intersections),
            cell_f(r.area_mm2, 2),
            cell_f(r.gi, 4),
            cell_f(r.mean_dist_to_seg_mm, 4),
        ];
        text += &(line.join("\t") + "\n");
    }
    std::fs::write(path, text)?;
    let sidecar = serde_json::json!({
        "surface": {"Description": "Surface décrite par la ligne : hemi-{L,R}_{white,pial}"},
        "n_vertices": {"Description": "Nombre de sommets"},
        "n_faces": {"Description": "Nombre de faces (triangles)"},
        "euler": {"Description": "Caractéristique d'Euler V − E + F (2 pour une surface fermée de genre 0)"},
        "self_intersections": {"Description": "Paires de faces qui se coupent (tests exacts)"},
        "area_mm2": {"Description": "Aire de la surface", "Units": "mm^2"},
        "gi": {"Description": "Indice de gyrification : aire de la surface / aire de son enveloppe convexe"},
        "mean_dist_to_seg_mm": {
            "Description": "Distance moyenne des sommets à la frontière voxelique de la segmentation : substance blanche pour white ; substance blanche et cortex, sillons enfouis ouverts par leur nappe médiane, pour pial",
            "Units": "mm"
        }
    });
    let text = serde_json::to_string_pretty(&sidecar).map_err(|e| IoError::Format(e.to_string()))?;
    std::fs::write(path.with_extension("json"), text + "\n")?;
    Ok(())
}

fn parse<T: std::str::FromStr>(s: &str, column: &str, line: usize) -> Result<Option<T>> {
    if s == "n/a" {
        return Ok(None);
    }
    s.parse().map(Some).map_err(|_| {
        IoError::Format(format!(
            "surfqc, ligne {line}, colonne {column} : valeur invalide « {s} »"
        ))
    })
}

/// Lit un `desc-surfqc.tsv`. Les colonnes sont repérées par leur nom (l'ordre peut varier, une colonne
/// inconnue est ignorée) ; une colonne manquante donne `None` sur toutes les lignes.
pub fn read_surfqc(path: impl AsRef<Path>) -> Result<Vec<SurfQc>> {
    let text = std::fs::read_to_string(path.as_ref())?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines
        .next()
        .ok_or_else(|| IoError::Format("surfqc : fichier vide".into()))?
        .split('\t')
        .map(str::trim)
        .collect();
    let col = |name: &str| header.iter().position(|&h| h == name);
    let surface = col("surface").ok_or_else(|| IoError::Format("surfqc : colonne « surface » absente".into()))?;
    let mut rows = Vec::new();
    for (k, line) in lines.enumerate() {
        let n = k + 2;
        let cells: Vec<&str> = line.split('\t').map(str::trim).collect();
        let get = |name: &str| col(name).and_then(|i| cells.get(i).copied()).unwrap_or("n/a");
        rows.push(SurfQc {
            surface: cells.get(surface).copied().unwrap_or_default().to_string(),
            n_vertices: parse(get("n_vertices"), "n_vertices", n)?,
            n_faces: parse(get("n_faces"), "n_faces", n)?,
            euler: parse(get("euler"), "euler", n)?,
            self_intersections: parse(get("self_intersections"), "self_intersections", n)?,
            area_mm2: parse(get("area_mm2"), "area_mm2", n)?,
            gi: parse(get("gi"), "gi", n)?,
            mean_dist_to_seg_mm: parse(get("mean_dist_to_seg_mm"), "mean_dist_to_seg_mm", n)?,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::tmp;

    #[test]
    fn aller_retour_avec_valeurs_absentes() {
        let rows = vec![
            SurfQc {
                surface: "hemi-L_white".into(),
                n_vertices: Some(1000),
                n_faces: Some(1996),
                euler: Some(2),
                self_intersections: Some(0),
                area_mm2: Some(4321.25),
                gi: Some(1.5432),
                mean_dist_to_seg_mm: Some(0.1234),
            },
            SurfQc {
                surface: "hemi-L_pial".into(),
                n_vertices: Some(1000),
                n_faces: Some(1996),
                euler: Some(2),
                self_intersections: Some(0),
                area_mm2: Some(5000.0),
                gi: None,
                mean_dist_to_seg_mm: None,
            },
        ];
        let p = tmp("sub-01_desc-surfqc.tsv");
        write_surfqc(&p, &rows).unwrap();
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with(
            "surface\tn_vertices\tn_faces\teuler\tself_intersections\tarea_mm2\tgi\tmean_dist_to_seg_mm\n"
        ));
        assert!(text.contains("hemi-L_pial\t1000\t1996\t2\t0\t5000.00\tn/a\tn/a"));
        assert_eq!(read_surfqc(&p).unwrap(), rows);
        let side: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(p.with_extension("json")).unwrap()).unwrap();
        assert_eq!(side["area_mm2"]["Units"], "mm^2");
    }

    #[test]
    fn colonnes_reperees_par_leur_nom() {
        let p = tmp("ordre-surfqc.tsv");
        std::fs::write(&p, "gi\tsurface\tautre\neuler_absent\n").unwrap();
        assert!(read_surfqc(&p).is_err());
        std::fs::write(&p, "gi\tsurface\tautre\n1.25\themi-R_white\tx\n").unwrap();
        let r = read_surfqc(&p).unwrap();
        assert_eq!(r[0].surface, "hemi-R_white");
        assert_eq!(r[0].gi, Some(1.25));
        assert_eq!(r[0].n_vertices, None);
    }
}
