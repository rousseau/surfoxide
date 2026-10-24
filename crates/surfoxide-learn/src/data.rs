//! Jeu d'entraînement : `index.tsv` et, par exemple, `<préfixe>_{classes,sdf}.nii.gz` et
//! `<préfixe>_{white,pial}.surf.gii`.

use std::path::{Path, PathBuf};

use surfoxide_core::{TriMesh, Volume};

/// Une ligne de l'index.
#[derive(Debug, Clone)]
pub struct Entry {
    pub id: String,
    pub split: String,
    pub ga_weeks: f64,
    /// Préfixe des fichiers de l'exemple, absolu.
    pub prefix: PathBuf,
}

/// Lit `<racine>/index.tsv`.
pub fn read_index(root: &Path) -> Result<Vec<Entry>, String> {
    let path = root.join("index.tsv");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().ok_or("index vide")?.split('\t').collect();
    let col = |name: &str| {
        header
            .iter()
            .position(|&h| h == name)
            .ok_or(format!("colonne « {name} » absente"))
    };
    let (id, split, ga, prefix) = (col("id")?, col("split")?, col("ga_weeks")?, col("prefix")?);
    Ok(lines
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            Entry {
                id: c[id].to_string(),
                split: c[split].to_string(),
                ga_weeks: c[ga].parse().unwrap_or(f64::NAN),
                prefix: root.join(c[prefix]),
            }
        })
        .collect())
}

/// Un exemple chargé.
pub struct Example {
    pub entry: Entry,
    /// Classes de tissu (0 fond, 1 cortex, 2 substance blanche, 3 nappe médiane) sur le cadrage.
    pub classes: Volume<u8>,
    /// Distance signée (mm) à la frontière des tissus à sillons ouverts, négative dedans.
    pub sdf: Volume<f64>,
    pub white: TriMesh,
    pub pial: TriMesh,
}

/// Nombre de classes de tissu.
pub const N_CLASSES: usize = 4;

pub fn load(entry: &Entry) -> Result<Example, String> {
    let p = entry.prefix.display();
    let path = format!("{p}_classes.nii.gz");
    let (labels, _) = surfoxide_io::nifti::read_labels(&path).map_err(|e| format!("{path}: {e}"))?;
    let classes = labels.map(|&c| c.clamp(0, N_CLASSES as i32 - 1) as u8);
    let path = format!("{p}_sdf.nii.gz");
    let (sdf, _) = surfoxide_io::nifti::read_scalar(&path).map_err(|e| format!("{path}: {e}"))?;
    let read = |name: &str| {
        let path = format!("{p}_{name}.surf.gii");
        surfoxide_io::gifti::read_surface(&path)
            .map(|(m, _)| m)
            .map_err(|e| format!("{path}: {e}"))
    };
    Ok(Example {
        entry: entry.clone(),
        classes,
        sdf: sdf.map(|&x| x as f64),
        white: read("white")?,
        pial: read("pial")?,
    })
}
