//! Chemins et fichiers des dérivées BIDS, selon le contrat de données de guioxide (`docs/donnees.md`) et les
//! mêmes règles que `medoxide-core::bids`.
//!
//! Les dérivées d'un pipeline sont dans `<étude>/derivatives/<pipeline>/sub-X[/ses-Y]/anat/`, avec un
//! `dataset_description.json` à la racine du pipeline. `<étude>` est la racine de l'étude ([`study_root`]).

use std::path::{Path, PathBuf};

use crate::{IoError, Result};

/// Dossier de dérivées de `surfx run` (surfaces, champs par sommet, contrôle des maillages).
pub const PIPELINE_SURFACES: &str = "surfx-surfaces";
/// Dossier de dérivées de `medx bounti` (segmentation en tissus), entrée de `surfx run`.
pub const PIPELINE_BOUNTI: &str = "medx-bounti";

/// Version de BIDS déclarée dans `dataset_description.json`.
const BIDS_VERSION: &str = "1.9.0";
/// Dépôt du code, déclaré dans `GeneratedBy`.
const CODE_URL: &str = "https://github.com/rousseau/surfoxide";

/// Racine de l'étude, là où vont les `derivatives/` : `output_dir` s'il est donné (`--output-dir`) ; sinon le
/// **parent** de la racine BIDS quand celle-ci s'appelle `sourcedata` (buckets Garage : `sourcedata/` et
/// `derivatives/` côte à côte) ; sinon la racine BIDS elle-même.
pub fn study_root(bids_root: &Path, output_dir: Option<&Path>) -> PathBuf {
    if let Some(dir) = output_dir {
        return dir.to_path_buf();
    }
    if bids_root.file_name().is_some_and(|name| name == "sourcedata") {
        if let Some(parent) = bids_root.parent() {
            return parent.to_path_buf();
        }
    }
    bids_root.to_path_buf()
}

/// Un identifiant BIDS : lettres et chiffres ASCII, non vide, sans préfixe `sub-` ni `ses-` (la valeur entre
/// dans un chemin : ni séparateur ni `..`).
pub fn check_label(value: &str) -> Result<()> {
    if !value.is_empty() && value.chars().all(|c| c.is_ascii_alphanumeric()) {
        Ok(())
    } else {
        Err(IoError::Format(format!(
            "identifiant BIDS invalide « {value} » : lettres et chiffres seulement, sans préfixe sub- ni ses-"
        )))
    }
}

/// Sous-dossier `sub-X[/ses-Y]/anat` et préfixe `sub-X[_ses-Y]` d'un participant et d'une session.
fn subject(participant: &str, session: Option<&str>) -> Result<(PathBuf, String)> {
    check_label(participant)?;
    let mut dir = PathBuf::from(format!("sub-{participant}"));
    let mut prefix = format!("sub-{participant}");
    if let Some(s) = session {
        check_label(s)?;
        dir.push(format!("ses-{s}"));
        prefix = format!("{prefix}_ses-{s}");
    }
    Ok((dir.join("anat"), prefix))
}

/// Où écrit un pipeline pour un participant (et une session).
#[derive(Debug, Clone, PartialEq)]
pub struct DerivativeDir {
    /// Racine du pipeline : `<étude>/derivatives/<pipeline>`. Elle reçoit `dataset_description.json`.
    pub pipeline: PathBuf,
    /// Dossier des fichiers : `<pipeline>/sub-X[/ses-Y]/anat`.
    pub anat: PathBuf,
    /// Début des noms de fichier : `sub-X[_ses-Y]`.
    pub prefix: String,
}

impl DerivativeDir {
    /// Préfixe complet des fichiers : `<anat>/sub-X[_ses-Y]`.
    pub fn file_prefix(&self) -> PathBuf {
        self.anat.join(&self.prefix)
    }
}

/// Dossier de dérivées de `pipeline` pour un participant et une session éventuelle, dans l'étude de
/// [`study_root`]`(bids_root, output_dir)`.
pub fn derivative_dir(
    bids_root: &Path,
    output_dir: Option<&Path>,
    pipeline: &str,
    participant: &str,
    session: Option<&str>,
) -> Result<DerivativeDir> {
    let (relative, prefix) = subject(participant, session)?;
    let pipeline = study_root(bids_root, output_dir).join("derivatives").join(pipeline);
    Ok(DerivativeDir {
        anat: pipeline.join(relative),
        pipeline,
        prefix,
    })
}

/// Segmentation BOUNTI d'une session, écrite par `medx bounti --bids` :
/// `derivatives/medx-bounti/sub-X[/ses-Y]/anat/sub-X[_ses-Y]_desc-bounti_dseg.nii.gz`. Cherchée d'abord sous
/// `output_dir` (une chaîne d'outils lancée avec le même `--output-dir`), puis dans l'étude.
pub fn bounti_segmentation(
    bids_root: &Path,
    output_dir: Option<&Path>,
    participant: &str,
    session: Option<&str>,
) -> Result<PathBuf> {
    let mut tried = Vec::new();
    let outs = if output_dir.is_some() {
        vec![output_dir, None]
    } else {
        vec![None]
    };
    for out in outs {
        let d = derivative_dir(bids_root, out, PIPELINE_BOUNTI, participant, session)?;
        let path = d.anat.join(format!("{}_desc-bounti_dseg.nii.gz", d.prefix));
        if path.is_file() {
            return Ok(path);
        }
        tried.push(path);
    }
    let list: Vec<String> = tried.iter().map(|p| p.display().to_string()).collect();
    Err(IoError::Format(format!(
        "segmentation BOUNTI introuvable : {}",
        list.join(", ")
    )))
}

/// Qui a écrit les fichiers : déclaré dans `GeneratedBy` de `dataset_description.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct Generator {
    /// Nom de l'outil, par exemple `surfx run`.
    pub name: String,
    /// Version de l'outil.
    pub version: String,
    /// Commit du code, s'il est connu.
    pub commit: Option<String>,
    /// Ligne de commande, si elle est connue.
    pub command_line: Option<String>,
}

impl Generator {
    fn to_json(&self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("Name".into(), self.name.clone().into());
        m.insert("Version".into(), self.version.clone().into());
        if let Some(c) = &self.commit {
            m.insert("Commit".into(), c.clone().into());
        }
        m.insert("CodeURL".into(), CODE_URL.into());
        if let Some(l) = &self.command_line {
            m.insert("CommandLine".into(), l.clone().into());
        }
        serde_json::Value::Object(m)
    }
}

/// Écrit `<pipeline>/dataset_description.json` : `DatasetType: derivative`, version de BIDS et `GeneratedBy` ;
/// `pipeline_name` est le nom du jeu (`surfx-surfaces`). Réécrit à chaque lancement (il décrit le dernier).
pub fn write_derivative_description(pipeline: &Path, pipeline_name: &str, generator: &Generator) -> Result<()> {
    let mut m = serde_json::Map::new();
    m.insert("Name".into(), pipeline_name.into());
    m.insert("BIDSVersion".into(), BIDS_VERSION.into());
    m.insert("DatasetType".into(), "derivative".into());
    m.insert(
        "GeneratedBy".into(),
        serde_json::Value::Array(vec![generator.to_json()]),
    );
    std::fs::create_dir_all(pipeline)?;
    let text =
        serde_json::to_string_pretty(&serde_json::Value::Object(m)).map_err(|e| IoError::Format(e.to_string()))?;
    std::fs::write(pipeline.join("dataset_description.json"), text + "\n")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Racine de l'étude : `sourcedata` remonte au parent (Garage), toute autre racine reste elle-même,
    /// `--output-dir` remplace tout.
    #[test]
    fn racine_de_l_etude() {
        assert_eq!(
            study_root(Path::new("/g/fetal/sourcedata"), None),
            Path::new("/g/fetal")
        );
        assert_eq!(
            study_root(Path::new("/g/fetal/sourcedata/"), None),
            Path::new("/g/fetal")
        );
        assert_eq!(
            study_root(Path::new("/g/sourcedata2"), None),
            Path::new("/g/sourcedata2")
        );
        assert_eq!(study_root(Path::new("/g/bids"), None), Path::new("/g/bids"));
        assert_eq!(
            study_root(Path::new("/g/fetal/sourcedata"), Some(Path::new("/tmp/o"))),
            Path::new("/tmp/o")
        );
    }

    /// Chemins de dérivées avec et sans session, et refus des identifiants qui sortiraient de l'arborescence.
    #[test]
    fn chemins_des_derivees() {
        let d = derivative_dir(Path::new("/g/sourcedata"), None, PIPELINE_SURFACES, "0avbc", Some("02")).unwrap();
        assert_eq!(d.pipeline, Path::new("/g/derivatives/surfx-surfaces"));
        assert_eq!(d.anat, Path::new("/g/derivatives/surfx-surfaces/sub-0avbc/ses-02/anat"));
        assert_eq!(
            d.file_prefix(),
            Path::new("/g/derivatives/surfx-surfaces/sub-0avbc/ses-02/anat/sub-0avbc_ses-02")
        );
        let sans = derivative_dir(Path::new("/d"), None, PIPELINE_SURFACES, "01", None).unwrap();
        assert_eq!(sans.anat, Path::new("/d/derivatives/surfx-surfaces/sub-01/anat"));
        assert_eq!(sans.prefix, "sub-01");
        for bad in ["", "sub-01", "../x", "a/b", "é"] {
            assert!(
                derivative_dir(Path::new("/d"), None, PIPELINE_SURFACES, bad, None).is_err(),
                "{bad}"
            );
        }
        assert!(derivative_dir(Path::new("/d"), None, PIPELINE_SURFACES, "01", Some("..")).is_err());
    }

    #[test]
    fn segmentation_cherchee_dans_la_sortie_puis_dans_l_etude() {
        let root = std::env::temp_dir().join(format!("surfoxide-bids-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let bids = root.join("sourcedata");
        let out = root.join("sortie");
        let in_study = root.join("derivatives/medx-bounti/sub-01/ses-1/anat/sub-01_ses-1_desc-bounti_dseg.nii.gz");
        std::fs::create_dir_all(in_study.parent().unwrap()).unwrap();
        std::fs::write(&in_study, b"").unwrap();
        assert_eq!(
            bounti_segmentation(&bids, Some(&out), "01", Some("1")).unwrap(),
            in_study
        );
        let in_out = out.join("derivatives/medx-bounti/sub-01/ses-1/anat/sub-01_ses-1_desc-bounti_dseg.nii.gz");
        std::fs::create_dir_all(in_out.parent().unwrap()).unwrap();
        std::fs::write(&in_out, b"").unwrap();
        assert_eq!(bounti_segmentation(&bids, Some(&out), "01", Some("1")).unwrap(), in_out);
        let err = bounti_segmentation(&bids, None, "02", None).unwrap_err().to_string();
        assert!(err.contains("sub-02_desc-bounti_dseg.nii.gz"), "{err}");
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn description_de_la_derivee() {
        let dir = std::env::temp_dir().join(format!("surfoxide-desc-{}", std::process::id()));
        let g = Generator {
            name: "surfx run".into(),
            version: "0.1.0".into(),
            commit: Some("abc1234".into()),
            command_line: Some("surfx run --bids x".into()),
        };
        write_derivative_description(&dir, PIPELINE_SURFACES, &g).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("dataset_description.json")).unwrap()).unwrap();
        assert_eq!(v["DatasetType"], "derivative");
        assert_eq!(v["Name"], "surfx-surfaces");
        let gb = &v["GeneratedBy"][0];
        assert_eq!(gb["Name"], "surfx run");
        assert_eq!(gb["Commit"], "abc1234");
        assert_eq!(gb["CodeURL"], CODE_URL);
        assert_eq!(gb["CommandLine"], "surfx run --bids x");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
