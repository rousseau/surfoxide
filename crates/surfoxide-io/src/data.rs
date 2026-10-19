//! Emplacement des données réelles rapatriées par `scripts/fetch-data.sh` (tests ignorés par défaut,
//! validation, figures).

use std::path::PathBuf;

/// Dossier des données : `$SURFOXIDE_DATA`, sinon `~/.cache/surfoxide/data`.
pub fn data_dir() -> PathBuf {
    match std::env::var_os("SURFOXIDE_DATA") {
        Some(d) => PathBuf::from(d),
        None => PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
            .join(".cache/surfoxide/data"),
    }
}

/// Segmentation d'une session : `labels` vaut `bounti19` ou `multibounti43`.
pub fn segmentation(sub: &str, ses: &str, labels: &str) -> PathBuf {
    data_dir()
        .join(sub)
        .join(ses)
        .join("anat")
        .join(format!("{sub}_{ses}_res-0p5mm_desc-{labels}_dseg.nii.gz"))
}
