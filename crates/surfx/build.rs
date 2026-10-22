//! Retient le commit du code compilé (variable `SURFX_COMMIT`, lue par `option_env!`), pour le déclarer dans
//! les `dataset_description.json` écrits par `surfx run --bids`. Sans `git` ou hors d'un dépôt (archive des
//! sources), la variable n'est pas définie et le commit est omis.

use std::path::PathBuf;
use std::process::Command;

/// Sortie de `git <arguments>` (texte sans espaces de bordure), si la commande réussit.
fn git(arguments: &[&str]) -> Option<String> {
    let output = Command::new("git").args(arguments).output().ok()?;
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (output.status.success() && !text.is_empty()).then_some(text)
}

fn main() {
    if let Some(commit) = git(&["rev-parse", "--short=12", "HEAD"]) {
        println!("cargo:rustc-env=SURFX_COMMIT={commit}");
    }
    println!("cargo:rerun-if-changed=build.rs");
    // Relancé à chaque nouveau commit : `HEAD` ne change qu'en changeant de branche ; un commit déplace la
    // référence de la branche (ou `packed-refs`).
    if let Some(git_dir) = git(&["rev-parse", "--absolute-git-dir"]).map(PathBuf::from) {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        println!("cargo:rerun-if-changed={}", git_dir.join("packed-refs").display());
        if let Some(reference) = git(&["symbolic-ref", "-q", "HEAD"]) {
            // En worktree, les branches sont dans le dossier commun.
            let common =
                git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]).map_or(git_dir, PathBuf::from);
            println!("cargo:rerun-if-changed={}", common.join(reference).display());
        }
    }
}
