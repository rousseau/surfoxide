//! `surfx --describe` contre la réalité : chaque sortie `required` déclarée est écrite par `surfx run --bids`.
//!
//! Test ignoré par défaut (il lance la chaîne de surfaces, plusieurs minutes) : il lit le jeu BIDS de test dans
//! `OXIDE_JEU_TEST` (le fantôme de guioxide, avec `derivatives/medx-bounti`), participant `OXIDE_JEU_SUB`
//! (défaut `fant01`) et session `OXIDE_JEU_SES` (défaut `01`), et écrit dans un dossier temporaire
//! (`--output-dir`). Lancer : `OXIDE_JEU_TEST=<jeu> cargo test --release -p surfx --test describe_sorties -- --ignored`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Résout un motif : `{x}` remplacé par sa valeur ; un groupe `[…]` est gardé (sans crochets) si toutes ses marques
/// ont une valeur, supprimé sinon.
fn resolve(pattern: &str, vars: &BTreeMap<&str, String>) -> String {
    let mut out = String::new();
    let mut rest = pattern;
    while let Some(i) = rest.find('[') {
        out.push_str(&substitute(&rest[..i], vars).expect("marque sans valeur hors d'un groupe facultatif"));
        let j = rest[i..].find(']').expect("crochet non fermé") + i;
        if let Some(inner) = substitute(&rest[i + 1..j], vars) {
            out.push_str(&inner);
        }
        rest = &rest[j + 1..];
    }
    out.push_str(&substitute(rest, vars).expect("marque sans valeur"));
    out
}

fn substitute(text: &str, vars: &BTreeMap<&str, String>) -> Option<String> {
    let mut out = String::new();
    let mut rest = text;
    while let Some(i) = rest.find('{') {
        out.push_str(&rest[..i]);
        let j = rest[i..].find('}')? + i;
        out.push_str(vars.get(&rest[i + 1..j])?);
        rest = &rest[j + 1..];
    }
    out.push_str(rest);
    Some(out)
}

#[test]
fn resolution_des_motifs() {
    let mut v = BTreeMap::new();
    v.insert("sub", "01".to_string());
    v.insert("ses", "02".to_string());
    v.insert("hemi", "L".to_string());
    let p = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_hemi-{hemi}_white.surf.gii";
    assert_eq!(resolve(p, &v), "sub-01/ses-02/anat/sub-01_ses-02_hemi-L_white.surf.gii");
    v.remove("ses");
    assert_eq!(resolve(p, &v), "sub-01/anat/sub-01_hemi-L_white.surf.gii");
}

#[test]
fn describe_sort_sans_rien_lire() {
    let out = Command::new(env!("CARGO_BIN_EXE_surfx"))
        .arg("--describe")
        .output()
        .unwrap();
    assert!(out.status.success());
    let d: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(d["schema"], "oxide-describe/1");
    assert!(out.stderr.is_empty());
}

#[test]
#[ignore = "jeu BIDS de test (OXIDE_JEU_TEST) et plusieurs minutes de calcul"]
fn chaque_sortie_declaree_est_ecrite() {
    let Some(jeu) = std::env::var_os("OXIDE_JEU_TEST").map(PathBuf::from) else {
        panic!("OXIDE_JEU_TEST (jeu BIDS de test avec derivatives/medx-bounti) n'est pas défini");
    };
    let sub = std::env::var("OXIDE_JEU_SUB").unwrap_or_else(|_| "fant01".into());
    let ses = std::env::var("OXIDE_JEU_SES").unwrap_or_else(|_| "01".into());
    let tmp = std::env::temp_dir().join(format!("surfx-describe-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);

    let doc = Command::new(env!("CARGO_BIN_EXE_surfx"))
        .arg("--describe")
        .output()
        .unwrap();
    let d: Value = serde_json::from_slice(&doc.stdout).unwrap();
    let cmd = &d["commands"][0];
    assert_eq!(cmd["argv"][0], "run");

    // La segmentation (entrée déclarée) est bien là où le document la place.
    let mut vars = BTreeMap::new();
    vars.insert("sub", sub.clone());
    vars.insert("ses", ses.clone());
    let entree = jeu
        .join("derivatives")
        .join(cmd["inputs"][0]["pipeline"].as_str().unwrap())
        .join(resolve(cmd["inputs"][0]["pattern"].as_str().unwrap(), &vars));
    assert!(entree.is_file(), "entrée déclarée absente : {}", entree.display());

    let status = Command::new(env!("CARGO_BIN_EXE_surfx"))
        .args(["run", "--bids"])
        .arg(&jeu)
        .args(["--participant", &sub, "--session", &ses, "--output-dir"])
        .arg(&tmp)
        .status()
        .unwrap();
    assert!(status.success(), "surfx run a échoué : {status}");

    let racine = tmp.join("derivatives").join(cmd["pipeline"].as_str().unwrap());
    let mut manquants = Vec::new();
    for o in cmd["outputs"].as_array().unwrap() {
        if o["required"] != true {
            continue;
        }
        let hemis: Vec<String> = match o.get("expand").and_then(|e| e.get("hemi")) {
            Some(h) => h["all"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_string())
                .collect(),
            None => vec![String::new()],
        };
        for h in hemis {
            let mut v = vars.clone();
            if !h.is_empty() {
                v.insert("hemi", h);
            }
            let chemin = racine.join(resolve(o["pattern"].as_str().unwrap(), &v));
            if !chemin.is_file() {
                manquants.push(chemin.display().to_string());
            }
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(
        manquants.is_empty(),
        "sorties déclarées mais non écrites : {manquants:#?}"
    );
}
