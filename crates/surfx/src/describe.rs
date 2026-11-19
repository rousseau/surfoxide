//! `surfx --describe` : description JSON des commandes que guioxide peut lancer (schéma `oxide-describe/1`,
//! `docs/donnees.md` de guioxide, section 5).
//!
//! Le document est généré, pas rédigé : les paramètres viennent des définitions clap (`Cli::command()`), les
//! entrées et sorties de la table `surfoxide_io::sorties` (celle d'où le code d'écriture tire ses noms de fichiers),
//! le reste de constantes de ce module. Aucun poids, aucun jeu de données, aucun GPU n'est touché.

use clap::{Arg, ArgAction, Command, CommandFactory};
use serde_json::{json, Map, Value};
use std::any::TypeId;
use surfoxide_io::bids::{PIPELINE_BOUNTI, PIPELINE_SURFACES};
use surfoxide_io::sorties;

use crate::Cli;

/// Version du schéma du document.
const SCHEMA: &str = "oxide-describe/1";
/// Version du contrat de données de guioxide respectée par l'outil.
const CONTRACT: &str = "1.0";
/// Dépôt du code.
const CODE_URL: &str = "https://github.com/rousseau/surfoxide";

/// Arguments BIDS de clap (ils font la ligne de commande BIDS de guioxide, ce ne sont pas des paramètres).
const BIDS_ARGS: [&str; 4] = ["bids", "participant", "session", "output_dir"];
/// Chemins du mode libre (hors BIDS) : rangés à part, pas des paramètres.
const FREE_IO: [&str; 2] = ["seg", "out"];
/// Paramètres affichés d'emblée dans un formulaire ; tout le reste est `advanced`.
const BASIC: [&str; 2] = ["labels", "hemi"];
/// Valeurs possibles des paramètres que clap analyse par `FromStr` (il ne les connaît pas).
const CHOICES: [(&str, &[&str]); 2] = [("labels", &["bounti19", "multibounti43"]), ("hemi", &["L", "R"])];
/// Bornes et unités connues de l'outil : (paramètre, minimum, unité). Rien n'est deviné.
const BOUNDS: [(&str, Option<f64>, Option<&str>); 4] = [
    ("sigma", Some(0.0), Some("mm")),
    ("edge", Some(0.0), Some("mm")),
    ("min_thickness", Some(0.0), Some("mm")),
    ("min_offset", None, Some("mm")),
];

/// Document complet : `{"schema", "contract", "tool", "commands"}`.
pub fn document() -> Value {
    let cli = Cli::command();
    let run = cli
        .get_subcommands()
        .find(|c| c.get_name() == "run")
        .expect("la sous-commande `run` existe");
    json!({
        "schema": SCHEMA,
        "contract": CONTRACT,
        "tool": {
            "name": "surfx",
            "version": env!("CARGO_PKG_VERSION"),
            "commit": option_env!("SURFX_COMMIT"),
            "code_url": CODE_URL,
        },
        "commands": [command_run(run)],
    })
}

/// Valeur JSON d'un défaut de clap, selon le type du paramètre.
fn default_value(kind: &str, arg: &Arg) -> Value {
    let Some(text) = arg.get_default_values().first().and_then(|v| v.to_str()) else {
        return if kind == "boolean" { json!(false) } else { Value::Null };
    };
    match kind {
        "number" => text.parse::<f64>().map_or(Value::Null, |v| json!(v)),
        "integer" => text.parse::<i64>().map_or(Value::Null, |v| json!(v)),
        "boolean" => json!(text == "true"),
        _ => json!(text),
    }
}

/// Type du schéma d'après l'analyseur de clap.
fn kind_of(arg: &Arg, has_choices: bool) -> &'static str {
    if matches!(arg.get_action(), ArgAction::SetTrue | ArgAction::SetFalse) {
        return "boolean";
    }
    if has_choices {
        return "enum";
    }
    let id = arg.get_value_parser().type_id();
    if id == TypeId::of::<f64>() || id == TypeId::of::<f32>() {
        "number"
    } else if [
        TypeId::of::<usize>(),
        TypeId::of::<u64>(),
        TypeId::of::<u32>(),
        TypeId::of::<i64>(),
        TypeId::of::<i32>(),
    ]
    .iter()
    .any(|t| id == *t)
    {
        "integer"
    } else if id == TypeId::of::<std::path::PathBuf>() {
        "path"
    } else {
        "string"
    }
}

/// Paramètre du schéma pour un argument clap.
fn parameter(arg: &Arg) -> Value {
    let id = arg.get_id().as_str();
    let mut choices: Vec<String> = arg
        .get_possible_values()
        .iter()
        .map(|v| v.get_name().to_string())
        .collect();
    if choices.is_empty() {
        if let Some((_, list)) = CHOICES.iter().find(|(name, _)| *name == id) {
            choices = list.iter().map(|s| s.to_string()).collect();
        }
    }
    let kind = kind_of(arg, !choices.is_empty());
    let mut m = Map::new();
    m.insert("id".into(), id.into());
    m.insert("flag".into(), format!("--{}", arg.get_long().unwrap_or(id)).into());
    m.insert("type".into(), kind.into());
    if !choices.is_empty() {
        m.insert("choices".into(), json!(choices));
    }
    m.insert("default".into(), default_value(kind, arg));
    if let Some((_, min, unit)) = BOUNDS.iter().find(|(name, _, _)| *name == id) {
        if let Some(min) = min {
            m.insert("min".into(), json!(min));
        }
        if let Some(unit) = unit {
            m.insert("unit".into(), json!(unit));
        }
    }
    m.insert("optional".into(), true.into());
    m.insert(
        "level".into(),
        (if BASIC.contains(&id) { "basic" } else { "advanced" }).into(),
    );
    m.insert(
        "help".into(),
        arg.get_help().map(|h| h.to_string()).unwrap_or_default().into(),
    );
    Value::Object(m)
}

/// Sortie du schéma pour une entrée de la table.
fn output(s: &sorties::Sortie) -> Value {
    let mut m = Map::new();
    m.insert("id".into(), s.id.into());
    m.insert("nature".into(), s.nature.into());
    m.insert("required".into(), s.required.into());
    m.insert("cardinality".into(), "session".into());
    if s.per_hemi {
        m.insert(
            "expand".into(),
            json!({ "hemi": { "param": "hemi", "all": ["L", "R"] } }),
        );
    }
    m.insert("pattern".into(), s.pattern().into());
    Value::Object(m)
}

/// `surfx run`.
fn command_run(run: &Command) -> Value {
    let parameters: Vec<Value> = run
        .get_arguments()
        .filter(|a| {
            let id = a.get_id().as_str();
            a.get_long().is_some() && !BIDS_ARGS.contains(&id) && !FREE_IO.contains(&id)
        })
        .map(parameter)
        .collect();
    json!({
        "id": "run",
        "argv": ["run"],
        "summary": run.get_about().map(|a| a.to_string()).unwrap_or_default(),
        "stage": "surfaces",
        "pipeline": PIPELINE_SURFACES,
        "generated_by": "surfx run",
        "bids": {
            "mode": "supported",
            "root": "--bids",
            "participant": "--participant",
            "session": "--session",
            "output_dir": "--output-dir",
        },
        "inputs": [{
            "id": "segmentation",
            "nature": "labels",
            "pipeline": PIPELINE_BOUNTI,
            "pattern": sorties::entree_segmentation_pattern(),
            "cardinality": "session",
            "count": "one",
            "required": true,
        }],
        "outputs": sorties::SORTIES.iter().map(output).collect::<Vec<_>>(),
        "parameters": parameters,
        "presets": [],
        "default_preset": null,
        "resources": { "gpu": "none", "device_param": null, "memory_gb": null },
        "progress": null,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Le document : forme, version, une commande `run`, motifs de la table.
    #[test]
    fn forme_du_document() {
        let d = document();
        assert_eq!(d["schema"], "oxide-describe/1");
        assert_eq!(d["tool"]["name"], "surfx");
        let run = &d["commands"][0];
        assert_eq!(run["id"], "run");
        assert_eq!(run["pipeline"], "surfx-surfaces");
        assert_eq!(run["generated_by"], "surfx run");
        assert_eq!(run["outputs"].as_array().unwrap().len(), sorties::SORTIES.len());
        assert_eq!(
            run["inputs"][0]["pattern"],
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_desc-bounti_dseg.nii.gz"
        );
        let white = &run["outputs"][0];
        assert_eq!(
            white["pattern"],
            "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]_hemi-{hemi}_white.surf.gii"
        );
    }

    /// Chaque option de `surfx run` (hors BIDS et mode libre) figure dans le document, avec le défaut de clap.
    #[test]
    fn tous_les_arguments_de_clap() {
        let cli = Cli::command();
        let run = cli.get_subcommands().find(|c| c.get_name() == "run").unwrap();
        let d = document();
        let params = d["commands"][0]["parameters"].as_array().unwrap();
        for a in run.get_arguments().filter(|a| a.get_long().is_some()) {
            let id = a.get_id().as_str();
            let present = params.iter().any(|p| p["id"] == id);
            let exclu = BIDS_ARGS.contains(&id) || FREE_IO.contains(&id) || id == "help" || id == "version";
            assert_eq!(present, !exclu, "paramètre {id}");
        }
        let by = |id: &str| params.iter().find(|p| p["id"] == id).unwrap().clone();
        assert_eq!(by("sigma")["type"], "number");
        assert_eq!(by("sigma")["default"], 0.25);
        assert_eq!(by("sigma")["unit"], "mm");
        assert_eq!(by("taubin")["type"], "integer");
        assert_eq!(by("taubin")["default"], 10);
        assert_eq!(by("no_skeleton")["type"], "boolean");
        assert_eq!(by("no_skeleton")["default"], false);
        assert_eq!(by("labels")["choices"], json!(["bounti19", "multibounti43"]));
        assert_eq!(by("labels")["default"], "bounti19");
        assert_eq!(by("hemi")["default"], Value::Null);
        assert_eq!(by("topology")["type"], "enum");
        assert_eq!(by("labels")["level"], "basic");
        assert_eq!(by("edge")["level"], "advanced");
    }

    /// Les entrées de la table de `help` et de `version` ne sont pas des paramètres.
    #[test]
    fn pas_d_aide_ni_de_version() {
        let d = document();
        let params = d["commands"][0]["parameters"].as_array().unwrap();
        assert!(params.iter().all(|p| p["id"] != "help" && p["id"] != "version"));
    }
}
