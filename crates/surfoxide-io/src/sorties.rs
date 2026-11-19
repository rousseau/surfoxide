//! Table des entrées et sorties de `surfx run` (mode BIDS), source unique des noms de fichiers : le code d'écriture
//! (`surfx::white`) en tire ses noms, et `surfx --describe` (schéma `oxide-describe/1`, `docs/donnees.md` de
//! guioxide, section 5) en tire ses motifs. Un nom ne s'écrit donc qu'ici.
//!
//! Un nom est un **suffixe** : `sub-X[_ses-Y]` + suffixe, dans `anat/` du dossier de dérivées. Hors BIDS, le préfixe
//! libre (`--out`) remplace `sub-X[_ses-Y]` et les noms restent ceux-ci.

/// Début commun des motifs de chemin (relatifs à la racine du pipeline) : dossier et préfixe d'une session.
pub const ANAT_PREFIX: &str = "sub-{sub}[/ses-{ses}]/anat/sub-{sub}[_ses-{ses}]";

/// Une sortie de `surfx run`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sortie {
    /// Identifiant stable (`white`, `pial`…), celui du document `--describe`.
    pub id: &'static str,
    /// Nature (`surface`, `field`, `metadata`, `table`), qui décide de l'affichage dans guioxide.
    pub nature: &'static str,
    /// Suffixe du nom de fichier ; `{hemi}` vaut `L` ou `R`.
    pub suffix: &'static str,
    /// Écrite à chaque exécution réussie (les sorties facultatives ne conditionnent pas l'avancement).
    pub required: bool,
    /// Une par hémisphère (`{hemi}` dans le suffixe).
    pub per_hemi: bool,
}

/// Sorties de `surfx run` : surfaces, champs par sommet, rapport de contrat par hémisphère, contrôle des maillages.
pub const SORTIES: &[Sortie] = &[
    Sortie {
        id: "white",
        nature: "surface",
        suffix: "_hemi-{hemi}_white.surf.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "pial",
        nature: "surface",
        suffix: "_hemi-{hemi}_pial.surf.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "curv",
        nature: "field",
        suffix: "_hemi-{hemi}_curv.shape.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "sulc",
        nature: "field",
        suffix: "_hemi-{hemi}_sulc.shape.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "thickness",
        nature: "field",
        suffix: "_hemi-{hemi}_thickness.shape.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "cortex",
        nature: "field",
        suffix: "_hemi-{hemi}_cortex.shape.gii",
        required: true,
        per_hemi: true,
    },
    Sortie {
        id: "report",
        nature: "metadata",
        suffix: "_hemi-{hemi}_report.json",
        required: false,
        per_hemi: true,
    },
    Sortie {
        id: "surfqc",
        nature: "table",
        suffix: "_desc-surfqc.tsv",
        required: true,
        per_hemi: false,
    },
];

/// Entrée de `surfx run --bids` : segmentation écrite par `medx bounti --bids` (pipeline [`crate::bids::PIPELINE_BOUNTI`]).
pub const ENTREE_SEGMENTATION: &str = "_desc-bounti_dseg.nii.gz";

/// La sortie d'identifiant `id` ; panique si elle n'existe pas (identifiant écrit dans le code, pas lu).
pub fn sortie(id: &str) -> &'static Sortie {
    SORTIES
        .iter()
        .find(|s| s.id == id)
        .unwrap_or_else(|| panic!("sortie inconnue : {id}"))
}

/// Suffixe du nom de fichier de la sortie `id` pour l'hémisphère `hemi` (`L` ou `R`, ignoré s'il n'y a pas de
/// `{hemi}`) : `suffixe("white", "L")` donne `_hemi-L_white.surf.gii`.
pub fn suffixe(id: &str, hemi: &str) -> String {
    sortie(id).suffix.replace("{hemi}", hemi)
}

impl Sortie {
    /// Motif de chemin relatif à la racine du pipeline, marques `{sub}`, `{ses}`, `{hemi}`.
    pub fn pattern(&self) -> String {
        format!("{ANAT_PREFIX}{}", self.suffix)
    }
}

/// Motif de la segmentation d'entrée, relatif à la racine du pipeline `medx-bounti`.
pub fn entree_segmentation_pattern() -> String {
    format!("{ANAT_PREFIX}{ENTREE_SEGMENTATION}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bids::derivative_dir;
    use std::path::Path;

    /// Résout un motif pour `sub-01`, `ses-01` (ou sans session) et l'hémisphère donné.
    fn resolu(pattern: &str, ses: Option<&str>, hemi: &str) -> String {
        let mut s = pattern.to_string();
        // Groupes facultatifs : gardés (marque présente) ou supprimés (marque absente).
        for (open, key) in [("[/ses-{ses}]", "ses"), ("[_ses-{ses}]", "ses")] {
            let valeur = if key == "ses" { ses } else { None };
            let remplacement = match valeur {
                Some(v) => open.trim_matches(['[', ']']).replace("{ses}", v),
                None => String::new(),
            };
            s = s.replace(open, &remplacement);
        }
        s.replace("{sub}", "01").replace("{hemi}", hemi)
    }

    /// Chaque motif, résolu, égale le chemin que construit `derivative_dir` + préfixe + suffixe (celui du code
    /// d'écriture), avec et sans session.
    #[test]
    fn motifs_egaux_aux_chemins_du_code_d_ecriture() {
        for ses in [Some("01"), None] {
            let d = derivative_dir(
                Path::new("/g/sourcedata"),
                None,
                crate::bids::PIPELINE_SURFACES,
                "01",
                ses,
            )
            .unwrap();
            for s in SORTIES {
                let attendu = d.anat.join(format!("{}{}", d.prefix, suffixe(s.id, "L")));
                let relatif = attendu.strip_prefix(&d.pipeline).unwrap().to_string_lossy().to_string();
                assert_eq!(resolu(&s.pattern(), ses, "L"), relatif, "sortie {}", s.id);
            }
        }
    }

    /// Les identifiants sont uniques et `{hemi}` figure exactement dans les sorties par hémisphère.
    #[test]
    fn table_coherente() {
        for (i, s) in SORTIES.iter().enumerate() {
            assert!(
                SORTIES[i + 1..].iter().all(|t| t.id != s.id),
                "identifiant en double : {}",
                s.id
            );
            assert_eq!(s.suffix.contains("{hemi}"), s.per_hemi, "sortie {}", s.id);
        }
    }

    /// Les noms d'avant la table (hors BIDS comme en BIDS) n'ont pas changé.
    #[test]
    fn noms_inchanges() {
        assert_eq!(suffixe("white", "L"), "_hemi-L_white.surf.gii");
        assert_eq!(suffixe("pial", "R"), "_hemi-R_pial.surf.gii");
        assert_eq!(suffixe("curv", "L"), "_hemi-L_curv.shape.gii");
        assert_eq!(suffixe("sulc", "L"), "_hemi-L_sulc.shape.gii");
        assert_eq!(suffixe("thickness", "L"), "_hemi-L_thickness.shape.gii");
        assert_eq!(suffixe("cortex", "R"), "_hemi-R_cortex.shape.gii");
        assert_eq!(suffixe("report", "L"), "_hemi-L_report.json");
        assert_eq!(suffixe("surfqc", "L"), "_desc-surfqc.tsv");
        assert_eq!(ENTREE_SEGMENTATION, "_desc-bounti_dseg.nii.gz");
    }
}
