#!/usr/bin/env bash
# Rapatrie depuis Garage les segmentations (BOUNTI-19 et multi-BOUNTI-43) utilisées par les tests sur
# données réelles, la validation et les figures, dans $SURFOXIDE_DATA (par défaut ~/.cache/surfoxide/data),
# rangées en <sub>/<ses>/anat/.
#
# Usage :
#   bash scripts/fetch-data.sh                         # segmentations des 193 sessions (~165 Mo)
#   bash scripts/fetch-data.sh --t2w sub-d6a1p/ses-01  # plus la T2w rééchantillonnée de ces sessions (~50 Mo chacune)
#   bash scripts/fetch-data.sh --clean                 # supprime le dossier local
#
# Les données vivent dans garage:crl2025 ; ce dossier n'est qu'une copie de travail, à nettoyer après usage
# (convention : rapatrier → traiter → renvoyer → nettoyer).

set -euo pipefail

DATA="${SURFOXIDE_DATA:-$HOME/.cache/surfoxide/data}"
REMOTE="garage:crl2025/derivatives/fetal-folding-geometry"

# Identifiants Garage, s'ils ne sont pas déjà dans l'environnement.
if [ -r "$HOME/.config/garage/env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HOME/.config/garage/env"
  set +a
fi

case "${1:-}" in
  --clean)
    rm -rf -- "$DATA"
    echo "supprimé : $DATA"
    exit 0
    ;;
  --t2w)
    shift
    [ $# -gt 0 ] || { echo "erreur : --t2w attend au moins une session sub-XX/ses-YY" >&2; exit 1; }
    for s in "$@"; do
      [[ "$s" =~ ^sub-[A-Za-z0-9]+/ses-[A-Za-z0-9]+$ ]] || { echo "erreur : session invalide « $s »" >&2; exit 1; }
      rclone copy "$REMOTE/$s/anat/" "$DATA/$s/anat/" --include "*_res-0p5mm_desc-resampled_T2w.nii.gz"
    done
    ;;
  "")
    ;;
  *)
    echo "usage : $0 [--t2w sub-XX/ses-YY ... | --clean]" >&2
    exit 1
    ;;
esac

rclone copy "$REMOTE/" "$DATA/" \
  --include "*_res-0p5mm_desc-bounti19_dseg.nii.gz" \
  --include "*_res-0p5mm_desc-multibounti43_dseg.nii.gz"
n=$(find "$DATA" -name "*_dseg.nii.gz" | wc -l)
echo "$n segmentations dans $DATA"
