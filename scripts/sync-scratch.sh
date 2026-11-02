#!/usr/bin/env bash
# Partage entre machines des résultats de travail de surfoxide (validations, jeu d'entraînement de P1, essais) :
# ~/scratch (ou $SURFOXIDE_SCRATCH) <-> $OXIDE_REMOTE/derivatives/surfoxide (Garage), un dossier par résultat.
#
# Usage :
#   bash scripts/sync-scratch.sh list                                # dossiers présents sur Garage, avec leur taille
#   bash scripts/sync-scratch.sh pull surfoxide-validation-10 ...    # rapatrie ces dossiers
#   bash scripts/sync-scratch.sh push surfoxide-validation-12 ...    # envoie ces dossiers
#   bash scripts/sync-scratch.sh push --all                          # envoie tout ~/scratch (sauf les target-*)
#
# Copie sans suppression (rclone copy) : un fichier effacé d'un côté n'est jamais effacé de l'autre.
# Les dossiers target-* (compilations Cargo) et les __pycache__ ne sont jamais envoyés.

set -euo pipefail

SCRATCH="${SURFOXIDE_SCRATCH:-$HOME/scratch}"

# Identifiants Garage, s'ils ne sont pas déjà dans l'environnement.
if [ -r "$HOME/.config/garage/env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$HOME/.config/garage/env"
  set +a
fi

# Remote rclone et bucket (« <remote>:<bucket> »), définis hors du dépôt, en général dans ~/.config/garage/env.
: "${OXIDE_REMOTE:?à définir : remote rclone et bucket, par exemple dans ~/.config/garage/env}"
REMOTE="$OXIDE_REMOTE/derivatives/surfoxide"

usage() {
  echo "usage : $0 list | pull <dossier>... | pull --all | push <dossier>... | push --all" >&2
  exit 1
}

valide() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ && "$1" != target-* && "$1" != . && "$1" != .. ]] \
    || { echo "erreur : dossier invalide « $1 »" >&2; exit 1; }
}

[ $# -ge 1 ] || usage
action="$1"
shift

case "$action" in
  list)
    rclone lsf --dirs-only "$REMOTE/" | while read -r d; do
      printf "%-32s %s\n" "$d" "$(rclone size "$REMOTE/$d" | awk -F': ' '/Total size/ {print $2}')"
    done
    ;;
  pull)
    [ $# -ge 1 ] || usage
    if [ "$1" = "--all" ]; then
      mkdir -p "$SCRATCH"
      rclone copy "$REMOTE/" "$SCRATCH/" --progress
    else
      for d in "$@"; do
        valide "$d"
        rclone copy "$REMOTE/$d" "$SCRATCH/$d" --progress
      done
    fi
    ;;
  push)
    [ $# -ge 1 ] || usage
    if [ "$1" = "--all" ]; then
      set --
      for p in "$SCRATCH"/*/; do
        d="$(basename "$p")"
        [[ "$d" == target-* ]] || set -- "$@" "$d"
      done
    fi
    for d in "$@"; do
      valide "$d"
      [ -d "$SCRATCH/$d" ] || { echo "erreur : $SCRATCH/$d n'existe pas" >&2; exit 1; }
      rclone copy "$SCRATCH/$d" "$REMOTE/$d" --exclude "__pycache__/**" --exclude "target/**" --progress
    done
    ;;
  *)
    usage
    ;;
esac
