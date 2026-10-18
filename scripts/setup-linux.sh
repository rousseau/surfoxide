#!/usr/bin/env bash
# Prépare une machine Ubuntu native avec GPU NVIDIA (PC Linux, RTX 2080 Ti) pour développer surfoxide :
#   1. paquets système, 2. Rust, 3. CUDA Toolkit, 4. compilation, tests et vérification du GPU.
#
# Usage, depuis le dépôt :
#   bash scripts/setup-linux.sh
#
# Le script s'arrête à la première erreur et peut être relancé : les étapes déjà faites sont sautées.
# Il demande le mot de passe sudo pour apt. Il ne touche jamais au pilote NVIDIA : seul le toolkit est
# installé, dans la version la plus récente acceptée à la fois par le pilote et par Burn.

set -euo pipefail

DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mErreur : %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
step "0. Vérifications"
grep -qi microsoft /proc/version && die "Ce système est WSL : utiliser scripts/setup-wsl.sh."
[ -r /etc/os-release ] && . /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Ubuntu attendu (trouvé : ${ID:-inconnu})."
ok "Ubuntu ${VERSION_ID:-?}"

command -v nvidia-smi >/dev/null || die "nvidia-smi introuvable : installer d'abord le pilote NVIDIA (ubuntu-drivers install)."
if ! SMI_OUT=$(nvidia-smi 2>&1); then
  printf '%s\n' "$SMI_OUT" >&2
  die "nvidia-smi échoue (sortie ci-dessus) : pilote mal chargé, redémarrer ou réinstaller le pilote."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
DRIVER_CUDA="${DRIVER_CUDA:-$(printf '%s\n' "$SMI_OUT" | grep -oE 'CUDA (UMD )?Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)}"
[ -n "$DRIVER_CUDA" ] || die "version CUDA du pilote introuvable dans nvidia-smi. Définir DRIVER_CUDA=13.2 (par exemple) et relancer."
ok "GPU : ${GPU_NAME:-inconnu} ; le pilote accepte CUDA jusqu'à $DRIVER_CUDA"

# ---------------------------------------------------------------------------------------------
step "1. Paquets système"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config git curl wget ca-certificates xz-utils
ok "paquets installés"

# ---------------------------------------------------------------------------------------------
step "2. Rust"
if ! command -v rustup >/dev/null && [ ! -x "$HOME/.cargo/bin/rustup" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile default
fi
# shellcheck disable=SC1091
. "$HOME/.cargo/env"
rustup update stable >/dev/null
rustup component add rustfmt clippy >/dev/null
ok "$(rustc --version)"

# ---------------------------------------------------------------------------------------------
step "3. CUDA Toolkit"
# Versions acceptées par cudarc 0.19 (dépendance de Burn 0.22), de la plus récente à la plus ancienne.
SUPPORTED="13.4 13.3 13.2 13.1 13.0 12.9 12.8 12.6 12.5 12.4 12.3 12.2 12.1 12.0"
version_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }
NVIDIA_REPOS="https://developer.download.nvidia.com/compute/cuda/repos"

# Choix du dépôt NVIDIA : celui de cette version d'Ubuntu s'il propose une version acceptée par le pilote,
# sinon celui d'Ubuntu 24.04 (les paquets du toolkit n'ont pas de dépendance au système qui gêne).
pick_version() {  # $1 = nom du dépôt (ex. ubuntu2404) ; affiche la version retenue, ou rien
  local listing v
  listing=$(curl -fsSL "$NVIDIA_REPOS/$1/x86_64/" 2>/dev/null) || return 0
  for v in $SUPPORTED; do
    if version_le "$v" "$DRIVER_CUDA" && grep -q "cuda-toolkit-${v/./-}_" <<<"$listing"; then
      echo "$v"; return 0
    fi
  done
}

CUDA_VER=""
for repo in "ubuntu${VERSION_ID//./}" ubuntu2404; do
  CUDA_VER=$(pick_version "$repo")
  if [ -n "$CUDA_VER" ]; then REPO="$repo"; break; fi
done
[ -n "$CUDA_VER" ] || die "Aucun toolkit compatible avec le pilote (CUDA $DRIVER_CUDA) : mettre à jour le pilote NVIDIA."

CUDA_HOME_DIR="/usr/local/cuda-$CUDA_VER"
if [ -x "$CUDA_HOME_DIR/bin/nvcc" ]; then
  ok "CUDA $CUDA_VER déjà installé dans $CUDA_HOME_DIR"
else
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cuda-keyring.deb" "$NVIDIA_REPOS/$REPO/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
    sudo apt-get update -qq
  fi
  # Uniquement le toolkit : le méta-paquet « cuda » remplacerait le pilote installé.
  sudo apt-get install -y "cuda-toolkit-${CUDA_VER/./-}"
  ok "CUDA $CUDA_VER installé dans $CUDA_HOME_DIR (dépôt $REPO)"
fi

# Variables d'environnement, ajoutées une seule fois à ~/.bashrc.
MARK="# >>> surfoxide CUDA >>>"
if ! grep -qF "$MARK" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF

$MARK
export CUDA_PATH="$CUDA_HOME_DIR"
export PATH="\$CUDA_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_PATH/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# <<< surfoxide CUDA <<<
EOF
  ok "variables CUDA ajoutées à ~/.bashrc"
fi
export CUDA_PATH="$CUDA_HOME_DIR"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ok "$(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------------------------
step "4. Compilation, tests et GPU"
cd "$DEST"
cargo test --release
cargo run --release -q -- selftest
# Première compilation CUDA : télécharge un paquet LLVM (~45 Mo) dans ~/.cache/tracel/.
cargo build --release --features cuda
./target/release/surfx device --backend cuda --size 2048

step "Terminé"
ok "Ouvrir un nouveau terminal (ou : source ~/.bashrc) pour retrouver les variables CUDA."
ok "Développement : cd $DEST && cargo run --release --features cuda -- --help"
