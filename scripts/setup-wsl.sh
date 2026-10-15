#!/usr/bin/env bash
# Prépare une machine WSL (Ubuntu) avec GPU NVIDIA pour développer surfoxide :
#   1. paquets système, 2. Rust, 3. CUDA Toolkit, 4. dépôt dans ~/Exp/surfoxide,
#   5. compilation, tests et vérification du GPU.
#
# Usage, dans WSL :
#   bash setup-wsl.sh
#
# Le script s'arrête à la première erreur et peut être relancé : les étapes déjà faites sont sautées.
# Il demande le mot de passe sudo pour apt. Il ne touche jamais au pilote NVIDIA : sous WSL, le
# pilote est celui de Windows, et il ne faut pas installer de pilote Linux dans WSL.

set -euo pipefail

REPO_SSH="git@github.com:rousseau/surfoxide.git"
REPO_HTTPS="https://github.com/rousseau/surfoxide.git"
DEST="${SURFOXIDE_DIR:-$HOME/Exp/surfoxide}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m    %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mErreur : %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------------------------
step "0. Vérifications"
grep -qi microsoft /proc/version || warn "Ce système ne semble pas être WSL ; je continue quand même."
[ -r /etc/os-release ] && . /etc/os-release
[ "${ID:-}" = "ubuntu" ] || die "Ubuntu attendu (trouvé : ${ID:-inconnu}). Le dépôt CUDA utilisé est celui de WSL-Ubuntu."
ok "Ubuntu ${VERSION_ID:-?}"

export PATH="/usr/lib/wsl/lib:$PATH"
command -v nvidia-smi >/dev/null || die "nvidia-smi introuvable. Installer ou mettre à jour le pilote NVIDIA côté Windows, puis relancer WSL (wsl --shutdown)."
if ! SMI_OUT=$(nvidia-smi 2>&1); then
  printf '%s\n' "$SMI_OUT" >&2
  die "nvidia-smi échoue (sortie ci-dessus). Sous Windows : mettre à jour le pilote NVIDIA, puis dans PowerShell « wsl --shutdown », et rouvrir WSL."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
DRIVER_CUDA="${DRIVER_CUDA:-$(printf '%s\n' "$SMI_OUT" | grep -oE 'CUDA Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || true)}"
if [ -z "$DRIVER_CUDA" ]; then
  printf '%s\n' "$SMI_OUT" >&2
  die "version CUDA du pilote introuvable dans la sortie de nvidia-smi (ci-dessus). Définir DRIVER_CUDA=12.8 (par exemple) et relancer."
fi
ok "GPU : ${GPU_NAME:-inconnu} ; le pilote Windows accepte CUDA jusqu'à $DRIVER_CUDA"

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

CUDA_VER=""
for v in $SUPPORTED; do
  if version_le "$v" "$DRIVER_CUDA"; then CUDA_VER="$v"; break; fi
done
[ -n "$CUDA_VER" ] || die "Le pilote Windows (CUDA $DRIVER_CUDA) est trop ancien : mettre à jour le pilote NVIDIA côté Windows."

CUDA_HOME_DIR="/usr/local/cuda-$CUDA_VER"
if [ -x "$CUDA_HOME_DIR/bin/nvcc" ]; then
  ok "CUDA $CUDA_VER déjà installé dans $CUDA_HOME_DIR"
else
  if ! dpkg -s cuda-keyring >/dev/null 2>&1; then
    tmp=$(mktemp -d)
    wget -q -O "$tmp/cuda-keyring.deb" \
      https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$tmp/cuda-keyring.deb"
    rm -rf "$tmp"
    sudo apt-get update -qq
  fi
  # Uniquement le toolkit : le méta-paquet « cuda » installerait un pilote Linux, à proscrire sous WSL.
  sudo apt-get install -y "cuda-toolkit-${CUDA_VER/./-}"
  ok "CUDA $CUDA_VER installé dans $CUDA_HOME_DIR"
fi

# Variables d'environnement, ajoutées une seule fois à ~/.bashrc.
MARK="# >>> surfoxide CUDA >>>"
if ! grep -qF "$MARK" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF

$MARK
export CUDA_PATH="$CUDA_HOME_DIR"
export PATH="\$CUDA_PATH/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:\$CUDA_PATH/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# <<< surfoxide CUDA <<<
EOF
  ok "variables CUDA ajoutées à ~/.bashrc"
fi
export CUDA_PATH="$CUDA_HOME_DIR"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ok "$(nvcc --version | tail -1)"

# ---------------------------------------------------------------------------------------------
step "4. Dépôt surfoxide dans $DEST"
if [ -d "$DEST/.git" ]; then
  git -C "$DEST" pull --ff-only || warn "git pull impossible (modifications locales ?) ; je garde l'état actuel."
elif [ -d "$DEST" ] && [ -n "$(ls -A "$DEST")" ]; then
  die "$DEST existe, n'est pas un dépôt git et n'est pas vide. Le déplacer ou le vider, puis relancer."
else
  mkdir -p "$(dirname "$DEST")"
  rmdir "$DEST" 2>/dev/null || true
  if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
    git clone "$REPO_SSH" "$DEST"
  elif command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh repo clone rousseau/surfoxide "$DEST"
  else
    die "Pas d'accès GitHub depuis WSL (dépôt privé). Au choix :
    - clé SSH : ssh-keygen -t ed25519, puis ajouter ~/.ssh/id_ed25519.pub sur github.com/settings/keys ;
    - ou : sudo apt-get install gh && gh auth login.
  Puis relancer ce script. (Clone HTTPS : $REPO_HTTPS)"
  fi
fi
[ -f "$DEST/Cargo.toml" ] || die "$DEST ne contient pas encore le code : pousser d'abord le squelette sur GitHub."
ok "dépôt prêt ($(git -C "$DEST" log -1 --format='%h %s'))"

# ---------------------------------------------------------------------------------------------
step "5. Compilation, tests et GPU"
cd "$DEST"
cargo test --release
cargo run --release -q -- selftest
# Première compilation CUDA : télécharge un paquet LLVM (~45 Mo) dans ~/.cache/tracel/.
cargo build --release --features cuda
./target/release/surfx device --backend cuda --size 2048

step "Terminé"
ok "Ouvrir un nouveau terminal (ou : source ~/.bashrc) pour retrouver les variables CUDA."
ok "Développement : cd $DEST && cargo run --release --features cuda -- --help"
