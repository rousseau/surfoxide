#!/usr/bin/env python3
"""Courbe d'étalonnage de `surfx profile` (plan qualité, Q0.4) sur les fantômes A.

Le flou de l'image déplace l'extremum de la dérivée de chaque bord d'un cortex mince (bande sombre entre substance
blanche et LCR clairs) : sur des surfaces vraies, le décalage médian n'est pas nul et grandit avec l'âge. Pour chaque
semaine de l'atlas fœtal dHCP et chaque hémisphère : fantôme A (réglages d'image par défaut de `surfx phantom`, sans
bruit de contour, graine --seed), `surfx profile` des surfaces vraies sur le masque vrai ; courbe = moyenne
gauche/droite par semaine et par surface.

Sortie : TSV `week  kind  offset_L  offset_R  offset` (mm), à passer à `surfx profile --calibration`.

Usage :
    python3 scripts/profile_calibration.py --out ~/scratch/plan-qualite/<dossier> [--seed 1] [--weeks 21-36]
        [--jobs 8] [-- options supplémentaires de surfx phantom]

Atlas : $OXIDE_DHCP_ATLAS (défaut data/atlas_dhcp/ du clone de gyroxide voisin). Bibliothèque standard seulement.
"""

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SURFX = Path(os.environ.get("SURFX", REPO / "target/release/surfx"))
ATLAS = Path(os.environ.get("OXIDE_DHCP_ATLAS", REPO.parent / "gyroxide/data/atlas_dhcp"))


def one(out, week, hemi, seed, extra):
    side = {"L": "left", "R": "right"}[hemi]
    stem = ATLAS / f"fetal.week{week}.{side}"
    prefix = out / f"s{seed}" / f"w{week}{hemi}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    if not Path(f"{prefix}_phantom.json").exists():
        subprocess.run([str(SURFX), "phantom", "--white", f"{stem}.wm.surf.gii", "--pial", f"{stem}.pial.surf.gii",
                        "--medial", f"{stem}.sulc.shape.gii", "--hemi", hemi, "--seed", str(seed), "-o", str(prefix),
                        *extra], check=True, stdout=subprocess.DEVNULL)
    truth = f"{prefix}_hemi-{hemi}_desc-truth"
    for kind in ("white", "pial"):
        js = Path(f"{prefix}_profile-{kind}.json")
        if not js.exists():
            subprocess.run([str(SURFX), "profile", "--t2w", f"{prefix}_T2w.nii.gz", "--surface",
                            f"{truth}_{kind}.surf.gii", "--kind", kind, "--mask", f"{truth}_mask.shape.gii",
                            "--json", str(js)], check=True)
        res[kind] = json.loads(js.read_text())["median_offset"]
    return week, hemi, res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--weeks", default="21-36")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("extra", nargs="*", help="options supplémentaires de surfx phantom (après --)")
    args = ap.parse_args()
    if not SURFX.exists():
        sys.exit(f"erreur : {SURFX} absent ; cargo build --release -p surfx")
    if not ATLAS.is_dir():
        sys.exit(f"erreur : atlas {ATLAS} absent ($OXIDE_DHCP_ATLAS)")
    lo, hi = (int(x) for x in args.weeks.split("-"))
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(lambda t: one(args.out, t[0], t[1], args.seed, args.extra),
                          [(w, h) for w in range(lo, hi + 1) for h in "LR"]))
    by = {(w, h): r for w, h, r in res}
    path = args.out / f"calibration_seed-{args.seed}.tsv"
    with open(path, "w") as f:
        f.write("week\tkind\toffset_L\toffset_R\toffset\n")
        for w in range(lo, hi + 1):
            for kind in ("white", "pial"):
                a, b = by[(w, "L")][kind], by[(w, "R")][kind]
                f.write(f"{w}\t{kind}\t{a:.4f}\t{b:.4f}\t{(a + b) / 2:.4f}\n")
    print(f"écrit : {path}")


if __name__ == "__main__":
    main()
