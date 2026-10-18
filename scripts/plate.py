#!/usr/bin/env python3
"""Planche de toutes les surfaces d'un dossier de sorties `surfx run`, de la plus jeune à la plus âgée.

Une vignette par session : vue latérale d'un hémisphère (piale par défaut), rendu à la FreeSurfer, même
échelle pour toutes, âge gestationnel en titre.

Usage :
    python3 scripts/plate.py --surfaces <dossier> --out planche.png [--kind pial|white] [--hemi L|R]

Dépendances : numpy, nibabel, matplotlib.
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from illustrate import SUBJECTS, load_surface, render_surface  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--surfaces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kind", choices=["pial", "white"], default="pial")
    ap.add_argument("--hemi", choices=["L", "R"], default="L")
    ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()

    ga = {(r["subject_id"], r["session_id"]): float(r["ga_weeks"]) for r in csv.DictReader(open(SUBJECTS))}
    pat = re.compile(rf"(sub-[^_]+)_(ses-[^_]+)_hemi-{args.hemi}_{args.kind}\.surf\.gii$")
    items = []
    for p in sorted(args.surfaces.iterdir()):
        m = pat.match(p.name)
        if m and (m[1], m[2]) in ga:
            items.append((ga[(m[1], m[2])], m[1], m[2], p))
    items.sort()
    if not items:
        sys.exit(f"aucune surface {args.kind} {args.hemi} dans {args.surfaces}")

    meshes = [load_surface(p) for *_, p in items]
    half = max((v.max(0) - v.min(0)).max() for v, _ in meshes) / 2 * 1.02

    n = len(items)
    cols = args.cols
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(3.4 * cols, 3.0 * rows + 0.8), facecolor="white")
    gs = fig.add_gridspec(rows, cols, wspace=0.0, hspace=0.12, left=0.01, right=0.99, top=0.93, bottom=0.02)
    side = "gauche" if args.hemi == "L" else "droit"
    azim = 180 if args.hemi == "L" else 0
    for k, ((g, sub, ses, _), (v, f)) in enumerate(zip(items, meshes)):
        ax = fig.add_subplot(gs[k // cols, k % cols], projection="3d")
        render_surface(ax, v, f, (v.min(0) + v.max(0)) / 2, half, azim=azim)
        ax.set_title(f"{g:.1f} SA", fontsize=13, pad=-4)
        ax.text2D(0.5, 0.02, f"{sub} {ses}", transform=ax.transAxes, ha="center", fontsize=8, color="#666a72")
        print(f"{g:.1f} SA {sub} {ses}", flush=True)
    # Barre d'échelle dans la dernière case libre (ou sous la dernière vignette).
    k = n if n < rows * cols else n - 1
    ax = fig.add_subplot(gs[k // cols, k % cols])
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.axis("off")
    # Les vues 3D sont agrandies (zoom 1,6) : la barre suit le même facteur.
    ax.plot([-5 * 1.6, 5 * 1.6], [0, 0], color="#2b2e33", lw=3, solid_capstyle="butt")
    ax.text(0, -half * 0.12, "10 mm", ha="center", va="top", fontsize=11)
    kind = "piales" if args.kind == "pial" else "blanches"
    fig.suptitle(f"Surfaces {kind} de l'hémisphère {side} (surfoxide), de {items[0][0]:.1f} à {items[-1][0]:.1f} SA",
                 fontsize=15, x=0.02, ha="left")
    fig.savefig(args.out, dpi=120)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
