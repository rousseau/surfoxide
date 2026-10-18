#!/usr/bin/env python3
"""Figure : image T2w, segmentation BOUNTI et surfaces surfoxide pour quelques âges gestationnels.

Une ligne par session. Colonnes :
  1. coupe coronale T2w ;
  2. segmentation BOUNTI-19 superposée (tissus regroupés) ;
  3. intersection des surfaces blanche et piale avec le plan de coupe ;
  4. vue latérale 3D de la piale gauche, colorée par l'épaisseur corticale.
Même échelle (mm) sur toutes les lignes, convention neurologique (gauche à gauche).

Usage :
    python3 scripts/illustrate.py --surfaces <dossier des sorties surfx run> --out figure.png \\
        sub-d6a1p/ses-01 sub-0fk6m/ses-01 sub-0avbc/ses-02

Dépendances : numpy, nibabel, matplotlib.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HOME = Path.home()
T2W = HOME / "Exp/fetal-folding-geometry/.scratch/derivatives/fetal-folding-geometry"
BOUNTI = HOME / "Exp/CRL-Cinema/data/raw/bounti"
SUBJECTS = HOME / "Exp/fetal-folding-geometry/data/subjects.csv"

# Tissus regroupés (labels BOUNTI-19) et couleurs (palette catégorielle de référence, ordre fixe).
GROUPS = [
    ("LCR extra-cérébral", [1, 2], "#2a78d6"),
    ("Cortex", [3, 4], "#eb6834"),
    ("Substance blanche", [5, 6], "#1baf7a"),
    ("Ventricules", [7, 8, 9, 18, 19], "#eda100"),
    ("Noyaux gris centraux, thalamus", [14, 15, 16, 17], "#e87ba4"),
    ("Tronc, cervelet", [10, 11, 12, 13], "#6250d6"),
]
WHITE_COLOR = "#46d2ff"
PIAL_COLOR = "#ff8a3d"
THICKNESS = LinearSegmentedColormap.from_list("epaisseur", ["#fbe7d6", "#f0a66f", "#c8541c", "#6e2306"])
TH_RANGE = (0.5, 2.5)


def load_surface(path):
    g = nib.load(path)
    return g.darrays[0].data.astype(float), g.darrays[1].data.astype(int)


def plane_segments(vertices, faces, y0):
    """Segments (x, z) de l'intersection du maillage avec le plan y = y0."""
    tri = vertices[faces]  # (F, 3, 3)
    d = tri[:, :, 1] - y0
    s = np.sign(d)
    cross = (s.min(1) < 0) & (s.max(1) > 0)
    tri, d = tri[cross], d[cross]
    segs = []
    for (a, b) in [(0, 1), (1, 2), (2, 0)]:
        da, db = d[:, a], d[:, b]
        hit = (da * db) < 0
        t = np.where(hit, da / np.where(hit, da - db, 1.0), np.nan)
        p = tri[:, a] + t[:, None] * (tri[:, b] - tri[:, a])
        segs.append(p[:, [0, 2]])
    pts = np.stack(segs, 1)  # (F, 3, 2), une arête sans intersection vaut NaN
    out = []
    for row in pts:
        ok = row[~np.isnan(row[:, 0])]
        if len(ok) >= 2:
            out.append(ok[:2])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sessions", nargs="+", help="sub-XX/ses-YY")
    ap.add_argument("--surfaces", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--slice", type=float, default=0.45,
                    help="position de la coupe coronale dans l'étendue antéro-postérieure du cerveau (0 : arrière)")
    args = ap.parse_args()

    ga = {(r["subject_id"], r["session_id"]): float(r["ga_weeks"]) for r in csv.DictReader(open(SUBJECTS))}
    n = len(args.sessions)
    fig = plt.figure(figsize=(16, 4.1 * n + 0.9), facecolor="white")
    gs = fig.add_gridspec(n, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.04, hspace=0.12,
                          left=0.11, right=0.99, top=0.93, bottom=0.08)
    # Même cadre (mm) pour toutes les lignes : celui du plus grand cerveau, centré sur chaque cerveau.
    data = []
    for s in args.sessions:
        sub, ses = s.split("/")
        t2 = nib.load(T2W / s / "anat" / f"{sub}_{ses}_res-0p5mm_desc-resampled_T2w.nii.gz")
        seg = nib.load(BOUNTI / s / "anat" / f"{sub}_{ses}_res-0p5mm_desc-bounti19_dseg.nii.gz")
        lab = np.asarray(seg.dataobj).astype(int)
        brain = np.argwhere(np.isin(lab, [3, 4, 5, 6, 7, 8, 14, 15, 16, 17]))
        lo, hi = brain.min(0), brain.max(0)
        data.append((sub, ses, t2, seg, lab, lo, hi))
    aff = data[0][3].affine
    half = max(((hi - lo)[[0, 2]].max() for *_, lo, hi in data)) * aff[0, 0] / 2 + 3

    # Demi-côté commun des vues 3D : la plus grande piale gauche.
    half3 = 0.0
    for sub, ses, *_ in data:
        v, _ = load_surface(args.surfaces / f"{sub}_{ses}_hemi-L_pial.surf.gii")
        half3 = max(half3, (v.max(0) - v.min(0)).max() / 2)
    half3 *= 1.02

    for row, (sub, ses, t2, seg, lab, lo, hi) in enumerate(data):
        a = seg.affine
        img = np.asarray(t2.dataobj).astype(float)
        j = int(round(lo[1] + args.slice * (hi[1] - lo[1])))
        y0 = a[1, 1] * j + a[1, 3]
        center = (lo + hi) / 2
        cx, cz = a[0, 0] * center[0] + a[0, 3], a[2, 2] * center[2] + a[2, 3]
        nx, nz = img.shape[0], img.shape[2]
        extent = [a[0, 3] - a[0, 0] / 2, a[0, 3] + a[0, 0] * (nx - 0.5),
                  a[2, 3] - a[2, 2] / 2, a[2, 3] + a[2, 2] * (nz - 0.5)]
        sl = img[:, j, :].T
        vmax = np.percentile(sl[sl > 0], 99.5) if (sl > 0).any() else 1
        sl_lab = lab[:, j, :].T

        surfs = {}
        for h in "LR":
            for kind in ("white", "pial"):
                p = args.surfaces / f"{sub}_{ses}_hemi-{h}_{kind}.surf.gii"
                surfs[(h, kind)] = load_surface(p)

        axes = [fig.add_subplot(gs[row, c]) for c in range(3)]
        for c, ax in enumerate(axes):
            ax.imshow(sl, cmap="gray", vmin=0, vmax=vmax, origin="lower", extent=extent, interpolation="nearest")
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cz - half, cz + half)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        # Segmentation.
        rgba = np.zeros(sl_lab.shape + (4,))
        for _, labels, color in GROUPS:
            m = np.isin(sl_lab, labels)
            rgba[m] = matplotlib.colors.to_rgba(color, 0.62)
        axes[1].imshow(rgba, origin="lower", extent=extent, interpolation="nearest")
        # Surfaces dans le plan de coupe.
        for (h, kind), (v, f) in surfs.items():
            segs = plane_segments(v, f, y0)
            if len(segs):
                axes[2].add_collection(LineCollection(
                    segs, colors=WHITE_COLOR if kind == "white" else PIAL_COLOR, linewidths=1.1))
        # Barre d'échelle de 10 mm.
        x_bar, z_bar = cx + half - 14, cz - half + 4
        axes[0].plot([x_bar, x_bar + 10], [z_bar, z_bar], color="white", lw=2.5, solid_capstyle="butt")
        axes[0].text(x_bar + 5, z_bar + 1.6, "10 mm", color="white", ha="center", va="bottom", fontsize=9)
        axes[0].text(cx - half + 2, cz + half - 3, "G", color="white", fontsize=10, va="top")
        axes[0].text(cx + half - 2, cz + half - 3, "D", color="white", fontsize=10, va="top", ha="right")
        g_weeks = ga.get((sub, ses))
        axes[0].set_ylabel(f"{g_weeks:.1f} SA\n{sub} {ses}", fontsize=11, rotation=0, ha="right", va="center", labelpad=8)

        # Vue 3D latérale de la piale gauche, colorée par l'épaisseur.
        ax3 = fig.add_subplot(gs[row, 3], projection="3d")
        v, f = surfs[("L", "pial")]
        th = nib.load(args.surfaces / f"{sub}_{ses}_hemi-L_thickness.shape.gii").darrays[0].data
        tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        light = np.array([-0.75, 0.25, 0.6]); light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
        tf = th[f].mean(1)
        col = THICKNESS((np.clip(tf, *TH_RANGE) - TH_RANGE[0]) / (TH_RANGE[1] - TH_RANGE[0]))
        col[:, :3] *= shade[:, None]
        pc = Poly3DCollection(tri, facecolors=col, edgecolors="none", linewidths=0)
        ax3.add_collection3d(pc)
        c3 = (v.min(0) + v.max(0)) / 2
        r3 = half3
        ax3.set_xlim(c3[0] - r3, c3[0] + r3); ax3.set_ylim(c3[1] - r3, c3[1] + r3); ax3.set_zlim(c3[2] - r3, c3[2] + r3)
        ax3.set_box_aspect((1, 1, 1), zoom=1.6)
        ax3.view_init(elev=8, azim=180)
        ax3.set_axis_off()
        ax3.set_proj_type("ortho")
        if row == 0:
            for ax, title in zip(axes + [ax3], ["T2w (coupe coronale)", "Segmentation BOUNTI-19",
                                                "Surfaces surfoxide dans la coupe", "Piale gauche, vue latérale"]):
                ax.set_title(title, fontsize=12, pad=8)
        print(f"{sub} {ses} : coupe y = {y0:.1f} mm")

    # Légendes.
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.62) for _, _, c in GROUPS]
    labels = [name for name, _, _ in GROUPS]
    handles += [plt.Line2D([0], [0], color=WHITE_COLOR, lw=2), plt.Line2D([0], [0], color=PIAL_COLOR, lw=2)]
    labels += ["Surface blanche (WM/GM)", "Surface piale (GM/LCR)"]
    fig.legend(handles, labels, loc="lower left", ncol=4, frameon=False, fontsize=10, bbox_to_anchor=(0.11, 0.005))
    cax = fig.add_axes([0.77, 0.035, 0.18, 0.012])
    sm = plt.cm.ScalarMappable(cmap=THICKNESS, norm=plt.Normalize(*TH_RANGE))
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("épaisseur corticale (mm)", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    fig.suptitle("Surfaces corticales fœtales estimées par surfoxide", fontsize=15, x=0.11, ha="left")
    fig.savefig(args.out, dpi=130)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
