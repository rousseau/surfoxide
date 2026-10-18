#!/usr/bin/env python3
"""Figure : image T2w, segmentation BOUNTI et surfaces surfoxide pour quelques âges gestationnels.

Une ligne par session. Colonnes :
  1. coupe coronale T2w ;
  2. segmentation BOUNTI-19 superposée (tissus regroupés) ;
  3. intersection des surfaces blanche et piale avec le plan de coupe ;
  4. vue latérale 3D de la piale gauche (rendu à la FreeSurfer : gyrus clairs, sillons foncés).
Même échelle (mm) sur toutes les lignes, convention neurologique (gauche à gauche).

Usage :
    python3 scripts/illustrate.py --data <dérivées> --surfaces <dossier des sorties surfx run> \\
        --out figure.png sub-d6a1p/ses-01 sub-0fk6m/ses-01 sub-0avbc/ses-02

<dérivées> contient <sub>/<ses>/anat/ avec la T2w rééchantillonnée à 0,5 mm et la segmentation
BOUNTI-19 sur la même grille (garage:crl2025/derivatives/fetal-folding-geometry, à rapatrier).

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

HOME = Path.home()
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


def load_surface(path):
    g = nib.load(path)
    return g.darrays[0].data.astype(float), g.darrays[1].data.astype(int)


def _neighbor_mean(faces, n):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(float)

    def mean(x):
        acc = np.zeros_like(x)
        np.add.at(acc, e[:, 0], x[e[:, 1]])
        return acc / (deg[:, None] if x.ndim == 2 else deg)

    return mean


def _vertex_normals(vertices, faces):
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vn = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)


def vertex_convexity(vertices, faces, scale_mm=5.0):
    """Mesure de type `sulc` : déplacement signé de chaque sommet par rapport à une version lissée de la
    surface, le long de la normale. Positif sur les crêtes (gyrus), négatif au fond des sillons. Le
    lissage (laplacien itéré) couvre environ `scale_mm`."""
    mean = _neighbor_mean(faces, len(vertices))
    edge = np.linalg.norm(vertices[faces[:, 1]] - vertices[faces[:, 0]], axis=1).mean()
    smooth = vertices.copy()
    for _ in range(int((scale_mm / edge) ** 2) + 1):
        smooth = mean(smooth)
    vn = _vertex_normals(vertices, faces)
    conv = ((vertices - smooth) * vn).sum(1)
    # Lissage de la carte elle-même : efface les ondulations des marches de voxels.
    for _ in range(15):
        conv = mean(conv)
    return conv


def render_surface(vertices, faces, center, half, size=600, azim=180, elev=8):
    """Image (niveaux de gris, fond blanc) de la surface vue de loin, rendu à la FreeSurfer : gris clair sur
    les gyrus, gris foncé dans les sillons (couleur presque binaire, comme la courbure dans Freeview),
    éclairage doux venant de l'observateur, normales lissées.

    Projection orthographique avec tampon de profondeur par pixel : le tri des faces de matplotlib laisse
    voir des faces cachées (l'insula à travers l'opercule temporal). Le cadre fait 2 `half` mm de côté
    autour de `center`, pour que toutes les vues partagent la même échelle."""
    az, el = np.radians(azim), np.radians(elev)
    view = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.cross([0.0, 0.0, 1.0], view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    rel = vertices - np.asarray(center)
    px = size / (2 * half)
    xy = np.stack([rel @ right, rel @ up], 1) * px + size / 2
    depth = rel @ view  # grand : proche de l'observateur

    conv = vertex_convexity(vertices, faces)
    mean = _neighbor_mean(faces, len(vertices))
    vn = _vertex_normals(vertices, faces)
    for _ in range(4):
        vn = mean(vn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    light = view + np.array([0.0, 0.0, 0.35])
    light /= np.linalg.norm(light)
    diffuse = np.clip(vn @ light, 0, 1)
    t = 0.5 + 0.5 * np.tanh(-conv / (np.percentile(np.abs(conv), 50) + 1e-12) * 1.5)  # 0 : gyrus
    shade = (0.82 - 0.48 * t) * (0.55 + 0.45 * diffuse)

    img = np.ones((size, size))
    zbuf = np.full((size, size), -np.inf)
    # Faces tournées vers l'observateur seulement (les autres sont cachées sur une surface fermée).
    tri_xy = xy[faces]
    area = ((tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
            - (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1]))
    for k in np.nonzero(area > 1e-12)[0]:
        tri = faces[k]
        a, b, c = tri_xy[k]
        x0, x1 = max(int(np.floor(min(a[0], b[0], c[0]))), 0), min(int(np.ceil(max(a[0], b[0], c[0]))), size - 1)
        y0, y1 = max(int(np.floor(min(a[1], b[1], c[1]))), 0), min(int(np.ceil(max(a[1], b[1], c[1]))), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        d = area[k]
        w0 = ((b[0] - X) * (c[1] - Y) - (c[0] - X) * (b[1] - Y)) / d
        w1 = ((c[0] - X) * (a[1] - Y) - (a[0] - X) * (c[1] - Y)) / d
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not m.any():
            continue
        z = (w0 * depth[tri[0]] + w1 * depth[tri[1]] + w2 * depth[tri[2]])[m]
        sh = (w0 * shade[tri[0]] + w1 * shade[tri[1]] + w2 * shade[tri[2]])[m]
        ys, xs = (Y[m] - 0.5).astype(int), (X[m] - 0.5).astype(int)
        closer = z > zbuf[ys, xs]
        zbuf[ys[closer], xs[closer]] = z[closer]
        img[ys[closer], xs[closer]] = sh[closer]
    return img[::-1]


def show_surface(ax, vertices, faces, center, half, **kw):
    """Affiche le rendu de [`render_surface`] dans des axes 2D."""
    ax.imshow(render_surface(vertices, faces, center, half, **kw), cmap="gray", vmin=0, vmax=1,
              interpolation="lanczos")
    ax.set_axis_off()


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
    ap.add_argument("--data", type=Path, required=True, help="dossier <sub>/<ses>/anat/ (T2w, BOUNTI)")
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
        t2 = nib.load(args.data / s / "anat" / f"{sub}_{ses}_res-0p5mm_desc-resampled_T2w.nii.gz")
        seg = nib.load(args.data / s / "anat" / f"{sub}_{ses}_res-0p5mm_desc-bounti19_dseg.nii.gz")
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
    half3 *= 1.08

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

        # Vue 3D latérale de la piale gauche, rendu à la FreeSurfer.
        ax3 = fig.add_subplot(gs[row, 3])
        v, f = surfs[("L", "pial")]
        show_surface(ax3, v, f, (v.min(0) + v.max(0)) / 2, half3)
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
    fig.suptitle("Surfaces corticales fœtales estimées par surfoxide", fontsize=15, x=0.11, ha="left")
    fig.savefig(args.out, dpi=130)
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
