#!/usr/bin/env python3
"""Benchmark de la profondeur sulcale (Dieudonné, Auzias, Lefèvre, Comput Biol Med 2026, expérience 1).

Pour chaque surface et sa texture de repères tracés à la main (Zenodo 10.5281/zenodo.14265228 : 500 crêtes
gyrales, 100 fonds de sillons, 10 à 90 lignes de paroi allant d'un fond à une crête, 0 ailleurs), calcule la
DPF* de surfoxide (`surfx depth`) pour plusieurs α_ref et les trois critères de l'article :

- StdCrest = écart-type de D sur les crêtes / IPR(D)        (idéal : 0)
- Sep      = (médiane de D sur les crêtes − médiane sur les fonds) / IPR(D)   (idéal : 1)
- Dev      = angle moyen (degrés) entre le gradient de D et la direction des lignes de paroi, orientées du
             fond vers la crête                                  (idéal : 0 ; 90 = aucun lien)

IPR = percentile 95 − percentile 5 de D sur toute la surface. Convention de surfoxide : D positive sur les
crêtes, d'où une orientation fond → crête.

    python3 scripts/dpf_benchmark.py --pairs paires.tsv --out resultats.tsv [--alpha 50 100 200 500 1000]
    python3 scripts/dpf_benchmark.py --selftest

`paires.tsv` : deux colonnes sans en-tête, surface (.surf.gii) puis repères (.gii), même nombre de sommets.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components

CREST, FUNDUS = 500, 100
WALLS = range(10, 100, 10)


def read_surface(path):
    g = nib.load(str(path))
    return g.agg_data("NIFTI_INTENT_POINTSET").astype(float), g.agg_data("NIFTI_INTENT_TRIANGLE").astype(np.int64)


def adjacency(n, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    return ((a + a.T) > 0).astype(np.int8).tocsr()


def vertex_gradient(v, faces, d):
    """Gradient de `d` aux sommets : moyenne des gradients des faces (linéaires par face) pondérée par l'aire."""
    p0, p1, p2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(n, axis=1, keepdims=True)
    nu = n / np.maximum(area2, 1e-30)
    # ∇d = Σ_i d_i (n̂ × e_i) / (2A), e_i arête opposée au sommet i, orientée dans le sens de la face.
    g = (d[faces[:, 0], None] * np.cross(nu, p2 - p1) + d[faces[:, 1], None] * np.cross(nu, p0 - p2)
         + d[faces[:, 2], None] * np.cross(nu, p1 - p0)) / np.maximum(area2, 1e-30)
    out = np.zeros_like(v)
    w = area2[:, 0]
    for k in range(3):
        np.add.at(out, faces[:, k], g * w[:, None])
    return out


def line_tangents(v, adj, labels):
    """Pour chaque sommet d'une ligne de paroi : tangente unitaire orientée du fond vers la crête (NaN sinon)."""
    t = np.full_like(v, np.nan)
    fundus = labels == FUNDUS
    crest = labels == CREST
    for w in WALLS:
        m = labels == w
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        sub = adj[idx][:, idx]
        ncomp, comp = connected_components(sub, directed=False)
        for c in range(ncomp):
            ids = np.flatnonzero(comp == c)
            if len(ids) < 3:
                continue
            verts = idx[ids]
            # Départ : le sommet de la ligne voisin d'un fond (sinon le plus proche d'un fond).
            touch = np.asarray(adj[verts][:, np.flatnonzero(fundus)].sum(axis=1)).ravel() > 0
            if touch.any():
                start = ids[np.argmax(touch)]
            else:
                fpts = v[fundus] if fundus.any() else v[crest]
                dist = np.min(np.linalg.norm(v[verts][:, None, :] - fpts[None, ::max(1, len(fpts) // 2000)], axis=2), axis=1)
                start = ids[np.argmin(dist)]
            order, pred = breadth_first_order(sub, start, directed=False, return_predecessors=True)
            rank = np.full(len(idx), -1)
            rank[order] = np.arange(len(order))
            for k in ids:
                nb = sub[k].indices
                nb = nb[comp[nb] == c]
                if len(nb) == 0:
                    continue
                ahead = nb[rank[nb] > rank[k]]
                behind = nb[rank[nb] < rank[k]]
                a = v[idx[ahead]].mean(axis=0) if len(ahead) else v[idx[k]]
                b = v[idx[behind]].mean(axis=0) if len(behind) else v[idx[k]]
                d = a - b
                nd = np.linalg.norm(d)
                if nd > 0:
                    t[idx[k]] = d / nd
    return t


def criteria(v, faces, labels, d):
    ipr = np.percentile(d, 95) - np.percentile(d, 5)
    crest, fundus = labels == CREST, labels == FUNDUS
    std_crest = np.std(d[crest]) / ipr
    sep = (np.median(d[crest]) - np.median(d[fundus])) / ipr
    adj = adjacency(len(v), faces)
    t = line_tangents(v, adj, labels)
    g = vertex_gradient(v, faces, d)
    ok = ~np.isnan(t[:, 0]) & (np.linalg.norm(g, axis=1) > 0)
    gu = g[ok] / np.linalg.norm(g[ok], axis=1, keepdims=True)
    dev = np.degrees(np.arccos(np.clip((gu * t[ok]).sum(axis=1), -1, 1))).mean() if ok.any() else np.nan
    return std_crest, sep, dev, int(ok.sum())


def dpf_star(surfx, surface, alpha, tmp):
    prefix = Path(tmp) / f"{Path(surface).name}_a{alpha}"
    subprocess.run([surfx, "depth", str(surface), "--alpha", str(alpha), "-o", str(prefix)], check=True,
                   capture_output=True)
    return nib.load(f"{prefix}_dpfstar.shape.gii").darrays[0].data.astype(float)


def make_bumpy(level=6):
    # Icosphère par subdivision, puis ondulation en azimut.
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5],
                  [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(level):
        verts = list(map(tuple, v))
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (np.array(verts[a]) + np.array(verts[b])) / 2
                verts.append(tuple(m / np.linalg.norm(m)))
                cache[key] = len(verts) - 1
            return cache[key]
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v, f = np.array(verts), np.array(nf)
    phi = np.arctan2(v[:, 1], v[:, 0])
    r = 30 * (1 + 0.08 * np.cos(8 * phi))
    return v * r[:, None], f, phi, v[:, 2]


def run_selftest(surfx):
    """Sphère ondulée r = 30 (1 + 0,08 cos 8φ) : crêtes aux maxima de r, fonds aux minima, parois le long de φ.
    Contrôle de vraisemblance (pas une reproduction de l'article) : Sep > 0,5, StdCrest < 0,1, Dev < 30°, et Dev
    > 150° pour −D (orientation des lignes du fond vers la crête)."""
    v, f, phi, z = make_bumpy()
    labels = np.zeros(len(v))
    band = np.abs(z) < 0.5
    c = np.cos(8 * phi)
    labels[band & (c > 0.995)] = CREST
    labels[band & (c < -0.995)] = FUNDUS
    # Parois : un parallèle (z ≈ 0,3) entre chaque fond et la crête suivante, 8 portions par label 10.
    s = np.sin(8 * phi)
    labels[(np.abs(z - 0.3) < 0.012) & (np.abs(c) < 0.9) & (s < 0)] = 10
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ondulee.surf.gii"
        img = nib.gifti.GiftiImage(darrays=[
            nib.gifti.GiftiDataArray(v.astype(np.float32), intent="NIFTI_INTENT_POINTSET"),
            nib.gifti.GiftiDataArray(f.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")])
        nib.save(img, str(path))
        for a in (50, 500):
            d = dpf_star(surfx, path, a, tmp)
            std_crest, sep, dev, n = criteria(v, f, labels, d)
            print(f"sphère ondulée, α_ref {a} : StdCrest {std_crest:.3f}, Sep {sep:.3f}, Dev {dev:.1f}° ({n} sommets de paroi)")
            if not (sep > 0.5 and std_crest < 0.1 and dev < 30):
                sys.exit("autotest : critères hors des valeurs attendues")
            # Signe : une profondeur inversée doit pointer à contresens des lignes de paroi.
            _, _, dev_inv, _ = criteria(v, f, labels, -d)
            if dev_inv < 150:
                sys.exit(f"autotest : orientation des lignes douteuse (Dev de −D = {dev_inv:.1f}°)")
    print("autotest réussi")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--alpha", type=float, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--surfx", default=str(Path(__file__).resolve().parent.parent / "target/release/surfx"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return run_selftest(args.surfx)
    if not args.pairs or not args.out:
        ap.error("--pairs et --out, ou --selftest")
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for line in open(args.pairs):
            if not line.strip():
                continue
            surface, marks = line.split()[:2]
            v, f = read_surface(surface)
            labels = nib.load(marks).darrays[0].data.astype(float)
            if len(labels) != len(v):
                sys.exit(f"{marks} : {len(labels)} valeurs pour {len(v)} sommets")
            for a in args.alpha:
                d = dpf_star(args.surfx, surface, a, tmp)
                std_crest, sep, dev, n = criteria(v, f, labels, d)
                rows.append((Path(surface).name, a, std_crest, sep, dev, n))
                print(f"{Path(surface).name} α_ref {a:g} : StdCrest {std_crest:.3f}, Sep {sep:.3f}, Dev {dev:.1f}°")
    with open(args.out, "w") as o:
        o.write("surface\talpha_ref\tstd_crest\tsep\tdev_deg\tn_wall_vertices\n")
        for r in rows:
            o.write("\t".join(str(x) for x in r) + "\n")
    print(f"écrit : {args.out}")


if __name__ == "__main__":
    main()
