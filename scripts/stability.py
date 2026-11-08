#!/usr/bin/env python3
"""Stabilité de la chaîne de surfoxide (plan qualité, Q0.5) : sensibilité à la grille et au contour des étiquettes.

Sur 8 sessions du développement (folds 0 à 3 de subjects.csv ; le fold 4, test, n'est jamais lu), réparties de 21 à
37 SA, et pour chaque table (BOUNTI-19, multi-BOUNTI-43, jamais mélangées) : 12 réplicats des étiquettes, 4 décalages
sous-voxel de la grille × 3 graines de bruit de contour. Un réplicat interpole chaque label (trilinéaire, un canal par
label) aux positions décalées et déplacées, puis prend l'argmax ; l'affine est translatée du décalage. Le bruit de
contour est un champ de déplacement 3D lisse (bruit blanc par composante, lissé par une gaussienne de --noise-length
mm, moyenne quadratique --noise-sigma mm par composante), qui fait basculer les voxels proches d'une frontière.

`surfx run` sur les étiquettes d'origine (référence) et sur chaque réplicat ; `surfx compare` de chaque réplicat à la
référence. Sorties dans --out : `<table>/<session>/{ref,r<décalage>-s<graine>}/`, `replicates.tsv` (une ligne par
réplicat et hémisphère), `stability.tsv` (par session, table et hémisphère : écart-type relatif des mesures sur les
réplicats, distance moyenne à la référence) et `concordance.tsv` (CCC de Lin BOUNTI-19 / multi-BOUNTI-43 des
références, par mesure).

Usage :
    python3 scripts/stability.py --out ~/scratch/plan-qualite/<dossier> [--jobs 5] [--sessions 8]
        [--tables bounti19,multibounti43] [--only-labels] [--check-determinism]

--only-labels écrit les étiquettes des réplicats sans lancer la chaîne. --check-determinism régénère les étiquettes
de la première session dans un second dossier et les compare (tableaux et affines).
Dépendances : numpy, scipy, nibabel. Données : bash scripts/fetch-data.sh.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

HOME = Path.home()
REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("SURFOXIDE_DATA", HOME / ".cache/surfoxide/data"))
SUBJECTS = Path(os.environ.get("OXIDE_SUBJECTS", DATA / "subjects.csv"))
SURFX = Path(os.environ.get("SURFX", REPO / "target/release/surfx"))
DEVELOPMENT = {"fold_0", "fold_1", "fold_2", "fold_3"}
SHIFTS = {"0": (0.0, 0.0, 0.0), "q": (0.25, 0.25, 0.25), "hx": (0.5, 0.0, 0.0), "h": (0.5, 0.5, 0.5)}
SEEDS = (1, 2, 3)
# Mesures par hémisphère : (colonne, source) ; « report » : rapport de surfx run, « compare » : compare.json.
MEASURES = ["area_white", "area_pial", "thickness", "gi_white", "gi_pial", "basins"]


def seg_path(sub, ses, table):
    return DATA / sub / ses / "anat" / f"{sub}_{ses}_res-0p5mm_desc-{table}_dseg.nii.gz"


def sessions(n, tables):
    rows = list(csv.DictReader(open(SUBJECTS)))
    dev = [r for r in rows if r["split"] in DEVELOPMENT and all(seg_path(r["subject_id"], r["session_id"], t).exists()
                                                                  for t in tables)]
    chosen, used = [], set()
    for target in np.linspace(21, 37, n):
        cands = sorted((abs(float(r["ga_weeks"]) - target), r["subject_id"], r["session_id"], float(r["ga_weeks"]))
                       for r in dev if r["subject_id"] not in used)
        _, sub, ses, ga = cands[0]
        used.add(sub)
        chosen.append((sub, ses, ga))
    return chosen


def replicate(labels, affine, shift, seed, sigma, length):
    """Étiquettes rééchantillonnées : décalage `shift` (voxels) et bruit de contour de graine `seed` (0 : aucun).

    Le déplacement total reste sous 1,3 voxel (décalage ≤ 0,5 ; bruit ≤ ~0,4 mm à 0,5 mm) : un voxel dont le voisinage
    5 × 5 × 5 ne porte qu'un label le garde, seuls les autres sont interpolés.
    """
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(0))
    border = ndimage.maximum_filter(labels, size=5, mode="nearest") != ndimage.minimum_filter(labels, size=5,
                                                                                             mode="nearest")
    idx = np.nonzero(border)
    coords = [idx[a].astype(np.float32) + np.float32(shift[a]) for a in range(3)]
    if seed:
        rng = np.random.default_rng(seed)
        for a in range(3):
            f = ndimage.gaussian_filter(rng.standard_normal(labels.shape).astype(np.float32),
                                        sigma=[length / s for s in spacing], mode="nearest")
            f *= np.float32(sigma / max(float(np.sqrt((f.astype(np.float64) ** 2).mean())), 1e-12))
            coords[a] += f[idx] / np.float32(spacing[a])
    best = np.full(len(idx[0]), -1.0, np.float32)
    lab_b = np.zeros(len(idx[0]), labels.dtype)
    for lab in np.unique(labels[idx]):
        p = ndimage.map_coordinates((labels == lab).astype(np.float32), coords, order=1, mode="nearest")
        take = p > best
        best[take] = p[take]
        lab_b[take] = lab
    out = labels.copy()
    out[idx] = lab_b
    t = np.eye(4)
    t[:3, 3] = shift
    return out, affine @ t


def write_replicates(out, sub, ses, table, args):
    img = nib.load(seg_path(sub, ses, table))
    labels = np.asarray(img.dataobj).round().astype(np.int16)
    d = out / table / f"{sub}_{ses}"
    d.mkdir(parents=True, exist_ok=True)
    paths = {"ref": seg_path(sub, ses, table)}
    for sname, shift in SHIFTS.items():
        for seed in SEEDS:
            name = f"r{sname}-s{seed}"
            p = d / name / "dseg.nii.gz"
            p.parent.mkdir(exist_ok=True)
            if not p.exists():
                lab, aff = replicate(labels, img.affine, shift, seed, args.noise_sigma, args.noise_length)
                hdr = img.header.copy()
                hdr.set_data_dtype(np.int16)
                nib.save(nib.Nifti1Image(lab, aff, hdr), p)
            paths[name] = p
    return d, paths


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def chain(d, name, seg, table):
    w = d / name
    w.mkdir(exist_ok=True)
    if not (w / "run_hemi-R_report.json").exists():
        t0 = time.time()
        run([SURFX, "run", seg, "--labels", table, "-o", w / "run"], w / "run.log")
        (w / "seconds.txt").write_text(f"{time.time() - t0:.1f}\n")
    return w


def compare(ref, w, hemi):
    out = w / f"compare_hemi-{hemi}.json"
    if not out.exists():
        r, e = ref / f"run_hemi-{hemi}", w / f"run_hemi-{hemi}"
        run([SURFX, "compare", "--white", f"{e}_white.surf.gii", "--pial", f"{e}_pial.surf.gii", "--truth-white",
             f"{r}_white.surf.gii", "--truth-pial", f"{r}_pial.surf.gii", "--json", out], w / f"compare-{hemi}.log")
    return json.loads(out.read_text()) if out.exists() else None


def measures(w, cmp, hemi, side):
    """Mesures d'un hémisphère : `side` = « estimated » (réplicat) ou « truth » (référence) dans compare.json."""
    rep = w / f"run_hemi-{hemi}_report.json"
    if cmp is None or not rep.exists():
        return None
    return {
        "area_white": cmp["area"]["white"][side],
        "area_pial": cmp["area"]["pial"][side],
        "thickness": json.loads(rep.read_text()).get("mean_thickness"),
        "gi_white": cmp["gi"]["white"][side],
        "gi_pial": cmp["gi"]["pial"][side],
        "basins": cmp["patches"][side]["basins"],
    }


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mx, my = x.mean(), y.mean()
    return 2 * ((x - mx) * (y - my)).mean() / (x.var() + y.var() + (mx - my) ** 2)


def check_determinism(out, sess, tables, args):
    sub, ses, _ = sess[0]
    bad = 0
    for table in tables:
        img = nib.load(seg_path(sub, ses, table))
        labels = np.asarray(img.dataobj).round().astype(np.int16)
        for sname, shift in SHIFTS.items():
            for seed in SEEDS:
                lab, aff = replicate(labels, img.affine, shift, seed, args.noise_sigma, args.noise_length)
                prev = nib.load(out / table / f"{sub}_{ses}" / f"r{sname}-s{seed}" / "dseg.nii.gz")
                same = np.array_equal(np.asarray(prev.dataobj), lab) and np.array_equal(prev.affine, aff)
                bad += not same
                print(f"{table} {sub}_{ses} r{sname}-s{seed} : {'identique' if same else 'DIFFÉRENT'}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--tables", default="bounti19,multibounti43")
    ap.add_argument("--noise-sigma", type=float, default=0.1)
    ap.add_argument("--noise-length", type=float, default=1.5)
    ap.add_argument("--only-labels", action="store_true")
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()
    if not SUBJECTS.exists():
        sys.exit(f"erreur : {SUBJECTS} absent ; lancer bash scripts/fetch-data.sh")
    tables = args.tables.split(",")
    sess = sessions(args.sessions, tables)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability.json").write_text(json.dumps({
        "argv": sys.argv, "sessions": sess, "shifts": SHIFTS, "seeds": SEEDS,
        "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True).stdout.strip()}, indent=1) + "\n")
    print("sessions :", ", ".join(f"{s}_{t} ({g} SA)" for s, t, g in sess), flush=True)
    with cf.ThreadPoolExecutor(min(args.jobs, 2)) as ex:
        dirs = list(ex.map(lambda t: write_replicates(out, t[0][0], t[0][1], t[1], args),
                           [(s, t) for s in sess for t in tables]))
    if args.check_determinism:
        bad = check_determinism(out, sess, tables, args)
        print(f"réplicats différents : {bad}")
        return 1 if bad else 0
    if args.only_labels:
        return 0
    if not SURFX.exists():
        sys.exit(f"erreur : {SURFX} absent ; cargo build --release -p surfx")
    keys = [(s, t) for s in sess for t in tables]
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        futs = {}
        for (s, t), (d, paths) in zip(keys, dirs):
            for name, p in paths.items():
                futs[ex.submit(chain, d, name, p, t)] = (s, t, name)
        for f in cf.as_completed(futs):
            s, t, name = futs[f]
            print(f"{t} {s[0]}_{s[1]} {name} fait", flush=True)
    rows = []
    for (s, t), (d, paths) in zip(keys, dirs):
        for hemi in "LR":
            ref = d / "ref"
            for name in paths:
                if name == "ref":
                    continue
                cmp = compare(ref, d / name, hemi)
                m = measures(d / name, cmp, hemi, "estimated")
                row = {"table": t, "session": f"{s[0]}_{s[1]}", "ga": s[2], "hemi": hemi, "replicate": name}
                if m:
                    row.update(m)
                    row["dist_white"] = cmp["white"]["mean"]
                    row["dist_pial"] = cmp["pial"]["mean"]
                rows.append(row)
            first = next(n for n in paths if n != "ref")
            cmp = compare(ref, d / first, hemi)
            m = measures(ref, cmp, hemi, "truth")
            rows.append({"table": t, "session": f"{s[0]}_{s[1]}", "ga": s[2], "hemi": hemi, "replicate": "ref",
                         **(m or {})})
    cols = ["table", "session", "ga", "hemi", "replicate"] + MEASURES + ["dist_white", "dist_pial"]
    with open(out / "replicates.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(out / "stability.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["table", "session", "ga", "hemi", "n"] + [f"rsd_{m}" for m in MEASURES]
                   + ["dist_white", "dist_pial"])
        for (s, t) in keys:
            for hemi in "LR":
                rs = [r for r in rows if r["table"] == t and r["session"] == f"{s[0]}_{s[1]}" and r["hemi"] == hemi
                      and r["replicate"] != "ref" and "area_white" in r]
                cells = []
                for m in MEASURES:
                    x = [r[m] for r in rs if r.get(m) is not None]
                    mean = statistics.fmean(x) if x else 0
                    cells.append(f"{statistics.stdev(x) / mean:.4g}" if len(x) > 1 and mean else "")
                dw = [r["dist_white"] for r in rs]
                dp = [r["dist_pial"] for r in rs]
                w.writerow([t, f"{s[0]}_{s[1]}", s[2], hemi, len(rs)] + cells
                           + [f"{statistics.fmean(dw):.4g}" if dw else "", f"{statistics.fmean(dp):.4g}" if dp else ""])
    if len(tables) == 2:
        with open(out / "concordance.tsv", "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["measure", "n", "ccc", "mean_" + tables[0], "mean_" + tables[1]])
            ref = {(r["table"], r["session"], r["hemi"]): r for r in rows if r["replicate"] == "ref"}
            for m in MEASURES:
                pairs = [(ref[(tables[0], k[1], k[2])].get(m), ref[(tables[1], k[1], k[2])].get(m))
                         for k in ref if k[0] == tables[0] and (tables[1], k[1], k[2]) in ref]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if len(pairs) > 1:
                    a, b = zip(*pairs)
                    w.writerow([m, len(pairs), f"{ccc(a, b):.4f}", f"{np.mean(a):.4g}", f"{np.mean(b):.4g}"])
    print(f"écrit : {out / 'replicates.tsv'}, {out / 'stability.tsv'}, {out / 'concordance.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
