#!/usr/bin/env python3
"""Banc de fantômes du plan qualité (Q0.3) : chaîne de surfoxide jugée contre la vérité.

Pour chaque fantôme × σ du bruit de contour × variante : `surfx phantom` (fantôme A, atlas fœtal dHCP, une semaine et
un hémisphère ; ou fantôme B, configuration analytique), `surfx run` (labels BOUNTI-19, options de la variante),
`surfx compare` contre les surfaces et champs vrais. Un fantôme est généré une fois pour toutes les variantes ; un
résultat déjà écrit n'est pas recalculé.

Variantes : `--variant nom=options` (options de `surfx run`, comme SURFX_RUN_ARGS dans validate.py), répétable ; sans
`--variant`, une seule variante « defaut » avec les options de $SURFX_RUN_ARGS (vide en général).

Sorties dans --out : `phantoms/v<voxel>_s<σ>/<fantôme>/ph_*`, `runs/<variante>/v<voxel>_s<σ>/<fantôme>/{run_*,
compare.json}`, `results.tsv` (une ligne par fantôme × σ × variante) et `summary.tsv` (médianes par variante × σ ×
tranche d'âge — 20–24, 24–28, 28–32, 32–38 SA — pour les fantômes A, par configuration pour les fantômes B).

Usage :
    python3 scripts/phantom_bench.py --out ~/scratch/plan-qualite/<dossier> [--sigmas 0,0.1,0.2,0.3]
        [--voxel 0.5] [--weeks 21-36] [--hemis L,R] [--analytic wide,narrow,buried,variable] [--no-atlas]
        [--seed 1] [--jobs 5] [--variant nom=options …]

Atlas : $OXIDE_DHCP_ATLAS (défaut data/atlas_dhcp/ du clone de gyroxide voisin). Bibliothèque standard seulement.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SURFX = Path(os.environ.get("SURFX", REPO / "target/release/surfx"))
ATLAS = Path(os.environ.get("OXIDE_DHCP_ATLAS", REPO.parent / "gyroxide/data/atlas_dhcp"))
BINS = [(20, 24), (24, 28), (28, 32), (32, 38)]
# Colonnes de results.tsv tirées de compare.json : (nom de colonne, chemin dans le JSON).
METRICS = [
    ("white_mean", ("white", "mean")),
    ("white_hd95", ("white", "hd95")),
    ("pial_mean", ("pial", "mean")),
    ("pial_hd95", ("pial", "hd95")),
    ("thickness_bias", ("thickness", "bias")),
    ("thickness_mad", ("thickness", "median_abs")),
    ("curv_pearson", ("curv", "pearson")),
    ("curv_spearman", ("curv", "spearman")),
    ("dpf_pearson", ("dpfstar", "pearson")),
    ("dpf_spearman", ("dpfstar", "spearman")),
    ("area_white_rel", ("area", "white", "relative_error")),
    ("area_pial_rel", ("area", "pial", "relative_error")),
    ("gi_white_rel", ("gi", "white", "relative_error")),
    ("gi_pial_rel", ("gi", "pial", "relative_error")),
    ("basins_est", ("patches", "estimated", "basins")),
    ("basins_true", ("patches", "truth", "basins")),
    ("small_est", ("patches", "estimated", "small")),
    ("small_true", ("patches", "truth", "small")),
]


def run(cmd, log):
    t0 = time.time()
    with open(log, "w") as f:
        p = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def phantoms(args):
    """(nom, âge ou None, hémisphère, options de surfx phantom)."""
    out = []
    if not args.no_atlas:
        lo, hi = (int(x) for x in args.weeks.split("-"))
        for w in range(lo, hi + 1):
            for h in args.hemis.split(","):
                side = {"L": "left", "R": "right"}[h]
                stem = ATLAS / f"fetal.week{w}.{side}"
                out.append((f"w{w}{h}", float(w), h, ["--white", f"{stem}.wm.surf.gii", "--pial",
                                                      f"{stem}.pial.surf.gii", "--medial", f"{stem}.sulc.shape.gii"]))
    for c in filter(None, args.analytic.split(",")):
        out.append((f"b-{c}", None, "L", ["--analytic", c]))
    return out


def make_phantom(root, ph, sigma, args):
    name, _, hemi, opts = ph
    d = root / "phantoms" / f"v{args.voxel}_s{sigma}" / name
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "ph_phantom.json").exists():
        code, _ = run([SURFX, "phantom", *opts, "--hemi", hemi, "--voxel", args.voxel, "--noise-sigma", sigma,
                       "--seed", args.seed, "-o", d / "ph"], d / "phantom.log")
        if code != 0:
            return None
    return d


def one(root, ph, sigma, variant, vargs, pdir, args):
    name, age, hemi, _ = ph
    d = root / "runs" / variant / f"v{args.voxel}_s{sigma}" / name
    d.mkdir(parents=True, exist_ok=True)
    row = {"variant": variant, "sigma": sigma, "voxel": args.voxel, "phantom": name, "age": age, "hemi": hemi}
    report = d / f"run_hemi-{hemi}_report.json"
    seconds = d / "seconds.txt"
    if not report.exists():
        code, secs = run([SURFX, "run", pdir / "ph_dseg.nii.gz", "--labels", "bounti19", "--hemi", hemi, "-o",
                          d / "run", *vargs], d / "run.log")
        seconds.write_text(f"{secs:.1f}\n")
        if not report.exists():
            row["valid"] = None
            row["returncode"] = code
            return row
    row["seconds"] = float(seconds.read_text()) if seconds.exists() else None
    row["valid"] = json.loads(report.read_text())["valid"]
    cmp = d / "compare.json"
    if not cmp.exists():
        t = pdir / f"ph_hemi-{hemi}_desc-truth"
        code, _ = run([SURFX, "compare", "--white", d / f"run_hemi-{hemi}_white.surf.gii", "--pial",
                       d / f"run_hemi-{hemi}_pial.surf.gii", "--truth-white", f"{t}_white.surf.gii", "--truth-pial",
                       f"{t}_pial.surf.gii", "--thickness", d / f"run_hemi-{hemi}_thickness.shape.gii",
                       "--truth-fields", *(f"{t}_{k}.shape.gii" for k in ("thickness", "curv", "dpfstar", "mask")),
                       "--json", cmp], d / "compare.log")
        if code != 0:
            return row
    c = json.loads(cmp.read_text())
    for col, path in METRICS:
        v = c
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        row[col] = v
    return row


def group(row):
    if row["age"] is None:
        return row["phantom"]
    for lo, hi in BINS:
        if lo <= row["age"] < hi:
            return f"{lo}-{hi}"
    return "hors"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sigmas", default="0,0.1,0.2,0.3")
    ap.add_argument("--voxel", default="0.5")
    ap.add_argument("--weeks", default="21-36")
    ap.add_argument("--hemis", default="L,R")
    ap.add_argument("--analytic", default="wide,narrow,buried,variable")
    ap.add_argument("--no-atlas", action="store_true")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--variant", action="append", default=[], help="nom=options de surfx run")
    args = ap.parse_args()
    if not SURFX.exists():
        sys.exit(f"erreur : {SURFX} absent ; cargo build --release -p surfx")
    if not args.no_atlas and not ATLAS.is_dir():
        sys.exit(f"erreur : atlas {ATLAS} absent ($OXIDE_DHCP_ATLAS)")
    variants = [v.split("=", 1) for v in args.variant] or [["defaut", os.environ.get("SURFX_RUN_ARGS", "")]]
    variants = [(n, shlex.split(o)) for n, o in variants]
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    (root / "bench.json").write_text(json.dumps({
        "argv": sys.argv, "surfx": str(SURFX),
        "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "variants": {n: o for n, o in variants}}, indent=1, ensure_ascii=False) + "\n")
    sigmas = args.sigmas.split(",")
    phs = phantoms(args)
    rows = []
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        pdirs = dict(zip([(p[0], s) for s in sigmas for p in phs],
                         ex.map(lambda t: make_phantom(root, t[0], t[1], args), [(p, s) for s in sigmas for p in phs])))
        tasks = [ex.submit(one, root, p, s, n, o, pdirs[(p[0], s)], args) for s in sigmas for p in phs
                 for n, o in variants if pdirs[(p[0], s)] is not None]
        failed = [(p[0], s) for (p, s) in [(p, s) for s in sigmas for p in phs] if pdirs[(p[0], s)] is None]
        for t in cf.as_completed(tasks):
            r = t.result()
            rows.append(r)
            print(f"{r['variant']} σ={r['sigma']} {r['phantom']} conforme={r.get('valid')}", flush=True)
    for name, s in failed:
        print(f"erreur : fantôme {name} σ={s} non généré", file=sys.stderr)
    cols = ["variant", "sigma", "voxel", "phantom", "age", "hemi", "valid", "seconds"] + [c for c, _ in METRICS]
    rows.sort(key=lambda r: (r["variant"], float(r["sigma"]), r["age"] or 99, r["phantom"]))
    with open(root / "results.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        groups.setdefault((r["variant"], r["sigma"], group(r)), []).append(r)
    with open(root / "summary.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["variant", "sigma", "group", "n", "valid"] + [c for c, _ in METRICS])
        for (v, s, g), rs in sorted(groups.items(), key=lambda k: (k[0][0], float(k[0][1]), k[0][2])):
            med = []
            for c, _ in METRICS:
                x = [r[c] for r in rs if isinstance(r.get(c), (int, float))]
                med.append(f"{statistics.median(x):.4g}" if x else "")
            w.writerow([v, s, g, len(rs), sum(1 for r in rs if r.get("valid"))] + med)
    print(f"écrit : {root / 'results.tsv'}, {root / 'summary.tsv'}")
    return 0 if not failed and all(r.get("valid") is not None for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
