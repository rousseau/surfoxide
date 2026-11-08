#!/usr/bin/env python3
"""Tableau de référence du plan qualité (Q0.7) : la chaîne actuelle mesurée par tous les instruments de la phase 0.

Familles (plan qualité) et sources :
- S1 contrat, S2 fidélité aux étiquettes, S8 maillage, S9 temps : `results.json` d'une validation (`validate.py
  --all`), cohorte complète, et séparément développement (folds 0 à 3) et test (fold 4) ;
- S3 exactitude : `summary.tsv` de `scripts/phantom_bench.py` (fantômes A par tranche d'âge, B par configuration,
  chaque σ du bruit de contour) ;
- S4 bruit de forme : petites taches de D* (< 5 mm²) et bassins (D* < 0, ≥ 5 mm²) de la blanche, par `surfx compare`
  de la blanche avec elle-même, développement ;
- S5 stabilité : `stability.tsv` et `concordance.tsv` de `scripts/stability.py` ;
- S6 image : `surfx profile` (blanche et piale, face médiale exclue par le champ `cortex`, décalage corrigé par la
  courbe d'étalonnage de Q0.4b) sur les surfaces de la validation et la T2w rééchantillonnée, développement ;
- repères DPF* : `resultats.tsv` de `scripts/dpf_benchmark.py` (médianes par α_ref).

Sorties dans --out : `profile/` et `s4/` (JSON par hémisphère, non recalculés s'ils existent), `reference.tsv`
(format long : famille, mesure, table, ensemble, groupe, n, médiane, q1, q3 ; pour taux_conforme, médiane = taux
et q1 = nombre de conformes) et `test-fold.tsv` (sessions du fold test). Tranches d'âge : 20–24, 24–28, 28–32, 32–38 SA, plus « toutes ».

Usage :
    python3 scripts/reference.py --validation ~/scratch/surfoxide-validation-15 \
        --phantoms <dossier de phantom_bench> --stability <dossier de stability.py> \
        --calibration <calibration_seed-1.tsv> --dpf <resultats.tsv> --out <dossier> [--jobs 4]

Données : bash scripts/fetch-data.sh, plus --t2w pour les sessions du développement. Bibliothèque standard
seulement.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SURFX = Path(os.environ.get("SURFX", REPO / "target/release/surfx"))
DATA = Path(os.environ.get("SURFOXIDE_DATA", Path.home() / ".cache/surfoxide/data"))
SUBJECTS = Path(os.environ.get("OXIDE_SUBJECTS", DATA / "subjects.csv"))
BINS = [(20, 24), (24, 28), (28, 32), (32, 38)]
TABLES = ["bounti19", "multibounti43"]


def age_bin(ga):
    for lo, hi in BINS:
        if lo <= ga < hi:
            return f"{lo}-{hi}"
    return "hors"


def quartiles(x):
    x = sorted(x)
    if not x:
        return None
    if len(x) == 1:
        return x[0], x[0], x[0]
    q = statistics.quantiles(x, n=4, method="inclusive")
    return q[1], q[0], q[2]


class Table:
    def __init__(self):
        self.rows = []

    def add(self, family, measure, table, subset, group, values):
        values = [v for v in values if isinstance(v, (int, float)) and v == v]
        q = quartiles(values)
        if q is None:
            return
        self.rows.append([family, measure, table, subset, group, len(values), *(f"{v:.4g}" for v in q)])

    def by_bins(self, family, measure, table, subset, items):
        """`items` : liste de (âge, valeur) ; une ligne par tranche et une pour toutes."""
        for lo, hi in BINS:
            self.add(family, measure, table, subset, f"{lo}-{hi}", [v for a, v in items if lo <= a < hi])
        self.add(family, measure, table, subset, "toutes", [v for _, v in items])

    def write(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["family", "measure", "table", "subset", "group", "n", "median", "q1", "q3"])
            w.writerows(self.rows)


def folds():
    with open(SUBJECTS) as f:
        return {(r["subject_id"], r["session_id"]): (float(r["ga_weeks"]), r["split"]) for r in csv.DictReader(f)}


def get(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def validation_measures(results, split, t):
    """S1, S2, S8, S9 par table × ensemble (cohorte, développement, test) × tranche d'âge."""
    hemi_measures = [
        ("S1", "conforme", lambda h: float(bool(get(h, "report", "valid")))),
        ("S1", "distance_min", lambda h: get(h, "report", "pair", "min_distance")),
        ("S1", "degagement_min_piale", lambda h: get(h, "report", "pial", "min_clearance")),
        ("S1", "degagement_min_blanche", lambda h: get(h, "report", "white", "min_clearance")),
        ("S1", "arete_min_piale", lambda h: get(h, "report", "pial_min_edge")),
        ("S1", "angle_min_piale", lambda h: get(h, "report", "pial_min_angle")),
        ("S2", "hd95_blanche", lambda h: get(h, "white_vs_white", "hausdorff95")),
        ("S2", "dist_moy_blanche", lambda h: get(h, "white_vs_white", "surface_to_boundary", "mean")),
        ("S2", "hd95_piale", lambda h: get(h, "pial_vs_opened", "hausdorff95")),
        ("S2", "dist_moy_piale", lambda h: get(h, "pial_vs_opened", "surface_to_boundary", "mean")),
        ("S7", "epaisseur_moyenne", lambda h: get(h, "report", "mean_thickness")),
        ("S7", "aire_blanche", lambda h: get(h, "report", "white", "area")),
        ("S8", "sommets_retenus_frac", lambda h: (get(h, "report", "pial_held_vertices") or 0)
         / max(get(h, "report", "pial", "n_vertices") or 1, 1)),
        ("S8", "manque_moyen_piale", lambda h: get(h, "report", "pial_mean_missing")),
        ("S8", "angle_min_moyen_blanche", lambda h: get(h, "white_vs_white", "quality", "mean_min_angle")),
        ("S8", "angle_min_moyen_piale", lambda h: get(h, "pial_vs_opened", "quality", "mean_min_angle")),
    ]
    subsets = {"cohorte": lambda s: True, "dev": lambda s: s != "fold_4", "test": lambda s: s == "fold_4"}
    for table in TABLES:
        rs = [r for r in results if r.get("labels") == table and r.get("method", "surfoxide") == "surfoxide"]
        for subset, keep in subsets.items():
            sel = [r for r in rs if keep(split.get((r["sub"], r["ses"]), (None, "?"))[1])]
            hemis = [(r["ga"], h) for r in sel for h in (r.get("hemi") or {}).values()]
            # Une session dont la chaîne a échoué compte comme deux hémisphères non conformes.
            hemis += [(r["ga"], {}) for r in sel if not r.get("hemi") for _ in range(2)]
            ok = [(a, bool(get(h, "report", "valid"))) for a, h in hemis]
            for lo, hi in BINS + [(0, 99)]:
                sel_ok = [v for a, v in ok if lo <= a < hi]
                if sel_ok:
                    g = "toutes" if hi == 99 else f"{lo}-{hi}"
                    t.rows.append(["S1", "taux_conforme", table, subset, g, len(sel_ok),
                                   f"{sum(sel_ok) / len(sel_ok):.4g}", f"{sum(sel_ok)}", ""])
            for fam, name, f in hemi_measures[1:]:
                items = []
                for a, h in hemis:
                    try:
                        items.append((a, f(h)))
                    except (TypeError, ZeroDivisionError):
                        pass
                t.by_bins(fam, name, table, subset, items)
            t.by_bins("S9", "secondes_session", table, subset, [(r["ga"], r.get("seconds")) for r in sel])


def run(cmd, log):
    with open(log, "w") as f:
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def real_one(out, val, table, sub, ses, ga, hemi, calibration):
    """S6 (blanche, piale) et S4 d'un hémisphère du développement ; None si les fichiers manquent."""
    stem = val / "surfoxide" / table / f"{sub}_{ses}_hemi-{hemi}"
    t2w = DATA / sub / ses / "anat" / f"{sub}_{ses}_res-0p5mm_desc-resampled_T2w.nii.gz"
    white, pial, cortex = (Path(f"{stem}_{k}") for k in ("white.surf.gii", "pial.surf.gii", "cortex.shape.gii"))
    if not (white.exists() and pial.exists()):
        return None
    res = {}
    if t2w.exists():
        for kind, surf in (("white", white), ("pial", pial)):
            j = out / "profile" / table / f"{sub}_{ses}_hemi-{hemi}_{kind}.json"
            j.parent.mkdir(parents=True, exist_ok=True)
            if not j.exists():
                cmd = [SURFX, "profile", "--t2w", t2w, "--surface", surf, "--kind", kind, "--calibration",
                       calibration, "--ga", ga, "--json", j]
                if cortex.exists():
                    cmd += ["--mask", cortex]
                run(cmd, j.with_suffix(".log"))
            if j.exists():
                res[kind] = json.loads(j.read_text())
    j = out / "s4" / table / f"{sub}_{ses}_hemi-{hemi}.json"
    j.parent.mkdir(parents=True, exist_ok=True)
    if not j.exists():
        run([SURFX, "compare", "--white", white, "--pial", pial, "--truth-white", white, "--truth-pial", pial,
             "--json", j], j.with_suffix(".log"))
    if j.exists():
        res["s4"] = json.loads(j.read_text())
    return res


def real_measures(out, val, split, calibration, jobs, t):
    dev = [(s, e, ga) for (s, e), (ga, f) in split.items() if f != "fold_4"]
    tasks = [(table, s, e, ga, h) for table in TABLES for s, e, ga in dev for h in ("L", "R")]
    with cf.ThreadPoolExecutor(jobs) as ex:
        res = list(ex.map(lambda k: real_one(out, val, k[0], k[1], k[2], k[3], k[4], calibration), tasks))
    missing_t2w = sum(1 for r in res if r is not None and "white" not in r)
    for table in TABLES:
        items = [(k[3], r) for k, r in zip(tasks, res) if k[0] == table and r]
        for kind, name in (("white", "blanche"), ("pial", "piale")):
            p = [(a, r[kind]) for a, r in items if kind in r]
            t.by_bins("S6", f"decalage_corrige_{name}", table, "dev", [(a, j.get("calibrated_median_offset"))
                                                                       for a, j in p])
            t.by_bins("S6", f"decalage_brut_{name}", table, "dev", [(a, j.get("median_offset")) for a, j in p])
            t.by_bins("S6", f"iqr_{name}", table, "dev", [(a, j.get("iqr_offset")) for a, j in p])
            t.by_bins("S6", f"sans_bord_net_{name}", table, "dev", [(a, j.get("not_sharp_fraction")) for a, j in p])
        s4 = [(a, r["s4"]) for a, r in items if "s4" in r]
        t.by_bins("S4", "petites_taches_blanche", table, "dev",
                  [(a, get(c, "patches", "estimated", "small")) for a, c in s4])
        t.by_bins("S4", "bassins_blanche", table, "dev", [(a, get(c, "patches", "estimated", "basins")) for a, c in s4])
    return missing_t2w


def phantom_measures(phantoms, t):
    path = phantoms / "summary.tsv"
    if not path.exists():
        print(f"attention : {path} absent, S3 non rapporté", file=sys.stderr)
        return
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            kind = "A" if r["group"][0].isdigit() else "B"
            for k, v in r.items():
                if k in ("variant", "sigma", "group", "n") or v == "":
                    continue
                n = int(r["n"])
                t.rows.append(["S3", f"{k}", "bounti19", f"fantomes{kind}_s{r['sigma']}", r["group"], n, v, "", ""])


def stability_measures(stab, t):
    path = stab / "stability.tsv"
    if not path.exists():
        print(f"attention : {path} absent, S5 non rapporté", file=sys.stderr)
        return
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for table in TABLES:
        rs = [r for r in rows if r["table"] == table]
        for k in rs[0]:
            if k.startswith("rsd_") or k.startswith("dist_"):
                t.by_bins("S5", k, table, "dev", [(float(r["ga"]), float(r[k])) for r in rs if r[k]])
    cpath = stab / "concordance.tsv"
    if cpath.exists():
        with open(cpath) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                t.rows.append(["S5", f"ccc_{r['measure']}", "les-deux", "dev", "toutes", int(r["n"]),
                               f"{float(r['ccc']):.4g}", "", ""])


def dpf_measures(dpf, t):
    if dpf is None or not dpf.exists():
        return
    with open(dpf) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        return
    for alpha in sorted({float(r["alpha_ref"]) for r in rows}):
        rs = [r for r in rows if float(r["alpha_ref"]) == alpha]
        for m, col in (("StdCrest", "std_crest"), ("Sep", "sep"), ("Dev_deg", "dev_deg")):
            t.add("DPF", f"{m}_alpha{alpha:g}", "dhcp-neonatal", "zenodo", "toutes",
                  [float(r[col]) for r in rs if r[col] not in ("", "nan")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validation", type=Path, required=True)
    ap.add_argument("--phantoms", type=Path, required=True)
    ap.add_argument("--stability", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--dpf", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    for p in (SURFX, SUBJECTS, args.validation / "results.json", args.calibration):
        if not p.exists():
            sys.exit(f"erreur : {p} absent")
    args.out.mkdir(parents=True, exist_ok=True)
    split = {k: v for k, v in folds().items()}
    t = Table()
    results = json.loads((args.validation / "results.json").read_text())
    validation_measures(results, split, t)
    missing = real_measures(args.out, args.validation, split, args.calibration, args.jobs, t)
    phantom_measures(args.phantoms, t)
    stability_measures(args.stability, t)
    dpf_measures(args.dpf, t)
    t.write(args.out / "reference.tsv")
    with open(args.out / "test-fold.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["subject_id", "session_id", "ga_weeks"])
        w.writerows(sorted((s, e, ga) for (s, e), (ga, fo) in split.items() if fo == "fold_4"))
    (args.out / "reference.json").write_text(json.dumps({
        "argv": sys.argv, "surfx": str(SURFX),
        "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "hemispheres_sans_t2w": missing}, indent=1, ensure_ascii=False) + "\n")
    if missing:
        print(f"attention : {missing} hémisphères du développement sans T2w (S6 incomplet)", file=sys.stderr)
    print(f"écrit : {args.out / 'reference.tsv'}, {args.out / 'test-fold.tsv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
