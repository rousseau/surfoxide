#!/usr/bin/env python3
"""Bilan de la correction topologique par défaut (plan qualité, Q1.1) sur une validation lancée avec
`--topology per-defect` (depuis le 10 novembre 2026 le défaut de `surfx run` ; avant, `SURFX_RUN_ARGS="--topology per-defect"`) : voxels modifiés contre la meilleure correction globale, hémisphères
gagnés ou perdus contre une validation de référence.

Voxels (développement = folds 0 à 3 de subjects.csv, par table et tranche d'âge) : le `report.json` de chaque
hémisphère donne `topology.voxels_modified` (stratégie par défaut) et `topology.voxels_best_global` (ce que la
stratégie `Best` aurait modifié sur le même masque). On rapporte la médiane de chaque, la variation relative des
médianes, la médiane de la variation par hémisphère (ceux dont `Best` modifie au moins un voxel) et les effectifs
« moins / autant / plus ».

Hémisphères : « perdu » = conforme (`report.valid`) dans la référence et non conforme ou absent dans la nouvelle
validation ; « gagné » = l'inverse.

Usage :
    python3 scripts/topologie.py --new ~/scratch/surfoxide-validation-16 --ref ~/scratch/surfoxide-validation-15
        [--out topologie.tsv]

Bibliothèque standard seulement.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

DATA = Path(os.environ.get("SURFOXIDE_DATA", Path.home() / ".cache/surfoxide/data"))
SUBJECTS = Path(os.environ.get("OXIDE_SUBJECTS", DATA / "subjects.csv"))
BINS = [(20, 24), (24, 28), (28, 32), (32, 38), (0, 99)]


def med(x):
    return statistics.median(x) if x else float("nan")


def load(path):
    return json.loads((Path(path) / "results.json").read_text())


def hemispheres(results):
    """{(table, sub, ses, hémisphère): (âge, rapport ou None)} pour les sessions surfoxide."""
    out = {}
    for r in results:
        if r.get("method", "surfoxide") != "surfoxide":
            continue
        for hemi in "LR":
            h = (r.get("hemi") or {}).get(hemi) or {}
            out[(r["labels"], r["sub"], r["ses"], hemi)] = (r["ga"], h.get("report"))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True, help="validation avec --topology per-defect (le défaut)")
    ap.add_argument("--ref", help="validation de référence (hémisphères gagnés et perdus)")
    ap.add_argument("--out", type=Path, help="TSV des mesures par table et tranche d'âge")
    args = ap.parse_args()
    split = {(r["subject_id"], r["session_id"]): r["split"] for r in csv.DictReader(open(SUBJECTS))}
    new = hemispheres(load(args.new))
    rows = []
    for table in ("bounti19", "multibounti43"):
        for lo, hi in BINS:
            sel = [(k, v) for k, v in new.items()
                   if k[0] == table and lo <= v[0] < hi and split.get((k[1], k[2])) not in (None, "fold_4")]
            pairs = []
            for _, (_, rep) in sel:
                t = (rep or {}).get("topology") or {}
                if "voxels_modified" in t and "voxels_best_global" in t:
                    pairs.append((t["voxels_modified"], t["voxels_best_global"], t.get("defects_count"),
                                  t.get("unrecovered_voxels")))
            if not pairs:
                continue
            nd = [a for a, _, _, _ in pairs]
            bs = [b for _, b, _, _ in pairs]
            rel = [(a - b) / b for a, b, _, _ in pairs if b > 0]
            rows.append({
                "table": table, "tranche": "toutes" if hi == 99 else f"{lo}-{hi}", "hemispheres": len(pairs),
                "med_defaut": med(nd), "med_best": med(bs),
                "variation_medianes": (med(nd) - med(bs)) / med(bs) if med(bs) else float("nan"),
                "hemis_best_non_nul": len(rel), "med_variation_appariee": med(rel),
                "moins": sum(a < b for a, b, _, _ in pairs), "autant": sum(a == b for a, b, _, _ in pairs),
                "plus": sum(a > b for a, b, _, _ in pairs),
                "somme_defaut": sum(nd), "somme_best": sum(bs),
                "med_defauts": med([d for _, _, d, _ in pairs if d is not None]),
                "non_poses": sum(u or 0 for _, _, _, u in pairs),
            })
    cols = list(rows[0]) if rows else []
    for r in rows:
        print("  ".join(f"{k}={r[k]:.4g}" if isinstance(r[k], float) else f"{k}={r[k]}" for k in cols))
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(cols)
            w.writerows([[f"{r[c]:.6g}" if isinstance(r[c], float) else r[c] for c in cols] for r in rows])
    if args.ref:
        ref = hemispheres(load(args.ref))
        ok = lambda rep: bool(rep and rep.get("valid"))
        lost = sorted(k for k, (_, rep) in ref.items() if ok(rep) and not ok(new.get(k, (0, None))[1]))
        won = sorted(k for k, (_, rep) in new.items() if ok(rep) and not ok(ref.get(k, (0, None))[1]))
        print(f"hémisphères conformes : référence {sum(ok(r) for _, r in ref.values())}/{len(ref)}, "
              f"nouvelle {sum(ok(r) for _, r in new.values())}/{len(new)} ; perdus {len(lost)}, gagnés {len(won)}")
        for k in lost:
            print("  perdu :", *k)
        for k in won:
            print("  gagné :", *k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
