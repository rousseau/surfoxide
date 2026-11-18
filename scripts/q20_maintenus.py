#!/usr/bin/env python3
"""Q2.0 : sommets maintenus pendant le lissage de la blanche, garde-fou binaire contre garde-fou au temps d'impact.

Pour chaque session du développement (folds 0 à 3 de subjects.csv ; le fold test n'est pas mesuré) et chaque table
(BOUNTI-19, multi-BOUNTI-43), `surfx white` est lancé deux fois (`--safe-update binary`, puis `impact`) ; on lit le
bloc `smoothing` du `report.json` de chaque hémisphère : sommets maintenus (`held`), sommets à moins de 10 % du pas
demandé (`held_tenth`, la mesure de Q2.0), fraction de déplacement perdu. Le temps mural de chaque appel est noté.

Usage : python3 scripts/q20_maintenus.py --out ~/scratch/plan-qualite/q20-impact [--jobs 3] [--limit N]

Bibliothèque standard seulement (script de mesure ponctuel).
"""

import argparse
import concurrent.futures as cf
import csv
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate  # noqa: E402

VARIANTES = ["binary", "impact"]


def une_session(out, sub, ses, labels):
    seg = validate.seg_path(sub, ses, labels)
    res = {"sub": sub, "ses": ses, "labels": labels, "variantes": {}}
    for v in VARIANTES:
        prefix = out / v / labels / f"{sub}_{ses}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        p = subprocess.run([validate.SURFX, "white", seg, "--labels", labels, "-o", prefix, "--safe-update", v],
                           capture_output=True, text=True)
        r = {"seconds": time.time() - t, "returncode": p.returncode, "hemi": {}}
        for h in "LR":
            rep = Path(f"{prefix}_hemi-{h}_report.json")
            if rep.exists():
                j = json.loads(rep.read_text())
                r["hemi"][h] = {"smoothing": j.get("smoothing"), "valid": j.get("white", {}).get("valid")}
        res["variantes"][v] = r
    return res


def mediane(xs):
    return st.median(xs) if xs else float("nan")


def resume(resultats):
    lignes = []
    for labels in ("bounti19", "multibounti43"):
        rs = [r for r in resultats if r["labels"] == labels]
        paires = []  # (binary, impact) par hémisphère
        for r in rs:
            for h in "LR":
                b = r["variantes"]["binary"]["hemi"].get(h)
                i = r["variantes"]["impact"]["hemi"].get(h)
                if b and i and b["smoothing"] and i["smoothing"]:
                    paires.append((b, i))
        rb = [p[0]["smoothing"]["held_tenth"] for p in paires]
        ri = [p[1]["smoothing"]["held_tenth"] for p in paires]
        lb = [p[0]["smoothing"]["lost_fraction"] for p in paires]
        li = [p[1]["smoothing"]["lost_fraction"] for p in paires]
        tb = [r["variantes"]["binary"]["seconds"] for r in rs if r["variantes"]["binary"]["returncode"] == 0]
        ti = [r["variantes"]["impact"]["seconds"] for r in rs if r["variantes"]["impact"]["returncode"] == 0]
        rapports = [r["variantes"]["impact"]["seconds"] / r["variantes"]["binary"]["seconds"] for r in rs
                    if r["variantes"]["binary"]["returncode"] == 0 and r["variantes"]["impact"]["returncode"] == 0]
        invalides = sum(1 for p in paires for k in (0, 1) if not p[k]["valid"])
        mb, mi = mediane(rb), mediane(ri)
        # Médiane du défaut nulle : critère mal défini ; on rapporte alors les hémisphères où il maintient au moins un sommet.
        avec = [(x, y) for x, y in zip(rb, ri) if x >= 1]
        lignes.append({
            "table": labels, "hemispheres": len(paires), "sessions": len(rs),
            "held_tenth_mediane_binary": mb, "held_tenth_mediane_impact": mi,
            "held_tenth_somme_binary": sum(rb), "held_tenth_somme_impact": sum(ri),
            "baisse_mediane": (1 - mi / mb) if mb > 0 else None,
            "baisse_somme": (1 - sum(ri) / sum(rb)) if sum(rb) > 0 else None,
            "hemispheres_avec_maintenus_binary": len(avec),
            "baisse_mediane_appariee": (1 - mediane([y for _, y in avec]) / mediane([x for x, _ in avec])) if avec else None,
            "perdu_mediane_binary": mediane(lb), "perdu_mediane_impact": mediane(li),
            "temps_mediane_binary_s": mediane(tb), "temps_mediane_impact_s": mediane(ti),
            "rapport_temps_mediane": mediane(rapports),
            "surfaces_invalides_binary_impact": invalides,
        })
    return lignes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="n'utiliser que les N premières sessions (essai)")
    a = ap.parse_args()
    folds = {(r["subject_id"], r["session_id"]): r["split"] for r in csv.DictReader(open(validate.SUBJECTS))}
    taches = []
    for sub, ses, _ in validate.all_sessions():
        if folds.get((sub, ses), "fold_4") == "fold_4":
            continue
        taches.append((sub, ses, "bounti19"))
        if validate.seg_path(sub, ses, "multibounti43").exists() and (sub, ses) not in validate.FAILED_MULTI:
            taches.append((sub, ses, "multibounti43"))
    if a.limit:
        taches = taches[: a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    resultats = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futs = [pool.submit(une_session, a.out, *t) for t in taches]
        for k, f in enumerate(cf.as_completed(futs), 1):
            r = f.result()
            resultats.append(r)
            print(f"[{k}/{len(taches)}] {r['labels']} {r['sub']} {r['ses']}", flush=True)
    (a.out / "resultats.json").write_text(json.dumps(resultats))
    lignes = resume(resultats)
    (a.out / "resume.json").write_text(json.dumps(lignes, indent=1))
    for l in lignes:
        print(json.dumps(l))


if __name__ == "__main__":
    main()
