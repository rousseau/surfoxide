#!/usr/bin/env python3
"""Validation de surfoxide sur la cohorte locale et comparaison à fetpype `surface_processing`.

Pour une session par semaine d'âge gestationnel (BOUNTI-19) : `surfx run` (WM et piale), puis
`surfx eval` contre trois références (masque blanc de surfoxide, tissu blanc brut, masque blanc ∪
cortex pour la piale). fetpype (image docker macatools/surf_proc:v0.0.1e) est lancé sur les mêmes
sessions et évalué de la même façon. Quelques sessions multi-BOUNTI-43 complètent.

Données : `bash scripts/fetch-data.sh` (segmentations dans $SURFOXIDE_DATA, par défaut
~/.cache/surfoxide/data).

Usage :
    python3 scripts/validate.py --out ~/scratch/surfoxide-validation [--jobs 6] [--no-fetpype]
        [--all --surf-jobs 4]
        [--reuse-fetpype ancien/results.json]

Bibliothèque standard seulement. Les résultats (surfaces, rapports) restent hors du dépôt.
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
# Segmentations rapatriées par scripts/fetch-data.sh.
DATA = Path(os.environ.get("SURFOXIDE_DATA", HOME / ".cache/surfoxide/data"))
SUBJECTS = HOME / "Exp/fetal-folding-geometry/data/subjects.csv"
FETPYPE_IMAGE = "macatools/surf_proc:v0.0.1e"
FETPYPE_LABELS = {"L": "5,7,14,16", "R": "6,8,15,17"}
FAILED_MULTI = {("sub-cq1rl", "ses-01")}
REPO = Path(__file__).resolve().parent.parent
SURFX = REPO / "target/release/surfx"


def seg_path(sub, ses, kind):
    return DATA / sub / ses / "anat" / f"{sub}_{ses}_res-0p5mm_desc-{kind}_dseg.nii.gz"


def all_sessions():
    """Toutes les sessions dont la segmentation BOUNTI-19 existe, triées par AG."""
    rows = csv.DictReader(open(SUBJECTS))
    out = [(r["subject_id"], r["session_id"], float(r["ga_weeks"])) for r in rows
           if seg_path(r["subject_id"], r["session_id"], "bounti19").exists()]
    return sorted(out, key=lambda s: (s[2], s[0], s[1]))


def select_sessions():
    """Une session par semaine entière d'AG (la première par identifiant), dont la segmentation existe."""
    by_week = {}
    for sub, ses, ga in sorted(all_sessions(), key=lambda s: (s[0], s[1])):
        by_week.setdefault(int(ga), (sub, ses, ga))
    return [by_week[w] for w in sorted(by_week)]


def run(cmd, **kw):
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, **kw)
    return p, time.time() - t


def evaluate(surface, seg, labels, hemi, boundary):
    p, _ = run([SURFX, "eval", surface, seg, "--labels", labels, "--hemi", hemi,
                "--boundary", boundary])
    if p.returncode != 0:
        return {"error": p.stderr.strip()}
    return json.loads(p.stdout)


def surfoxide_session(out, sub, ses, ga, labels):
    seg = seg_path(sub, ses, labels)
    prefix = out / "surfoxide" / labels / f"{sub}_{ses}"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    p, secs = run([SURFX, "run", seg, "--labels", labels, "-o", prefix])
    res = {"sub": sub, "ses": ses, "ga": ga, "labels": labels, "method": "surfoxide",
           "seconds": secs, "returncode": p.returncode, "log": p.stdout + p.stderr, "hemi": {}}
    for hemi in "LR":
        h = {}
        rep = Path(f"{prefix}_hemi-{hemi}_report.json")
        if rep.exists():
            h["report"] = json.loads(rep.read_text())
        white = f"{prefix}_hemi-{hemi}_white.surf.gii"
        pial = f"{prefix}_hemi-{hemi}_pial.surf.gii"
        if Path(white).exists():
            h["white_vs_white"] = evaluate(white, seg, labels, hemi, "white")
            h["white_vs_tissue"] = evaluate(white, seg, labels, hemi, "tissue")
        if Path(pial).exists():
            h["pial_vs_pial"] = evaluate(pial, seg, labels, hemi, "pial")
            h["pial_vs_opened"] = evaluate(pial, seg, labels, hemi, "pial-opened")
        res["hemi"][hemi] = h
    return res


def fetpype_hemi(out, sub, ses, hemi):
    seg = seg_path(sub, ses, "bounti19")
    dest = out / "fetpype"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{sub}_{ses}_hemi-{hemi}_white.gii"
    p, secs = run(["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
                   "-e", "HOME=/tmp", "-e", "NUMBA_CACHE_DIR=/tmp",
                   "-v", f"{seg.parent}:/in:ro", "-v", f"{dest}:/out", FETPYPE_IMAGE,
                   "python3", "generate_mesh.py", "-s", f"/in/{seg.name}",
                   "-l", FETPYPE_LABELS[hemi], "-m", f"/out/{name}"])
    h = {"seconds": secs, "returncode": p.returncode}
    if (dest / name).exists():
        h["white_vs_white"] = evaluate(dest / name, seg, "bounti19", hemi, "white")
        h["white_vs_tissue"] = evaluate(dest / name, seg, "bounti19", hemi, "tissue")
    else:
        h["error"] = p.stderr[-2000:]
    return hemi, h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=6, help="conteneurs fetpype en parallèle")
    ap.add_argument("--no-fetpype", action="store_true")
    ap.add_argument("--multi", type=int, default=4, help="sessions multi-BOUNTI-43")
    ap.add_argument("--all", action="store_true",
                    help="toutes les sessions (BOUNTI-19 et multi-BOUNTI-43) au lieu d'une par semaine ; "
                         "fetpype reste sur une session par semaine")
    ap.add_argument("--surf-jobs", type=int, default=1, help="sessions surfoxide traitées en parallèle")
    ap.add_argument("--reuse-fetpype", type=Path,
                    help="results.json d'une exécution précédente dont on reprend les résultats fetpype")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not SURFX.exists():
        sys.exit("compiler d'abord : cargo build --release")
    weekly = select_sessions()
    sessions = all_sessions() if args.all else weekly
    print(f"{len(sessions)} sessions BOUNTI-19 ({sessions[0][2]:.1f} à {sessions[-1][2]:.1f} SA)")
    results = []

    fetpype_futures = {}
    pool = cf.ThreadPoolExecutor(max_workers=args.jobs)
    if not args.no_fetpype:
        for sub, ses, ga in weekly:
            for hemi in "LR":
                fetpype_futures[pool.submit(fetpype_hemi, args.out, sub, ses, hemi)] = (sub, ses, ga)

    # Multi-BOUNTI : toutes les sessions avec --all, sinon quelques-unes réparties sur la plage d'AG.
    multi = [s for s in sessions if (s[0], s[1]) not in FAILED_MULTI and seg_path(s[0], s[1], "multibounti43").exists()]
    if not args.all:
        step = max(1, len(multi) // max(1, args.multi))
        multi = multi[::step][: args.multi]
    jobs = [(s, "bounti19") for s in sessions] + [(s, "multibounti43") for s in multi]
    with cf.ThreadPoolExecutor(max_workers=args.surf_jobs) as surf_pool:
        futures = {surf_pool.submit(surfoxide_session, args.out, sub, ses, ga, labels): (sub, ses, ga, labels)
                   for (sub, ses, ga), labels in jobs}
        for k, fut in enumerate(cf.as_completed(futures), 1):
            sub, ses, ga, labels = futures[fut]
            r = fut.result()
            print(f"[{k}/{len(jobs)}] surfoxide {labels} {sub} {ses} ({ga:.1f} SA) : {r['seconds']:.0f} s, "
                  f"code {r['returncode']}", flush=True)
            results.append(r)

    fet = {}
    for fut in cf.as_completed(fetpype_futures):
        sub, ses, ga = fetpype_futures[fut]
        hemi, h = fut.result()
        key = (sub, ses)
        fet.setdefault(key, {"sub": sub, "ses": ses, "ga": ga, "labels": "bounti19",
                             "method": "fetpype", "seconds": 0.0, "hemi": {}})
        fet[key]["hemi"][hemi] = h
        fet[key]["seconds"] += h["seconds"]
        print(f"fetpype {sub} {ses} {hemi} : {h['seconds']:.0f} s", flush=True)
    results.extend(fet.values())
    pool.shutdown()
    if args.reuse_fetpype:
        results.extend(r for r in json.loads(args.reuse_fetpype.read_text()) if r["method"] == "fetpype")

    dest = args.out / "results.json"
    if dest.exists():
        dest.rename(args.out / "results.previous.json")
    dest.write_text(json.dumps(results, indent=1))
    print(f"écrit : {dest}")


if __name__ == "__main__":
    main()
