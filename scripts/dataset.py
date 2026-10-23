#!/usr/bin/env python3
"""Jeu d'entraînement des réseaux de déformation (P1) à partir d'une validation complète.

Pour chaque hémisphère conforme au contrat (rapport `valid`) de chaque table de labels, `surfx dataset` écrit
les classes de tissu, la distance signée et le couple blanche–piale ; un index TSV donne la session, l'âge
gestationnel et le fold (découpage par sujet de `subjects.csv` : folds 0–2 entraînement, 3 validation, 4 test).

    python3 scripts/dataset.py --surfaces ~/scratch/surfoxide-validation-7/surfoxide \\
        --out ~/scratch/surfoxide-dataset [--jobs 8]
"""
import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SUBJECTS = Path.home() / "Exp/fetal-folding-geometry/data/subjects.csv"
SPLITS = {"fold_0": "train", "fold_1": "train", "fold_2": "train", "fold_3": "val", "fold_4": "test"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces", type=Path, required=True, help="dossier <labels>/<sub>_<ses>_hemi-X_* d'une validation")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("SURFOXIDE_DATA", Path.home() / ".cache/surfoxide/data")))
    ap.add_argument("--surfx", type=Path, default=Path(__file__).resolve().parent.parent / "target/release/surfx")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(SUBJECTS)))
    jobs = []
    for labels in ("bounti19", "multibounti43"):
        for r in rows:
            sub, ses = r["subject_id"], r["session_id"]
            seg = next((args.data / sub / ses / "anat").glob(f"*desc-{labels}_dseg.nii.gz"), None)
            for hemi in "LR":
                stem = f"{sub}_{ses}_hemi-{hemi}"
                report = args.surfaces / labels / f"{stem}_report.json"
                if seg is None or not report.exists() or not json.load(open(report)).get("valid"):
                    continue
                jobs.append(dict(labels=labels, sub=sub, ses=ses, hemi=hemi, ga=r["ga_weeks"], fold=r["split"],
                                 split=SPLITS[r["split"]], seg=seg, src=args.surfaces / labels / stem,
                                 prefix=args.out / labels / stem))

    def run(j):
        cmd = [str(args.surfx), "dataset", str(j["seg"]), "--labels", j["labels"], "--hemi", j["hemi"],
               "--white", f"{j['src']}_white.surf.gii", "--pial", f"{j['src']}_pial.surf.gii", "-o", str(j["prefix"])]
        p = subprocess.run(cmd, capture_output=True, text=True)
        return j, p.returncode, (p.stdout + p.stderr).strip()

    args.out.mkdir(parents=True, exist_ok=True)
    done = []
    with ThreadPoolExecutor(args.jobs) as ex:
        for k, (j, code, msg) in enumerate(ex.map(run, jobs), 1):
            if code != 0:
                print(f"[{k}/{len(jobs)}] ÉCHEC {j['prefix'].name} : {msg}")
            else:
                done.append(j)
            if k % 50 == 0:
                print(f"[{k}/{len(jobs)}]")
    with open(args.out / "index.tsv", "w") as f:
        f.write("id\tlabels\tsubject\tsession\themi\tga_weeks\tfold\tsplit\tprefix\n")
        for j in done:
            rel = j["prefix"].relative_to(args.out)
            f.write(f"{j['labels']}/{j['prefix'].name}\t{j['labels']}\t{j['sub']}\t{j['ses']}\t{j['hemi']}\t{j['ga']}\t"
                    f"{j['fold']}\t{j['split']}\t{rel}\n")
    counts = {s: sum(j["split"] == s for j in done) for s in ("train", "val", "test")}
    print(f"écrit : {args.out / 'index.tsv'} ({len(done)} exemples ; {counts})")


if __name__ == "__main__":
    main()
