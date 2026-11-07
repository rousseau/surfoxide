#!/usr/bin/env python3
"""Essaie des hémisphères d'une validation dans gyroxide : maillage TetGen, court run, contrôle de la surface simulée.

Usage : python3 scripts/gyroxide_check.py --validation <dossier> --cases <cas.tsv> --out <dossier>
        [--gyroxide <binaire>] [--steps 20000] [--growth uniform,cortex] [--surfx <binaire>]

<cas.tsv> : colonnes labels, sub, ses, hemi, role (en-tête obligatoire ; role libre, recopié dans le résultat).
Pour chaque cas, les surfaces de la validation (blanche, piale, masque du cortex) sont exposées par liens dans une
étude BIDS minimale (<out>/<labels>/derivatives/surfx-surfaces/…), puis : `gyroxide mesh`, `gyroxide run` de
--steps pas par mode de croissance (`--uniform-growth` ou `--growth-field cortex`, toujours `--no-midplane`), et
`surfx check` des surfaces simulées au premier et au dernier pas. Résultat : <out>/results.tsv (une ligne par cas
et par mode), journaux dans <out>/logs/. Un cas déjà terminé (status.json « done ») n'est pas relancé.

Colonnes : arête min et dégagement de la piale d'entrée (rapport de la validation), espacement a et pas de temps
de gyroxide, nombre de pas pour t ∈ [0, 1], ms par pas, t atteint, état final, tétraèdres inversés (max du
journal), contacts au premier pas écrit et au dernier, auto-intersections et dégagement de la surface simulée.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

FLOAT = r"([-+0-9.eE]+|NaN|nan|inf)"


def run(cmd, log):
    t0 = time.time()
    with open(log, "w") as f:
        f.write(" ".join(map(str, cmd)) + "\n")
        f.flush()
        code = subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    return code, time.time() - t0


def expose(validation, out, case):
    """Liens vers les surfaces de la validation dans <out>/<labels>/derivatives/surfx-surfaces/sub/ses/anat."""
    labels, sub, ses, hemi = case["labels"], case["sub"], case["ses"], case["hemi"]
    src = validation / "surfoxide" / labels
    anat = out / labels / "derivatives" / "surfx-surfaces" / sub / ses / "anat"
    anat.mkdir(parents=True, exist_ok=True)
    for name in ("white.surf.gii", "pial.surf.gii", "cortex.shape.gii"):
        f = f"{sub}_{ses}_hemi-{hemi}_{name}"
        if not (src / f).is_file():
            sys.exit(f"erreur : {src / f} absent")
        link = anat / f
        if not link.exists():
            link.symlink_to((src / f).resolve())
    report = json.loads((src / f"{sub}_{ses}_hemi-{hemi}_report.json").read_text())
    return out / labels, report


def parse_run_log(text):
    r = {}
    m = re.search(r"\| a = " + FLOAT, text)
    r["a"] = float(m.group(1)) if m else None
    m = re.search(r"dt = " + FLOAT + r", (\d+) steps for t in", text)
    if m:
        r["dt"], r["steps_t1"] = float(m.group(1)), int(m.group(2))
    m = re.search(r"done: (\d+) steps, t = " + FLOAT + r", " + FLOAT + r"s \(" + FLOAT + r" ms/step\), (\d+) contact",
                  text)
    if m:
        r["steps"], r["t"], r["ms_step"], r["rebuilds"] = int(m.group(1)), float(m.group(2)), float(m.group(4)), int(
            m.group(5))
    ninv = [int(line.split()[8]) for line in text.splitlines()
            if re.match(r"\s+\d+\s+" + FLOAT + r"\s", line) and len(line.split()) >= 10]
    r["ninv_max"] = max(ninv) if ninv else None
    r["nan"] = bool(re.search(r"\bNaN\b|\bnan\b", text))
    return r


def check(surfx, surface, log):
    code, _ = run([surfx, "check", surface, "--json", log.with_suffix(".json")], log)
    if code != 0 and not log.with_suffix(".json").exists():
        return {}
    s = json.loads(log.with_suffix(".json").read_text())["surface"]
    return {"self_intersections": s["self_intersections"], "min_clearance": s["min_clearance"],
            "low_clearance_vertices": s.get("low_clearance_vertices")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validation", required=True, type=Path)
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gyroxide", default="gyroxide")
    ap.add_argument("--surfx", default=str(Path(__file__).resolve().parent.parent / "target/release/surfx"))
    ap.add_argument("--tetgen", default=None, help="exécutable TetGen (défaut : recherche de gyroxide)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--growth", default="uniform,cortex", help="modes de croissance, séparés par des virgules")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    validation, out = args.validation.expanduser(), args.out.expanduser()
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(open(args.cases), delimiter="\t"))
    rows = []
    for case in cases:
        root, report = expose(validation, out, case)
        sub, ses, hemi = case["sub"], case["ses"], case["hemi"]
        part, sess = sub.removeprefix("sub-"), ses.removeprefix("ses-")
        tag = f"{case['labels']}_{sub}_{ses}_{hemi}"
        anat = root / "derivatives" / "gyroxide" / sub / ses / "anat"
        vtu = anat / f"{sub}_{ses}_hemi-{hemi}_desc-tetgen_tet.vtu"
        if not vtu.exists():
            cmd = [args.gyroxide, "mesh", "--bids", root, "--participant", part, "--session", sess, "--hemi", hemi]
            if args.tetgen:
                cmd += ["--tetgen", args.tetgen]
            code, sec = run(cmd, logs / f"{tag}_mesh.log")
            print(f"{tag} : maillage {sec:.0f} s, code {code}", flush=True)
            if code != 0:
                rows.append(dict(case, growth="", state="mesh-failed"))
                continue
        for growth in args.growth.split(","):
            label = growth
            rundir = anat / f"run-{label}"
            log = logs / f"{tag}_run-{label}.log"
            status = rundir / "status.json"
            done = status.exists() and json.loads(status.read_text()).get("state") == "done"
            if not done:
                cmd = [args.gyroxide, "run", "--bids", root, "--participant", part, "--session", sess, "--hemi", hemi,
                       "--label", label, "--steps", args.steps, "--no-midplane", "--backend", args.backend,
                       "--print-every", 500, "--surface-every", args.steps]
                cmd += ["--uniform-growth"] if growth == "uniform" else ["--growth-field", growth]
                code, sec = run(cmd, log)
                print(f"{tag} run-{label} : {sec:.0f} s, code {code}", flush=True)
            r = dict(case, growth=growth, pial_min_edge=report.get("pial_min_edge"),
                     pial_min_clearance=report["pial"]["min_clearance"], pial_vertices=report["pial"]["n_vertices"])
            r.update(parse_run_log(log.read_text()) if log.exists() else {})
            st = json.loads(status.read_text()) if status.exists() else {}
            r["state"], r["message"] = st.get("state"), st.get("message")
            seq = rundir / "sequence.tsv"
            if seq.exists():
                lines = list(csv.DictReader(open(seq), delimiter="\t"))
                if lines:
                    r["contacts_first"], r["contacts_last"] = lines[0]["n_contacts"], lines[-1]["n_contacts"]
                    r["gi_first"], r["gi_last"] = lines[0]["gi"], lines[-1]["gi"]
                    for which, line in (("first", lines[0]), ("last", lines[-1])):
                        surf = rundir / line["file"]
                        if surf.exists():
                            c = check(args.surfx, surf, logs / f"{tag}_run-{label}_check-{which}.log")
                            for k, v in c.items():
                                r[f"{k}_{which}"] = v
            rows.append(r)
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(out / "results.tsv", "w") as f:
        w = csv.DictWriter(f, keys, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    print(f"écrit : {out / 'results.tsv'}")


if __name__ == "__main__":
    main()
