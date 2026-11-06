#!/usr/bin/env python3
"""Compare deux validations de `scripts/validate.py` (results.json), hémisphère par hémisphère.

Usage : python3 scripts/compare_validations.py <avant> <après> [--short 0.05] [--list 20]
(<avant> et <après> : dossiers de validation ou fichiers results.json).

Par table de labels : contrat (report.valid) avant → après, hémisphères gagnés et perdus, rapports
identiques, médianes de HD95 (blanche contre son masque, piale contre la frontière à sillons ouverts) et de
l'épaisseur moyenne (report.mean_thickness), distribution de report.pial_min_edge et de
report.pial.min_clearance, piales sous le seuil d'arête courte, puis hémisphères dégradés (arête min de la
piale en baisse de plus de 10 % et sous 0,1 mm, dégagement divisé par plus de deux et sous 0,05 mm, ou contrat
perdu).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

QUANTILES = (0.0, 0.01, 0.05, 0.25, 0.5)


def load(path):
    p = Path(path)
    if p.is_dir():
        p = p / "results.json"
    if not p.exists():
        sys.exit(f"erreur : {p} absent")
    out = {}
    for r in json.loads(p.read_text()):
        if r.get("method") != "surfoxide":
            continue
        for hemi in "LR":
            h = r.get("hemi", {}).get(hemi)
            if h is not None:
                out[(r["labels"], r["sub"], r["ses"], hemi)] = dict(h, ga=r["ga"])
    return out


def get(h, *keys):
    for k in keys:
        if not isinstance(h, dict) or k not in h:
            return None
        h = h[k]
    return h


def quantiles(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return "—"
    return " ".join(f"q{int(q * 100)}={v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]:.4f}" for q in QUANTILES)


def median(values):
    v = [x for x in values if x is not None]
    return f"{statistics.median(v):.3f}" if v else "—"


def fmt(x):
    return "—" if x is None else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--short", type=float, default=0.05, help="seuil d'arête courte de la piale (mm)")
    ap.add_argument("--list", type=int, default=30, help="nombre maximal d'hémisphères listés par rubrique")
    args = ap.parse_args()
    a, b = load(args.before), load(args.after)
    print(f"avant : {args.before} ({len(a)} hémisphères) ; après : {args.after} ({len(b)} hémisphères)")
    for labels in sorted({k[0] for k in a} | {k[0] for k in b}):
        keys = sorted(k for k in set(a) | set(b) if k[0] == labels)
        both = [k for k in keys if k in a and k in b]
        va = sum(bool(get(a[k], "report", "valid")) for k in keys if k in a)
        vb = sum(bool(get(b[k], "report", "valid")) for k in keys if k in b)
        na = sum(k in a for k in keys)
        nb = sum(k in b for k in keys)
        gained = [k for k in both if not get(a[k], "report", "valid") and get(b[k], "report", "valid")]
        lost = [k for k in both if get(a[k], "report", "valid") and not get(b[k], "report", "valid")]
        same = sum(get(a[k], "report") == get(b[k], "report") for k in both)
        print(f"\n== {labels}")
        print(f"contrat : {va}/{na} → {vb}/{nb} ; gagnés {len(gained)}, perdus {len(lost)} ; "
              f"rapports identiques {same}/{len(both)}")
        for name, ks in (("gagné", gained), ("perdu", lost)):
            for k in ks[: args.list]:
                print(f"  {name} : {' '.join(k[1:])}")
        for title, path in (("HD95 blanche", ("white_vs_white", "hausdorff95")),
                            ("HD95 piale (sillons ouverts)", ("pial_vs_opened", "hausdorff95")),
                            ("épaisseur moyenne", ("report", "mean_thickness"))):
            print(f"{title} médian(e) : {median(get(a[k], *path) for k in both)} → "
                  f"{median(get(b[k], *path) for k in both)}")
        for title, path in (("arête min de la piale", ("report", "pial_min_edge")),
                            ("dégagement de la piale", ("report", "pial", "min_clearance"))):
            print(f"{title} : avant {quantiles(get(a[k], *path) for k in both)}")
            print(f"{' ' * len(title)}   après {quantiles(get(b[k], *path) for k in both)}")
        for t in (args.short, 0.1):
            sa = sum((get(a[k], "report", "pial_min_edge") or 1.0) < t for k in both)
            sb = sum((get(b[k], "report", "pial_min_edge") or 1.0) < t for k in both)
            print(f"piales d'arête min < {t:g} mm : {sa} → {sb}")
        degraded = []
        for k in both:
            ea, eb = get(a[k], "report", "pial_min_edge"), get(b[k], "report", "pial_min_edge")
            worse_edge = ea is not None and eb is not None and eb < 0.9 * ea and eb < 0.1
            ca, cb = get(a[k], "report", "pial", "min_clearance"), get(b[k], "report", "pial", "min_clearance")
            worse_clearance = ca is not None and cb is not None and cb < 0.5 * ca and cb < 0.05
            if worse_edge or worse_clearance or k in lost:
                degraded.append(k)
        print(f"hémisphères dégradés : {len(degraded)}")
        for k in degraded[: args.list]:
            ra, rb = a[k], b[k]
            print(f"  {' '.join(k[1:])} ({ra['ga']:.1f} SA) : arête min {fmt(get(ra, 'report', 'pial_min_edge'))} → "
                  f"{fmt(get(rb, 'report', 'pial_min_edge'))}, dégagement "
                  f"{fmt(get(ra, 'report', 'pial', 'min_clearance'))} → {fmt(get(rb, 'report', 'pial', 'min_clearance'))}"
                  f", contrat {get(ra, 'report', 'valid')} → {get(rb, 'report', 'valid')}")


if __name__ == "__main__":
    main()
