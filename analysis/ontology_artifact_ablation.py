#!/usr/bin/env python3
"""
ontology_artifact_ablation.py

Re-runs the concept-extractor artifact audit against the current generation run
and the current top-two systems by text similarity (MedGemma-27B-it and
Llama3-OpenBioLLM-8B), replacing the earlier audit that was anchored on GPT-5 Pro.

The question: MetaMapLite maps a great deal of non-clinical text to UMLS
concepts. Because those spurious matches occur in the gold standard notes and
the generated notes alike, they land in the intersection and inflate ontology
F1. How much of the reported agreement is real?

Artifact CUIs are identified by three rules, applied to every CUI detected
anywhere in the corpus. The rules are deliberately mechanical so the audit is
reproducible rather than a matter of taste:

  A. Semantic type is a gene/genome (gngm) or a language (lang). Neither has any
     plausible role in a rehabilitation note; these arise from abbreviation
     collisions such as WWOX and Minangkabau.

  B. No semantic type is recorded. In this extractor these are parser fragments
     ("Preposition For"), not concepts.

  C. The concept name matches a function word, discourse connective, or
     dosing-instruction fragment, and its semantic type is one of the abstract
     classes (ftcn, idcn, tmco, spco, qlco, qnco, inpr). Both conditions are
     required, so "Distance" (qnco) survives while "In addition to" (ftcn) does
     not.

  D. An explicit list of rehabilitation-shorthand collisions confirmed by
     inspection: min A -> Minute / Minangkabau, STS -> sodium tetradecyl
     sulfate, INAD -> Infantile Neuroaxonal Dystrophy.

Filtering is applied symmetrically: artifact CUIs are removed from the gold set
and the predicted set before recomputing per-note coverage. Any asymmetric
filter would manufacture the result it is meant to test.

  python analysis/ontology_artifact_ablation.py
  python analysis/ontology_artifact_ablation.py --list   # print every filtered CUI
  python analysis/ontology_artifact_ablation.py --export resources/artifact_cuis.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Overridable with --metrics_dir. Run from the repository root.
RESULTS = Path("metrics_test")
POOLED = RESULTS / "results_pooled"

MODELS = ["gpt5-pro", "healthscribe", "jsl-text2soap", "medgemma-27b",
          "med42-70b", "asclepius-llama3-8b", "openbiollm-8b"]
NAME = {"gpt5-pro": "GPT-5 Pro", "healthscribe": "AWS HealthScribe",
        "jsl-text2soap": "JSL Text2SOAP", "medgemma-27b": "MedGemma-27B-it",
        "med42-70b": "Llama3-Med42-70B",
        "asclepius-llama3-8b": "Asclepius-Llama3-8B",
        "openbiollm-8b": "Llama3-OpenBioLLM-8B"}

ABSTRACT = {"ftcn", "idcn", "tmco", "spco", "qlco", "qnco", "inpr"}

# rule C: function words, discourse connectives, dosing-instruction fragments
FUNCTION_WORD = re.compile(
    r"^(in addition to|preposition\b|with\b.*dosing|during\b.*dosing|during$|"
    r"then$|from$|following$|adjacent$|due to$|approximate$|about$|"
    r"per\b|and$|or$|the$|of$|for$|to$|by$|as$|at$|on$|in$|"
    r"status\s*-|act mood|objective \(goal\)|planned$|includes$|"
    r"other$|unspecified$|not otherwise specified|nos$|"
    r"^\W*$)", re.I)

# rule D: confirmed rehabilitation-shorthand collisions
NAMED_COLLISIONS = {
    "C2347166",  # Minute Unit of Plane Angle   <- "min A"
    "C0574377",  # Minangkabau Language         <- "min A"
    "C0037556",  # sodium tetradecyl sulfate    <- "STS"
    "C0270724",  # Infantile Neuroaxonal Dystrophy <- "INAD"
    "C4321252",  # WWOX wt Allele
}


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_semtypes_and_names():
    sem, name = {}, {}
    for corpus in ("encompass", "adapts"):
        for m in MODELS:
            f = RESULTS / f"results_{corpus}/metric_reports/{m}_{corpus}_cui_coverage.json"
            d = load(f)
            if d:
                for c, s in (d.get("cui_semtypes") or {}).items():
                    sem.setdefault(c, s)
    cf = load(POOLED / "16_concept_frequency_tables.json")
    if cf:
        for corpus in ("encompass", "adapts"):
            blk = cf["corpora"][corpus]
            for e in blk["gold"]:
                name.setdefault(e["cui"], e["name"])
            for ents in blk["systems"].values():
                for e in ents:
                    name.setdefault(e["cui"], e["name"])
    return sem, name


def classify(sem, name):
    """-> {cui: rule} for every CUI judged an artifact."""
    art = {}
    for cui, nm in name.items():
        st = sem.get(cui)
        st = set(st) if isinstance(st, list) else (set() if st is None else {st})
        if st & {"gngm", "lang"}:
            art[cui] = "A gene/language collision"
        elif not st:
            art[cui] = "B no semantic type (parser fragment)"
        elif (st & ABSTRACT) and FUNCTION_WORD.match(nm.strip()):
            art[cui] = "C function word / discourse / dosing fragment"
    for cui in NAMED_COLLISIONS:
        if cui in name:
            art[cui] = "D confirmed shorthand collision"
    return art


def f1(pred, gold):
    if not pred or not gold:
        return None
    inter = len(pred & gold)
    if not inter:
        return 0.0
    r, p = inter / len(gold), inter / len(pred)
    return 2 * r * p / (r + p)


def per_note(model, artifacts):
    """[(unfiltered_f1, filtered_f1)] over all encounters in both corpora."""
    out = []
    for corpus in ("encompass", "adapts"):
        d = load(RESULTS / f"results_{corpus}/metric_reports/{model}_{corpus}_cui_coverage.json")
        if not d:
            continue
        for it in d["per_item"]:
            pg, gg = set(it["pred_cuis"]), set(it["gold_cuis"])
            a = f1(pg, gg)
            b = f1(pg - artifacts.keys(), gg - artifacts.keys())
            if a is not None and b is not None:
                out.append((a, b))
    return out


def boot(vals, n=10000, seed=0):
    rng = random.Random(seed)
    if not vals:
        return None, None, None
    mean = sum(vals) / len(vals)
    idx = range(len(vals))
    draws = sorted(sum(vals[rng.choice(idx)] for _ in idx) / len(vals) for _ in range(n))
    return mean, draws[int(0.025 * n)], draws[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print every filtered CUI")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--metrics_dir", default="metrics_test",
                    help="Folder holding results_<corpus>/ and results_pooled/.")
    ap.add_argument("--export", default=None,
                    help="Also write the artifact-CUI list (CUI, rule, name) to this JSON path.")
    args = ap.parse_args()
    global RESULTS, POOLED
    RESULTS = Path(args.metrics_dir)
    POOLED = RESULTS / "results_pooled"

    sem, name = build_semtypes_and_names()
    art = classify(sem, name)

    from collections import Counter
    by_rule = Counter(v.split()[0] for v in art.values())
    print(f"  CUIs detected anywhere in the corpus : {len(name):,}")
    print(f"  classified as extraction artifacts    : {len(art):,} "
          f"({len(art)/len(name)*100:.1f}%)")
    for r in sorted(by_rule):
        lbl = next(v for v in art.values() if v.startswith(r))
        print(f"      rule {lbl:<44} {by_rule[r]:>4}")

    if args.list:
        print("\n  filtered CUIs:")
        for cui, rule in sorted(art.items(), key=lambda x: (x[1], name.get(x[0], ""))):
            print(f"    {cui}  {rule[0]}  {name.get(cui,'?')}")

    print(f"\n  {'system':<22}{'F1 unfiltered':>26}{'F1 artifact-filtered':>28}"
          f"{'rel. drop':>11}")
    rows = []
    for m in MODELS:
        pn = per_note(m, art)
        if not pn:
            continue
        u = boot([a for a, _ in pn], args.resamples)
        fl = boot([b for _, b in pn], args.resamples)
        drop = (u[0] - fl[0]) / u[0] * 100 if u[0] else 0
        rows.append((m, u, fl, drop, len(pn)))
        print(f"  {NAME[m]:<22}{u[0]:>8.3f} [{u[1]:.3f}, {u[2]:.3f}]"
              f"{fl[0]:>10.3f} [{fl[1]:.3f}, {fl[2]:.3f}]{drop:>10.1f}%")

    out = POOLED / "ontology_artifact_ablation.json"
    out.write_text(json.dumps({
        "n_cuis_total": len(name),
        "n_artifacts": len(art),
        "artifacts": {c: {"rule": r, "name": name.get(c)} for c, r in art.items()},
        "systems": {m: {"f1_unfiltered": u[0], "ci_unfiltered": [u[1], u[2]],
                        "f1_filtered": f[0], "ci_filtered": [f[1], f[2]],
                        "relative_drop_pct": d, "n_notes": n}
                    for m, u, f, d, n in rows},
    }, indent=2), encoding="utf-8")
    print(f"\n  saved -> {out}")
    if args.export:
        Path(args.export).write_text(json.dumps(
            {c: {"rule": r, "name": name.get(c)} for c, r in sorted(art.items())},
            indent=2), encoding="utf-8")
        print(f"  artifact list -> {args.export}")


if __name__ == "__main__":
    raise SystemExit(main())
