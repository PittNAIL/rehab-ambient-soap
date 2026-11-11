#!/usr/bin/env python3
"""
paper_statistics.py

In-text statistics reported in the Results and Discussion that are not a table
cell. Reads cached pipeline outputs only -- no LLM calls.

  1. GPT-5 Pro vs AWS HealthScribe, per-encounter claim recall, pooled over
     153 encounters: median paired difference and Wilcoxon signed-rank test.
  2. Median system claim recall, and closed-source vs open-weight aggregate
     recall (mean of system means; descriptive, no test -- see Methods).
  3. Performance by therapy discipline within Study 1: claim recall by
     discipline (Kruskal-Wallis, all systems pooled and per system), and the
     share of gold claims with no transcript support, by discipline
     (percentile bootstrap over encounters).
  4. Transcript-observability ceiling: share of DISTINCT gold claims missed by
     at least one system that have no transcript support, pooled and per study.

Expected layout (written by the earlier pipeline stages), run from repo root:

  metrics_test/results_<corpus>/11_factehr_exact_claims.json
  metrics_test/results_<corpus>/grounding/16_missed_fact_grounding.json

with <corpus> in {encompass (Study 1), adapts (Study 2)}.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, wilcoxon

CORPORA = [("encompass", "Study 1"), ("adapts", "Study 2")]
SYSTEMS = ["gpt5-pro", "healthscribe", "jsl-text2soap", "medgemma-27b",
           "med42-70b", "asclepius-llama3-8b", "openbiollm-8b"]
CLOSED = {"gpt5-pro", "healthscribe", "jsl-text2soap"}
NAME = {
    "gpt5-pro": "GPT-5 Pro", "healthscribe": "AWS HealthScribe",
    "jsl-text2soap": "JSL Text2SOAP", "medgemma-27b": "MedGemma-27B-it",
    "med42-70b": "Llama3-Med42-70B", "asclepius-llama3-8b": "Asclepius-Llama3-8B",
    "openbiollm-8b": "Llama3-OpenBioLLM-8B",
}
IN, OUT = "in_transcript_not_in_ai_note", "not_in_transcript"
N_BOOT, SEED = 10000, 12345


def load(root: Path, corpus: str, rel: str):
    p = root / f"results_{corpus}" / rel
    return json.loads(p.read_text(encoding="utf-8"))


def discipline(idx: str) -> str:
    return idx.rsplit("_", 1)[-1].upper()


def recall_by_encounter(root):
    """{system: {(corpus, idx): recall}} over non-degenerate notes."""
    out = defaultdict(dict)
    for corpus, _ in CORPORA:
        d = load(root, corpus, "11_factehr_exact_claims.json")
        for s in SYSTEMS:
            m = d["models"].get(f"{s}_{corpus}")
            if not m:
                continue
            for it in m["per_item"]:
                if it.get("degenerate_pred_output") or it.get("recall") is None:
                    continue
                out[s][(corpus, it["idx"])] = it["recall"]
    return out


def grounding_by_fact(root, corpus):
    """{(idx, fact): classification}; identical across systems for one fact."""
    g = load(root, corpus, "grounding/16_missed_fact_grounding.json")
    out = {}
    for node in g["models"].values():
        for it in node["per_item"]:
            for x in it.get("groundings", []):
                out[(it["idx"], x["fact"])] = x["classification"]
    return out


def boot_ratio(num, den, rng):
    """Percentile bootstrap of sum(num)/sum(den), resampling encounters."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    r = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics_dir", default="metrics_test")
    args = ap.parse_args()
    root = Path(args.metrics_dir)
    rng = np.random.default_rng(SEED)

    rec = recall_by_encounter(root)

    # 1. GPT-5 Pro vs HealthScribe -----------------------------------------
    keys = sorted(set(rec["gpt5-pro"]) & set(rec["healthscribe"]))
    a = np.array([rec["gpt5-pro"][k] for k in keys])
    b = np.array([rec["healthscribe"][k] for k in keys])
    w = wilcoxon(a, b)
    print("== 1. GPT-5 Pro vs AWS HealthScribe (per-encounter claim recall)")
    print(f"   paired encounters      {len(keys)}")
    print(f"   median paired diff     {np.median(a - b):+.3f}")
    print(f"   Wilcoxon signed-rank   W={w.statistic:.1f}  p={w.pvalue:.2e}")

    # 2. System-level summaries --------------------------------------------
    means = {s: float(np.mean(list(rec[s].values()))) for s in SYSTEMS}
    print("\n== 2. System-level claim recall (mean over non-degenerate notes)")
    for s in SYSTEMS:
        print(f"   {NAME[s]:<22} {means[s]:.3f}  (n={len(rec[s])})")
    print(f"   median system          {np.median(list(means.values())):.3f}")
    # Pooled over notes (each non-degenerate note weighted once), not a mean of
    # system means.
    closed = np.mean([r for s in SYSTEMS if s in CLOSED for r in rec[s].values()])
    openw = np.mean([r for s in SYSTEMS if s not in CLOSED for r in rec[s].values()])
    print(f"   closed-source          {closed:.3f}   open-weight {openw:.3f}"
          "   (pooled over notes; descriptive, no test)")

    # 3. Discipline, Study 1 only ------------------------------------------
    print("\n== 3. Discipline (Study 1)")
    groups = defaultdict(list)
    for s in SYSTEMS:
        for (corpus, idx), r in rec[s].items():
            if corpus == "encompass":
                groups[discipline(idx)].append(r)
    order = ["OT", "PT", "SLP"]
    for dsc in order:
        print(f"   recall {dsc:<4} {np.mean(groups[dsc]):.3f}  (n notes={len(groups[dsc])})")
    print(f"   Kruskal-Wallis, all systems   p={kruskal(*[groups[d] for d in order]).pvalue:.3f}")
    for s in SYSTEMS:
        g = defaultdict(list)
        for (corpus, idx), r in rec[s].items():
            if corpus == "encompass":
                g[discipline(idx)].append(r)
        print(f"     {NAME[s]:<22} p={kruskal(*[g[d] for d in order]).pvalue:.3f}")

    claims = load(root, "encompass", "11_factehr_exact_claims.json")
    gmap = grounding_by_fact(root, "encompass")
    missed_any = defaultdict(set)   # idx -> distinct gold facts missed by >=1 system
    for node in claims["models"].values():
        for it in node["per_item"]:
            missed_any[it["idx"]].update(it.get("missed_gold_facts") or [])
    print("   share of distinct missed gold claims with no transcript support:")
    for dsc in order:
        num, den = [], []
        for idx, facts in missed_any.items():
            if discipline(idx) != dsc:
                continue
            cls = [gmap.get((idx, f)) for f in facts]
            cls = [c for c in cls if c in (IN, OUT)]
            if cls:
                num.append(sum(c == OUT for c in cls)); den.append(len(cls))
        m, lo, hi = boot_ratio(num, den, rng)
        print(f"     {dsc:<4} {m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]  (encounters={len(num)})")

    # 4. Observability ceiling ---------------------------------------------
    print("\n== 4. Observability ceiling (distinct gold claims missed by >=1 system)")
    tot_in = tot_out = verified = 0
    for corpus, label in CORPORA:
        claims = load(root, corpus, "11_factehr_exact_claims.json")
        gmap = grounding_by_fact(root, corpus)
        uniq, gold_verified = {}, {}
        for node in claims["models"].values():
            for it in node["per_item"]:
                gold_verified[it["idx"]] = len(it.get("verified_gold_facts") or [])
                for f in it.get("missed_gold_facts") or []:
                    c = gmap.get((it["idx"], f))
                    if c in (IN, OUT):
                        uniq[(it["idx"], f)] = c
        n_out = sum(c == OUT for c in uniq.values())
        tot_in += len(uniq) - n_out; tot_out += n_out
        verified += sum(gold_verified.values())
        print(f"   {label}: {n_out}/{len(uniq)} = {n_out/len(uniq)*100:.1f}% beyond transcript")
    n = tot_in + tot_out
    print(f"   Pooled : {tot_out}/{n} = {tot_out/n*100:.1f}% beyond transcript; "
          f"{n} distinct = {n/verified*100:.1f}% of {verified} verified gold claims")


if __name__ == "__main__":
    main()
