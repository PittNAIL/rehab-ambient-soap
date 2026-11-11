#!/usr/bin/env python3
"""
render_table1.py

Render Table 1 in the paper's format -- Mean [95% CI] for ROUGE-1, ROUGE-L,
BERTScore, ontology F1 and ontology IoU, with the non-empty denominator.

Writes two files:
  TABLE1_unpooled.md   one table per corpus
  TABLE1_pooled.md     ENCOMPASS + ADAPTS combined, one row per system

Pooling is over per-encounter values, never by averaging the two corpus means:
ADAPTS has 2.4x the encounters, so averaging means would silently weight each
ENCOMPASS encounter 2.4x an ADAPTS one.

All CIs are recomputed here by percentile bootstrap from per-encounter values,
including the text metrics. The metrics pipeline stores its own CIs, but those
cannot be pooled, so recomputing everything the same way keeps the unpooled and
pooled tables directly comparable.

Per the caption the denominator is each system's NON-EMPTY outputs, which
includes outputs later classified as degenerate; non-degenerate counts belong
to Table 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RNG_SEED = 12345
N_BOOT = 10000

NAME = {
    "gpt5-pro": "GPT-5 Pro",
    "healthscribe": "AWS HealthScribe",
    "jsl-text2soap": "JSL Text2SOAP",
    "medgemma-27b": "MedGemma-27B-it",
    "med42-70b": "Llama3-Med42-70B",
    "asclepius-llama3-8b": "Asclepius-Llama3-8B",
    "openbiollm-8b": "Llama3-OpenBioLLM-8B",
}
ORDER = ["gpt5-pro", "healthscribe", "jsl-text2soap", "medgemma-27b",
         "med42-70b", "asclepius-llama3-8b", "openbiollm-8b"]
CORPORA = [("encompass", 45), ("adapts", 108)]
METRICS = [("rouge1_f1", "ROUGE-1"), ("rougeL_f1", "ROUGE-L"),
           ("bertscore_f1", "BERTScore")]


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def boot(vals, rng):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if a.size == 0:
        return None
    idx = rng.integers(0, a.size, size=(N_BOOT, a.size))
    m = a[idx].mean(axis=1)
    return float(a.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell(stat, nd=3):
    if stat is None:
        return "--"
    m, lo, hi = stat
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def per_encounter(system, corpus):
    """Return per-encounter value lists for one system on one corpus."""
    out = {k: [] for k, _ in METRICS}
    out["ontf1"], out["iou"] = [], []
    n = 0

    txt = load(f"metrics_test/results_{corpus}/conventional_text_metrics_emptyonly.json")
    rec = (txt or {}).get("results_by_filename", {}).get(f"{system}_{corpus}.json")
    if rec:
        n = rec.get("num_valid", len(rec["per_note"]))
        for note in rec["per_note"]:
            for k, _ in METRICS:
                if note.get(k) is not None:
                    out[k].append(note[k])

    cov = load(f"metrics_test/results_{corpus}/metric_reports/{system}_{corpus}_cui_coverage.json")
    if cov:
        out["ontf1"] = [i["f1_coverage"] for i in cov["per_item"]]
    iou = load(f"metrics_test/results_{corpus}/concept_iou/{system}_{corpus}_cui_coverage_conceptual_iou.json")
    if iou:
        out["iou"] = [i["conceptual_iou"] for i in iou["per_item"]]
    return out, n


HEADER = ("| System | ROUGE-1 | ROUGE-L | BERTScore | Ontology F1 | Ontology IoU | n non-empty |\n"
          "|---|---|---|---|---|---|---|")


def row(label, vals, n, total, rng):
    cells = [cell(boot(vals[k], rng)) for k, _ in METRICS]
    cells.append(cell(boot(vals["ontf1"], rng)))
    cells.append(cell(boot(vals["iou"], rng)))
    return f"| {label} | " + " | ".join(cells) + f" | {n}/{total} |"


def unpooled():
    rng = np.random.default_rng(RNG_SEED)
    parts = []
    for corpus, total in CORPORA:
        parts += [f"\n## {corpus.upper()} (n = {total} encounters)\n", HEADER]
        for s in ORDER:
            vals, n = per_encounter(s, corpus)
            if not any(vals[k] for k, _ in METRICS):
                continue
            parts.append(row(NAME[s], vals, n, total, rng))
    return "\n".join(parts)


def pooled():
    rng = np.random.default_rng(RNG_SEED)
    parts = ["\n## Pooled: ENCOMPASS + ADAPTS (n = 153 encounters)\n", HEADER]
    for s in ORDER:
        merged = {k: [] for k, _ in METRICS}
        merged["ontf1"], merged["iou"] = [], []
        n_tot = 0
        split = []
        for corpus, total in CORPORA:
            vals, n = per_encounter(s, corpus)
            if not any(vals[k] for k, _ in METRICS):
                split.append(0)
                continue
            for k in merged:
                merged[k] += vals[k]
            n_tot += n
            split.append(n)
        if n_tot == 0:
            continue
        cells = [cell(boot(merged[k], rng)) for k, _ in METRICS]
        cells.append(cell(boot(merged["ontf1"], rng)))
        cells.append(cell(boot(merged["iou"], rng)))
        parts.append(f"| {NAME[s]} | " + " | ".join(cells)
                     + f" | {n_tot}/153 ({'+'.join(str(x) for x in split)}) |")
    return "\n".join(parts)


CAPTION = (
    "Mean [95% CI]. Metrics are computed over each system's non-empty outputs, "
    "which include outputs subsequently classified as degenerate; the final "
    "column gives that denominator. Non-degenerate counts are reported "
    "separately in Table 3 and are not the denominator for any value here.\n\n"
    f"_All CIs are percentile bootstrap over per-encounter values "
    f"({N_BOOT} resamples, seed {RNG_SEED})._"
)


def main():
    out = Path("metrics_test/results_pooled")
    out.mkdir(parents=True, exist_ok=True)

    a = ("# Table 1. Lexical, semantic, and ontology-based similarity "
         "(per corpus)\n\n" + CAPTION + "\n" + unpooled() + "\n")
    (out / "TABLE1_unpooled.md").write_text(a, encoding="utf-8")

    b = ("# Table 1. Lexical, semantic, and ontology-based similarity "
         "(pooled)\n\n" + CAPTION + "\n\nThe n column shows the split as "
         "total (ENCOMPASS+ADAPTS). Pooling is over per-encounter values, not "
         "an average of corpus means.\n" + pooled() + "\n")
    (out / "TABLE1_pooled.md").write_text(b, encoding="utf-8")

    print(a)
    print(b)
    print(f"Saved -> {out/'TABLE1_unpooled.md'}")
    print(f"Saved -> {out/'TABLE1_pooled.md'}")


if __name__ == "__main__":
    main()
