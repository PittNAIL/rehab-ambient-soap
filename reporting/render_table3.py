#!/usr/bin/env python3
"""
render_table3.py

Render Table 3 -- claim-level factuality -- pooled and per corpus.

  TABLE3_unpooled.md   one table per corpus
  TABLE3_pooled.md     ENCOMPASS + ADAPTS combined, one row per system

Per the caption, values are means over each system's NON-DEGENERATE notes, and
the final column reports that denominator. This differs from Table 1, whose
denominator is non-empty outputs.

F1 is computed per note as the harmonic mean of that note's recall and
precision and then averaged, so it is deliberately NOT the harmonic mean of the
tabulated means.

Pooling is over per-note values, never an average of corpus means: ADAPTS has
2.4x the encounters and would otherwise be under-weighted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

RNG_SEED = 12345
N_BOOT = 10000
MIN_FOR_BOLD = 30      # caption: exclude systems with <30 non-degenerate notes

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


def cell(stat, nd=3, bold=False):
    if stat is None:
        return "—"
    m, lo, hi = stat
    s = f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"
    return f"**{s}**" if bold else s


def per_note(system, corpus):
    """Per-note recall/precision/F1 over NON-DEGENERATE notes, plus counts."""
    d = load(f"metrics_test/results_{corpus}/11_factehr_exact_claims.json")
    m = (d or {}).get("models", {}).get(f"{system}_{corpus}")
    if not m:
        return None
    rec, prec, f1, gold, pred = [], [], [], [], []
    degen = 0
    for it in m["per_item"]:
        if it.get("degenerate_pred_output"):
            degen += 1
            continue
        if it.get("recall") is None:
            continue
        rec.append(it["recall"]); prec.append(it["precision"]); f1.append(it["f1"])
        gold.append(it.get("gold_facts_total")); pred.append(it.get("pred_facts_total"))
    return {"recall": rec, "precision": prec, "f1": f1,
            "gold": gold, "pred": pred, "n": len(rec), "degen": degen}


HEADER = ("| System | Claim recall | Claim precision | Claim F1 | "
          "Gold claims/note | Pred claims/note | Non-degenerate |\n"
          "|---|---|---|---|---|---|---|")


def table(rows, total, split=False):
    # bold the best value per metric, excluding low-n systems per the caption
    best = {}
    for k in ("recall", "precision", "f1"):
        cands = [(r["stats"][k][0], r["label"]) for r in rows
                 if r["stats"][k] and r["n"] >= MIN_FOR_BOLD]
        if cands:
            best[k] = max(cands)[1]
    out = [HEADER]
    for r in rows:
        s = r["stats"]
        g = f"{np.mean([x for x in r['gold'] if x is not None]):.1f}" if r["gold"] else "—"
        p = f"{np.mean([x for x in r['pred'] if x is not None]):.1f}" if r["pred"] else "—"
        denom = (f"{r['n']}/{total} ({'+'.join(str(x) for x in r['split'])})"
                 if split else f"{r['n']}/{total}")
        out.append(
            f"| {r['label']} "
            f"| {cell(s['recall'], bold=best.get('recall') == r['label'])} "
            f"| {cell(s['precision'], bold=best.get('precision') == r['label'])} "
            f"| {cell(s['f1'], bold=best.get('f1') == r['label'])} "
            f"| {g} | {p} | {denom} |")
    return "\n".join(out)


def unpooled():
    rng = np.random.default_rng(RNG_SEED)
    parts = []
    for corpus, total in CORPORA:
        rows = []
        for s in ORDER:
            d = per_note(s, corpus)
            if not d:
                continue
            rows.append({"label": NAME[s], "n": d["n"], "split": [d["n"]],
                         "gold": d["gold"], "pred": d["pred"],
                         "stats": {k: boot(d[k], rng) for k in ("recall", "precision", "f1")}})
        parts += [f"\n## {corpus.upper()} (n = {total} encounters)\n", table(rows, total)]
    return "\n".join(parts)


def pooled():
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for s in ORDER:
        merged = {"recall": [], "precision": [], "f1": [], "gold": [], "pred": []}
        n_tot, split = 0, []
        for corpus, _ in CORPORA:
            d = per_note(s, corpus)
            if not d:
                split.append(0)
                continue
            for k in merged:
                merged[k] += d[k]
            n_tot += d["n"]; split.append(d["n"])
        if n_tot == 0:
            continue
        rows.append({"label": NAME[s], "n": n_tot, "split": split,
                     "gold": merged["gold"], "pred": merged["pred"],
                     "stats": {k: boot(merged[k], rng) for k in ("recall", "precision", "f1")}})
    return ("\n## Pooled: ENCOMPASS + ADAPTS (n = 153 encounters)\n"
            + table(rows, 153, split=True))


CAPTION = (
    "Claim recall is the share of gold standard claims entailed by the generated "
    "note; claim precision is the share of generated claims entailed by the gold "
    "standard note. F1 was computed per note as the harmonic mean of that note's "
    "recall and precision and then averaged across notes; it is therefore not the "
    "harmonic mean of the tabulated means. Values are means over each system's "
    "non-degenerate notes; the final column reports that denominator. Note this "
    "differs from Table 1, whose denominator is non-empty outputs.\n\n"
    f"_Mean [95% CI], percentile bootstrap over per-note values ({N_BOOT} "
    f"resamples, seed {RNG_SEED}). Best value per metric in bold, excluding "
    f"systems with fewer than {MIN_FOR_BOLD} non-degenerate notes._"
)


def main():
    out = Path("metrics_test/results_pooled")
    out.mkdir(parents=True, exist_ok=True)

    a = ("# Table 3. Claim-level factuality of AI-generated notes (per corpus)\n\n"
         + CAPTION + "\n" + unpooled() + "\n")
    (out / "TABLE3_unpooled.md").write_text(a, encoding="utf-8")

    b = ("# Table 3. Claim-level factuality of AI-generated notes (pooled)\n\n"
         + CAPTION + "\n\nThe final column shows the split as total "
         "(ENCOMPASS+ADAPTS). Pooling is over per-note values, not an average "
         "of corpus means.\n" + pooled() + "\n")
    (out / "TABLE3_pooled.md").write_text(b, encoding="utf-8")

    print(a)
    print(b)
    print(f"Saved -> {out/'TABLE3_unpooled.md'}")
    print(f"Saved -> {out/'TABLE3_pooled.md'}")


if __name__ == "__main__":
    main()
