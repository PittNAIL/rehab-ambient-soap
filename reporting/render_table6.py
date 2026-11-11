#!/usr/bin/env python3
"""
render_table6.py

Table 6 -- transcript grounding of note-exclusive claims.

Two directions, deliberately asymmetric:

  AI-exclusive claims   present in the AI note, not entailed by the gold note.
                        Those the transcript also fails to support are FABRICATED.
                        Source: grounding/13_hallucination_grounding.json

  Human-exclusive       present in the gold note, not entailed by the AI note.
                        Those the transcript fails to support are documentation
                        BEYOND THE TRANSCRIPT -- clinical inference, direct
                        observation, or off-audio information.
                        Source: grounding/16_missed_fact_grounding.json

Percentages are of that system's own note-exclusive total, so rows are
independent of one another. Confidence intervals are percentile bootstrap over
ENCOUNTERS, not over claims: claims within an encounter share a transcript and
a note and are not independent, so resampling claims would understate the
interval. Encounters carry their claim counts with them, which is what makes
the resample honest.

  python render_table6.py                    # pooled + per corpus
  python render_table6.py --resamples 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

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


def ai_per_encounter(corpus, sysname):
    """[(n_claims, n_fabricated)] per encounter."""
    d = load(f"metrics_test/results_{corpus}/grounding/13_hallucination_grounding.json")
    m = (d or {}).get("models", {}).get(f"{sysname}_{corpus}")
    if not m:
        return []
    out = []
    for item in m["per_item"]:
        g = item.get("groundings", [])
        # entailment 0 => transcript does not support it => fabricated
        out.append((len(g), sum(1 for x in g if not x.get("entailment"))))
    return out


def human_per_encounter(corpus, sysname):
    """[(n_claims, n_beyond_transcript)] per encounter."""
    d = load(f"metrics_test/results_{corpus}/grounding/16_missed_fact_grounding.json")
    m = (d or {}).get("models", {}).get(f"{sysname}_{corpus}")
    if not m:
        return []
    out = []
    for item in m["per_item"]:
        g = item.get("groundings", [])
        out.append((len(g), sum(1 for x in g if not x.get("entailment"))))
    return out


def boot(pairs, resamples, rng):
    """Percentile bootstrap CI for hits/total, resampling encounters."""
    tot = sum(n for n, _ in pairs)
    hit = sum(k for _, k in pairs)
    if not tot:
        return hit, tot, None, None, None
    pct = hit / tot
    if len(pairs) < 2:
        return hit, tot, pct, None, None
    draws = []
    idx = range(len(pairs))
    for _ in range(resamples):
        s = [pairs[rng.choice(idx)] for _ in idx]
        t = sum(n for n, _ in s)
        if t:
            draws.append(sum(k for _, k in s) / t)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws))]
    return hit, tot, pct, lo, hi


def cell(hit, tot, pct, lo, hi):
    if not tot:
        return "—"
    s = f"{hit:,} ({pct*100:.1f}%)"
    if lo is not None:
        s += f" [{lo*100:.1f}, {hi*100:.1f}]"
    return s


def rows_for(corpora, resamples, seed=0):
    rng = random.Random(seed)
    out = []
    ai_tot, hu_tot = [], []
    for s in ORDER:
        ai, hu = [], []
        for c in corpora:
            ai += ai_per_encounter(c, s)
            hu += human_per_encounter(c, s)
        if not ai and not hu:
            continue
        ai_tot += ai
        hu_tot += hu
        out.append((NAME[s], boot(ai, resamples, rng), boot(hu, resamples, rng)))
    return out, boot(ai_tot, resamples, rng), boot(hu_tot, resamples, rng)


CAPTION = (
    "AI-exclusive claims (present in the AI note, not entailed by the gold "
    "standard note) that the transcript also fails to support are classified as "
    "fabricated. Human-exclusive claims (present in the gold standard note, not "
    "entailed by the AI note) that the transcript fails to support reflect "
    "documentation beyond the recorded dialogue: clinical inference, direct "
    "observation, or off-audio information. Values are n (% of that system's "
    "note-exclusive claims) [95% CI]. Confidence intervals are percentile "
    "bootstrap over encounters rather than claims, because claims within an "
    "encounter share a transcript and a note and are not independent."
)


def md(corpora, title, resamples):
    rows, ai_t, hu_t = rows_for(corpora, resamples)
    L = [f"\n### {title}\n",
         "| System | AI-exclusive claims | Fabricated, n (%) [95% CI] "
         "| Human-exclusive claims | Beyond transcript, n (%) [95% CI] |",
         "|---|---|---|---|---|"]
    for label, ai, hu in rows:
        L.append(f"| {label} | {ai[1]:,} | {cell(*ai)} | {hu[1]:,} | {cell(*hu)} |")
    L.append(f"| **Total** | **{ai_t[1]:,}** | **{cell(*ai_t)}** | "
             f"**{hu_t[1]:,}** | **{cell(*hu_t)}** |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="metrics_test/results_pooled")
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    pooled = ["# Table 6. Transcript grounding of note-exclusive claims (pooled)\n",
              CAPTION,
              md(["encompass", "adapts"],
                 "Pooled: ENCOMPASS + ADAPTS (153 encounters)", args.resamples)]
    (out / "TABLE6_pooled.md").write_text("\n".join(pooled) + "\n", encoding="utf-8")

    per = ["# Table 6. Transcript grounding of note-exclusive claims (per corpus)\n",
           CAPTION]
    for c, n in CORPORA:
        per.append(md([c], f"{c.upper()} ({n} encounters)", args.resamples))
    (out / "TABLE6_unpooled.md").write_text("\n".join(per) + "\n", encoding="utf-8")

    print("\n".join(pooled))
    print()
    print("\n".join(per[2:]))
    print(f"\nSaved -> {out/'TABLE6_pooled.md'}")
    print(f"Saved -> {out/'TABLE6_unpooled.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
