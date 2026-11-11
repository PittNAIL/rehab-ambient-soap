#!/usr/bin/env python3
"""
render_table4_unique.py

Table 4 with an added distinct-claim column, answering reviewer comment HM22.

The published Table 4 counts a claim once per system that missed it, so systems
missing the most content drive the percentages. This script reproduces that
table exactly and adds one column: the transcript-present share recomputed over
distinct gold claims, each counted once regardless of how many systems missed
it.

Collection logic is copied from render_table4.py unchanged, so the instance
columns are identical to the published table. The only addition is `uniq`,
which dedupes on (corpus, encounter, fact) before tallying.

  TABLE4_unique_pooled.md     ENCOMPASS + ADAPTS combined
  TABLE4_unique_unpooled.md   one table per corpus

No LLM calls; reads cached artifacts only. Run from the repo root.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CORPORA = ["encompass", "adapts"]
UNCAT = "uncategorized_by_creola"
CAT_LABEL = {
    "current_issues": "Current issues",
    "information_and_plan": "Information & plan",
    "pmfs": "PMFS history",
}
CAT_ORDER = ["current_issues", "information_and_plan", "pmfs"]

IN = "in_transcript_not_in_ai_note"
OUT = "not_in_transcript"

EX_N = 3
EX_CHARS = 190


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect(corpora):
    """fact -> category, pooled per-fact tallies, and distinct-claim tallies."""
    cat_of = {}
    topic_of = (load("metrics_test/results_pooled/15_uncategorized_topic_model.json")
                or {}).get("fact_to_topic", {})

    missed = Counter()                       # key -> times missed (instances)
    grounded = defaultdict(Counter)          # key -> {in, out} over instances
    examples = defaultdict(Counter)
    uniq = {}                                # (corpus, idx, fact) -> (key, cls)

    for corpus in corpora:
        creola = load(f"metrics_test/results_{corpus}/creola/14_creola_category_gaps.json")
        claims = load(f"metrics_test/results_{corpus}/11_factehr_exact_claims.json")
        ground = load(f"metrics_test/results_{corpus}/grounding/16_missed_fact_grounding.json")
        if not (creola and claims):
            continue

        for m in creola["models"].values():
            for item in m["per_item"]:
                for f in item.get("gold_fact_classes", []):
                    cat_of.setdefault(f["fact"], f.get("category"))

        gmap = {}
        for name, node in (ground or {}).get("models", {}).items():
            for item in node.get("per_item", []):
                for g in item.get("groundings", []):
                    gmap[(name, item["idx"], g["fact"])] = g["classification"]

        for name, node in claims["models"].items():
            for item in node["per_item"]:
                for fact in (item.get("missed_gold_facts") or []):
                    cat = cat_of.get(fact)
                    key = ("topic", topic_of.get(fact, "(unassigned)")) if cat == UNCAT \
                        else ("cat", cat)
                    missed[key] += 1
                    examples[key][fact] += 1
                    cls = gmap.get((name, item["idx"], fact))
                    if cls:
                        grounded[key][cls] += 1
                        # One entry per distinct gold claim, however many
                        # systems missed it. Same key, same classification
                        # for every system, so last write is harmless.
                        uniq[(corpus, item["idx"], fact)] = (key, cls)

    uniq_t = defaultdict(Counter)
    for key, cls in uniq.values():
        uniq_t[key][cls] += 1
    return missed, grounded, examples, uniq_t


def pct(n, d):
    return f"{n / d * 100:.0f}%" if d else "—"


def render(corpora, title):
    missed, grounded, examples, uniq_t = collect(corpora)
    total = sum(missed.values())
    uniq_total = sum(sum(v.values()) for v in uniq_t.values())
    cat_rows = [(k, v) for k, v in missed.items() if k[0] == "cat"]
    topic_rows = sorted([(k, v) for k, v in missed.items() if k[0] == "topic"],
                        key=lambda kv: -kv[1])
    uncat_total = sum(v for _, v in topic_rows)
    uncat_uniq = sum(sum(uniq_t[k].values()) for k, _ in topic_rows)

    out = [
        f"\n## {title}",
        f"\n_n = {total:,} model-encounter claim instances; "
        f"{uniq_total:,} distinct gold claims._\n",
        "| Category | Missed, n (%) | In transcript | Beyond transcript "
        "| Distinct claims, n | In transcript, distinct | Common examples |",
        "|---|---|---|---|---|---|---|",
    ]

    def row(label, key, n, indent=False):
        g = grounded.get(key, Counter())
        din, dout = g.get(IN, 0), g.get(OUT, 0)
        tot = din + dout
        u = uniq_t.get(key, Counter())
        uin, uout = u.get(IN, 0), u.get(OUT, 0)
        utot = uin + uout
        name = f"&nbsp;&nbsp;{label}" if indent else f"**{label}**"
        ex = "; ".join(f'"{f}"' for f, _ in examples[key].most_common(EX_N))
        if len(ex) > EX_CHARS:
            ex = ex[:EX_CHARS].rsplit(";", 1)[0] + " […]"
        # Difference of the two displayed (rounded) percentages, so the
        # arithmetic in the printed row is self-consistent.
        delta = (f" ({round(uin / utot * 100) - round(din / tot * 100):+d})"
                 if tot and utot else "")
        return (f"| {name} | {n:,} ({n / total * 100:.0f}%) | {pct(din, tot)} "
                f"| {pct(dout, tot)} | {utot:,} | {pct(uin, utot)}{delta} | {ex} |")

    for c in CAT_ORDER:
        key = ("cat", c)
        if key in missed:
            out.append(row(CAT_LABEL[c], key, missed[key]))
    other = [(k, v) for k, v in cat_rows
             if k[1] not in CAT_ORDER and k[1] != UNCAT]
    for k, v in sorted(other, key=lambda kv: -kv[1]):
        # A claim carrying no CREOLA label at all. render_table4.py emits this
        # row too; it was dropped when the published table was typeset.
        label = "No CREOLA label recorded" if k[1] is None else str(k[1])
        out.append(row(label, k, v))

    out.append(f"| **Uncategorized** | {uncat_total:,} "
               f"({uncat_total / total * 100:.0f}%) | | | {uncat_uniq:,} | | |")
    for k, v in topic_rows:
        out.append(row(k[1], k, v, indent=True))

    # Pooled row, for the caption's headline comparison.
    gi = Counter()
    for v in grounded.values():
        gi.update(v)
    gu = Counter()
    for v in uniq_t.values():
        gu.update(v)
    out.append(
        f"| **All missed claims** | {total:,} (100%) | {pct(gi[IN], sum(gi.values()))} "
        f"| {pct(gi[OUT], sum(gi.values()))} | {uniq_total:,} "
        f"| {pct(gu[IN], sum(gu.values()))} | |")
    return "\n".join(out)


CAPTION = (
    "Claims were classified with the CREOLA taxonomy; uncategorized claims were "
    "characterized by single-level TopicGPT modelling. \"In transcript\" is the "
    "share of missed claims entailed by the session transcript (true AI "
    "omissions); \"beyond transcript\" is the share absent from the transcript "
    "(documentation beyond the recorded dialogue -- clinical inference, direct "
    "observation, or off-audio information).\n\n"
    "_Counts in the first three columns are model-encounter pairs: a claim missed "
    "by four systems counts four times, so systems that miss the most content "
    "contribute most to those shares. The two distinct-claim columns recompute "
    "the same quantity over distinct gold claims, each counted once regardless of "
    "how many systems missed it; the parenthetical is the change in percentage "
    "points._"
)


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    out = Path("metrics_test/results_pooled")
    out.mkdir(parents=True, exist_ok=True)

    a = ["# Table 4 with distinct-claim column (per corpus)\n", CAPTION]
    for c in CORPORA:
        a.append(render([c], c.upper()))
    (out / "TABLE4_unique_unpooled.md").write_text("\n".join(a) + "\n", encoding="utf-8")

    b = ["# Table 4 with distinct-claim column (pooled)\n", CAPTION,
         render(CORPORA, "Pooled: Study 1 + Study 2")]
    (out / "TABLE4_unique_pooled.md").write_text("\n".join(b) + "\n", encoding="utf-8")

    print("\n".join(b))


if __name__ == "__main__":
    main()
