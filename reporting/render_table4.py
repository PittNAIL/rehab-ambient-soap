#!/usr/bin/env python3
"""
render_table4.py

Render Table 4 -- content categories of claims missed by AI-generated notes,
pooled across models, with transcript grounding.

  TABLE4_unpooled.md   one table per corpus
  TABLE4_pooled.md     ENCOMPASS + ADAPTS combined

Rows are the three CREOLA content categories, then the uncategorized bucket
broken out into TopicGPT topics. Columns give how often each was missed and
what share of those misses the transcript supports.

"In transcript" means the missed claim WAS spoken in the session and the model
omitted it anyway -- a true omission. "Beyond transcript" means it never
appears in the dialogue, so the human documented it from observation,
measurement, or off-audio knowledge, and no transcript-driven system could
have recovered it. Separating the two is the point of the table: they are
different failures with different remedies.

Counts are model-encounter pairs pooled across systems, matching the paper's
"pooled across models" framing: a claim missed by four systems counts four
times.
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

EX_N = 3          # example claims shown per row
EX_CHARS = 190    # truncate the example cell so the table stays readable


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def collect(corpora):
    """fact -> category, and pooled per-fact miss/grounding tallies."""
    cat_of = {}
    topic_of = (load("metrics_test/results_pooled/15_uncategorized_topic_model.json")
                or {}).get("fact_to_topic", {})

    missed = Counter()                       # key -> times missed
    grounded = defaultdict(Counter)          # key -> {in_transcript, not_in_transcript}
    examples = defaultdict(Counter)          # key -> fact text, by how often missed

    for corpus in corpora:
        creola = load(f"metrics_test/results_{corpus}/creola/14_creola_category_gaps.json")
        claims = load(f"metrics_test/results_{corpus}/11_factehr_exact_claims.json")
        ground = load(f"metrics_test/results_{corpus}/grounding/16_missed_fact_grounding.json")
        if not (creola and claims):
            continue

        # gold fact -> CREOLA category (model-invariant, so any model will do)
        for m in creola["models"].values():
            for item in m["per_item"]:
                for f in item.get("gold_fact_classes", []):
                    cat_of.setdefault(f["fact"], f.get("category"))

        # grounding classification per (model, idx, fact)
        gmap = {}
        for name, node in (ground or {}).get("models", {}).items():
            for item in node.get("per_item", []):
                for g in item.get("groundings", []):
                    gmap[(name, item["idx"], g["fact"])] = g["classification"]

        # pooled misses across systems
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
    return missed, grounded, examples


def pct(n, d):
    return f"{n / d * 100:.0f}%" if d else "—"


def render(corpora, title):
    missed, grounded, examples = collect(corpora)
    total = sum(missed.values())
    cat_rows = [(k, v) for k, v in missed.items() if k[0] == "cat"]
    topic_rows = sorted([(k, v) for k, v in missed.items() if k[0] == "topic"],
                        key=lambda kv: -kv[1])
    uncat_total = sum(v for _, v in topic_rows)

    out = [f"\n## {title}  (n = {total:,} gold-standard-exclusive claims, pooled across systems)\n",
           "| Category | Missed, n (%) | In transcript | Beyond transcript | Common examples |",
           "|---|---|---|---|---|"]

    def row(label, key, n, indent=False):
        g = grounded.get(key, Counter())
        din = g.get("in_transcript_not_in_ai_note", 0)
        dout = g.get("not_in_transcript", 0)
        tot = din + dout
        name = f"&nbsp;&nbsp;{label}" if indent else f"**{label}**"
        # Most frequently missed claims verbatim, so the reader can see what
        # kind of content each row actually contains.
        ex = "; ".join(f'"{f}"' for f, _ in examples[key].most_common(EX_N))
        if len(ex) > EX_CHARS:
            ex = ex[:EX_CHARS].rsplit(";", 1)[0] + " […]"
        return (f"| {name} | {n:,} ({n / total * 100:.0f}%) | {pct(din, tot)} "
                f"| {pct(dout, tot)} | {ex} |")

    for c in CAT_ORDER:
        key = ("cat", c)
        if key in missed:
            out.append(row(CAT_LABEL[c], key, missed[key]))
    other = [(k, v) for k, v in cat_rows
             if k[1] not in CAT_ORDER and k[1] != UNCAT]
    for k, v in sorted(other, key=lambda kv: -kv[1]):
        out.append(row(str(k[1]), k, v))

    out.append(f"| **Uncategorized** | {uncat_total:,} "
               f"({uncat_total / total * 100:.0f}%) | | | |")
    for k, v in topic_rows:
        out.append(row(k[1], k, v, indent=True))
    return "\n".join(out)


CAPTION = (
    "Claims were classified with the CREOLA taxonomy; uncategorized claims were "
    "characterized by single-level TopicGPT modelling. \"In transcript\" is the "
    "share of missed claims entailed by the session transcript (true AI "
    "omissions); \"beyond transcript\" is the share absent from the transcript "
    "(documentation beyond the recorded dialogue -- clinical inference, direct "
    "observation, or off-audio information). Percentages in those two columns "
    "are of the grounded subset, given in the final column.\n\n"
    "_Counts are model-encounter pairs pooled across systems: a claim missed by "
    "four systems counts four times._"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0, help="unused; all topics are listed")
    ap.parse_args()

    out = Path("metrics_test/results_pooled")
    out.mkdir(parents=True, exist_ok=True)

    a = ["# Table 4. Content categories of claims missed by AI-generated notes "
         "(per corpus)\n", CAPTION]
    for c in CORPORA:
        a.append(render([c], c.upper()))
    (out / "TABLE4_unpooled.md").write_text("\n".join(a) + "\n", encoding="utf-8")

    b = ["# Table 4. Content categories of claims missed by AI-generated notes "
         "(pooled)\n", CAPTION,
         render(CORPORA, "Pooled: ENCOMPASS + ADAPTS")]
    (out / "TABLE4_pooled.md").write_text("\n".join(b) + "\n", encoding="utf-8")

    print("\n".join(b))
    print(f"\nSaved -> {out/'TABLE4_unpooled.md'}")
    print(f"Saved -> {out/'TABLE4_pooled.md'}")


if __name__ == "__main__":
    main()
