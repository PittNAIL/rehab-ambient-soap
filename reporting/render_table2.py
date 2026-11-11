#!/usr/bin/env python3
"""
render_table2.py

Table 2 -- the twenty most frequent UMLS concepts in gold standard notes and in
two chosen systems, side by side.

Defaults to MedGemma-27B-it and OpenBioLLM-8B: on the corrected data these are
the two highest-scoring systems on the surface and ontology metrics that Table 2
audits, which is the comparison the table exists to interrogate.

Counts are TOTAL DETECTIONS, repeats within a note included, per the caption.
Concepts flagged as audited artifacts are marked with * and italicised in the
markdown -- function words, discourse connectives, dosing-instruction
fragments, and spurious mappings such as gene alleles, language names, and
unrelated diseases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SRC = Path("metrics_test/results_pooled/16_concept_frequency_tables.json")

NAME = {
    "gpt5-pro": "GPT-5 Pro", "healthscribe": "AWS HealthScribe",
    "jsl-text2soap": "JSL Text2SOAP", "medgemma-27b": "MedGemma-27B-it",
    "med42-70b": "Llama3-Med42-70B", "asclepius-llama3-8b": "Asclepius-Llama3-8B",
    "openbiollm-8b": "Llama3-OpenBioLLM-8B",
}

# Audited artifact classes. Matched case-insensitively on the concept name.
ARTIFACT_EXACT = {
    "in addition to", "with - dosing instruction fragment", "preposition for",
    "for (preposition)", "from", "then", "during",
    "during - dosing instruction fragment", "following", "adjacent",
    "one", "not", "today", "continuous", "approximate", "out (direction)",
    "removed", "observed", "global", "generalized", "act mood - goal",
    "minute unit of plane angle", "minute", "minangkabau language",
    "carib language", "wwox wt allele", "infantile neuroaxonal dystrophy",
    "sodium tetradecyl sulfate", "diabetes mellitus, non-insulin-dependent",
    "safe-biopharma standard", "automobiles", "date property (qualifier value)",
    "date in time", "therapy object (animal model)", "negation",
    "reports (qualifier value)", "reporting", "base sequence", "awareness",
    "breast feeding",
}


def is_artifact(name: str) -> bool:
    return (name or "").strip().lower() in ARTIFACT_EXACT


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def pool(d, picker):
    """Sum detections across corpora for one column.

    A CUI seen in both corpora is one concept with two partial counts, so the
    pooled figure is the sum -- not an average, and not a re-ranking of one
    corpus. 450 of the 1,375 gold CUIs appear in both.
    """
    from collections import defaultdict
    det, enc, meta = defaultdict(int), defaultdict(int), {}
    for corpus in ("encompass", "adapts"):
        node = d["corpora"].get(corpus)
        if not node:
            continue
        for r in picker(node, corpus):
            det[r["cui"]] += r["total_detections"]
            enc[r["cui"]] += r["encounters_present"]
            meta.setdefault(r["cui"], r)
    rows = [{"cui": c, "name": meta[c].get("name"),
             "semtypes": meta[c].get("semtypes", []),
             "total_detections": n, "encounters_present": enc[c]}
            for c, n in det.items()]
    rows.sort(key=lambda r: (-r["total_detections"], -r["encounters_present"], r["cui"]))
    return rows


def render_pooled(d, sys_a, sys_b, top):
    gold = pool(d, lambda node, c: node["gold"])

    def sys_rows(name):
        return pool(d, lambda node, c, n=name: node["systems"].get(f"{n}_{c}", []))

    a, b = sys_rows(sys_a), sys_rows(sys_b)
    n_art = sum(is_artifact(r["name"] or "") for r in gold[:top])
    lines = ["\n### Pooled: ENCOMPASS + ADAPTS (n = 153 encounters)\n",
             f"| Rank | Gold standard notes | {NAME[sys_a]} | {NAME[sys_b]} |",
             "|---|---|---|---|"]
    ga, aa, bb = column(gold, top), column(a, top), column(b, top)
    for i in range(top):
        lines.append(f"| {i+1} | {ga[i]} | {aa[i]} | {bb[i]} |")
    lines.append(f"\n_Artifacts among the top {top} pooled gold concepts: "
                 f"**{n_art}/{top}**; genuine clinical content: "
                 f"**{top - n_art}/{top}**._")
    return "\n".join(lines)


def column(rows, top):
    out = []
    for r in rows[:top]:
        label = r["name"] or r["cui"]
        mark = "*" if is_artifact(label) else ""
        out.append(f"{mark}{label}{mark} ({r['total_detections']})")
    while len(out) < top:
        out.append("")
    return out


def render(node, corpus_label, sys_a, sys_b, top, n_enc):
    key_a = [k for k in node["systems"] if k.startswith(sys_a + "_")]
    key_b = [k for k in node["systems"] if k.startswith(sys_b + "_")]
    if not (key_a and key_b):
        return f"\n_{corpus_label}: system not found_\n"
    gold = column(node["gold"], top)
    a = column(node["systems"][key_a[0]], top)
    b = column(node["systems"][key_b[0]], top)

    n_art = sum(is_artifact((r["name"] or "")) for r in node["gold"][:top])
    lines = [f"\n### {corpus_label} (n = {n_enc} encounters)\n",
             f"| Rank | Gold standard notes | {NAME[sys_a]} | {NAME[sys_b]} |",
             "|---|---|---|---|"]
    for i in range(top):
        lines.append(f"| {i+1} | {gold[i]} | {a[i]} | {b[i]} |")
    lines.append(f"\n_Artifacts among the top {top} gold concepts: "
                 f"**{n_art}/{top}**; genuine clinical content: "
                 f"**{top - n_art}/{top}**._")
    return "\n".join(lines)


CAPTION = (
    "Concepts were extracted with MetaMapLite against the UMLS 2022AB ontology; "
    "parenthetical values are total detection counts across the encounters "
    "(repeats within a note included). Concepts marked with * are audited "
    "artifacts -- function words, discourse connectives, dosing-instruction "
    "fragments, and spurious mappings such as gene alleles, language names and "
    "unrelated diseases -- that do not plausibly reflect note content. Because "
    "these artifacts appear on both sides of the comparison, they inflate the "
    "measured concept overlap, so the ontology scores in Table 1 are upper "
    "bounds."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="medgemma-27b")
    ap.add_argument("--b", default="openbiollm-8b")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="metrics_test/results_pooled/TABLE2_medgemma_vs_openbio.md")
    args = ap.parse_args()

    d = load()
    parts = [f"# Table 2. Twenty most frequent UMLS concepts — gold standard vs "
             f"{NAME[args.a]} and {NAME[args.b]}\n", CAPTION]
    parts.append(render_pooled(d, args.a, args.b, args.top))
    for corpus, n in (("encompass", 45), ("adapts", 108)):
        node = d["corpora"].get(corpus)
        if node:
            parts.append(render(node, corpus.upper(), args.a, args.b,
                                args.top, n))

    text = "\n".join(parts) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
