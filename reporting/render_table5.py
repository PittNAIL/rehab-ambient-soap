#!/usr/bin/env python3
"""
render_table5.py

Table 5 -- transcript coverage. Every verified transcript claim is tested for
entailment by the gold standard note and by the AI note for that encounter, and
falls into one of four mutually exclusive cells.

Writes:
  TABLE5_unpooled.md   one table per corpus
  TABLE5_pooled.md     ENCOMPASS + ADAPTS combined
  TABLE5.docx          both, as Word tables

The Total row sums the per-system rows, so the denominator is model-encounter
pairs (transcript claims x systems), not transcript claims. That is the same
convention as the published table.
"""

from __future__ import annotations

import argparse
import json
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

# summary key -> column
CELLS = [("in_gold_in_pred", "Both notes"),
         ("not_in_gold_in_pred", "AI note only"),
         ("in_gold_not_in_pred", "Human note only"),
         ("cell_1_3", "Neither note")]


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def rows_for(corpora):
    """(label, {cell: n}, checked) per system, plus the pooled total."""
    out = []
    tot = {k: 0 for k, _ in CELLS}
    tot_checked = 0
    for s in ORDER:
        agg = {k: 0 for k, _ in CELLS}
        checked = 0
        found = False
        for corpus in corpora:
            d = load(f"metrics_test/results_{corpus}/transcript/18_transcript_cell_1_3.json")
            m = (d or {}).get("models", {}).get(f"{s}_{corpus}")
            if not m:
                continue
            found = True
            u = m["summary"]
            checked += u["transcript_facts_checked"]
            for k, _ in CELLS:
                agg[k] += u[k]["count"]
        if not found:
            continue
        out.append((NAME[s], agg, checked))
        for k in agg:
            tot[k] += agg[k]
        tot_checked += checked
    return out, tot, tot_checked


def cell(n, d):
    return f"{n:,} ({n / d * 100:.0f}%)" if d else "—"


def md(corpora, title):
    rows, tot, tot_checked = rows_for(corpora)
    lines = [f"\n### {title}\n",
             "| System | " + " | ".join(c for _, c in CELLS) + " | Transcript claims |",
             "|---|---|---|---|---|---|"]
    for label, agg, checked in rows:
        cells = " | ".join(cell(agg[k], checked) for k, _ in CELLS)
        lines.append(f"| {label} | {cells} | {checked:,} |")
    cells = " | ".join(f"**{cell(tot[k], tot_checked)}**" for k, _ in CELLS)
    lines.append(f"| **Total (pooled)** | {cells} | **{tot_checked:,}** |")
    return "\n".join(lines)


CAPTION = (
    "Each verified transcript claim was tested for entailment by the gold "
    "standard note and by the AI-generated note for that encounter, and assigned "
    "to one of four mutually exclusive categories. Values are n (% of that "
    "system's transcript claims). Every system is scored over all encounters "
    "against the same denominator; no encounters are excluded, and empty and "
    "degenerate outputs were retained and scored as emitted, so rows are "
    "directly comparable.\n\n"
    "Two consequences follow for systems with low reliability. Empty outputs "
    "entail nothing, so their transcript claims fall into the human-note-only or "
    "neither category. Conversely, outputs that continue or reproduce the "
    "transcript entail transcript claims verbatim, so for systems with "
    "substantial degeneracy the AI-note columns measure textual overlap with the "
    "transcript rather than documentation, and should be read alongside the "
    "non-degenerate counts in Table 3.\n\n"
    "The Total row sums the per-system rows, so its denominator is "
    "model-encounter pairs rather than transcript claims."
)


def write_docx(out_path):
    from docx import Document
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.shared import Pt

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10)
    doc.add_heading("Table 5. Transcript coverage", level=1)
    r = doc.add_paragraph().add_run(CAPTION)
    r.font.size = Pt(9)

    def block(corpora, title):
        rows, tot, tot_checked = rows_for(corpora)
        doc.add_paragraph()
        h = doc.add_paragraph().add_run(title)
        h.bold = True
        h.font.size = Pt(11)
        headers = ["System"] + [c for _, c in CELLS] + ["Transcript claims"]
        tbl = doc.add_table(rows=len(rows) + 2, cols=len(headers))
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, htxt in enumerate(headers):
            c_ = tbl.rows[0].cells[i]
            c_.text = ""
            rr = c_.paragraphs[0].add_run(htxt)
            rr.bold = True
            rr.font.size = Pt(9)
        for ri, (label, agg, checked) in enumerate(rows, start=1):
            vals = [label] + [cell(agg[k], checked) for k, _ in CELLS] + [f"{checked:,}"]
            for ci, v in enumerate(vals):
                c_ = tbl.rows[ri].cells[ci]
                c_.text = ""
                rr = c_.paragraphs[0].add_run(v)
                rr.font.size = Pt(9)
        vals = (["Total (pooled)"] + [cell(tot[k], tot_checked) for k, _ in CELLS]
                + [f"{tot_checked:,}"])
        for ci, v in enumerate(vals):
            c_ = tbl.rows[len(rows) + 1].cells[ci]
            c_.text = ""
            rr = c_.paragraphs[0].add_run(v)
            rr.bold = True
            rr.font.size = Pt(9)

    block(["encompass", "adapts"], "Pooled: ENCOMPASS + ADAPTS (153 encounters)")
    for c, n in CORPORA:
        block([c], f"{c.upper()} ({n} encounters)")
    doc.save(out_path)
    print(f"Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="metrics_test/results_pooled")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    a = ["# Table 5. Transcript coverage (per corpus)\n", CAPTION]
    for c, n in CORPORA:
        a.append(md([c], f"{c.upper()} ({n} encounters)"))
    (out / "TABLE5_unpooled.md").write_text("\n".join(a) + "\n", encoding="utf-8")

    b = ["# Table 5. Transcript coverage (pooled)\n", CAPTION,
         md(["encompass", "adapts"], "Pooled: ENCOMPASS + ADAPTS (153 encounters)")]
    (out / "TABLE5_pooled.md").write_text("\n".join(b) + "\n", encoding="utf-8")

    print("\n".join(b))
    print()
    print("\n".join(a[2:]))
    write_docx(out / "TABLE5.docx")
    print(f"Saved -> {out / 'TABLE5_pooled.md'}")
    print(f"Saved -> {out / 'TABLE5_unpooled.md'}")


if __name__ == "__main__":
    main()
