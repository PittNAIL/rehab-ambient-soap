#!/usr/bin/env python3
"""
Assemble the comprehensive paper table by joining:
  - conventional text metrics (ROUGE-1/L, BERTScore, Jaccard, token counts,
    lexical entropy, readability) produced by conventional_text_metrics.py, and
  - ontology metrics (CUI-coverage F1, conceptual IoU) from metrics_test/metric_reports,
adding BCa bootstrap 95% CIs to the ontology columns as well.

Emits, for each filter convention (empty-only = reproduces the current Table 1;
degenerate = internally consistent with the 'Valid notes' N and accuracy analysis):
  - a Markdown preview
  - a LaTeX (booktabs) table, two panels

Run after conventional_text_metrics.py has written both JSON outputs.
"""

import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import bootstrap

# Repo root is the parent of this reporting/ folder; data lives under it.
ROOT = Path(__file__).resolve().parent.parent
SOAP = ROOT / "metrics_test" / "soap_prompts"
REPORTS = ROOT / "metrics_test" / "metric_reports"
QUANT = ROOT / "metrics_test" / "quantitative_metrics"

# file stem -> (display name, ontology-report stem)
MODELS = [
    ("gpt5-pro_soap_preds",        "GPT-5-Pro"),
    ("healthscribe_summaries",     "AWS HealthScribe"),
    ("jsl_text2soap_predictions",  "JSL Text2SOAP"),
    ("medgemma-27b-it",            "MedGemma-27B"),
    ("Llama3-Med42-70B",           "Med42-70B"),
    ("asclepius_13b_predictions",  "Asclepius-13B"),
    ("Llama3-OpenBioLLM-8B",       "OpenBioLLM-8B"),
]


def bca(values, seed=0):
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"), float("nan"), 0)
    m = float(arr.mean())
    if n < 2 or arr.std() == 0:
        return (m, m, m, n)
    try:
        res = bootstrap((arr,), np.mean, method="BCa", n_resamples=10000,
                        confidence_level=0.95, random_state=np.random.default_rng(seed))
        return (m, float(res.confidence_interval.low), float(res.confidence_interval.high), n)
    except Exception:
        return (m, m, m, n)


def load_positions(stem):
    """position -> string idx, and non-empty / non-degenerate flags."""
    soap = json.load(open(SOAP / f"{stem}.json", encoding="utf-8"))
    return soap


def ontology_by_position(stem):
    cov = json.load(open(REPORTS / f"{stem}_cui_coverage.json", encoding="utf-8"))
    iou = json.load(open(REPORTS / "concept_iou" / f"{stem}_cui_coverage_conceptual_iou.json", encoding="utf-8"))
    f1 = {int(x["idx"]): x["f1_coverage"] for x in cov["per_item"]}
    io = {int(x["idx"]): x["conceptual_iou"] for x in iou["per_item"]}
    return f1, io


def build(convention):
    """convention in {'empty-only','degenerate'}"""
    conv_file = ("conventional_text_metrics_emptyonly.json"
                 if convention == "empty-only" else "conventional_text_metrics.json")
    conv = json.load(open(QUANT / conv_file, encoding="utf-8"))["results_by_filename"]

    rows = []
    for stem, name in MODELS:
        res = conv[f"{stem}.json"]
        soap = load_positions(stem)
        idx_to_pos = {str(r.get("idx", i)): i for i, r in enumerate(soap)}
        f1_by_pos, iou_by_pos = ontology_by_position(stem)

        per = res.get("per_note", [])
        # collect ontology values for the SAME set of notes used for conventional metrics
        ont_f1, ont_iou = [], []
        for note in per:
            pos = idx_to_pos.get(str(note["idx"]))
            if pos in f1_by_pos:
                ont_f1.append(f1_by_pos[pos])
            if pos in iou_by_pos:
                ont_iou.append(iou_by_pos[pos])

        s = res["summary"]
        def g(k):  # (mean,lo,hi,n) from conventional summary
            v = s[k]; return (v["mean"], v["ci_low"], v["ci_high"], v["n"])

        rows.append({
            "name": name,
            # empty-only: count of non-empty notes; degenerate: count of non-degenerate notes.
            # Either way this is the denominator the metrics in the row were averaged over.
            "n_valid": res["num_valid"],
            "n_total": res["num_records"],
            "rouge1": g("rouge1_f1"),
            "rougeL": g("rougeL_f1"),
            "bert":   g("bertscore_f1"),
            "jacc":   g("jaccard_no_stopwords"),
            "ontf1":  bca(ont_f1),
            "ontiou": bca(ont_iou),
            "ptok":   g("prediction_num_tokens"),
            "rtok":   g("reference_num_tokens"),
            "ent":    g("lexical_entropy_bits"),
            "fre":    g("flesch_reading_ease"),
            "fkgl":   g("flesch_kincaid_grade"),
            "fog":    g("gunning_fog"),
        })
    return rows


def cell(t, nd):
    m, lo, hi, n = t
    if n == 0 or math.isnan(m):
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def n_header(convention):
    """The count column names the filter that produced it, not a generic 'Valid'."""
    return "Non-degenerate notes" if convention == "degenerate" else "Non-empty notes"


def md_table(rows, convention):
    out = []
    out.append("**Panel A — Similarity to therapist reference note** (mean [95% CI]; higher = better)\n")
    out.append(f"| Model | {n_header(convention)} | ROUGE-1 | ROUGE-L | BERTScore | Jaccard | Ontology F1 | Ontology IoU |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        out.append("| {name} | {v} | {a} | {b} | {c} | {d} | {e} | {f} |".format(
            name=r["name"], v=f"{r['n_valid']}/{r['n_total']}",
            a=cell(r["rouge1"],3), b=cell(r["rougeL"],3), c=cell(r["bert"],3),
            d=cell(r["jacc"],3), e=cell(r["ontf1"],3), f=cell(r["ontiou"],3)))
    out.append("\n**Panel B — Note characterization** (mean [95% CI])\n")
    out.append("| Model | Pred tokens | Ref tokens | Lexical entropy (bits) | Flesch Reading Ease | FK Grade | Gunning Fog |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append("| {name} | {a} | {b} | {c} | {d} | {e} | {f} |".format(
            name=r["name"], a=cell(r["ptok"],1), b=cell(r["rtok"],1),
            c=cell(r["ent"],2), d=cell(r["fre"],1), e=cell(r["fkgl"],1), f=cell(r["fog"],1)))
    return "\n".join(out)


def latex_table(rows, convention):
    def c(t, nd):
        m, lo, hi, n = t
        if n == 0 or math.isnan(m):
            return r"--"
        return rf"{m:.{nd}f} \tiny[{lo:.{nd}f}, {hi:.{nd}f}]"
    valid_word = "non-degenerate" if convention == "degenerate" else "non-empty"
    nh = n_header(convention)
    lines = []
    lines.append(r"% Comprehensive table (" + convention + r" convention). Requires \usepackage{booktabs}.")
    lines.append(r"\begin{table*}[t]\centering\footnotesize")
    lines.append(r"\caption{Lexical, semantic, and ontology-based similarity between AI-generated and therapist reference notes, "
                 r"and note-characterization metrics. Values are per-note means with 95\% BCa bootstrap CIs (10{,}000 resamples) "
                 r"over the " + valid_word + r" notes per system.}")
    lines.append(r"\label{tab:comprehensive}")
    # Panel A
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\toprule")
    lines.append("Model & " + nh + r" & ROUGE-1 & ROUGE-L & BERTScore & Jaccard & Ontology F1 & Ontology IoU \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append("{name} & {v} & {a} & {b} & {cc} & {d} & {e} & {f} \\\\".format(
            name=r["name"], v=f"{r['n_valid']}/{r['n_total']}",
            a=c(r["rouge1"],3), b=c(r["rougeL"],3), cc=c(r["bert"],3),
            d=c(r["jacc"],3), e=c(r["ontf1"],3), f=c(r["ontiou"],3)))
    lines.append(r"\midrule")
    lines.append(r"Model & \multicolumn{2}{c}{Pred / Ref tokens} & Entropy (bits) & Flesch RE & FK Grade & \multicolumn{2}{c}{Gunning Fog} \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append("{name} & {a} & {b} & {cc} & {d} & {e} & \\multicolumn{{2}}{{c}}{{{f}}} \\\\".format(
            name=r["name"], a=c(r["ptok"],1), b=c(r["rtok"],1),
            cc=c(r["ent"],2), d=c(r["fre"],1), e=c(r["fkgl"],1), f=c(r["fog"],1)))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    for conv in ["degenerate", "empty-only"]:
        rows = build(conv)
        md = md_table(rows, conv)
        tex = latex_table(rows, conv)
        (QUANT / f"comprehensive_table_{conv}.md").write_text(md, encoding="utf-8")
        (QUANT / f"comprehensive_table_{conv}.tex").write_text(tex, encoding="utf-8")
        print("\n" + "#" * 90)
        print(f"# CONVENTION: {conv}")
        print("#" * 90)
        print(md)
    print(f"\nSaved .md and .tex for both conventions to {QUANT}")
