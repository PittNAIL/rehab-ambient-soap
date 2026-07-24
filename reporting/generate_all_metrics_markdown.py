#!/usr/bin/env python3
"""
Emit ONE large markdown report with every metric for every session (encounter),
over ALL non-empty notes (degenerate generations INCLUDED; only truly empty
predictions are dropped, since they have no ROUGE/undefined readability). This
matches the convention behind the existing Table 1 and avoids the survivorship
bias of filtering out failed generations.

Sections:
  1. Per-model summary (mean [95% BCa CI]) - similarity + note characterization.
  2. Per-model, per-session detail tables: one row per scored encounter with all
     per-note metrics (ROUGE-1/L, BERTScore, Jaccard, Ontology F1/IoU, token
     counts, lexical entropy, readability).

Reads the empty-only output of conventional_text_metrics.py and the ontology reports.
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
CONV_JSON = QUANT / "conventional_text_metrics_emptyonly.json"  # all non-empty notes
OUT = QUANT / "all_metrics_include_degenerate.md"

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


def ontology_by_position(stem):
    cov = json.load(open(REPORTS / f"{stem}_cui_coverage.json", encoding="utf-8"))
    iou = json.load(open(REPORTS / "concept_iou" / f"{stem}_cui_coverage_conceptual_iou.json", encoding="utf-8"))
    f1 = {int(x["idx"]): x["f1_coverage"] for x in cov["per_item"]}
    io = {int(x["idx"]): x["conceptual_iou"] for x in iou["per_item"]}
    return f1, io


def ci(t, nd):
    m, lo, hi, n = t
    if n == 0 or math.isnan(m):
        return "--"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def num(v, nd):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{nd}f}"


def main():
    conv = json.load(open(CONV_JSON, encoding="utf-8"))["results_by_filename"]

    md = []
    md.append("# Comprehensive metrics — all sessions (all non-empty notes; degenerate included)\n")
    md.append("_AI-generated therapy/clinical notes vs therapist-authored reference notes._\n")
    md.append("All notes are scored, including degenerate generations (repetition loops, "
              "dialogue leaks, truncated text); only truly **empty** predictions are dropped, "
              "since they have no ROUGE and undefined readability. This matches the convention "
              "behind the existing Table 1 and avoids the survivorship bias of excluding failed "
              "generations. Per-model aggregates are means with 95% BCa bootstrap CIs "
              "(10,000 resamples over encounters). Metrics: ROUGE-1/L F1 (stemmed), BERTScore F1 "
              "(roberta-large), Jaccard (stopwords removed), ontology CUI-coverage F1 and "
              "conceptual IoU (MetaMap), predicted/reference token counts (stopwords removed), "
              "lexical entropy (bits), and readability (Flesch Reading Ease, Flesch-Kincaid "
              "Grade, Gunning Fog).\n")
    md.append("> **Caveat on readability/entropy:** Flesch/Fog indices divide by sentence and "
              "syllable counts, which break on repetition-loop text; extreme values (e.g. large "
              "negative Flesch Reading Ease or grade levels in the hundreds) are arithmetic "
              "artifacts of degenerate output, not genuine complexity. Treat readability and "
              "entropy as descriptive of note text, not as quality scores.\n")

    # ---- Section 1: summary tables ----
    md.append("## 1. Per-model summary\n")
    md.append("### Panel A — Similarity to reference (mean [95% CI])\n")
    md.append("| Model | N | ROUGE-1 | ROUGE-L | BERTScore | Jaccard | Ontology F1 | Ontology IoU |")
    md.append("|---|---|---|---|---|---|---|---|")
    panelB = []
    for stem, name in MODELS:
        res = conv[f"{stem}.json"]
        s = res["summary"]
        soap = json.load(open(SOAP / f"{stem}.json", encoding="utf-8"))
        idx_to_pos = {str(r.get("idx", i)): i for i, r in enumerate(soap)}
        f1_by_pos, iou_by_pos = ontology_by_position(stem)
        ont_f1 = [f1_by_pos[idx_to_pos[str(n["idx"])]] for n in res["per_note"] if idx_to_pos.get(str(n["idx"])) in f1_by_pos]
        ont_iou = [iou_by_pos[idx_to_pos[str(n["idx"])]] for n in res["per_note"] if idx_to_pos.get(str(n["idx"])) in iou_by_pos]

        def g(k, _s=s):
            v = _s[k]; return (v["mean"], v["ci_low"], v["ci_high"], v["n"])

        md.append(f"| {name} | {res['num_valid']}/{res['num_records']} | "
                  f"{ci(g('rouge1_f1'),3)} | {ci(g('rougeL_f1'),3)} | {ci(g('bertscore_f1'),3)} | "
                  f"{ci(g('jaccard_no_stopwords'),3)} | {ci(bca(ont_f1),3)} | {ci(bca(ont_iou),3)} |")
        panelB.append((name, s))

    md.append("\n### Panel B — Note characterization (mean [95% CI])\n")
    md.append("| Model | Pred tokens | Ref tokens | Lexical entropy (bits) | Flesch RE | FK Grade | Gunning Fog |")
    md.append("|---|---|---|---|---|---|---|")
    for name, s in panelB:
        def g(k, _s=s):
            v = _s[k]; return (v["mean"], v["ci_low"], v["ci_high"], v["n"])
        md.append(f"| {name} | {ci(g('prediction_num_tokens'),1)} | {ci(g('reference_num_tokens'),1)} | "
                  f"{ci(g('lexical_entropy_bits'),2)} | {ci(g('flesch_reading_ease'),1)} | "
                  f"{ci(g('flesch_kincaid_grade'),1)} | {ci(g('gunning_fog'),1)} |")

    # ---- Section 2: per-session detail ----
    md.append("\n## 2. Per-session detail\n")
    md.append("One row per scored encounter. `idx` is the encounter ID. "
              "Rouge/BERT/Jaccard/Ontology are similarity to the reference; "
              "tokens/entropy/readability characterize the generated note.\n")

    for stem, name in MODELS:
        res = conv[f"{stem}.json"]
        md.append(f"\n### {name}  ({res['num_valid']}/{res['num_records']} scored)\n")
        if res["num_valid"] == 0:
            md.append("_No non-empty notes; omitted from all estimates._\n")
            continue
        soap = json.load(open(SOAP / f"{stem}.json", encoding="utf-8"))
        idx_to_pos = {str(r.get("idx", i)): i for i, r in enumerate(soap)}
        f1_by_pos, iou_by_pos = ontology_by_position(stem)

        md.append("| idx | ROUGE-1 | ROUGE-L | BERTScore | Jaccard | Ont F1 | Ont IoU | Pred tok | Ref tok | Entropy | FRE | FKGL | Fog |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for n in res["per_note"]:
            pos = idx_to_pos.get(str(n["idx"]))
            of1 = f1_by_pos.get(pos); oio = iou_by_pos.get(pos)
            md.append("| {idx} | {r1} | {rl} | {bt} | {jc} | {of1} | {oio} | {pt} | {rt} | {en} | {fre} | {fk} | {fog} |".format(
                idx=n["idx"],
                r1=num(n.get("rouge1_f1"), 3), rl=num(n.get("rougeL_f1"), 3),
                bt=num(n.get("bertscore_f1"), 3), jc=num(n.get("jaccard_no_stopwords"), 3),
                of1=num(of1, 3), oio=num(oio, 3),
                pt=num(n.get("prediction_num_tokens"), 0), rt=num(n.get("reference_num_tokens"), 0),
                en=num(n.get("lexical_entropy_bits"), 2), fre=num(n.get("flesch_reading_ease"), 1),
                fk=num(n.get("flesch_kincaid_grade"), 1), fog=num(n.get("gunning_fog"), 1)))

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT}  ({len(md)} lines)")


if __name__ == "__main__":
    main()
