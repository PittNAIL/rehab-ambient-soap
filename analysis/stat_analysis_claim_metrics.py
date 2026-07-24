#!/usr/bin/env python3
"""
Statistical analysis for the claim-level factuality metrics.

- Per-note metrics (claim recall/precision/F1) summarized as mean with 95% CI via
  BCa bootstrap over encounters.
- Pre-specified between-system comparisons:
    (1) GPT-5 Pro vs AWS HealthScribe on claim recall  -> paired Wilcoxon signed-rank
        (natural pairing: same encounters, both non-degenerate).
    (2) closed-source vs open-weight on claim recall    -> group contrast.
        Reported as note-level Mann-Whitney U (primary, correct for an unpaired
        group contrast) AND an encounter-level paired signed-rank on per-encounter
        group means (secondary, to honor the paired template).
- Holm-Bonferroni correction across the k=2 pre-specified comparisons.
"""
import json
import numpy as np
from scipy.stats import bootstrap, wilcoxon, mannwhitneyu

RNG = np.random.default_rng(20260721)
N_BOOT = 10000

FILE = "metrics_test/gap_analysis/11_factehr_exact_claims.json"

DISPLAY = {
    "gpt5-pro_soap_preds": "GPT-5 Pro",
    "healthscribe_summaries": "AWS HealthScribe",
    "jsl_text2soap_predictions": "JSL Text2SOAP",
    "medgemma-27b-it": "MedGemma-27B",
    "Llama3-Med42-70B": "Med42-70B",
    "asclepius_13b_predictions": "Asclepius-13B",
    "Llama3-OpenBioLLM-8B": "OpenBioLLM-8B",
}
CLOSED = ["gpt5-pro_soap_preds", "healthscribe_summaries", "jsl_text2soap_predictions"]
OPEN = ["medgemma-27b-it", "Llama3-Med42-70B", "asclepius_13b_predictions", "Llama3-OpenBioLLM-8B"]

data = json.load(open(FILE, encoding="utf-8"))["models"]


def nondegen_items(model_key):
    """Return dict idx -> {recall,precision,f1} for non-degenerate notes."""
    out = {}
    for it in data[model_key]["per_item"]:
        if not it.get("degenerate_pred_output", False):
            out[it["idx"]] = {"recall": it["recall"], "precision": it["precision"], "f1": it["f1"]}
    return out


def bca_ci(vals):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2 or np.allclose(vals, vals[0]):
        return (float(np.mean(vals)), float(np.mean(vals)))
    res = bootstrap((vals,), np.mean, n_resamples=N_BOOT, method="BCa",
                    confidence_level=0.95, random_state=RNG)
    return (float(res.confidence_interval.low), float(res.confidence_interval.high))


print("=" * 78)
print(f"PER-NOTE METRICS: mean [95% BCa CI], non-degenerate notes, {N_BOOT} bootstrap resamples")
print("=" * 78)
print(f"{'System':<18}{'n':>4}  {'Recall':>22}  {'Precision':>22}  {'F1':>22}")
items = {mk: nondegen_items(mk) for mk in DISPLAY}
for mk in CLOSED + OPEN:
    d = items[mk]
    n = len(d)
    if n == 0:
        print(f"{DISPLAY[mk]:<18}{n:>4}  {'-- (no valid notes)':>22}")
        continue
    cells = []
    for m in ("recall", "precision", "f1"):
        v = [d[k][m] for k in d]
        lo, hi = bca_ci(v)
        cells.append(f"{np.mean(v):.3f} [{lo:.3f},{hi:.3f}]")
    flag = "  <- small n" if n < 15 else ""
    print(f"{DISPLAY[mk]:<18}{n:>4}  {cells[0]:>22}  {cells[1]:>22}  {cells[2]:>22}{flag}")

# ---------------------------------------------------------------- comparison 1
print("\n" + "=" * 78)
print("COMPARISON 1: GPT-5 Pro vs AWS HealthScribe, claim RECALL (paired signed-rank)")
print("=" * 78)
g = items["gpt5-pro_soap_preds"]
h = items["healthscribe_summaries"]
shared = sorted(set(g) & set(h))
gx = np.array([g[k]["recall"] for k in shared])
hx = np.array([h[k]["recall"] for k in shared])
w1 = wilcoxon(gx, hx, alternative="two-sided", zero_method="wilcox")
p1 = w1.pvalue
print(f"paired encounters (both non-degenerate): n = {len(shared)}")
print(f"GPT-5 Pro mean recall  = {gx.mean():.3f}   median = {np.median(gx):.3f}")
print(f"HealthScribe mean recall = {hx.mean():.3f}   median = {np.median(hx):.3f}")
print(f"median paired difference (GPT-HS) = {np.median(gx-hx):+.3f}")
print(f"Wilcoxon W = {w1.statistic:.1f},  raw p = {p1:.4g}")

# ---------------------------------------------------------------- comparison 2
print("\n" + "=" * 78)
print("COMPARISON 2: closed-source vs open-weight, claim RECALL")
print("=" * 78)
closed_notes = [items[mk][k]["recall"] for mk in CLOSED for k in items[mk]]
open_notes = [items[mk][k]["recall"] for mk in OPEN for k in items[mk]]
mw = mannwhitneyu(closed_notes, open_notes, alternative="two-sided")
p2 = mw.pvalue
print("PRIMARY (note-level, unpaired Mann-Whitney U):")
print(f"  closed: {len(closed_notes)} notes, mean recall {np.mean(closed_notes):.3f}, median {np.median(closed_notes):.3f}")
print(f"  open:   {len(open_notes)} notes, mean recall {np.mean(open_notes):.3f}, median {np.median(open_notes):.3f}")
print(f"  Mann-Whitney U = {mw.statistic:.1f},  raw p = {p2:.4g}")

# secondary: encounter-level paired means
enc_closed, enc_open = {}, {}
for mk in CLOSED:
    for k, v in items[mk].items():
        enc_closed.setdefault(k, []).append(v["recall"])
for mk in OPEN:
    for k, v in items[mk].items():
        enc_open.setdefault(k, []).append(v["recall"])
paired_enc = sorted(set(enc_closed) & set(enc_open))
cx = np.array([np.mean(enc_closed[k]) for k in paired_enc])
ox = np.array([np.mean(enc_open[k]) for k in paired_enc])
if len(paired_enc) >= 1 and not np.allclose(cx, ox):
    w2 = wilcoxon(cx, ox, alternative="two-sided", zero_method="wilcox")
    print("SECONDARY (encounter-level paired signed-rank on group means):")
    print(f"  paired encounters (>=1 valid note each group): n = {len(paired_enc)}")
    print(f"  closed mean {cx.mean():.3f}  vs  open mean {ox.mean():.3f}   median diff {np.median(cx-ox):+.3f}")
    print(f"  Wilcoxon W = {w2.statistic:.1f},  raw p = {w2.pvalue:.4g}")

# ---------------------------------------------------------------- correction
print("\n" + "=" * 78)
print("HOLM-BONFERRONI CORRECTION across k=2 pre-specified comparisons, alpha=0.05 two-sided")
print("=" * 78)
tests = [("GPT-5 Pro vs HealthScribe (recall)", p1),
         ("closed vs open (recall, Mann-Whitney)", p2)]
order = sorted(tests, key=lambda t: t[1])
k = len(order)
for i, (name, p) in enumerate(order):
    thr = 0.05 / (k - i)
    padj = min(p * (k - i), 1.0)
    print(f"  {name:<42} raw p={p:.4g}  Holm p={padj:.4g}  {'SIG' if padj < 0.05 else 'ns'}")
