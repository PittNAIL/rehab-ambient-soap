#!/usr/bin/env python3
"""
visualize_category_severity.py

Visualizations for 12_category_severity_gaps.json:
1. Aggregate miss rate by category (across all valid models)
2. Category x severity stacked bar for missed facts
3. Per-model heatmap: category (rows) x model (columns), cell = miss rate

Usage
-----
python visualize_category_severity.py \
    --claims_json metrics_test/gap_analysis/12_category_severity_gaps.json \
    --output_dir  metrics_test/gap_analysis/meeting_update_factehr_exact_claims/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short(name: str) -> str:
    return (
        name.replace("_predictions", "")
        .replace("_soap_preds", "")
        .replace("_summaries", "")
        .replace("_text2soap", "")
    )


def _derive_gold_total(count: int, rate: float | None) -> int:
    """rate_of_category = count / gold_total -> recover gold_total."""
    if not rate:
        return 0
    return round(count / rate)


def figure_1_aggregate_miss_rate(data: Dict[str, Any], output_dir: Path) -> None:
    agg_missed = defaultdict(int)
    agg_total = defaultdict(int)

    for model, r in data["models"].items():
        for cat, v in r["summary"]["missed_by_category"].items():
            agg_missed[cat] += v["count"]
            agg_total[cat] += _derive_gold_total(v["count"], v.get("rate_of_category"))

    # Order by absolute missed count (operational impact), not raw rate --
    # categories with very few total occurrences (e.g. n<15) can show a
    # 100% miss rate that is statistically noisy and would otherwise
    # visually dominate over high-volume categories like "functional".
    cats_all = [c for c in agg_total if agg_total[c] > 0]
    cats_all.sort(key=lambda c: -agg_missed[c])
    rates = [agg_missed[c] / agg_total[c] for c in cats_all]
    counts = [agg_missed[c] for c in cats_all]
    totals = [agg_total[c] for c in cats_all]
    low_n = [n < 15 for n in totals]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#bdbdbd" if low else c for low, c in zip(
        low_n, plt.cm.Reds(np.clip(np.array(rates) * 0.6 + 0.3, 0, 1)))]
    bars = ax.bar(cats_all, counts, color=colors, edgecolor="black", linewidth=1)
    for bar, rate, cnt, total, low in zip(bars, rates, counts, totals, low_n):
        tag = " (low n)" if low else ""
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{rate:.0%} of {total}{tag}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Missed fact count (all models combined)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_title("Which categories are most frequently missed?\n(bar = count, label = miss rate; gray = fewer than 15 total occurrences)",
                 fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "04_category_miss_rate.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def figure_2_category_severity_stack(data: Dict[str, Any], output_dir: Path) -> None:
    major = defaultdict(int)
    minor = defaultdict(int)
    for model, r in data["models"].items():
        cxs = r["summary"]["missed_category_x_severity"]
        for cat, sevs in cxs.items():
            major[cat] += sevs.get("major", 0)
            minor[cat] += sevs.get("minor", 0)

    cats = sorted(set(list(major.keys()) + list(minor.keys())),
                  key=lambda c: -(major[c] + minor[c]))
    major_vals = [major[c] for c in cats]
    minor_vals = [minor[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cats))
    ax.bar(x, major_vals, 0.6, label="Major (impacts dx/management)", color="#c0392b")
    ax.bar(x, minor_vals, 0.6, bottom=major_vals, label="Minor", color="#95a5a6")

    for i, (mj, mn) in enumerate(zip(major_vals, minor_vals)):
        if mj:
            ax.text(i, mj / 2, str(mj), ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        if mn:
            ax.text(i, mj + mn / 2, str(mn), ha="center", va="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=30, ha="right")
    ax.set_ylabel("Missed fact count", fontsize=12, fontweight="bold")
    ax.set_title("Missed facts by category and clinical severity (all models combined)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "05_category_severity_stack.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def figure_3_per_model_heatmap(data: Dict[str, Any], output_dir: Path) -> None:
    models = [m for m in data["models"] if data["models"][m]["summary"]["total_missed"] > 0]
    all_cats = sorted({
        cat for m in models
        for cat in data["models"][m]["summary"]["missed_by_category"]
    })

    matrix = np.full((len(all_cats), len(models)), np.nan)
    for j, model in enumerate(models):
        by_cat = data["models"][model]["summary"]["missed_by_category"]
        for i, cat in enumerate(all_cats):
            v = by_cat.get(cat)
            if v and v.get("rate_of_category") is not None:
                matrix[i, j] = v["rate_of_category"]

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.3), max(6, len(all_cats) * 0.5)))
    im = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([_short(m) for m in models], rotation=30, ha="right")
    ax.set_yticks(range(len(all_cats)))
    ax.set_yticklabels(all_cats)

    for i in range(len(all_cats)):
        for j in range(len(models)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center",
                         fontsize=8, color="black" if matrix[i, j] < 0.6 else "white")

    ax.set_title("Miss rate by category x model", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Miss rate")
    plt.tight_layout()
    out = output_dir / "06_category_model_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize category/severity gap analysis.")
    parser.add_argument("--claims_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    data = load_json(Path(args.claims_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_1_aggregate_miss_rate(data, output_dir)
    figure_2_category_severity_stack(data, output_dir)
    figure_3_per_model_heatmap(data, output_dir)


if __name__ == "__main__":
    main()
