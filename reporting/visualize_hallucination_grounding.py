#!/usr/bin/env python3
"""
visualize_hallucination_grounding.py

Visualizations for 13_hallucination_grounding.json, optionally crossed with
severity from 12_category_severity_gaps.json:
1. Per-model + aggregate: fabricated vs omitted-but-true rate
2. Fabricated/omitted-but-true x major/minor severity cross-tab
   ("fabricated + major" is the highest clinical-risk bucket)

Usage
-----
python visualize_hallucination_grounding.py \
    --grounding_json metrics_test/gap_analysis/13_hallucination_grounding.json \
    --category_json  metrics_test/gap_analysis/12_category_severity_gaps.json \
    --output_dir     metrics_test/gap_analysis/meeting_update_factehr_exact_claims/figures
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


def figure_1_per_model(data: Dict[str, Any], output_dir: Path) -> None:
    models = [m for m in data["models"] if data["models"][m]["summary"]["total_hallucinated_checked"] > 0]
    fab = [data["models"][m]["summary"]["fabricated"]["count"] for m in models]
    om = [data["models"][m]["summary"]["omitted_but_true"]["count"] for m in models]

    agg_fab, agg_om = sum(fab), sum(om)
    labels = [_short(m) for m in models] + ["ALL (aggregate)"]
    fab_all = fab + [agg_fab]
    om_all = om + [agg_om]

    order = sorted(range(len(labels)), key=lambda i: -(fab_all[i] + om_all[i]))
    labels = [labels[i] for i in order]
    fab_all = [fab_all[i] for i in order]
    om_all = [om_all[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(labels))
    ax.bar(x, om_all, 0.6, label="Omitted but true (grounded in transcript)", color="#27ae60")
    ax.bar(x, fab_all, 0.6, bottom=om_all, label="Fabricated (unsupported by any source)", color="#c0392b")

    for i, (f, o) in enumerate(zip(fab_all, om_all)):
        total = f + o
        if total:
            ax.text(i, o / 2, f"{o}\n({o/total:.0%})", ha="center", va="center", fontsize=8, color="white")
            ax.text(i, o + f / 2, f"{f}\n({f/total:.0%})", ha="center", va="center", fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    for tick, lab in zip(ax.get_xticklabels(), labels):
        if lab == "ALL (aggregate)":
            tick.set_fontweight("bold")
    ax.set_ylabel("Hallucinated fact count", fontsize=12, fontweight="bold")
    ax.set_title("Of the facts flagged 'hallucinated' (unsupported by gold note),\n"
                 "how many are actually grounded in the conversation transcript?",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "07_hallucination_grounding.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def figure_2_severity_cross(grounding: Dict[str, Any], category: Dict[str, Any], output_dir: Path) -> None:
    cross = defaultdict(int)
    for model in grounding["models"]:
        if model not in category["models"]:
            continue
        g_items = {it["idx"]: it for it in grounding["models"][model]["per_item"]}
        c_items = {it["idx"]: it for it in category["models"][model]["per_item"]}
        for idx, gitem in g_items.items():
            if idx not in c_items:
                continue
            sev_lookup = {pc["fact"]: pc["severity"] for pc in c_items[idx]["pred_fact_classes"]}
            for grounding_entry in gitem["groundings"]:
                sev = sev_lookup.get(grounding_entry["fact"], "unknown")
                if sev == "unknown":
                    continue
                cross[(grounding_entry["classification"], sev)] += 1

    cats = ["fabricated", "omitted_but_true"]
    sevs = ["major", "minor"]
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(cats))
    bottom = np.zeros(len(cats))
    colors = {"major": "#c0392b", "minor": "#95a5a6"}
    for sev in sevs:
        vals = [cross[(cat, sev)] for cat in cats]
        ax.bar(x, vals, 0.5, bottom=bottom, label=sev.capitalize(), color=colors[sev])
        for i, v in enumerate(vals):
            if v:
                ax.text(i, bottom[i] + v / 2, str(v), ha="center", va="center", fontsize=10,
                         color="white" if sev == "major" else "black", fontweight="bold")
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels(["Fabricated\n(unsupported by any source)", "Omitted but true\n(grounded in transcript)"])
    ax.set_ylabel("Hallucinated fact count", fontsize=12, fontweight="bold")
    ax.set_title("Hallucination type x clinical severity (all models combined)\n"
                 "'Fabricated + Major' is the highest-risk bucket",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = output_dir / "08_hallucination_severity_cross.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize hallucination grounding results.")
    parser.add_argument("--grounding_json", required=True)
    parser.add_argument("--category_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    grounding = load_json(Path(args.grounding_json))
    category = load_json(Path(args.category_json))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_1_per_model(grounding, output_dir)
    figure_2_severity_cross(grounding, category, output_dir)


if __name__ == "__main__":
    main()
