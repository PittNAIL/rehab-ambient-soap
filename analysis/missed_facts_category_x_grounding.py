#!/usr/bin/env python3
"""
missed_facts_category_x_grounding.py

Pure aggregation, no new LLM calls. Joins three already-computed outputs by
(model, idx, fact text) to answer: of the facts AI misses, which clinical
categories are "fair" misses (the content was right there in the transcript)
vs. structurally unrecoverable (never said aloud, so no text-only system
could ever have caught it)?

  - 16_missed_fact_grounding.json   -> transcript-presence classification
                                        per missed fact, per model
  - 14_creola_category_gaps.json    -> gold_fact_classes (content category,
                                        per item, per model)
  - 15_uncategorized_topic_model.json -> fact_to_topic (global, dedup'd by
                                        fact text) for finer topic labels
                                        when category == uncategorized_by_creola
                                        (this is where mood/affect content
                                        lives, under the "Session
                                        Participation" topic)

Usage
-----
python missed_facts_category_x_grounding.py \
    --grounding_json metrics_test/gap_analysis/16_missed_fact_grounding.json \
    --category_json  metrics_test/gap_analysis/14_creola_category_gaps.json \
    --topic_json     metrics_test/gap_analysis/15_uncategorized_topic_model.json \
    --output_dir     metrics_test/gap_analysis
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-tab missed-fact category x transcript-grounding.")
    parser.add_argument("--grounding_json", required=True)
    parser.add_argument("--category_json", required=True)
    parser.add_argument("--topic_json", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    grounding_data = load_json(Path(args.grounding_json))
    category_data = load_json(Path(args.category_json))
    topic_data = load_json(Path(args.topic_json))
    fact_to_topic: Dict[str, str] = topic_data.get("fact_to_topic", {})

    # label = category, or "uncategorized_by_creola: <topic>" when a topic is known
    cross_tab: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unmatched = 0
    total = 0

    for model_name, model_grounding in grounding_data["models"].items():
        category_items = {
            item["idx"]: {c["fact"]: c for c in item.get("gold_fact_classes", [])}
            for item in category_data["models"].get(model_name, {}).get("per_item", [])
        }
        for item in model_grounding["per_item"]:
            idx = item["idx"]
            cat_lookup = category_items.get(idx, {})
            for g in item["groundings"]:
                fact = g["fact"]
                transcript_label = g["classification"]
                cat_entry = cat_lookup.get(fact)
                total += 1
                if cat_entry is None:
                    unmatched += 1
                    continue
                category = cat_entry.get("category", "other")
                if category == "uncategorized_by_creola":
                    topic = fact_to_topic.get(fact)
                    label = f"uncategorized_by_creola: {topic}" if topic else "uncategorized_by_creola: (no topic)"
                else:
                    label = category
                cross_tab[label][transcript_label] += 1

    rows = []
    for label, counts in cross_tab.items():
        in_t = counts.get("in_transcript_not_in_ai_note", 0)
        out_t = counts.get("not_in_transcript", 0)
        n = in_t + out_t
        rows.append({
            "label": label,
            "total": n,
            "in_transcript_not_in_ai_note": {"count": in_t, "pct": round(in_t / n, 3) if n else 0},
            "not_in_transcript": {"count": out_t, "pct": round(out_t / n, 3) if n else 0},
        })
    rows.sort(key=lambda r: -r["total"])

    mood_row = next((r for r in rows if r["label"] == "uncategorized_by_creola: Session Participation"), None)

    output = {
        "metadata": {
            "script": "missed_facts_category_x_grounding.py",
            "method": "Pure join of 16_missed_fact_grounding.json (transcript-presence) with "
                      "14_creola_category_gaps.json (content category) and "
                      "15_uncategorized_topic_model.json (topic, for the uncategorized_by_creola "
                      "subset) by (model, idx, fact text). No new LLM calls.",
            "total_facts_joined": total,
            "unmatched_facts": unmatched,
        },
        "category_x_grounding": rows,
        "mood_affect_row": mood_row,
    }
    out_path = Path(args.output_dir) / "17_missed_facts_category_x_grounding.json"
    save_json(output, out_path)
    print(f"Saved -> {out_path}")
    print(f"\nJoined {total} facts ({unmatched} unmatched, {unmatched/total:.1%})" if total else "no facts")
    print("\nCategory x transcript-presence (sorted by volume):")
    for r in rows:
        print(f"  {r['total']:4d}  {r['label']:<55s} "
              f"in_transcript={r['in_transcript_not_in_ai_note']['pct']:.0%}  "
              f"not_in_transcript={r['not_in_transcript']['pct']:.0%}")
    if mood_row:
        print(f"\nMood/affect (Session Participation): {mood_row['total']} facts, "
              f"{mood_row['not_in_transcript']['pct']:.0%} not_in_transcript")


if __name__ == "__main__":
    main()
