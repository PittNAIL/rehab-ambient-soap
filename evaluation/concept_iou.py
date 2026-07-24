#!/usr/bin/env python3
"""
conceptual_iou_folder.py

Compute Conceptual Intersection over Union (C-IoU) for all JSON/JSONL files
inside a given folder.

Expected per-item format:
[
    {
        "pred_cuis": ["C0030193", "C0011849"],
        "gold_cuis": ["C0030193", "C0020538"]
    },
    ...
]

C-IoU = |pred_cuis ∩ gold_cuis| / |pred_cuis ∪ gold_cuis|

Example:
python concept_iou.py \
    --input_dir ./metrics_test/metric_reports \
    --pred_key pred_cuis \
    --gold_key gold_cuis \
    --output_dir ./metrics_test/concept_iou/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.stdout.reconfigure(encoding="utf-8")


def load_json_or_jsonl(path: Path) -> Any:
    """
    Load .json or .jsonl file.

    .json expected:
        [
            {"pred_cuis": [...], "gold_cuis": [...]},
            ...
        ]

    .jsonl expected:
        {"pred_cuis": [...], "gold_cuis": [...]}
        {"pred_cuis": [...], "gold_cuis": [...]}
    """
    if path.suffix.lower() == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_cui_list(value: Any) -> Set[str]:
    """
    Convert different possible CUI formats into a clean set of CUI strings.

    Supported:
    - ["C0030193", "C0011849"]
    - {"C0030193": "Pain", "C0011849": "Diabetes Mellitus"}
    - "C0030193"
    - None
    """
    if value is None:
        return set()

    if isinstance(value, list):
        return {
            str(x).strip()
            for x in value
            if x is not None and str(x).strip()
        }

    if isinstance(value, dict):
        # Useful if the field is something like:
        # {"C0030193": "Pain", "C0011849": "Diabetes Mellitus"}
        return {
            str(k).strip()
            for k in value.keys()
            if k is not None and str(k).strip()
        }

    if isinstance(value, str):
        value = value.strip()
        return {value} if value else set()

    return set()


def conceptual_iou(pred_cuis: Set[str], gold_cuis: Set[str]) -> Dict[str, Any]:
    """
    Compute conceptual IoU for one item.

    Returns C-IoU plus intersection/union statistics.
    """
    intersection = pred_cuis & gold_cuis
    union = pred_cuis | gold_cuis

    if not union:
        # If both prediction and gold have no CUIs, define C-IoU as 0.0.
        # You can change this to 1.0 if you want empty-empty to count as perfect.
        ciou = 0.0
    else:
        ciou = len(intersection) / len(union)

    return {
        "conceptual_iou": ciou,
        "num_pred_cuis": len(pred_cuis),
        "num_gold_cuis": len(gold_cuis),
        "num_intersection_cuis": len(intersection),
        "num_union_cuis": len(union),
        "intersection_cuis": sorted(intersection),
        "union_cuis": sorted(union),
    }


def get_items_from_loaded_json(data: Any, item_key: str | None = None) -> List[Dict[str, Any]]:
    """
    Convert loaded JSON into a list of item dictionaries.

    Supports:
    1. Top-level list:
        [
            {"pred_cuis": [...], "gold_cuis": [...]},
            ...
        ]

    2. Top-level dict containing a list under item_key:
        {
            "per_item": [
                {"pred_cuis": [...], "gold_cuis": [...]}
            ]
        }

    If item_key is not given and data is a dict, this function tries common keys:
    - "per_item"
    - "items"
    - "examples"
    - "results"
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        candidate_keys = []

        if item_key is not None:
            candidate_keys.append(item_key)

        candidate_keys.extend(["per_item", "items", "examples", "results"])

        for key in candidate_keys:
            if key in data and isinstance(data[key], list):
                return [x for x in data[key] if isinstance(x, dict)]

        # If the dict itself looks like one example, treat it as a single item.
        return [data]

    raise ValueError("Unsupported JSON structure. Expected list or dict.")


def evaluate_items(
    items: List[Dict[str, Any]],
    pred_key: str,
    gold_key: str,
    skip_empty_gold: bool = False,
    skip_empty_pred: bool = False,
) -> Dict[str, Any]:
    """
    Compute C-IoU for all items in one file.
    """
    per_item_results = []

    for idx, item in enumerate(items):
        pred_cuis = normalize_cui_list(item.get(pred_key))
        gold_cuis = normalize_cui_list(item.get(gold_key))

        if skip_empty_gold and not gold_cuis:
            continue

        if skip_empty_pred and not pred_cuis:
            continue

        result = conceptual_iou(pred_cuis, gold_cuis)

        per_item_results.append(
            {
                "idx": idx,
                **result,
            }
        )

    ciou_values = [r["conceptual_iou"] for r in per_item_results]

    if ciou_values:
        mean_ciou = sum(ciou_values) / len(ciou_values)
    else:
        mean_ciou = 0.0

    # Micro C-IoU pools all item-level intersections and unions by count.
    total_intersection = sum(r["num_intersection_cuis"] for r in per_item_results)
    total_union = sum(r["num_union_cuis"] for r in per_item_results)

    micro_ciou = total_intersection / total_union if total_union else 0.0

    return {
        "num_evaluated": len(per_item_results),
        "mean_conceptual_iou": mean_ciou, # this is important
        "micro_conceptual_iou": micro_ciou,
        "total_intersection_cuis": total_intersection,
        "total_union_cuis": total_union,
        "per_item": per_item_results,
    }


def save_json(obj: Any, path: Path) -> None:
    """Save object as pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save one-row-per-file summary CSV."""
    fieldnames = [
        "file",
        "num_evaluated",
        "mean_conceptual_iou",
        "micro_conceptual_iou",
        "total_intersection_cuis",
        "total_union_cuis",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Conceptual IoU for all JSON/JSONL files in a folder."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Folder containing .json or .jsonl files.",
    )

    parser.add_argument(
        "--pred_key",
        default="pred_cuis",
        help="JSON key containing predicted/generated CUIs.",
    )

    parser.add_argument(
        "--gold_key",
        default="gold_cuis",
        help="JSON key containing gold/reference CUIs.",
    )

    parser.add_argument(
        "--item_key",
        default=None,
        help=(
            "Optional key containing the per-item list if JSON is a dict. "
            "Example: --item_key per_item"
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional folder to save detailed per-file C-IoU results.",
    )

    parser.add_argument(
        "--skip_empty_gold",
        action="store_true",
        help="Skip items where the gold/reference CUI set is empty.",
    )

    parser.add_argument(
        "--skip_empty_pred",
        action="store_true",
        help="Skip items where the predicted/generated CUI set is empty.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input_dir must be a directory: {input_dir}")

    json_files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    
    json_files = [file for file in json_files if not file.name.endswith('summary_cui_coverage_by_file.json')]

    if not json_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {input_dir}")

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for json_path in json_files:
        print("\n" + "=" * 80)
        print(f"Evaluating: {json_path.name}")
        print("=" * 80)

        try:
            data = load_json_or_jsonl(json_path)
            items = get_items_from_loaded_json(data, item_key=args.item_key)

            results = evaluate_items(
                items=items,
                pred_key=args.pred_key,
                gold_key=args.gold_key,
                skip_empty_gold=args.skip_empty_gold,
                skip_empty_pred=args.skip_empty_pred,
            )

            summary = {
                "file": json_path.name,
                "num_evaluated": results["num_evaluated"],
                "mean_conceptual_iou": results["mean_conceptual_iou"],
                "micro_conceptual_iou": results["micro_conceptual_iou"],
                "total_intersection_cuis": results["total_intersection_cuis"],
                "total_union_cuis": results["total_union_cuis"],
            }

            summary_rows.append(summary)

            print(f"Number evaluated: {summary['num_evaluated']}")
            print(f"Mean C-IoU:       {summary['mean_conceptual_iou']:.6f}")
            print(f"Micro C-IoU:      {summary['micro_conceptual_iou']:.6f}")
            print(f"Total |∩|:        {summary['total_intersection_cuis']}")
            print(f"Total |∪|:        {summary['total_union_cuis']}")

            if output_dir is not None:
                output_path = output_dir / f"{json_path.stem}_conceptual_iou.json"
                save_json(results, output_path)
                print(f"Saved detailed results to: {output_path}")

        except Exception as e:
            print(f"Failed to evaluate {json_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Summary across files")
    print("=" * 80)

    for row in summary_rows:
        print(
            f"{row['file']}: "
            f"n={row['num_evaluated']}, "
            f"mean_cIoU={row['mean_conceptual_iou']:.6f}, "
            f"micro_cIoU={row['micro_conceptual_iou']:.6f}"
        )

    if output_dir is not None:
        summary_json_path = output_dir / "summary_conceptual_iou_by_file.json"
        summary_csv_path = output_dir / "summary_conceptual_iou_by_file.csv"

        save_json(summary_rows, summary_json_path)
        save_summary_csv(summary_rows, summary_csv_path)

        print(f"\nSaved summary JSON to: {summary_json_path}")
        print(f"Saved summary CSV to:  {summary_csv_path}")


if __name__ == "__main__":
    main()