#!/usr/bin/env python3
"""
plot_top_concepts.py

Plot the top-N most commonly used UMLS concepts from a JSON file
that already contains CUI -> concept name mappings.

Example input fields:
    "pred_cui_names": {
        "C0030193": "Pain",
        "C0011849": "Diabetes Mellitus"
    }

or:
    "pred_cui_names": {
        "C0030193": ["Pain", "Pain, unspecified"],
        "C0011849": ["Diabetes Mellitus"]
    }

Example usage:
    python metamap_concepts_histogram.py \
        --input metrics_test/metric_reports/complete_concept_data/gpt5-pro_soap_preds_cui_coverage.json \
        --name_keys pred_cui_names \
        --top_n 20 \
        --output metrics_test/metric_reports/complete_concept_data/gpt5-pro_soap_preds_top20_pred_concepts.png

You can also count gold/common concepts:
    python metamap_concepts_histogram.py \
        --input metrics_test/metric_reports/complete_concept_data/gpt5-pro_soap_preds_cui_coverage.json \
        --name_keys gold_cui_names common_cui_names \
        --top_n 20 \
        --output metrics_test/metric_reports/complete_concept_data/gpt5-pro_soap_preds_top20_gold_common_concepts.png

    python metamap_concepts_histogram.py \
        --input metrics_test/metric_reports/complete_concept_data/healthscribe_summaries_cui_coverage.json \
        --name_keys gold_cui_names pred_cui_names \
        --top_n 20 \
        --output metrics_test/metric_reports/complete_concept_data/healthscribe_summaries_top20_gold_pred_concepts.png

"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, List, Optional

import matplotlib.pyplot as plt


DEFAULT_NAME_KEYS = [
    "cui_names",
    "pred_cui_names",
    "gold_cui_names",
    "common_cui_names",
]


def load_json(path: str) -> Any:
    """Load JSON data from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_name(name: str) -> str:
    """
    Normalize concept names before counting.

    This keeps the displayed name readable while avoiding trivial duplicates
    due to extra whitespace.
    """
    return " ".join(name.strip().split())


def choose_display_name(names: Any) -> Optional[str]:
    """
    Convert a value from a CUI-name mapping into one displayable concept name.

    Supported formats:
    - "Pain"
    - ["Pain", "Pain, unspecified"]
    - null

    If multiple names are present for one CUI, this function chooses the first
    non-empty name. This avoids counting the same CUI multiple times because
    it has several synonyms.
    """
    if names is None:
        return None

    if isinstance(names, str):
        cleaned = normalize_name(names)
        return cleaned if cleaned else None

    if isinstance(names, list):
        for name in names:
            if isinstance(name, str):
                cleaned = normalize_name(name)
                if cleaned:
                    return cleaned

    return None


def collect_concept_names_from_obj(
    obj: Any,
    name_keys: set[str],
    counter: Counter,
    unique_per_item: bool = False,
) -> None:
    """
    Recursively collect concept names from a nested JSON object.

    Parameters
    ----------
    obj:
        Any nested JSON structure.

    name_keys:
        Keys that contain CUI -> concept-name mappings.

    counter:
        Counter updated in place.

    unique_per_item:
        If False, count every occurrence found.
        If True, this function should be called at the item level so each
        concept is counted at most once per item.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in name_keys:
                names_found = extract_names_from_name_mapping(value)

                if unique_per_item:
                    names_found = list(set(names_found))

                counter.update(names_found)
            else:
                collect_concept_names_from_obj(
                    value,
                    name_keys=name_keys,
                    counter=counter,
                    unique_per_item=False,
                )

    elif isinstance(obj, list):
        for item in obj:
            collect_concept_names_from_obj(
                item,
                name_keys=name_keys,
                counter=counter,
                unique_per_item=False,
            )


def extract_names_from_name_mapping(value: Any) -> List[str]:
    """
    Extract displayable concept names from a CUI-name mapping.

    Expected input:
        {
            "C0030193": "Pain",
            "C0011849": ["Diabetes Mellitus"]
        }

    Returns:
        ["Pain", "Diabetes Mellitus"]
    """
    concept_names = []

    if not isinstance(value, dict):
        return concept_names

    for cui, names in value.items():
        display_name = choose_display_name(names)

        if display_name is not None:
            concept_names.append(display_name)

    return concept_names


def count_concepts(
    data: Any,
    name_keys: Iterable[str],
    unique_per_item: bool = False,
) -> Counter:
    """
    Count concept-name frequencies across the JSON.

    Parameters
    ----------
    data:
        Loaded JSON data.

    name_keys:
        JSON keys to read concept names from.

    unique_per_item:
        If True, count each concept at most once per top-level item.
        This is useful if each item is one clinical note or one example.

    Returns
    -------
    Counter
        Mapping from concept name -> frequency.
    """
    name_keys = set(name_keys)
    counter = Counter()

    if unique_per_item and isinstance(data, list):
        for item in data:
            item_counter = Counter()
            collect_concept_names_from_obj(
                item,
                name_keys=name_keys,
                counter=item_counter,
                unique_per_item=False,
            )
            counter.update(set(item_counter.keys()))
    else:
        collect_concept_names_from_obj(
            data,
            name_keys=name_keys,
            counter=counter,
            unique_per_item=False,
        )

    return counter


def plot_top_concepts(
    concept_counter: Counter,
    top_n: int,
    output_path: str,
    title: str,
) -> None:
    """
    Plot a horizontal bar chart for the top-N concepts.
    """
    top_items = concept_counter.most_common(top_n)

    if not top_items:
        raise ValueError("No concept names found to plot.")

    concepts = [item[0] for item in top_items]
    counts = [item[1] for item in top_items]

    # Reverse so the most common concept appears at the top.
    concepts = concepts[::-1]
    counts = counts[::-1]

    # Increase height for readability with long concept names.
    fig_height = max(6, 0.4 * len(concepts))
    plt.figure(figsize=(12, fig_height))

    plt.barh(concepts, counts)
    plt.xlabel("Count")
    plt.ylabel("Concept name")
    plt.title(title)

    # Add count labels beside bars.
    for i, count in enumerate(counts):
        plt.text(count, i, f" {count}", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_top_concepts_csv(
    concept_counter: Counter,
    top_n: int,
    output_csv_path: str,
) -> None:
    """
    Save top-N concept counts to CSV.
    """
    with open(output_csv_path, "w", encoding="utf-8") as f:
        f.write("rank,concept_name,count\n")
        for rank, (concept_name, count) in enumerate(
            concept_counter.most_common(top_n),
            start=1,
        ):
            safe_name = concept_name.replace('"', '""')
            f.write(f'{rank},"{safe_name}",{count}\n')


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot top-N commonly used UMLS concepts by concept name."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON file with CUI-name mappings.",
    )

    parser.add_argument(
        "--name_keys",
        nargs="*",
        default=DEFAULT_NAME_KEYS,
        help=(
            "JSON keys containing CUI-to-name mappings. "
            "Examples: pred_cui_names gold_cui_names common_cui_names"
        ),
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top concepts to plot.",
    )

    parser.add_argument(
        "--output",
        default="top_concepts.png",
        help="Output plot path, e.g. top20_concepts.png.",
    )

    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional CSV file to save top concept counts.",
    )

    parser.add_argument(
        "--unique_per_item",
        action="store_true",
        help=(
            "Count each concept at most once per top-level JSON item. "
            "Useful if each item is one note/example."
        ),
    )

    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    data = load_json(str(input_path))

    concept_counter = count_concepts(
        data=data,
        name_keys=args.name_keys,
        unique_per_item=args.unique_per_item,
    )

    if not concept_counter:
        raise ValueError(
            "No concepts found. Check that --name_keys match your JSON fields. "
            f"Tried keys: {args.name_keys}"
        )

    title = args.title
    if title is None:
        title = f"Top {args.top_n} Most Common Concepts"

    plot_top_concepts(
        concept_counter=concept_counter,
        top_n=args.top_n,
        output_path=args.output,
        title=title,
    )

    print(f"Saved histogram to: {args.output}")

    print("\nTop concepts:")
    for rank, (concept_name, count) in enumerate(
        concept_counter.most_common(args.top_n),
        start=1,
    ):
        print(f"{rank:02d}. {concept_name}: {count}")

    if args.output_csv:
        save_top_concepts_csv(
            concept_counter=concept_counter,
            top_n=args.top_n,
            output_csv_path=args.output_csv,
        )
        print(f"\nSaved CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()