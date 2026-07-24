#!/usr/bin/env python3
"""
Calculate text metrics for every JSON/JSONL file in an input folder and save
all results into one output JSON file grouped by filename.

Metrics:
1. Jaccard similarity index using NLTK tokenization
2. Word-level information entropy using NLTK tokenization
3. Gunning Fog using textstat
4. Flesch-Kincaid Grade Level using textstat
5. Flesch Reading Ease using textstat

Stopwords are removed for:
- Jaccard similarity
- Word-level information entropy

Example:

python quantitative_metrics.py \
  --input-folder metrics_test/soap_prompts \
  --reference-key gold_note \
  --prediction-key pred_note \
  --output-path metrics_test/quantitative_metrics/all_text_metrics.json
"""

import argparse
import json
import math
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat


def ensure_nltk_resources() -> None:
    """
    Download required NLTK resources if missing.

    Newer NLTK versions may require punkt_tab in addition to punkt.
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]

    for resource_path, package_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package_name)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON list, JSONL file, or dict with an 'items' list."""
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]

    if isinstance(data, dict) and isinstance(data.get("metrics"), list):
        return data["metrics"]

    raise ValueError(
        f"Unsupported JSON structure in {path}. "
        "Expected JSON list, JSONL, dict with 'items', or dict with 'metrics'."
    )


def preprocess_tokens(
    text: str,
    remove_stopwords: bool = True,
    lowercase: bool = True,
) -> List[str]:
    """
    Tokenize text using NLTK, remove punctuation/non-word tokens, and optionally remove stopwords.

    Used for Jaccard similarity and word-level entropy.
    """
    if not isinstance(text, str):
        text = ""

    if lowercase:
        text = text.lower()

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english")) if remove_stopwords else set()
    punctuation = set(string.punctuation)

    cleaned_tokens = []
    for token in tokens:
        token = token.strip()

        # Keep alphabetic tokens only.
        # This removes punctuation, numbers, and mixed symbols.
        if not token.isalpha():
            continue

        if token in punctuation:
            continue

        if remove_stopwords and token in stop_words:
            continue

        cleaned_tokens.append(token)
    return cleaned_tokens


def jaccard_similarity(
    text_a: str,
    text_b: str,
    remove_stopwords: bool = True,
) -> float:
    """
    Jaccard similarity over unique non-stopword word sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    """
    tokens_a = set(preprocess_tokens(text_a, remove_stopwords=remove_stopwords))
    tokens_b = set(preprocess_tokens(text_b, remove_stopwords=remove_stopwords))

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))

def compute_pair_metrics(
    reference: str,
    prediction: str,
    remove_stopwords: bool = True,
) -> Dict[str, Any]:
    """
    Compute metrics for one reference-prediction pair.

    Jaccard compares reference vs prediction after stopword removal.
    """

    pred_tokens_no_stop = preprocess_tokens(
        prediction,
        remove_stopwords=remove_stopwords,
    )
    ref_tokens_no_stop = preprocess_tokens(
        reference,
        remove_stopwords=remove_stopwords,
    )

    return {
        "jaccard_similarity_no_stopwords": jaccard_similarity(
            reference,
            prediction,
            remove_stopwords=remove_stopwords,
        ),

        "prediction_num_tokens_no_stopwords": len(pred_tokens_no_stop),
        "reference_num_tokens_no_stopwords": len(ref_tokens_no_stop),
    }


def summarize_metric_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Calculate mean, min, and max for numeric metric columns."""
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )

    summary = {}

    for key in numeric_keys:
        values = [
            row[key]
            for row in rows
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        ]

        if not values:
            continue

        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    return summary


def compute_metrics_for_file(
    file_path: Path,
    reference_key: str,
    prediction_key: str,
    id_key: str,
    remove_stopwords: bool,
) -> Dict[str, Any]:
    """
    Compute all metrics for one JSON/JSONL prediction file.

    Output is grouped by filename.
    """
    rows = load_json_or_jsonl(file_path)
    metric_rows = []

    skipped_rows = []

    for i, row in enumerate(rows):
        reference = row.get(reference_key, "")
        prediction = row.get(prediction_key, "")

        if not isinstance(reference, str):
            reference = ""

        if not isinstance(prediction, str):
            prediction = ""

        if not reference.strip() and not prediction.strip():
            skipped_rows.append(
                {
                    id_key: row.get(id_key, i),
                    "reason": "Both reference and prediction are empty.",
                }
            )
            continue

        metrics = compute_pair_metrics(
            reference=reference,
            prediction=prediction,
            remove_stopwords=remove_stopwords,
        )

        metric_rows.append(
            {
                id_key: row.get(id_key, i),
                **metrics,
            }
        )

    summary = summarize_metric_rows(metric_rows)

    return {
        "filename": file_path.name,
        "file_path": str(file_path),
        "num_rows_loaded": len(rows),
        "num_rows_evaluated": len(metric_rows),
        "num_rows_skipped": len(skipped_rows),
        "skipped_rows": skipped_rows,
        "metrics": metric_rows,
        "summary": summary,
    }


def find_input_files(input_folder: Path, recursive: bool = False) -> List[Path]:
    """Find JSON and JSONL files in an input folder."""
    if recursive:
        json_files = list(input_folder.rglob("*.json"))
        jsonl_files = list(input_folder.rglob("*.jsonl"))
    else:
        json_files = list(input_folder.glob("*.json"))
        jsonl_files = list(input_folder.glob("*.jsonl"))

    return sorted(json_files + jsonl_files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Jaccard similarity"
            "in a folder. Save all results in one JSON file grouped by filename."
        )
    )

    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Folder containing JSON/JSONL files.",
    )

    parser.add_argument(
        "--reference-key",
        type=str,
        default="gold_note",
        help="Field containing the reference/gold note.",
    )

    parser.add_argument(
        "--prediction-key",
        type=str,
        default="pred_note",
        help="Field containing the generated/predicted note.",
    )

    parser.add_argument(
        "--id-key",
        type=str,
        default="idx",
        help="ID field to preserve in output.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Single output JSON file for all metrics.",
    )

    parser.add_argument(
        "--keep-stopwords",
        action="store_true",
        help=(
            "Do not remove stopwords for Jaccard similarity and entropy. "
            "By default, stopwords are removed."
        ),
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for JSON/JSONL files.",
    )

    args = parser.parse_args()

    # ensure_nltk_resources()

    input_folder = Path(args.input_folder)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_folder.exists() or not input_folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist or is not a folder: {input_folder}")

    input_files = find_input_files(
        input_folder=input_folder,
        recursive=args.recursive,
    )

    if not input_files:
        raise FileNotFoundError(f"No .json or .jsonl files found in {input_folder}")

    remove_stopwords = not args.keep_stopwords

    all_results = {}
    file_summaries = {}

    for file_path in input_files:
        print(f"Processing: {file_path}")

        try:
            file_result = compute_metrics_for_file(
                file_path=file_path,
                reference_key=args.reference_key,
                prediction_key=args.prediction_key,
                id_key=args.id_key,
                remove_stopwords=remove_stopwords,
            )

            all_results[file_path.name] = file_result

            file_summaries[file_path.name] = {
                "num_rows_loaded": file_result["num_rows_loaded"],
                "num_rows_evaluated": file_result["num_rows_evaluated"],
                "num_rows_skipped": file_result["num_rows_skipped"],
                "summary": file_result["summary"],
            }

        except Exception as e:
            all_results[file_path.name] = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "error": str(e),
            }

            file_summaries[file_path.name] = {
                "error": str(e),
            }

            print(f"  ERROR: {e}")

    output = {
        "input_folder": str(input_folder),
        "num_files": len(input_files),
        "remove_stopwords_for_jaccard_and_entropy": remove_stopwords,
        "reference_key": args.reference_key,
        "prediction_key": args.prediction_key,
        "id_key": args.id_key,
        "file_summaries": file_summaries,
        "results_by_filename": all_results,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all metrics to: {output_path}")

    print("\nPer-file summary:")
    for filename, summary in file_summaries.items():
        if "error" in summary:
            print(f"{filename}: ERROR - {summary['error']}")
            continue

        print(
            f"{filename}: "
            f"loaded={summary['num_rows_loaded']}, "
            f"evaluated={summary['num_rows_evaluated']}, "
            f"skipped={summary['num_rows_skipped']}"
        )


if __name__ == "__main__":
    main()