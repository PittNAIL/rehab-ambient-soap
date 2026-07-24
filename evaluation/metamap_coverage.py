"""
cui_coverage_metamaplite.py

Compute clinical concept coverage between generated/predicted notes and gold notes
using UMLS CUIs extracted with pymetamaplite.

Expected item format:
[
    {
        "pred_note": "... generated clinical note ...",
        "gold_note": "... reference clinical note ..."
    },
    ...
]

Main metrics:
- recall_coverage:   fraction of gold CUIs covered by prediction
- precision_coverage: fraction of predicted CUIs also present in gold
- f1_coverage: harmonic mean of CUI precision and CUI recall

Notes:
- Matching by CUI is preferred over matching by raw surface strings because
  different strings may refer to the same UMLS concept.
- You must set IVF_DIR to your local MetaMapLite IVF index path.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import nltk
import numpy as np

from metamaplite import MetaMapLite, result_utils
from pathlib import Path


# MetaMapLite expects token objects with these fields.
Token = namedtuple("Token", ["text", "tag_", "idx", "start"])


@dataclass
class CUICoverageResult:
    """Coverage scores for one prediction/gold pair."""

    recall_coverage: float
    precision_coverage: float
    f1_coverage: float
    pred_cuis: Set[str]
    gold_cuis: Set[str]
    common_cuis: Set[str]


def ensure_nltk_resources() -> None:
    """
    Ensure that NLTK POS tagger resources are available.

    Depending on your NLTK version, you may need either:
    - averaged_perceptron_tagger
    - averaged_perceptron_tagger_eng

    This function tries to download both quietly. If you already have them,
    NLTK will skip re-downloading.
    """
    try:
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except Exception:
        # Do not hard-fail here; nltk.pos_tag will raise a clearer error later
        # if the resource is truly unavailable.
        pass


def add_spans(postokenlist: Sequence[Tuple[str, str]]) -> List[Token]:
    """
    Convert NLTK POS-tagged tokens into the token format expected by MetaMapLite.

    Parameters
    ----------
    postokenlist:
        List of tuples: [(token_text, pos_tag), ...]

    Returns
    -------
    List[Token]
        Tokens with text, POS tag, token index, and character start offset.
    """
    tokenlist: List[Token] = []
    start = 0

    for idx, (token_text, pos_tag) in enumerate(postokenlist):
        tokenlist.append(
            Token(
                text=token_text,
                tag_=pos_tag,
                idx=idx,
                start=start,
            )
        )
        # This assumes whitespace-separated tokenization.
        # Good enough for many clinical-note experiments, but you can replace
        # this with a tokenizer that preserves exact offsets if needed.
        start += len(token_text) + 1

    return tokenlist


def make_metamaplite(
    ivf_dir: str,
    use_sources: Optional[List[str]] = None,
    use_semtypes: Optional[List[str]] = None,
    stopwords: Optional[List[str]] = None,
    excluded_terms: Optional[List[str]] = None,
    use_cache: bool = True,
) -> MetaMapLite:
    """
    Initialize a MetaMapLite instance.

    Parameters
    ----------
    ivf_dir:
        Path to your MetaMapLite IVF index directory, for example:
        /path/to/public_mm_lite/data/ivf/2020AA/USAbase

    use_sources:
        Optional UMLS vocabulary filters, e.g. ["SNOMEDCT_US", "MSH"].
        Empty list means use all available sources.

    use_semtypes:
        Optional semantic type filters, e.g. ["dsyn", "sosy", "phsu"].
        Empty list means use all available semantic types.

    stopwords:
        Optional stopword list.

    excluded_terms:
        Optional excluded term list. Format expected by pymetamaplite:
        ["C0004002:got", "C0006104:bra", ...]

    use_cache:
        Cache lookups for speed. Useful for large datasets.

    Returns
    -------
    MetaMapLite
        Initialized MetaMapLite object.
    """
    # POS tags recommended in pymetamaplite README.
    postags = {
        "CD",
        "FW",
        "RB",
        "IN",
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "JJ",
        "JJR",
        "JJS",
        "LS",
    }

    return MetaMapLite(
        ivf_dir,
        use_sources or [],
        use_semtypes or [],
        postags,
        stopwords or [],
        excluded_terms or [],
        use_cache=use_cache,
    )


def extract_cuis(
    text: str,
    metamap_lite: MetaMapLite,
    lowercase_text: bool = False,
    return_debug: bool = False,
) -> Set[str] | Tuple[Set[str], List[Dict[str, Any]]]:
    """
    Extract unique UMLS CUIs from text using pymetamaplite.

    Parameters
    ----------
    text:
        Input clinical note text.

    metamap_lite:
        Initialized MetaMapLite object.

    lowercase_text:
        Whether to lowercase text before extraction.
        Usually False is safer because some clinical abbreviations are case-sensitive.

    return_debug:
        If True, also return a list of matched terms/postings for inspection.

    Returns
    -------
    Set[str]
        Unique CUIs found in the text.

    Or, if return_debug=True:
    Tuple[Set[str], List[Dict[str, Any]]]
        CUIs and debug information.
    """
    if not text or not text.strip():
        return (set(), []) if return_debug else set()

    if lowercase_text:
        text = text.lower()

    # Simple whitespace tokenization.
    # You can replace this with nltk.word_tokenize or a clinical tokenizer,
    # but make sure character offsets remain consistent.
    text_tokens = text.split()
    pos_tokens = nltk.pos_tag(text_tokens)
    tokenlist = add_spans(pos_tokens)

    # Get raw entity matches.
    raw_matches = metamap_lite.get_entities(tokenlist, span_info=True)

    # Convert postings to structured objects with fields such as:
    # cui, sui, str, src, termtype, semtypeset.
    matches = result_utils.add_semantic_types(metamap_lite, raw_matches)

    cuis: Set[str] = set()
    debug_rows: List[Dict[str, Any]] = []

    for term in matches:
        term_text = getattr(term, "text", None)
        term_start = getattr(term, "start", None)
        term_end = getattr(term, "end", None)

        for post in getattr(term, "postings", []):
            cui = getattr(post, "cui", None)

            if not cui:
                continue

            cuis.add(cui)

            if return_debug:
                debug_rows.append(
                    {
                        "matched_text": term_text,
                        "start": term_start,
                        "end": term_end,
                        "cui": cui,
                        "umls_string": getattr(post, "str", None),
                        "source": getattr(post, "src", None),
                        "term_type": getattr(post, "termtype", None),
                        "semantic_types": getattr(post, "semtypeset", None),
                    }
                )

    return (cuis, debug_rows) if return_debug else cuis


def calculate_cui_coverage(pred_cuis: Set[str], gold_cuis: Set[str]) -> CUICoverageResult:
    """
    Calculate CUI-level coverage between predicted and gold concepts.

    Parameters
    ----------
    pred_cuis:
        CUIs extracted from generated/predicted note.

    gold_cuis:
        CUIs extracted from reference/gold note.

    Returns
    -------
    CUICoverageResult
        Recall, precision, F1, and CUI sets.
    """
    common_cuis = pred_cuis & gold_cuis

    # Recall-style coverage:
    # "How much of the gold clinical content did the prediction cover?"
    recall_coverage = len(common_cuis) / len(gold_cuis) if gold_cuis else 0.0

    # Precision-style coverage:
    # "How much of the predicted clinical content is supported by gold?"
    precision_coverage = len(common_cuis) / len(pred_cuis) if pred_cuis else 0.0

    if precision_coverage + recall_coverage > 0:
        f1_coverage = (
            2 * precision_coverage * recall_coverage
        ) / (precision_coverage + recall_coverage)
    else:
        f1_coverage = 0.0

    return CUICoverageResult(
        recall_coverage=recall_coverage,
        precision_coverage=precision_coverage,
        f1_coverage=f1_coverage,
        pred_cuis=pred_cuis,
        gold_cuis=gold_cuis,
        common_cuis=common_cuis,
    )


def evaluate_items(
    items: Iterable[Dict[str, Any]],
    metamap_lite: MetaMapLite,
    pred_key: str = "pred_note",
    gold_key: str = "gold_note",
    skip_empty_pred: bool = True,
    skip_empty_gold: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate CUI coverage over a dataset.

    Parameters
    ----------
    items:
        Iterable of dicts containing predicted and gold notes.

    metamap_lite:
        Initialized MetaMapLite object.

    pred_key:
        Key for predicted/generated note.

    gold_key:
        Key for gold/reference note.

    skip_empty_pred:
        If True, skip examples with empty predicted note.

    skip_empty_gold:
        If True, skip examples with empty gold note.

    Returns
    -------
    Dict[str, Any]
        Aggregate metrics and per-example results.
    """
    per_item_results: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        pred_note = item.get(pred_key, "") or ""
        gold_note = item.get(gold_key, "") or ""
        

        if skip_empty_pred and not pred_note.strip():
            continue

        if skip_empty_gold and not gold_note.strip():
            continue

        pred_cuis = extract_cuis(pred_note, metamap_lite)
        gold_cuis = extract_cuis(gold_note, metamap_lite)

        result = calculate_cui_coverage(pred_cuis, gold_cuis)

        per_item_results.append(
            {
                "idx": idx,
                "recall_coverage": result.recall_coverage,
                "precision_coverage": result.precision_coverage,
                "f1_coverage": result.f1_coverage,
                "num_pred_cuis": len(result.pred_cuis),
                "num_gold_cuis": len(result.gold_cuis),
                "num_common_cuis": len(result.common_cuis),
                "pred_cuis": sorted(result.pred_cuis),
                "gold_cuis": sorted(result.gold_cuis),
                "common_cuis": sorted(result.common_cuis),
            }
        )

    recall_values = [r["recall_coverage"] for r in per_item_results]
    precision_values = [r["precision_coverage"] for r in per_item_results]
    f1_values = [r["f1_coverage"] for r in per_item_results]

    aggregate = {
        "num_evaluated": len(per_item_results),
        "mean_recall_coverage": float(np.mean(recall_values)) if recall_values else 0.0,
        "mean_precision_coverage": float(np.mean(precision_values)) if precision_values else 0.0,
        "mean_f1_coverage": float(np.mean(f1_values)) if f1_values else 0.0,
        "per_item": per_item_results,
    }

    return aggregate


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from .json or .jsonl.

    .json expected format:
        [
            {"pred_note": "...", "gold_note": "..."},
            ...
        ]

    .jsonl expected format:
        {"pred_note": "...", "gold_note": "..."}
        {"pred_note": "...", "gold_note": "..."}
    """
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON file to contain a list of dicts.")

    return data

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CUI-level coverage using pymetamaplite for each JSON/JSONL file in a folder."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to folder containing .json or .jsonl files.",
    )
    parser.add_argument(
        "--ivf_dir",
        required=True,
        help="Path to MetaMapLite IVF index directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional folder to save one output metrics JSON per input file.",
    )
    parser.add_argument(
        "--pred_key",
        default="pred_note",
        help="Key containing predicted/generated note.",
    )
    parser.add_argument(
        "--gold_key",
        default="gold_note",
        help="Key containing gold/reference note.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=[],
        help=(
            "Optional UMLS source filters, e.g. --sources SNOMEDCT_US MSH. "
            "Default uses all sources."
        ),
    )
    parser.add_argument(
        "--semtypes",
        nargs="*",
        default=[],
        help=(
            "Optional semantic type filters, e.g. --semtypes dsyn sosy phsu. "
            "Default uses all semantic types."
        ),
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable MetaMapLite lookup cache.",
    )

    args = parser.parse_args()

    ensure_nltk_resources()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"--input_dir must be a directory: {input_dir}")

    json_files = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
        ]
    )

    if not json_files:
        raise FileNotFoundError(
            f"No .json or .jsonl files found in input directory: {input_dir}"
        )

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None

    metamap_lite = make_metamaplite(
        ivf_dir=args.ivf_dir,
        use_sources=args.sources,
        use_semtypes=args.semtypes,
        use_cache=not args.no_cache,
    )

    all_file_summaries = []
    existing_files = os.listdir(output_dir)

    for json_path in json_files:
        
        # skip existing outputs
        if json_path.name.replace('.json', '_cui_coverage.json') in existing_files:
            ALREADY_EXISTS = True
            if ALREADY_EXISTS:
                summary_path = output_dir / "summary_cui_coverage_by_file.json"
                with open(summary_path, 'r', encoding='utf-8') as f:
                    all_file_summaries = json.load(f)

            continue
        else:
            ALREADY_EXISTS = False
        
        
        print("\n" + "=" * 80)
        print(f"Evaluating file: {json_path.name}")
        print("=" * 80)

        try:
            items = load_json_or_jsonl(str(json_path))

            results = evaluate_items(
                items=items,
                metamap_lite=metamap_lite,
                pred_key=args.pred_key,
                gold_key=args.gold_key,
            )

            file_summary = {
                "file": json_path.name,
                "num_evaluated": results["num_evaluated"],
                "mean_recall_coverage": results["mean_recall_coverage"],
                "mean_precision_coverage": results["mean_precision_coverage"],
                "mean_f1_coverage": results["mean_f1_coverage"],
            }

            all_file_summaries.append(file_summary)

            print("Number evaluated:", results["num_evaluated"])
            print("Mean CUI recall coverage:", results["mean_recall_coverage"])
            print("Mean CUI precision coverage:", results["mean_precision_coverage"])
            print("Mean CUI F1 coverage:", results["mean_f1_coverage"])

            if output_dir is not None:
                output_path = output_dir / f"{json_path.stem}_cui_coverage.json"
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                print(f"Saved detailed results to: {output_path}")

        except Exception as e:
            print(f"Failed to evaluate {json_path.name}: {e}")

    print("\n" + "=" * 80)
    print("Summary across files")
    print("=" * 80)

    for summary in all_file_summaries:
        print(
            f"{summary['file']}: "
            f"n={summary['num_evaluated']}, "
            f"recall={summary['mean_recall_coverage']:.4f}, "
            f"precision={summary['mean_precision_coverage']:.4f}, "
            f"f1={summary['mean_f1_coverage']:.4f}"
        )

    if output_dir is not None:
        summary_path = output_dir / "summary_cui_coverage_by_file.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_file_summaries, f, indent=2)
        print(f"\nSaved summary to: {summary_path}")

if __name__ == "__main__":
    main()
    
# python metamap_coverage.py \
#   --input_dir metrics_test/soap_prompts \
#   --ivf_dir /path/to/pymetamaplite/data/public_mm_lite/data/ivf/2022AB/USAbase \
#   --output_dir metrics_test/metric_reports
