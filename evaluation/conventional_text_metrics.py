#!/usr/bin/env python3
"""
Conventional text-similarity metrics for AI-generated vs therapist reference notes.

For every model prediction file in an input folder, this computes per-note:

  Lexical / semantic similarity (pred vs reference)
    - ROUGE-1 F1            (n-gram overlap)            [Lin 2004]
    - ROUGE-L F1            (longest common subsequence) [Lin 2004]
    - BERTScore F1          (contextual embedding sim.)  [Zhang et al. 2020]
    - Jaccard similarity    (token sets, stopwords removed)

  Note-characterisation (computed on the generated note; ref token count kept
  for length comparison)
    - prediction / reference token counts (stopwords removed)
    - lexical entropy       (Shannon entropy, base 2, over non-stopword tokens)
    - Flesch Reading Ease
    - Flesch-Kincaid Grade Level
    - Gunning Fog Index

Degenerate generations are excluded exactly as in factehr_pipeline.is_degenerate
(empty / < 50 chars / dialogue leak / repetition loop), so per-model N matches
the "Valid notes" column of Table 1.

Per-model aggregates are reported as mean with a 95% confidence interval from
bias-corrected and accelerated (BCa) bootstrap resampling over encounters
(10,000 resamples), matching the paper's methods.

Example:

  python conventional_text_metrics.py \
    --input-folder metrics_test/soap_prompts \
    --output-path metrics_test/quantitative_metrics/conventional_text_metrics.json
"""

import argparse
import json
import math
import re
import string
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat
from rouge_score import rouge_scorer
from bert_score import score as bertscore
from scipy.stats import bootstrap


# ── NLTK bootstrap ───────────────────────────────────────────────────────────

def ensure_nltk_resources() -> None:
    for resource_path, package in [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


# ── Degeneracy filter (verbatim from factehr_pipeline.is_degenerate) ──────────

_SPEAKER_TURN_RE = re.compile(r"\b(?:Doctor|Client|Patient|SPK_\d+)\s*:")


def is_degenerate(text: str, min_len: int = 50) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return "empty"
    if len(text) < min_len:
        return "too_short"
    if len(_SPEAKER_TURN_RE.findall(text)) >= 3:
        return "dialogue_leak"
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.4:
        return "repetition_loop"
    words = text.split()
    if len(words) >= 15:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) / len(trigrams) < 0.3:
            return "repetition_loop"
    return None


# ── Tokenisation (matches quantitative_metrics.preprocess_tokens) ─────────────

_STOP = None
_PUNCT = set(string.punctuation)


def preprocess_tokens(text: str, remove_stopwords: bool = True,
                      lowercase: bool = True) -> List[str]:
    global _STOP
    if _STOP is None:
        _STOP = set(stopwords.words("english"))
    if not isinstance(text, str):
        text = ""
    if lowercase:
        text = text.lower()
    tokens = word_tokenize(text)
    cleaned = []
    for tok in tokens:
        tok = tok.strip()
        if not tok.isalpha():          # drops punctuation, numbers, mixed symbols
            continue
        if tok in _PUNCT:
            continue
        if remove_stopwords and tok in _STOP:
            continue
        cleaned.append(tok)
    return cleaned


def jaccard_similarity(a: str, b: str, remove_stopwords: bool = True) -> float:
    ta = set(preprocess_tokens(a, remove_stopwords))
    tb = set(preprocess_tokens(b, remove_stopwords))
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def lexical_entropy(text: str, remove_stopwords: bool = True) -> float:
    """Shannon entropy (bits) of the non-stopword token frequency distribution."""
    toks = preprocess_tokens(text, remove_stopwords)
    if not toks:
        return 0.0
    n = len(toks)
    counts = Counter(toks)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


# ── Per-note lexical metrics (everything except BERTScore) ────────────────────

_rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def per_note_lexical(reference: str, prediction: str) -> Dict[str, float]:
    # ROUGE F1 is order-invariant; use conventional (target=ref, prediction=pred).
    r = _rouge.score(reference, prediction)
    pred_tokens = preprocess_tokens(prediction)
    ref_tokens = preprocess_tokens(reference)
    return {
        "rouge1_f1": r["rouge1"].fmeasure,
        "rougeL_f1": r["rougeL"].fmeasure,
        "jaccard_no_stopwords": jaccard_similarity(reference, prediction),
        "prediction_num_tokens": len(pred_tokens),
        "reference_num_tokens": len(ref_tokens),
        "lexical_entropy_bits": lexical_entropy(prediction),
        "flesch_reading_ease": textstat.flesch_reading_ease(prediction),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(prediction),
        "gunning_fog": textstat.gunning_fog(prediction),
    }


# ── BCa bootstrap aggregation ─────────────────────────────────────────────────

def mean_with_bca_ci(values: List[float], n_resamples: int = 10000,
                     confidence_level: float = 0.95,
                     seed: int = 0) -> Dict[str, float]:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
                     dtype=float)
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    mean = float(arr.mean())
    if n < 2 or float(arr.std()) == 0.0:
        # BCa is undefined without variance; report a point interval.
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": n}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = bootstrap((arr,), np.mean, method="BCa",
                            n_resamples=n_resamples,
                            confidence_level=confidence_level,
                            random_state=np.random.default_rng(seed))
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        lo, hi = mean, mean
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": n}


# ── Per-file driver ───────────────────────────────────────────────────────────

def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _skip_reason(pred: str, filter_mode: str) -> Optional[str]:
    """empty-only reproduces metrics.py (Table 1); degenerate uses is_degenerate."""
    if filter_mode == "empty-only":
        return None if pred.strip() else "empty"
    return is_degenerate(pred)


def process_file(path: Path, reference_key: str, prediction_key: str,
                 id_key: str, n_resamples: int,
                 filter_mode: str = "degenerate") -> Dict[str, Any]:
    records = load_records(path)
    valid_notes: List[Dict[str, Any]] = []
    ref_texts: List[str] = []
    pred_texts: List[str] = []
    skipped: List[Dict[str, Any]] = []

    for i, row in enumerate(records):
        ref = row.get(reference_key, "") or ""
        pred = row.get(prediction_key, "") or ""
        reason = _skip_reason(pred, filter_mode)
        if reason:
            skipped.append({id_key: row.get(id_key, i), "reason": reason})
            continue
        valid_notes.append({id_key: row.get(id_key, i),
                            **per_note_lexical(ref, pred)})
        ref_texts.append(ref.strip())
        pred_texts.append(pred.strip())

    # BERTScore F1 in one batch per file (roberta-large, lang='en') to match metrics.py.
    if pred_texts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, _, f1 = bertscore(pred_texts, ref_texts, lang="en", verbose=False)
        for note, val in zip(valid_notes, f1.tolist()):
            note["bertscore_f1"] = float(val)

    metric_keys = [
        "rouge1_f1", "rougeL_f1", "bertscore_f1", "jaccard_no_stopwords",
        "prediction_num_tokens", "reference_num_tokens", "lexical_entropy_bits",
        "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
    ]
    summary = {}
    for key in metric_keys:
        vals = [note[key] for note in valid_notes if key in note]
        summary[key] = mean_with_bca_ci(vals, n_resamples=n_resamples)

    return {
        "filename": path.name,
        "num_records": len(records),
        "num_valid": len(valid_notes),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "per_note": valid_notes,
        "summary": summary,
    }


# ── CLI / reporting ───────────────────────────────────────────────────────────

_DISPLAY = [
    ("rouge1_f1", "ROUGE-1", 3),
    ("rougeL_f1", "ROUGE-L", 3),
    ("bertscore_f1", "BERTScore", 3),
    ("jaccard_no_stopwords", "Jaccard", 3),
    ("prediction_num_tokens", "PredTok", 1),
    ("reference_num_tokens", "RefTok", 1),
    ("lexical_entropy_bits", "Entropy", 2),
    ("flesch_reading_ease", "FRE", 1),
    ("flesch_kincaid_grade", "FKGL", 1),
    ("gunning_fog", "Fog", 1),
]


def print_report(results: Dict[str, Any]) -> None:
    for fname, res in results.items():
        if "error" in res:
            print(f"\n### {fname}: ERROR - {res['error']}")
            continue
        print(f"\n### {fname}  (valid {res['num_valid']}/{res['num_records']})")
        if res["num_valid"] == 0:
            print("  no valid notes; omitted from estimates")
            continue
        for key, label, nd in _DISPLAY:
            s = res["summary"][key]
            print(f"  {label:10s} {s['mean']:.{nd}f}  "
                  f"[{s['ci_low']:.{nd}f}, {s['ci_high']:.{nd}f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-folder", required=True)
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--reference-key", default="gold_note")
    ap.add_argument("--prediction-key", default="pred_note")
    ap.add_argument("--id-key", default="idx")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--filter", choices=["degenerate", "empty-only"],
                    default="degenerate",
                    help="degenerate = is_degenerate() (matches 'Valid notes' N); "
                         "empty-only = drop empty strings only (reproduces metrics.py / Table 1).")
    args = ap.parse_args()

    ensure_nltk_resources()

    in_dir = Path(args.input_folder)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files in {in_dir}")

    results: Dict[str, Any] = {}
    for path in files:
        print(f"Processing {path.name} ...", flush=True)
        try:
            results[path.name] = process_file(
                path, args.reference_key, args.prediction_key,
                args.id_key, args.n_resamples, args.filter)
        except Exception as e:
            results[path.name] = {"filename": path.name, "error": str(e)}
            print(f"  ERROR: {e}")

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "input_folder": str(in_dir),
            "n_resamples": args.n_resamples,
            "reference_key": args.reference_key,
            "prediction_key": args.prediction_key,
            "results_by_filename": results,
        }, f, indent=2, ensure_ascii=False)

    print_report(results)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
