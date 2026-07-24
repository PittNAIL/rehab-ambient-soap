#!/usr/bin/env python3
"""
cui_claim_crosscheck.py

Independent cross-check for Analysis 7 (claim_overlap.py).

Analysis 7 asks a single LLM (GPT-4o) to decide, in free text, whether each
reference claim is "covered" or "missed" by the AI note, and whether each AI
claim is "hallucinated". That's one fallible judgment per decision.

This script re-derives the same covered / missed / hallucinated labels from a
second, non-LLM signal: UMLS concept (CUI) overlap, extracted with scispaCy's
biomedical NER + UMLS entity linker. Two claims are considered to refer to the
same clinical fact if they link to overlapping UMLS concepts, regardless of
surface wording.

It does NOT call any LLM — it only reads the disk cache that claim_overlap.py
already populated (gold claims + pred claims + LLM verdicts), so it is cheap to
re-run and safe to run alongside the original analysis for comparison.

Outputs, per model and overall:
  - CUI-grounded recall / precision / F1 (directly comparable to the
    mean_claim_recall / mean_claim_precision / mean_claim_f1 in
    7_claim_decomposition_gaps.json)
  - Agreement between the LLM judge and the CUI-grounded check, broken into a
    2x2 confusion table, restricted to claims that scispaCy could actually link
    to >=1 UMLS concept ("groundable" claims) -- claims with no linkable
    concept are reported separately since CUI overlap can't arbitrate them.

Usage
-----
python cui_claim_crosscheck.py \
    --claim_cache metrics_test/gap_analysis/.claim_cache \
    --output_dir  metrics_test/gap_analysis_cui_crosscheck \
    [--linker_threshold 0.75] \
    [--spacy_model en_core_sci_sm]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import spacy
from scispacy.linking import EntityLinker  # noqa: F401  (registers the "scispacy_linker" pipe)


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── UMLS linking (scispaCy), with a per-claim disk cache ───────────────────────
#
# Gold claims are shared across all 7 models, so caching by a hash of the claim
# text (rather than by model/idx) means each unique claim is only ever linked
# once, no matter how many models reuse it.

def load_nlp(spacy_model: str, threshold: float):
    nlp = spacy.load(spacy_model)
    nlp.add_pipe(
        "scispacy_linker",
        config={"resolve_abbreviations": True, "linker_name": "umls", "threshold": threshold},
    )
    return nlp


def _claim_cache_key(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()


def link_claim(
    text: str,
    nlp,
    threshold: float,
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    """Return UMLS concepts linked from `text`, each >= `threshold` similarity.

    Cached to disk per unique claim string so re-runs (and overlapping claims
    across models) don't re-pay NER + linking cost.
    """
    cache_path = cache_dir / f"{_claim_cache_key(text)}.json"
    if cache_path.exists():
        return load_json(cache_path)

    linker = nlp.get_pipe("scispacy_linker")
    hits: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    doc = nlp(text)
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            if score < threshold or cui in seen:
                continue
            seen.add(cui)
            entity = linker.kb.cui_to_entity.get(cui)
            hits.append({
                "cui": cui,
                "name": entity.canonical_name if entity else None,
                "score": round(float(score), 3),
                "matched_text": ent.text,
            })

    save_json(hits, cache_path)
    return hits


def cui_set(hits: List[Dict[str, Any]]) -> Set[str]:
    return {h["cui"] for h in hits}


# ── Per-model cross-check ──────────────────────────────────────────────────────

def crosscheck_model(
    model_stem: str,
    gold_cache_dir: Path,
    compare_dir: Path,
    nlp,
    threshold: float,
    cui_cache_dir: Path,
) -> Dict[str, Any]:
    per_item: List[Dict[str, Any]] = []

    # Confusion counts: gold-claim coverage verdicts, restricted to claims
    # scispaCy could link to >=1 UMLS concept ("groundable").
    cov_both_covered = cov_both_missed = 0
    cov_llm_only = cov_cui_only = 0          # disagreements
    n_groundable_gold = n_ungroundable_gold = 0

    # Confusion counts: pred-claim hallucination verdicts, restricted to
    # groundable pred claims.
    hall_both_yes = hall_both_no = 0
    hall_llm_only = hall_cui_only = 0
    n_groundable_pred = n_ungroundable_pred = 0

    # Recomputed CUI-grounded recall/precision/F1 (over groundable claims only,
    # so they're an apples-to-apples second opinion on the same fact set).
    cui_recall_acc: List[float] = []
    cui_precision_acc: List[float] = []
    cui_f1_acc: List[float] = []

    gold_files = sorted(gold_cache_dir.glob("*.json"))
    for gold_file in gold_files:
        idx = gold_file.stem
        cmp_file = compare_dir / f"{idx}.json"
        if not cmp_file.exists():
            continue

        gold_result = load_json(gold_file)
        cmp_result = load_json(cmp_file)

        gold_claims: List[str] = gold_result.get("claims", [])
        pred_claims: List[str] = cmp_result.get("pred_claims", [])
        llm_covered: Set[str] = set(cmp_result.get("covered_gold_claims", []))
        llm_missed: Set[str] = set(cmp_result.get("missed_gold_claims", []))
        llm_hallucinated: Set[str] = set(cmp_result.get("hallucinated_pred_claims", []))

        gold_cuis = {c: cui_set(link_claim(c, nlp, threshold, cui_cache_dir)) for c in gold_claims}
        pred_cuis = {c: cui_set(link_claim(c, nlp, threshold, cui_cache_dir)) for c in pred_claims}

        pred_cui_union: Set[str] = set().union(*pred_cuis.values()) if pred_cuis else set()
        gold_cui_union: Set[str] = set().union(*gold_cuis.values()) if gold_cuis else set()

        item_gold_rows = []
        item_n_groundable = item_n_cui_covered = 0
        for c in gold_claims:
            cuis = gold_cuis[c]
            groundable = bool(cuis)
            llm_verdict = "covered" if c in llm_covered else ("missed" if c in llm_missed else "unscored")

            if not groundable:
                n_ungroundable_gold += 1
                cui_verdict = "ungroundable"
            else:
                n_groundable_gold += 1
                item_n_groundable += 1
                cui_says_covered = bool(cuis & pred_cui_union)
                cui_verdict = "covered" if cui_says_covered else "missed"
                if cui_says_covered:
                    item_n_cui_covered += 1

                if llm_verdict == "covered" and cui_verdict == "covered":
                    cov_both_covered += 1
                elif llm_verdict == "missed" and cui_verdict == "missed":
                    cov_both_missed += 1
                elif llm_verdict == "covered" and cui_verdict == "missed":
                    cov_llm_only += 1          # LLM says covered, CUIs disagree
                elif llm_verdict == "missed" and cui_verdict == "covered":
                    cov_cui_only += 1          # CUIs say covered, LLM disagrees

            item_gold_rows.append({
                "claim": c,
                "cuis": [{"cui": h["cui"], "name": h["name"]} for h in link_claim(c, nlp, threshold, cui_cache_dir)],
                "llm_verdict": llm_verdict,
                "cui_verdict": cui_verdict,
                "agree": (llm_verdict == cui_verdict) if groundable else None,
            })

        item_hall_rows = []
        item_n_pred_groundable = item_n_cui_hallucinated = 0
        for c in pred_claims:
            cuis = pred_cuis[c]
            groundable = bool(cuis)
            llm_verdict = "hallucinated" if c in llm_hallucinated else "not_hallucinated"

            if not groundable:
                n_ungroundable_pred += 1
                cui_verdict = "ungroundable"
            else:
                n_groundable_pred += 1
                item_n_pred_groundable += 1
                cui_says_hallucinated = not bool(cuis & gold_cui_union)
                cui_verdict = "hallucinated" if cui_says_hallucinated else "not_hallucinated"
                if cui_says_hallucinated:
                    item_n_cui_hallucinated += 1

                if llm_verdict == "hallucinated" and cui_verdict == "hallucinated":
                    hall_both_yes += 1
                elif llm_verdict == "not_hallucinated" and cui_verdict == "not_hallucinated":
                    hall_both_no += 1
                elif llm_verdict == "hallucinated" and cui_verdict == "not_hallucinated":
                    hall_llm_only += 1
                elif llm_verdict == "not_hallucinated" and cui_verdict == "hallucinated":
                    hall_cui_only += 1

            item_hall_rows.append({
                "claim": c,
                "cuis": [{"cui": h["cui"], "name": h["name"]} for h in link_claim(c, nlp, threshold, cui_cache_dir)],
                "llm_verdict": llm_verdict,
                "cui_verdict": cui_verdict,
                "agree": (llm_verdict == cui_verdict) if groundable else None,
            })

        # Per-item CUI-grounded recall/precision/F1 (groundable subset only).
        if item_n_groundable:
            recall = item_n_cui_covered / item_n_groundable
            n_pred_g = item_n_pred_groundable
            precision = (n_pred_g - item_n_cui_hallucinated) / n_pred_g if n_pred_g else 0.0
            f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
            cui_recall_acc.append(recall)
            cui_precision_acc.append(precision)
            cui_f1_acc.append(f1)

        per_item.append({
            "idx": idx,
            "n_gold_claims": len(gold_claims),
            "n_groundable_gold_claims": item_n_groundable,
            "n_pred_claims": len(pred_claims),
            "n_groundable_pred_claims": item_n_pred_groundable,
            "gold_claim_crosscheck": item_gold_rows,
            "pred_claim_crosscheck": item_hall_rows,
        })

    n_groundable_total = max(n_groundable_gold, 1)
    n_groundable_pred_total = max(n_groundable_pred, 1)

    summary = {
        "coverage_crosscheck": {
            "groundable_gold_claims": n_groundable_gold,
            "ungroundable_gold_claims": n_ungroundable_gold,
            "agreement_rate": round((cov_both_covered + cov_both_missed) / n_groundable_total, 4),
            "confusion": {
                "both_covered":            cov_both_covered,
                "both_missed":             cov_both_missed,
                "llm_covered_cui_missed":  cov_llm_only,
                "llm_missed_cui_covered":  cov_cui_only,
            },
        },
        "hallucination_crosscheck": {
            "groundable_pred_claims": n_groundable_pred,
            "ungroundable_pred_claims": n_ungroundable_pred,
            "agreement_rate": round((hall_both_yes + hall_both_no) / n_groundable_pred_total, 4),
            "confusion": {
                "both_hallucinated":            hall_both_yes,
                "both_not_hallucinated":        hall_both_no,
                "llm_hallucinated_cui_not":     hall_llm_only,
                "llm_not_cui_hallucinated":     hall_cui_only,
            },
        },
        "cui_grounded_metrics": {
            "mean_claim_recall":    round(sum(cui_recall_acc)    / len(cui_recall_acc),    4) if cui_recall_acc    else None,
            "mean_claim_precision": round(sum(cui_precision_acc) / len(cui_precision_acc), 4) if cui_precision_acc else None,
            "mean_claim_f1":        round(sum(cui_f1_acc)        / len(cui_f1_acc),        4) if cui_f1_acc        else None,
            "n_items_scored": len(cui_recall_acc),
        },
    }

    return {"summary": summary, "per_item": per_item}


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_comparison(model_stem: str, llm_summary: Dict[str, Any] | None, crosscheck: Dict[str, Any]) -> None:
    s = crosscheck["summary"]
    cov = s["coverage_crosscheck"]
    hall = s["hallucination_crosscheck"]
    cui = s["cui_grounded_metrics"]

    print(f"\n{'='*72}\n{model_stem}\n{'='*72}")
    if llm_summary:
        print(
            f"  LLM-judge   (Analysis 7) : R={llm_summary.get('mean_claim_recall')} "
            f"P={llm_summary.get('mean_claim_precision')} F1={llm_summary.get('mean_claim_f1')}"
        )
    print(
        f"  CUI-grounded (this script): R={cui['mean_claim_recall']} "
        f"P={cui['mean_claim_precision']} F1={cui['mean_claim_f1']}  "
        f"[{cui['n_items_scored']} notes scored]"
    )
    print(
        f"\n  Coverage agreement   : {cov['agreement_rate']*100:.1f}% "
        f"over {cov['groundable_gold_claims']} groundable / "
        f"{cov['groundable_gold_claims'] + cov['ungroundable_gold_claims']} gold claims "
        f"({cov['ungroundable_gold_claims']} had no linkable UMLS concept)"
    )
    c = cov["confusion"]
    print(f"    both say covered          : {c['both_covered']}")
    print(f"    both say missed           : {c['both_missed']}")
    print(f"    LLM says covered, CUIs don't (possible LLM over-credit) : {c['llm_covered_cui_missed']}")
    print(f"    CUIs overlap, LLM says missed (possible LLM miss)       : {c['llm_missed_cui_covered']}")

    print(
        f"\n  Hallucination agreement: {hall['agreement_rate']*100:.1f}% "
        f"over {hall['groundable_pred_claims']} groundable / "
        f"{hall['groundable_pred_claims'] + hall['ungroundable_pred_claims']} pred claims "
        f"({hall['ungroundable_pred_claims']} had no linkable UMLS concept)"
    )
    h = hall["confusion"]
    print(f"    both say hallucinated     : {h['both_hallucinated']}")
    print(f"    both say not hallucinated : {h['both_not_hallucinated']}")
    print(f"    LLM flags hallucination, CUIs match gold (possible LLM false alarm): {h['llm_hallucinated_cui_not']}")
    print(f"    CUIs find no match, LLM doesn't flag (possible LLM miss)           : {h['llm_not_cui_hallucinated']}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check Analysis 7's LLM claim verdicts against UMLS CUI overlap (scispaCy)."
    )
    parser.add_argument("--claim_cache", required=True,
                        help="Path to claim_overlap.py's cache dir (contains gold/ and compare/<model>/).")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write cui_claim_crosscheck.json.")
    parser.add_argument("--llm_results", default=None,
                        help="Optional path to 7_claim_decomposition_gaps.json, for side-by-side R/P/F1.")
    parser.add_argument("--spacy_model", default="en_core_sci_sm")
    parser.add_argument("--linker_threshold", type=float, default=0.75,
                        help="Minimum scispaCy UMLS-linker similarity score to accept a CUI match.")
    parser.add_argument("--cui_cache_dir", default=None,
                        help="Per-claim CUI link cache (default: <output_dir>/.cui_cache).")
    args = parser.parse_args()

    claim_cache = Path(args.claim_cache)
    gold_cache_dir = claim_cache / "gold"
    compare_root = claim_cache / "compare"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cui_cache_dir = Path(args.cui_cache_dir) if args.cui_cache_dir else output_dir / ".cui_cache"
    cui_cache_dir.mkdir(parents=True, exist_ok=True)

    llm_results = load_json(Path(args.llm_results)) if args.llm_results else {}

    print(f"Loading {args.spacy_model} + scispaCy UMLS linker (threshold={args.linker_threshold}) ...")
    nlp = load_nlp(args.spacy_model, args.linker_threshold)
    print("Ready.\n")

    model_dirs = sorted(p for p in compare_root.iterdir() if p.is_dir())
    all_results: Dict[str, Any] = {}

    for model_dir in model_dirs:
        model_stem = model_dir.name
        print(f"Cross-checking {model_stem} ...")
        result = crosscheck_model(
            model_stem=model_stem,
            gold_cache_dir=gold_cache_dir,
            compare_dir=model_dir,
            nlp=nlp,
            threshold=args.linker_threshold,
            cui_cache_dir=cui_cache_dir,
        )
        all_results[model_stem] = result
        llm_summary = (llm_results.get(model_stem) or {}).get("summary")
        print_comparison(model_stem, llm_summary, result)

    out_path = output_dir / "cui_claim_crosscheck.json"
    save_json(all_results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
