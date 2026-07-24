#!/usr/bin/env python3
"""
claim_gap_analysis.py

Uses an LLM to extract atomic clinical facts from human (gold) and AI
(pred) notes, then identifies:
  - missed_claims       : in gold but absent from pred
  - hallucinated_claims : in pred but absent from gold

Gold claims are cached per-idx so they are only extracted once regardless
of how many models are evaluated.

Usage
-----
python claim_gap_analysis.py \
    --soap_dir  metrics_test/soap_prompts \
    --output_dir metrics_test/gap_analysis \
    [--keys      generation/keys.json] \
    [--model     gpt-4o] \
    [--cache_dir metrics_test/gap_analysis/.claim_cache] \
    [--force]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYS = (
    "You are a clinical NLP expert. You extract atomic clinical facts from "
    "medical notes. Always return valid JSON exactly as specified — no markdown, "
    "no explanation, no code fences."
)

_EXTRACT_USER = """\
Extract every atomic clinical fact from the clinical note below.

An atomic fact is ONE specific, self-contained piece of clinical information.
  Good: "patient arrived in wheelchair", "prescribed metformin 500 mg BID", "BP 132/84"
  Bad : "patient has multiple conditions", "medications were adjusted"

Assign each claim to ONE category:
  symptoms       – patient-reported symptoms or complaints
  diagnoses      – diagnoses, conditions, or clinical impressions
  medications    – drugs, dosages, frequencies, routes
  procedures     – tests or interventions performed during the visit
  observations   – objective exam findings, clinician observations
  vitals         – vital signs and numeric measurements
  history        – past medical, surgical, or psychiatric history
  social_history – social, family, or lifestyle information
  plan           – treatment plans, follow-ups, referrals, orders
  functional     – functional status, ADL performance, mobility

Return ONLY valid JSON:
{{
  "claims": ["claim1", "claim2"],
  "categories": {{
    "symptoms": [], "diagnoses": [], "medications": [], "procedures": [],
    "observations": [], "vitals": [], "history": [], "social_history": [],
    "plan": [], "functional": []
  }}
}}

Clinical note:
{note}"""

_COMPARE_USER = """\
Compare the AI-generated note against the pre-extracted reference claims.

REFERENCE CLAIMS (from human-written note):
{gold_claims_json}

AI-GENERATED NOTE:
{pred_note}

Tasks:
1. Extract all atomic clinical claims from the AI note (one fact per string).
2. For each reference claim decide:
   - "covered"  — the AI note contains this fact (exact or paraphrased)
   - "missed"   — the AI note omits this fact entirely
3. Mark AI claims that introduce information absent from the reference as
   "hallucinated".

Be strict: "covered" only if the SAME specific fact is present, not a related topic.

Return ONLY valid JSON:
{{
  "pred_claims": [],
  "covered_gold_claims": [],
  "missed_gold_claims": [],
  "hallucinated_pred_claims": []
}}"""


# ── LLM wrapper ────────────────────────────────────────────────────────────────

def _call_llm(
    client: openai.OpenAI,
    user_prompt: str,
    model: str,
    max_retries: int = 4,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYS},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except openai.RateLimitError:
            wait = 60 * (attempt + 1)
            print(f"\n  [rate limit] waiting {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as exc:
            print(f"\n  [API error attempt {attempt+1}/{max_retries}] {exc}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _parse(raw: Optional[str]) -> Optional[Dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Gold claim extraction (with per-idx disk cache) ────────────────────────────

def extract_gold_claims(
    note: str,
    idx: str,
    client: openai.OpenAI,
    model: str,
    gold_cache_dir: Path,
    force: bool,
) -> Optional[Dict]:
    cache_path = gold_cache_dir / f"{idx}.json"
    if not force and cache_path.exists():
        return load_json(cache_path)

    result = _parse(_call_llm(client, _EXTRACT_USER.format(note=note), model))
    if result:
        save_json(result, cache_path)
    return result


# ── Pred comparison (with per-model/idx disk cache) ────────────────────────────

def compare_pred(
    gold_claims: List[str],
    pred_note: str,
    idx: str,
    client: openai.OpenAI,
    model: str,
    compare_cache_dir: Path,
    force: bool,
) -> Optional[Dict]:
    cache_path = compare_cache_dir / f"{idx}.json"
    if not force and cache_path.exists():
        return load_json(cache_path)

    prompt = _COMPARE_USER.format(
        gold_claims_json=json.dumps(gold_claims, indent=2, ensure_ascii=False),
        pred_note=pred_note,
    )
    result = _parse(_call_llm(client, prompt, model))
    if result:
        save_json(result, cache_path)
    return result


# ── Per-model analysis ─────────────────────────────────────────────────────────

_CATEGORIES = [
    "symptoms", "diagnoses", "medications", "procedures",
    "observations", "vitals", "history", "social_history", "plan", "functional",
]


def analyse_model(
    soap_path: Path,
    gold_cache_dir: Path,
    compare_root_dir: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Dict[str, Any]:
    data = load_json(soap_path)
    cmp_dir = compare_root_dir / soap_path.stem
    cmp_dir.mkdir(parents=True, exist_ok=True)

    recall_acc: List[float] = []
    precision_acc: List[float] = []
    f1_acc: List[float] = []
    total_gold = total_pred = total_missed = total_hall = 0
    missed_by_cat: Dict[str, int] = defaultdict(int)
    per_item: List[Dict] = []

    for i, item in enumerate(data):
        idx = item.get("idx", str(i))
        gold_note = item.get("gold_note", "") or ""
        pred_note = item.get("pred_note", "") or ""

        print(f"  [{i+1:2d}/{len(data)}] {idx}", end=" … ", flush=True)

        # Step 1: extract gold claims
        gold_result = extract_gold_claims(
            gold_note, idx, client, model, gold_cache_dir, force
        )
        if not gold_result:
            print("SKIP (gold extraction failed)")
            per_item.append({"idx": idx, "error": "gold_extraction_failed"})
            continue

        gold_claims: List[str] = gold_result.get("claims", [])
        gold_categories: Dict[str, List[str]] = gold_result.get("categories", {})

        # Step 2: compare pred against gold claims
        cmp_result = compare_pred(
            gold_claims, pred_note, idx, client, model, cmp_dir, force
        )
        if not cmp_result:
            print("SKIP (comparison failed)")
            per_item.append({"idx": idx, "error": "comparison_failed"})
            continue

        pred_claims: List[str] = cmp_result.get("pred_claims", [])
        covered: List[str] = cmp_result.get("covered_gold_claims", [])
        missed: List[str] = cmp_result.get("missed_gold_claims", [])
        hallucinated: List[str] = cmp_result.get("hallucinated_pred_claims", [])

        n_gold = len(gold_claims)
        n_pred = len(pred_claims)
        n_cov = len(covered)
        n_miss = len(missed)
        n_hall = len(hallucinated)

        recall = n_cov / n_gold if n_gold else 0.0
        precision = (n_pred - n_hall) / n_pred if n_pred else 0.0
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) else 0.0
        )

        recall_acc.append(recall)
        precision_acc.append(precision)
        f1_acc.append(f1)
        total_gold += n_gold
        total_pred += n_pred
        total_missed += n_miss
        total_hall += n_hall

        # Track missed claims by category using reverse lookup
        claim_to_cat = {
            c: cat
            for cat, claims in gold_categories.items()
            for c in claims
        }
        for mc in missed:
            missed_by_cat[claim_to_cat.get(mc, "other")] += 1

        per_item.append({
            "idx": idx,
            "gold_claim_count": n_gold,
            "pred_claim_count": n_pred,
            "covered_count": n_cov,
            "missed_count": n_miss,
            "hallucinated_count": n_hall,
            "claim_recall": round(recall, 4),
            "claim_precision": round(precision, 4),
            "claim_f1": round(f1, 4),
            "gold_claims": gold_claims,
            "gold_categories": gold_categories,
            "pred_claims": pred_claims,
            "missed_claims": missed,
            "hallucinated_claims": hallucinated,
        })

        print(
            f"R={recall:.2f} P={precision:.2f} F1={f1:.2f}  "
            f"gold={n_gold} pred={n_pred} miss={n_miss} hall={n_hall}"
        )

    n_ok = sum(1 for x in per_item if "error" not in x)

    missed_by_cat_summary = {
        cat: {
            "count": cnt,
            "pct_of_total_missed": round(cnt / total_missed, 3) if total_missed else 0,
        }
        for cat, cnt in sorted(missed_by_cat.items(), key=lambda kv: -kv[1])
    }

    summary = {
        "total_examples": len(data),
        "successful_examples": n_ok,
        "mean_claim_recall":    round(sum(recall_acc)    / len(recall_acc),    4) if recall_acc    else None,
        "mean_claim_precision": round(sum(precision_acc) / len(precision_acc), 4) if precision_acc else None,
        "mean_claim_f1":        round(sum(f1_acc)        / len(f1_acc),        4) if f1_acc        else None,
        "total_gold_claims":        total_gold,
        "total_pred_claims":        total_pred,
        "total_missed_claims":      total_missed,
        "total_hallucinated_claims": total_hall,
        "missed_by_category":       missed_by_cat_summary,
    }

    return {"summary": summary, "per_item": per_item}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomic clinical claim decomposition gap analysis."
    )
    parser.add_argument("--soap_dir",   required=True, help="Folder with model prediction JSONs.")
    parser.add_argument("--output_dir", required=True, help="Where to write 7_claim_decomposition_gaps.json.")
    parser.add_argument("--keys",  default="generation/keys.json", help="OpenAI keys JSON.")
    parser.add_argument("--model", default="gpt-4o",               help="OpenAI model to use.")
    parser.add_argument("--cache_dir",  default=None,
                        help="LLM response cache (default: <output_dir>/.claim_cache).")
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and re-run all LLM calls.")
    args = parser.parse_args()

    soap_dir   = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".claim_cache"
    gold_cache_dir   = cache_dir / "gold"
    compare_root_dir = cache_dir / "compare"
    gold_cache_dir.mkdir(parents=True, exist_ok=True)
    compare_root_dir.mkdir(parents=True, exist_ok=True)

    keys   = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"])

    soap_files = sorted(
        p for p in soap_dir.iterdir()
        if p.suffix.lower() in {".json", ".jsonl"} and not p.name.startswith(".")
    )
    print(f"SOAP files ({len(soap_files)}): {[p.name for p in soap_files]}")
    print(f"LLM model : {args.model}")
    print(f"Cache dir : {cache_dir}\n")

    all_results: Dict[str, Any] = {}

    for soap_path in soap_files:
        print(f"\n{'='*60}")
        print(f"Model: {soap_path.stem}")
        print(f"{'='*60}")
        result = analyse_model(
            soap_path=soap_path,
            gold_cache_dir=gold_cache_dir,
            compare_root_dir=compare_root_dir,
            client=client,
            force=args.force,
        )
        all_results[soap_path.stem] = result

        s = result["summary"]
        print(
            f"\n  Summary: R={s['mean_claim_recall']} "
            f"P={s['mean_claim_precision']} F1={s['mean_claim_f1']} | "
            f"gold={s['total_gold_claims']} miss={s['total_missed_claims']} "
            f"hall={s['total_hallucinated_claims']}"
        )

    out_path = output_dir / "7_claim_decomposition_gaps.json"
    save_json(all_results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
