#!/usr/bin/env python3
"""
category_severity_gap_analysis.py

Classifies the already-extracted, FactEHR-verified facts from
11_factehr_exact_claims.json by:
  - category : the 10-label clinical taxonomy used in claim_overlap.py / file 7
               (domain-adapted for PT/OT/SLP rehab notes — an in-house scheme,
               not a literature citation)
  - severity : major / minor, per the definition used in the npj Digital
               Medicine 2025 clinical-safety framework — major = "would impact
               the diagnosis or management of the patient if not corrected"

No new decomposition or entailment calls are made. Each note's full fact list
is classified ONCE (one batched call per note); missed_gold_facts and
hallucinated_pred_facts are subsets of those already-classified lists, so
their category/severity are derived by lookup, not a second call.

Usage
-----
python category_severity_gap_analysis.py \
    --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
    --soap_dir    metrics_test/soap_prompts \
    --output_dir  metrics_test/gap_analysis \
    --keys        generation/keys_foundry.json \
    --model       gpt-4o \
    [--cache_dir  metrics_test/gap_analysis/.category_severity_cache] \
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


_CATEGORIES = [
    "symptoms", "diagnoses", "medications", "procedures",
    "observations", "vitals", "history", "social_history", "plan", "functional",
]


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYS = (
    "You are a clinical NLP expert classifying pre-extracted clinical facts. "
    "Always return valid JSON exactly as specified — no markdown, no explanation."
)

_CLASSIFY_USER = """\
Each fact below was already extracted from the clinical note. For each fact, assign:

1. category — exactly ONE of:
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

2. severity — exactly ONE of:
   major – this fact would impact the diagnosis or management of the patient
           if it were incorrect or omitted (e.g. medications, safety risks,
           diagnoses, vital abnormalities, falls/safety incidents)
   minor – anything else (e.g. routine observations, pleasantries, minor
           procedural detail) that would NOT change clinical decision-making
           if incorrect or omitted

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 10 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}

CLINICAL NOTE (for context):
{note}

FACTS TO CLASSIFY ({n} total):
{facts_json}"""


# ── LLM wrapper ────────────────────────────────────────────────────────────────

def _call_llm(client: openai.OpenAI, user_prompt: str, model: str, max_retries: int = 4) -> Optional[str]:
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
        except openai.RateLimitError as exc:
            if getattr(exc, "code", None) == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI/Foundry account has insufficient quota/funds. "
                    "Add funds and rerun — cached progress is preserved."
                ) from None
            time.sleep(30 * (attempt + 1))
        except Exception as exc:
            print(f"\n  [API error attempt {attempt+1}/{max_retries}] {exc}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def classify_facts(
    note: str,
    facts: List[str],
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Optional[List[Dict[str, str]]]:
    if not force and cache_path.exists():
        return load_json(cache_path)
    if not facts:
        save_json([], cache_path)
        return []

    prompt = _CLASSIFY_USER.format(
        note=note, n=len(facts),
        facts_json=json.dumps(facts, indent=2, ensure_ascii=False),
    )
    raw = _call_llm(client, prompt, model)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
        classifications = parsed.get("classifications", [])
    except json.JSONDecodeError:
        return None

    save_json(classifications, cache_path)
    return classifications


# ── Per-model analysis ───────────────────────────────────────────────────────────

def _build_lookup(classifications: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {c["fact"]: c for c in classifications if "fact" in c}


def analyse_model(
    model_name: str,
    per_item_claims: List[Dict[str, Any]],
    soap_lookup: Dict[str, Dict[str, str]],
    gold_cache_dir: Path,
    pred_cache_dir: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Dict[str, Any]:
    pred_cache_dir.mkdir(parents=True, exist_ok=True)

    missed_by_cat: Dict[str, int] = defaultdict(int)
    hall_by_cat: Dict[str, int] = defaultdict(int)
    gold_total_by_cat: Dict[str, int] = defaultdict(int)
    pred_total_by_cat: Dict[str, int] = defaultdict(int)
    missed_by_sev: Dict[str, int] = defaultdict(int)
    hall_by_sev: Dict[str, int] = defaultdict(int)
    missed_cat_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hall_cat_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_item_out: List[Dict[str, Any]] = []

    for item in per_item_claims:
        idx = item.get("idx")
        if item.get("degenerate_pred_output") or "error" in item or idx not in soap_lookup:
            continue

        gold_note = soap_lookup[idx]["gold_note"]
        pred_note = soap_lookup[idx]["pred_note"]
        gold_facts = item.get("gold_facts", [])
        pred_facts = item.get("pred_facts", [])

        print(f"  [{idx}]", end=" ", flush=True)

        gold_classes = classify_facts(
            gold_note, gold_facts, gold_cache_dir / f"{idx}.json", client, model, force
        )
        pred_classes = classify_facts(
            pred_note, pred_facts, pred_cache_dir / f"{idx}.json", client, model, force
        )
        if gold_classes is None or pred_classes is None:
            print("SKIP (classification failed)")
            continue

        gold_lookup = _build_lookup(gold_classes)
        pred_lookup = _build_lookup(pred_classes)

        for f in gold_facts:
            c = gold_lookup.get(f, {})
            gold_total_by_cat[c.get("category", "other")] += 1
        for f in pred_facts:
            c = pred_lookup.get(f, {})
            pred_total_by_cat[c.get("category", "other")] += 1

        for f in item.get("missed_gold_facts", []):
            c = gold_lookup.get(f, {})
            cat, sev = c.get("category", "other"), c.get("severity", "minor")
            missed_by_cat[cat] += 1
            missed_by_sev[sev] += 1
            missed_cat_sev[cat][sev] += 1

        for f in item.get("hallucinated_pred_facts", []):
            c = pred_lookup.get(f, {})
            cat, sev = c.get("category", "other"), c.get("severity", "minor")
            hall_by_cat[cat] += 1
            hall_by_sev[sev] += 1
            hall_cat_sev[cat][sev] += 1

        per_item_out.append({
            "idx": idx,
            "gold_fact_classes": gold_classes,
            "pred_fact_classes": pred_classes,
        })

        print("ok")

    total_missed = sum(missed_by_cat.values())
    total_hall = sum(hall_by_cat.values())

    def _cat_summary(by_cat, total_by_cat, total):
        return {
            cat: {
                "count": by_cat.get(cat, 0),
                "pct_of_total": round(by_cat.get(cat, 0) / total, 3) if total else 0,
                "rate_of_category": round(by_cat.get(cat, 0) / total_by_cat[cat], 3) if total_by_cat.get(cat) else None,
            }
            for cat in set(list(by_cat.keys()) + list(total_by_cat.keys()))
        }

    summary = {
        "missed_by_category": dict(sorted(
            _cat_summary(missed_by_cat, gold_total_by_cat, total_missed).items(),
            key=lambda kv: -kv[1]["count"],
        )),
        "hallucinated_by_category": dict(sorted(
            _cat_summary(hall_by_cat, pred_total_by_cat, total_hall).items(),
            key=lambda kv: -kv[1]["count"],
        )),
        "missed_by_severity": {
            sev: {"count": missed_by_sev.get(sev, 0),
                  "pct_of_total": round(missed_by_sev.get(sev, 0) / total_missed, 3) if total_missed else 0}
            for sev in ("major", "minor")
        },
        "hallucinated_by_severity": {
            sev: {"count": hall_by_sev.get(sev, 0),
                  "pct_of_total": round(hall_by_sev.get(sev, 0) / total_hall, 3) if total_hall else 0}
            for sev in ("major", "minor")
        },
        "missed_category_x_severity": {cat: dict(sevs) for cat, sevs in missed_cat_sev.items()},
        "hallucinated_category_x_severity": {cat: dict(sevs) for cat, sevs in hall_cat_sev.items()},
        "total_missed": total_missed,
        "total_hallucinated": total_hall,
    }

    return {"summary": summary, "per_item": per_item_out}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Category + severity classification of FactEHR-exact facts.")
    parser.add_argument("--claims_json", required=True, help="Path to 11_factehr_exact_claims.json")
    parser.add_argument("--soap_dir", required=True, help="Folder with model prediction JSONs (for note text).")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys", default="generation/keys_foundry.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model names to process (default: all).")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    claims_data = load_json(Path(args.claims_json))
    soap_dir = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".category_severity_cache"
    gold_cache_dir = cache_root / "gold"
    pred_cache_root = cache_root / "pred"

    keys = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"], base_url=keys.get("base_url"))

    model_names = args.models if args.models else list(claims_data["models"].keys())

    all_results: Dict[str, Any] = {}
    for model_name in model_names:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
        soap_path = soap_dir / f"{model_name}.json"
        soap_items = load_json(soap_path)
        soap_lookup = {
            item.get("idx"): {
                "gold_note": (item.get("gold_note") or "").strip(),
                "pred_note": (item.get("pred_note") or "").strip(),
            }
            for item in soap_items
        }

        per_item_claims = claims_data["models"][model_name]["per_item"]
        result = analyse_model(
            model_name=model_name,
            per_item_claims=per_item_claims,
            soap_lookup=soap_lookup,
            gold_cache_dir=gold_cache_dir,
            pred_cache_dir=pred_cache_root / model_name,
            client=client,
            model=args.model,
            force=args.force,
        )
        all_results[model_name] = result
        s = result["summary"]
        print(f"\n  missed={s['total_missed']} hallucinated={s['total_hallucinated']}")

    output = {
        "metadata": {
            "script": "category_severity_gap_analysis.py",
            "source": str(Path(args.claims_json)),
            "category_taxonomy": "in-house 10-label scheme (claim_overlap.py), domain-adapted for rehab notes",
            "severity_definition": "npj Digital Medicine 2025 framework: major = impacts diagnosis/management if uncorrected",
            "judge_model": args.model,
        },
        "models": all_results,
    }
    out_path = output_dir / "12_category_severity_gaps.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
