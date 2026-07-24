#!/usr/bin/env python3
"""
missed_fact_grounding_analysis.py

For every fact already flagged as "missed" (a gold fact not entailed by
pred_note) in 11_factehr_exact_claims.json, checks whether it is actually
grounded in the full conversation transcript. This splits "missed" into two
materially different cases:

  - in_transcript_not_in_ai_note : entailed by the transcript -- the AI had
                                    access to this content and still failed
                                    to capture it. A genuine, fair AI miss.
  - not_in_transcript            : not entailed by the transcript either --
                                    structurally unavailable to any text-only
                                    system (e.g. patient mood/affect that the
                                    human note-writer observed directly but
                                    was never said aloud, so it appears in
                                    the gold note but nowhere in the spoken
                                    transcript).

This is the missed-fact counterpart to hallucination_grounding_analysis.py
(which does the same check for hallucinated_pred_facts, classifying them as
"omitted_but_true"/"fabricated" -- labels that don't apply to missed facts,
since a missed gold fact is real, human-written content and can't be
"fabricated"). Mechanism, model, and caching pattern are reused verbatim by
importing hallucination_grounding_analysis.py's helpers directly (confirmed
lightweight: only openai/json/time/concurrent.futures, no heavy deps).

Usage
-----
python missed_fact_grounding_analysis.py \
    --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
    --soap_dir    metrics_test/soap_prompts \
    --output_dir  metrics_test/gap_analysis \
    --keys        generation/keys_foundry.json \
    --model       gpt-4o \
    --workers     8 \
    [--models model1 model2 ...] \
    [--force]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai

from hallucination_grounding_analysis import (
    QuotaExhausted,
    _entail_one,
    load_json,
    save_json,
)


def ground_missed_facts(
    conversation: str,
    facts: List[str],
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    workers: int,
    force: bool,
) -> List[Dict[str, Any]]:
    if not force and cache_path.exists():
        return load_json(cache_path)
    if not facts:
        save_json([], cache_path)
        return []

    results: List[Optional[int]] = [None] * len(facts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_entail_one, client, model, conversation, f): i for i, f in enumerate(facts)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    out = [
        {"fact": f, "entailment": e,
         "classification": "in_transcript_not_in_ai_note" if e == 1 else "not_in_transcript"}
        for f, e in zip(facts, results)
    ]
    save_json(out, cache_path)
    return out


# ── Per-model analysis ──────────────────────────────────────────────────────────

def analyse_model(
    model_name: str,
    per_item_claims: List[Dict[str, Any]],
    conv_lookup: Dict[str, str],
    cache_dir: Path,
    client: openai.OpenAI,
    model: str,
    workers: int,
    force: bool,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_item_out: List[Dict[str, Any]] = []
    in_transcript = not_in_transcript = 0

    for item in per_item_claims:
        idx = item.get("idx")
        missed_facts = item.get("missed_gold_facts", [])
        if item.get("degenerate_pred_output") or "error" in item or not missed_facts or idx not in conv_lookup:
            continue

        print(f"  [{idx}] {len(missed_facts)} facts", end=" ", flush=True)
        results = ground_missed_facts(
            conv_lookup[idx], missed_facts, cache_dir / f"{idx}.json", client, model, workers, force
        )
        n_in = sum(1 for r in results if r["classification"] == "in_transcript_not_in_ai_note")
        n_out = sum(1 for r in results if r["classification"] == "not_in_transcript")
        in_transcript += n_in
        not_in_transcript += n_out
        per_item_out.append({"idx": idx, "groundings": results})
        print(f"-> in_transcript_not_in_ai_note={n_in} not_in_transcript={n_out}")

    total = in_transcript + not_in_transcript
    summary = {
        "total_missed_checked": total,
        "in_transcript_not_in_ai_note": {
            "count": in_transcript, "pct": round(in_transcript / total, 3) if total else 0,
        },
        "not_in_transcript": {
            "count": not_in_transcript, "pct": round(not_in_transcript / total, 3) if total else 0,
        },
    }
    return {"summary": summary, "per_item": per_item_out}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify missed gold facts as in_transcript_not_in_ai_note vs not_in_transcript."
    )
    parser.add_argument("--claims_json", required=True)
    parser.add_argument("--soap_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys", default="generation/keys_foundry.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    claims_data = load_json(Path(args.claims_json))
    soap_dir = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".missed_fact_grounding_cache"

    keys = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"], base_url=keys.get("base_url"))

    model_names = args.models if args.models else list(claims_data["models"].keys())

    all_results: Dict[str, Any] = {}
    try:
        for model_name in model_names:
            print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")
            soap_items = load_json(soap_dir / f"{model_name}.json")
            conv_lookup = {
                item.get("idx"): (item.get("conversation") or "").strip()
                for item in soap_items
            }
            per_item_claims = claims_data["models"][model_name]["per_item"]
            result = analyse_model(
                model_name=model_name,
                per_item_claims=per_item_claims,
                conv_lookup=conv_lookup,
                cache_dir=cache_root / model_name,
                client=client,
                model=args.model,
                workers=args.workers,
                force=args.force,
            )
            all_results[model_name] = result
            s = result["summary"]
            print(f"\n  in_transcript_not_in_ai_note={s['in_transcript_not_in_ai_note']['count']} "
                  f"({s['in_transcript_not_in_ai_note']['pct']:.0%})  "
                  f"not_in_transcript={s['not_in_transcript']['count']} "
                  f"({s['not_in_transcript']['pct']:.0%})")
    except QuotaExhausted as exc:
        print(f"\n*** STOPPED: {exc} ***")

    output = {
        "metadata": {
            "script": "missed_fact_grounding_analysis.py",
            "method": "Binary entailment (FactEHR entailment_binary6.tmpl, same mechanism as "
                      "hallucination_grounding_analysis.py) of each missed gold fact against the "
                      "full conversation transcript. Splits missed facts into "
                      "in_transcript_not_in_ai_note (AI had access via transcript, missed it "
                      "anyway -- a fair/fixable miss) vs not_in_transcript (structurally "
                      "unavailable to a text-only system, e.g. patient mood/affect observed "
                      "directly by the human note-writer but never said aloud).",
            "judge_model": args.model,
        },
        "models": all_results,
    }
    out_path = output_dir / "16_missed_fact_grounding.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
