#!/usr/bin/env python3
"""
transcript_fact_extraction.py

Computes cell 1.3 of the 2x3 transcript-grounding framework:

  In transcript; NOT in human note; NOT in AI note.

Method: applies FactEHR's atomic fact decomposition prompt (_DECOMP_TEMPLATE,
verbatim from factehr_pipeline.py lines 73-93) to each of the 45 conversation
transcripts, then runs binary entailment checks (same mechanism as files 13 and
16) against gold_note (shared across models) and pred_note (per model).

Cell 1.3 = transcript facts where BOTH checks return 0.

Full 2x2 cross-tab also reported per model:
  (gold=0, pred=0) -> cell_1_3          : neither note captured it
  (gold=0, pred=1) -> not_in_gold_in_pred : AI captured what human missed
  (gold=1, pred=0) -> in_gold_not_in_pred : human captured, AI missed
  (gold=1, pred=1) -> in_gold_in_pred    : both captured

Usage
-----
python transcript_fact_extraction.py \\
    --soap_dir    metrics_test/soap_prompts \\
    --output_dir  metrics_test/gap_analysis \\
    --keys        generation/keys_foundry.json \\
    --model       gpt-4o \\
    --workers     8 \\
    [--models model1 model2 ...] \\
    [--force]
"""

from __future__ import annotations

import argparse
import time
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

# ── FactEHR decomposition prompt (verbatim from factehr_pipeline.py lines 73-93) ──

_DECOMP_TEMPLATE = """\
Please breakdown the following text into independent facts as a string delimited by "//" to separate the facts

Example 1 :
Note: "There is a dense consolidation in the left lower lobe."

Atomic facts:
There is a consolidation. // The consolidation is dense. // The consolidation is on the left. // The consolidation is in a lobe. // The consolidation is in the lower portion of the left lobe.


Example 2:

Note: "The patient has been having intermittent shortness of breath for the last two years."

Atomic facts:
The patient has been having shortness of breath. // The shortness of breath is intermittent. // The shortness of breath has been present for the last two years.

Do not include any other text, or say "Here is the list..."

Note:
{text}"""


# ── Non-JSON LLM call for decomposition (plain text response) ──────────────────

def _call_llm_text(
    client: openai.OpenAI,
    user_prompt: str,
    model: str,
    max_retries: int = 4,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content
        except openai.RateLimitError as exc:
            if getattr(exc, "code", None) == "insufficient_quota":
                raise QuotaExhausted(
                    "Insufficient quota/funds. Add funds and rerun — cache is preserved."
                ) from None
            time.sleep(30 * (attempt + 1))
        except Exception as exc:
            print(f"\n  [API error attempt {attempt+1}/{max_retries}] {exc}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── Decomposition ───────────────────────────────────────────────────────────────

def decompose_transcript(
    transcript: str,
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Optional[List[str]]:
    if not force and cache_path.exists():
        return load_json(cache_path)["facts"]
    if not transcript.strip():
        save_json({"facts": [], "raw": ""}, cache_path)
        return []
    raw = _call_llm_text(client, _DECOMP_TEMPLATE.format(text=transcript), model)
    if raw is None:
        return None
    facts = [f.strip() for f in raw.split("//") if f.strip()]
    save_json({"facts": facts, "raw": raw}, cache_path)
    return facts


# ── Entailment batch ────────────────────────────────────────────────────────────

def entail_facts_vs_note(
    note: str,
    facts: List[str],
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    workers: int,
    force: bool,
) -> List[Dict[str, Any]]:
    """Returns [{fact, entailment}] — same cache pattern as files 13 and 16."""
    if not force and cache_path.exists():
        return load_json(cache_path)
    if not facts:
        save_json([], cache_path)
        return []
    results: List[Optional[int]] = [None] * len(facts)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_entail_one, client, model, note, f): i for i, f in enumerate(facts)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    out = [{"fact": f, "entailment": e} for f, e in zip(facts, results)]
    save_json(out, cache_path)
    return out


# ── Per-model analysis ──────────────────────────────────────────────────────────

def analyse_model(
    model_name: str,
    pred_lookup: Dict[str, str],
    transcript_facts_lookup: Dict[str, List[str]],
    gold_entail_lookup: Dict[str, List[Dict[str, Any]]],
    cache_dir: Path,
    client: openai.OpenAI,
    judge_model: str,
    workers: int,
    force: bool,
) -> Dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_item_out = []
    g0p0 = g0p1 = g1p0 = g1p1 = 0

    for idx in sorted(transcript_facts_lookup):
        t_facts = transcript_facts_lookup[idx]
        gold_results = gold_entail_lookup.get(idx)
        pred_note = pred_lookup.get(idx, "")
        if not t_facts or gold_results is None or not pred_note.strip():
            continue

        print(f"  [{idx}] {len(t_facts)} facts", end=" ", flush=True)
        pred_results = entail_facts_vs_note(
            pred_note, t_facts,
            cache_dir / f"{idx}.json",
            client, judge_model, workers, force,
        )
        gold_map = {r["fact"]: r["entailment"] for r in gold_results}
        pred_map = {r["fact"]: r["entailment"] for r in pred_results}

        item_g0p0 = item_g0p1 = item_g1p0 = item_g1p1 = 0
        cell_1_3_facts = []
        for fact in t_facts:
            ge = gold_map.get(fact)
            pe = pred_map.get(fact)
            if ge == 0 and pe == 0:
                item_g0p0 += 1
                cell_1_3_facts.append(fact)
            elif ge == 0 and pe == 1:
                item_g0p1 += 1
            elif ge == 1 and pe == 0:
                item_g1p0 += 1
            elif ge == 1 and pe == 1:
                item_g1p1 += 1

        g0p0 += item_g0p0
        g0p1 += item_g0p1
        g1p0 += item_g1p0
        g1p1 += item_g1p1
        per_item_out.append({
            "idx": idx,
            "transcript_facts_count": len(t_facts),
            "cell_1_3_count": item_g0p0,
            "cell_1_3_facts": cell_1_3_facts,
        })
        print(f"-> 1.3={item_g0p0} ai_only={item_g0p1} human_only={item_g1p0} both={item_g1p1}")

    total = g0p0 + g0p1 + g1p0 + g1p1

    def pct(n: int) -> float:
        return round(n / total, 3) if total else 0.0

    summary = {
        "transcript_facts_checked": total,
        "cell_1_3":              {"count": g0p0, "pct": pct(g0p0)},
        "not_in_gold_in_pred":   {"count": g0p1, "pct": pct(g0p1)},
        "in_gold_not_in_pred":   {"count": g1p0, "pct": pct(g1p0)},
        "in_gold_in_pred":       {"count": g1p1, "pct": pct(g1p1)},
    }
    return {"summary": summary, "per_item": per_item_out}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cell 1.3: transcript facts captured by neither note."
    )
    parser.add_argument("--soap_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys",       default="generation/keys_foundry.json")
    parser.add_argument("--model",      default="gpt-4o")
    parser.add_argument("--cache_dir",  default=None)
    parser.add_argument("--workers",    type=int, default=8)
    parser.add_argument("--models",     nargs="*", default=None)
    parser.add_argument("--force",      action="store_true")
    args = parser.parse_args()

    soap_dir = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".transcript_cache"

    keys = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"], base_url=keys.get("base_url"))

    soap_files: Dict[str, Path] = {p.stem: p for p in soap_dir.glob("*.json")}
    model_names = args.models if args.models else sorted(soap_files)

    # ── Build shared lookup: idx -> transcript, idx -> gold_note ───────────────
    # gold_note and conversation are the same for a given idx across all models.
    # Collect from all models so we don't miss any encounter.
    transcript_lookup: Dict[str, str] = {}
    gold_note_lookup: Dict[str, str] = {}
    for mname in model_names:
        if mname not in soap_files:
            continue
        for item in load_json(soap_files[mname]):
            idx = item.get("idx")
            if not idx:
                continue
            if idx not in transcript_lookup:
                transcript_lookup[idx] = (item.get("conversation") or "").strip()
            if idx not in gold_note_lookup:
                gold_note_lookup[idx] = (item.get("gold_note") or "").strip()

    print(f"Loaded {len(transcript_lookup)} unique encounters across {len(model_names)} models")

    # ── Stage 1: Transcript decomposition (once per encounter) ─────────────────
    decomp_cache = cache_root / ".decomp"
    decomp_cache.mkdir(parents=True, exist_ok=True)
    transcript_facts_lookup: Dict[str, List[str]] = {}

    print(f"\n{'='*60}\nStage 1: Transcript decomposition ({len(transcript_lookup)} encounters)\n{'='*60}")
    for idx in sorted(transcript_lookup):
        facts = decompose_transcript(
            transcript_lookup[idx],
            decomp_cache / f"{idx}.json",
            client, args.model, args.force,
        )
        if facts is not None:
            transcript_facts_lookup[idx] = facts
            print(f"  [{idx}] {len(facts)} facts")

    total_t_facts = sum(len(f) for f in transcript_facts_lookup.values())
    print(f"\nTotal transcript facts: {total_t_facts} across {len(transcript_facts_lookup)} encounters")

    # ── Stage 2: Gold entailment (shared across models, once per encounter) ─────
    gold_cache = cache_root / ".gold_entail"
    gold_cache.mkdir(parents=True, exist_ok=True)
    gold_entail_lookup: Dict[str, List[Dict[str, Any]]] = {}

    print(f"\n{'='*60}\nStage 2: Gold entailment ({total_t_facts} facts, shared)\n{'='*60}")
    for idx in sorted(transcript_facts_lookup):
        t_facts = transcript_facts_lookup[idx]
        gold_note = gold_note_lookup.get(idx, "")
        if not gold_note:
            print(f"  [{idx}] skipped — no gold_note")
            continue
        print(f"  [{idx}] {len(t_facts)} facts", end=" ", flush=True)
        results = entail_facts_vs_note(
            gold_note, t_facts,
            gold_cache / f"{idx}.json",
            client, args.model, args.workers, args.force,
        )
        gold_entail_lookup[idx] = results
        n_in = sum(1 for r in results if r["entailment"] == 1)
        n_out = len(results) - n_in
        print(f"-> in_gold={n_in} not_in_gold={n_out}")

    # ── Stage 3: Per-model pred entailment ──────────────────────────────────────
    all_results: Dict[str, Any] = {}
    try:
        for model_name in model_names:
            if model_name not in soap_files:
                print(f"\nSkipping {model_name} — no soap file found")
                continue
            print(f"\n{'='*60}\nStage 3 [{model_name}]\n{'='*60}")
            pred_lookup = {
                item.get("idx"): (item.get("pred_note") or "").strip()
                for item in load_json(soap_files[model_name])
                if item.get("idx")
            }
            result = analyse_model(
                model_name=model_name,
                pred_lookup=pred_lookup,
                transcript_facts_lookup=transcript_facts_lookup,
                gold_entail_lookup=gold_entail_lookup,
                cache_dir=cache_root / model_name,
                client=client,
                judge_model=args.model,
                workers=args.workers,
                force=args.force,
            )
            all_results[model_name] = result
            s = result["summary"]
            print(
                f"\n  cell_1_3={s['cell_1_3']['count']} ({s['cell_1_3']['pct']:.0%})"
                f"  ai_only={s['not_in_gold_in_pred']['count']} ({s['not_in_gold_in_pred']['pct']:.0%})"
                f"  human_only={s['in_gold_not_in_pred']['count']} ({s['in_gold_not_in_pred']['pct']:.0%})"
                f"  both={s['in_gold_in_pred']['count']} ({s['in_gold_in_pred']['pct']:.0%})"
            )
    except QuotaExhausted as exc:
        print(f"\n*** STOPPED: {exc} ***")

    output = {
        "metadata": {
            "script": "transcript_fact_extraction.py",
            "method": (
                "FactEHR _DECOMP_TEMPLATE (factehr_pipeline.py lines 73-93) applied to each "
                "conversation transcript to extract atomic facts, then binary entailment checks "
                "(entailment_binary6 mechanism, same as files 13 and 16) of each transcript "
                "fact against gold_note (shared across models, Stage 2) and pred_note (per "
                "model, Stage 3). Cell 1.3 = gold_entail=0 AND pred_entail=0."
            ),
            "judge_model": args.model,
            "total_encounters": len(transcript_facts_lookup),
            "total_transcript_facts": total_t_facts,
        },
        "models": all_results,
    }
    out_path = output_dir / "18_transcript_cell_1_3.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")

    # Aggregate cross-tab across all models
    agg: Dict[str, int] = {
        "cell_1_3": 0,
        "not_in_gold_in_pred": 0,
        "in_gold_not_in_pred": 0,
        "in_gold_in_pred": 0,
    }
    for r in all_results.values():
        for k in agg:
            agg[k] += r["summary"].get(k, {}).get("count", 0)
    tot = sum(agg.values())
    print("\nAggregate cross-tab (all models):")
    labels = {
        "cell_1_3":            "cell 1.3 (neither note)",
        "not_in_gold_in_pred": "AI-only  (not in human, in AI)",
        "in_gold_not_in_pred": "human-only (in human, not in AI)",
        "in_gold_in_pred":     "both     (in both notes)",
    }
    for k, v in agg.items():
        pct_str = f"{v/tot:.1%}" if tot else "N/A"
        print(f"  {labels[k]:<38s} {v:5d}  ({pct_str})")


if __name__ == "__main__":
    main()
