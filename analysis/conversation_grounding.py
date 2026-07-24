#!/usr/bin/env python3
"""
conversation_grounding.py

Analysis 8: Conversation grounding / coverage.

For each patient-doctor transcript, segments the conversation into 4-6
clinical phases and checks whether each segment's clinical content was
captured in the human note (gold) vs the AI note (pred).

This answers: "which parts of the conversation did the physician document
but the AI ignored?" and surfaces temporal / speaker / phase blind spots.

Two-step LLM pipeline:
  Step 1 (per-idx, cached)  : segment conversation + extract claims + check gold coverage
  Step 2 (per-model/idx, cached): check pred coverage against the cached segments

Usage
-----
python conversation_grounding.py \
    --soap_dir  metrics_test/soap_prompts \
    --output_dir metrics_test/gap_analysis \
    [--keys      generation/keys.json] \
    [--model     gpt-4o] \
    [--cache_dir metrics_test/gap_analysis/.conv_cache] \
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
    "You are a clinical NLP expert analyzing medical conversations and notes. "
    "Always return valid JSON exactly as specified — no markdown, no code fences."
)

# Step 1: segment conversation + extract claims + check gold coverage
_SEGMENT_USER = """\
Analyze this clinical conversation transcript and compare it to the human-written note.

NOTE: The transcript is an ASR output — consecutive SPK_0 / SPK_1 lines often belong to
the same sentence. Read them as a flowing conversation between two speakers.

CONVERSATION TRANSCRIPT:
{conversation}

HUMAN-WRITTEN CLINICAL NOTE:
{gold_note}

Tasks:
1. Divide the conversation into 4-6 meaningful SEGMENTS based on clinical phase or topic.
   Use phases like: intake, symptom_review, physical_examination, assessment_discussion,
   plan_discussion, follow_up, social_history, or other descriptive labels.
2. For each segment extract every atomic clinical claim discussed.
3. Mark which claims the human note actually documents.
4. Note which speaker primarily provides the clinical information in each segment.
5. Mark the temporal position of each segment (early = first third, middle = second third,
   late = final third of the conversation).

Return ONLY valid JSON:
{{
  "segments": [
    {{
      "segment_id": 0,
      "phase": "intake",
      "conversation_position": "early",
      "primary_speaker": "doctor",
      "text_excerpt": "(first ~80 chars of this segment)",
      "clinical_claims": ["claim1", "claim2"],
      "in_gold_note": [true, false]
    }}
  ]
}}"""

# Step 2: check pred coverage against cached segments
_PRED_COVERAGE_USER = """\
Given these pre-extracted conversation segments and their clinical claims, assess how
well the AI-generated note covers them.

CONVERSATION SEGMENTS WITH CLAIMS:
{segments_json}

AI-GENERATED NOTE:
{pred_note}

For each segment, mark which claims the AI note covers (true/false list aligned with
the segment's clinical_claims list). Also list any claims that are missed.

Return ONLY valid JSON:
{{
  "segment_coverage": [
    {{
      "segment_id": 0,
      "claim_coverage": [true, false],
      "missed_claims": ["claim not in AI note"]
    }}
  ]
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


# ── Step 1: segment conversation + gold coverage (cached per idx) ──────────────

def segment_and_check_gold(
    conversation: str,
    gold_note: str,
    idx: str,
    client: openai.OpenAI,
    model: str,
    cache_dir: Path,
    force: bool,
) -> Optional[Dict]:
    cache_path = cache_dir / f"{idx}.json"
    if not force and cache_path.exists():
        return load_json(cache_path)

    # Truncate very long conversations at ~18000 chars (~4500 tokens) to stay safe
    MAX_CONV = 18000
    conv_text = conversation[:MAX_CONV]
    if len(conversation) > MAX_CONV:
        conv_text += "\n[transcript truncated for length]"

    prompt = _SEGMENT_USER.format(conversation=conv_text, gold_note=gold_note)
    result = _parse(_call_llm(client, prompt, model))
    if result:
        save_json(result, cache_path)
    return result


# ── Step 2: pred coverage (cached per model/idx) ───────────────────────────────

def check_pred_coverage(
    segments: List[Dict],
    pred_note: str,
    idx: str,
    client: openai.OpenAI,
    model: str,
    cache_dir: Path,
    force: bool,
) -> Optional[Dict]:
    cache_path = cache_dir / f"{idx}.json"
    if not force and cache_path.exists():
        return load_json(cache_path)

    # Pass only what the LLM needs (id, phase, claims) — not the full text
    slim_segments = [
        {
            "segment_id": s.get("segment_id", i),
            "phase": s.get("phase", "unknown"),
            "conversation_position": s.get("conversation_position", "unknown"),
            "primary_speaker": s.get("primary_speaker", "unknown"),
            "clinical_claims": s.get("clinical_claims", []),
        }
        for i, s in enumerate(segments)
    ]

    prompt = _PRED_COVERAGE_USER.format(
        segments_json=json.dumps(slim_segments, indent=2, ensure_ascii=False),
        pred_note=pred_note,
    )
    result = _parse(_call_llm(client, prompt, model))
    if result:
        save_json(result, cache_path)
    return result


# ── Metrics helpers ────────────────────────────────────────────────────────────

def _safe_div(n: float, d: float) -> Optional[float]:
    return round(n / d, 4) if d else None


def _coverage_rate(coverage_list: List[bool]) -> Optional[float]:
    if not coverage_list:
        return None
    return round(sum(coverage_list) / len(coverage_list), 4)


# ── Per-model analysis ─────────────────────────────────────────────────────────

def analyse_model(
    soap_path: Path,
    seg_cache_dir: Path,
    pred_cache_root: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Dict[str, Any]:
    data = load_json(soap_path)
    model_name = soap_path.stem
    pred_cache_dir = pred_cache_root / model_name
    pred_cache_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by phase, position, speaker
    gold_by_phase: Dict[str, List[bool]] = defaultdict(list)
    pred_by_phase: Dict[str, List[bool]] = defaultdict(list)
    gold_by_pos: Dict[str, List[bool]] = defaultdict(list)
    pred_by_pos: Dict[str, List[bool]] = defaultdict(list)
    gold_by_speaker: Dict[str, List[bool]] = defaultdict(list)
    pred_by_speaker: Dict[str, List[bool]] = defaultdict(list)

    all_gold_cov: List[bool] = []
    all_pred_cov: List[bool] = []

    per_item: List[Dict] = []

    for i, item in enumerate(data):
        idx = item.get("idx", str(i))
        gold_note = item.get("gold_note", "") or ""
        pred_note = item.get("pred_note", "") or ""
        conversation = item.get("conversation", "") or ""

        print(f"  [{i+1:2d}/{len(data)}] {idx}", end=" … ", flush=True)

        # ── Step 1: segment + gold coverage (cached globally) ─────────────
        seg_result = segment_and_check_gold(
            conversation, gold_note, idx, client, model, seg_cache_dir, force
        )
        if not seg_result:
            print("SKIP (segmentation failed)")
            per_item.append({"idx": idx, "error": "segmentation_failed"})
            continue

        segments: List[Dict] = seg_result.get("segments", [])
        if not segments:
            print("SKIP (no segments returned)")
            per_item.append({"idx": idx, "error": "no_segments"})
            continue

        # ── Step 2: pred coverage ──────────────────────────────────────────
        pred_result = check_pred_coverage(
            segments, pred_note, idx, client, model, pred_cache_dir, force
        )

        # Build per-segment merged view
        # Index pred coverage by segment_id for safe merge
        pred_by_id: Dict[int, Dict] = {}
        if pred_result:
            for sc in pred_result.get("segment_coverage", []):
                pred_by_id[sc.get("segment_id", -1)] = sc

        item_segments: List[Dict] = []
        item_gold_cov: List[bool] = []
        item_pred_cov: List[bool] = []

        for seg in segments:
            seg_id = seg.get("segment_id", 0)
            phase = (seg.get("phase") or "unknown").lower().replace(" ", "_")
            position = (seg.get("conversation_position") or "unknown").lower()
            speaker = (seg.get("primary_speaker") or "unknown").lower()
            claims = seg.get("clinical_claims", [])
            gold_flags: List[bool] = seg.get("in_gold_note", [])

            # Align gold flags length to claims
            while len(gold_flags) < len(claims):
                gold_flags.append(False)
            gold_flags = gold_flags[:len(claims)]

            # Pred coverage
            pc_entry = pred_by_id.get(seg_id, {})
            pred_flags_raw: List[bool] = pc_entry.get("claim_coverage", [])
            while len(pred_flags_raw) < len(claims):
                pred_flags_raw.append(False)
            pred_flags: List[bool] = pred_flags_raw[:len(claims)]
            missed_by_pred: List[str] = pc_entry.get("missed_claims", [])

            g_cov = _coverage_rate(gold_flags)
            p_cov = _coverage_rate(pred_flags) if pred_result else None

            # Accumulate aggregates
            for f in gold_flags:
                gold_by_phase[phase].append(f)
                gold_by_pos[position].append(f)
                gold_by_speaker[speaker].append(f)
                all_gold_cov.append(f)

            if pred_result:
                for f in pred_flags:
                    pred_by_phase[phase].append(f)
                    pred_by_pos[position].append(f)
                    pred_by_speaker[speaker].append(f)
                    all_pred_cov.append(f)

            item_gold_cov.extend(gold_flags)
            item_pred_cov.extend(pred_flags)

            item_segments.append({
                "segment_id": seg_id,
                "phase": phase,
                "conversation_position": position,
                "primary_speaker": speaker,
                "text_excerpt": seg.get("text_excerpt", ""),
                "clinical_claims": claims,
                "gold_coverage_flags": gold_flags,
                "pred_coverage_flags": pred_flags if pred_result else None,
                "gold_coverage_rate": g_cov,
                "pred_coverage_rate": p_cov,
                "coverage_gap": round(g_cov - p_cov, 4) if (g_cov is not None and p_cov is not None) else None,
                "claims_missed_by_pred": missed_by_pred,
            })

        n_g = len(item_gold_cov)
        n_p = len(item_pred_cov)
        item_gold_rate = _coverage_rate(item_gold_cov)
        item_pred_rate = _coverage_rate(item_pred_cov) if pred_result else None

        per_item.append({
            "idx": idx,
            "total_segments": len(segments),
            "total_conv_claims": n_g,
            "gold_coverage_rate": item_gold_rate,
            "pred_coverage_rate": item_pred_rate,
            "coverage_gap": round(item_gold_rate - item_pred_rate, 4)
                            if (item_gold_rate is not None and item_pred_rate is not None) else None,
            "segments": item_segments,
        })

        g_str = f"{item_gold_rate:.2f}" if item_gold_rate is not None else "?"
        p_str = f"{item_pred_rate:.2f}" if item_pred_rate is not None else "?"
        print(f"gold_cov={g_str} pred_cov={p_str} ({len(segments)} segs, {n_g} claims)")

    # ── Build summary ─────────────────────────────────────────────────────────
    def _phase_summary(g_dict, p_dict):
        all_phases = set(g_dict) | set(p_dict)
        out = {}
        for ph in sorted(all_phases):
            gv = g_dict.get(ph, [])
            pv = p_dict.get(ph, [])
            g_rate = _coverage_rate(gv)
            p_rate = _coverage_rate(pv) if pv else None
            out[ph] = {
                "n_claims": len(gv),
                "gold_coverage": g_rate,
                "pred_coverage": p_rate,
                "gap": round(g_rate - p_rate, 4) if (g_rate is not None and p_rate is not None) else None,
            }
        return out

    summary = {
        "total_examples": len(data),
        "successful_examples": sum(1 for x in per_item if "error" not in x),
        "overall_gold_coverage": _coverage_rate(all_gold_cov),
        "overall_pred_coverage": _coverage_rate(all_pred_cov) if all_pred_cov else None,
        "overall_gap": round(_coverage_rate(all_gold_cov) - _coverage_rate(all_pred_cov), 4)
                       if (all_gold_cov and all_pred_cov) else None,
        "by_phase":            _phase_summary(gold_by_phase, pred_by_phase),
        "by_conversation_position": _phase_summary(gold_by_pos, pred_by_pos),
        "by_primary_speaker":  _phase_summary(gold_by_speaker, pred_by_speaker),
    }

    return {"summary": summary, "per_item": per_item}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conversation grounding / coverage gap analysis."
    )
    parser.add_argument("--soap_dir",   required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys",  default="generation/keys.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--cache_dir",  default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    soap_dir   = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir     = Path(args.cache_dir) if args.cache_dir else output_dir / ".conv_cache"
    seg_cache_dir = cache_dir / "segments"
    pred_cache_root = cache_dir / "pred_coverage"
    seg_cache_dir.mkdir(parents=True, exist_ok=True)
    pred_cache_root.mkdir(parents=True, exist_ok=True)

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
            seg_cache_dir=seg_cache_dir,
            pred_cache_root=pred_cache_root,
            client=client,
            model=args.model,
            force=args.force,
        )
        all_results[soap_path.stem] = result

        s = result["summary"]
        print(
            f"\n  Summary: gold_cov={s['overall_gold_coverage']} "
            f"pred_cov={s['overall_pred_coverage']} gap={s['overall_gap']}"
        )
        print("  By conversation position:")
        for pos, v in s["by_conversation_position"].items():
            print(f"    {pos:<10} gold={v['gold_coverage']} pred={v['pred_coverage']} gap={v['gap']}")

    out_path = output_dir / "8_conversation_grounding.json"
    save_json(all_results, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
