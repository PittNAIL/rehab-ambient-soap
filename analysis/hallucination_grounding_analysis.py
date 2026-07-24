#!/usr/bin/env python3
"""
hallucination_grounding_analysis.py

For every fact already flagged as "hallucinated" (unsupported by gold_note) in
11_factehr_exact_claims.json, checks whether it is actually grounded in the
full conversation transcript. This splits "hallucinated" into two materially
different failure modes (Maynez et al. 2020 intrinsic/extrinsic hallucination
framing):

  - omitted_but_true : entailed by the transcript -- the visit really
                        included this, the human note-writer just left it out.
  - fabricated        : not entailed by the transcript either -- unsupported
                         by any available source.

Mechanism reused verbatim from factehr_exact_claims.py: FactEHR's
entailment_binary6.tmpl single premise/hypothesis binary entailment prompt.
This adapts DocLens's (Xie et al., ACL 2024) citation-precision metric, which
entails generated note sentences against their *cited* conversational turns;
since our pred_notes carry no inline citation markers, the whole transcript
is used as premise instead of a targeted citation -- a disclosed
simplification, not a methodology substitution.

Usage
-----
python hallucination_grounding_analysis.py \
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
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


# ── FactEHR binary entailment prompt (verbatim, reused from factehr_exact_claims.py) ──

_ENTAIL_TEMPLATE = """\
You are an expert on natural language entailment.
Your task is to deduce whether premise statements entail hypotheses.
Return only '1' if the hypothesis can be fully entailed by the premise.
Return only '0' if the hypothesis contains information that cannot be entailed by the premise.
Generate the answer in JSON format with the following keys:
'entailment_prediction': 1 or 0, whether the claim can be entailed.
Only return the JSON-formatted answer and nothing else.

Premise: {premise}
Hypothesis: {hypothesis}

Here is the JSON-formatted answer:
"""


class QuotaExhausted(Exception):
    pass


def _call_llm(client: openai.OpenAI, user_prompt: str, model: str, max_retries: int = 4) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except openai.RateLimitError as exc:
            if getattr(exc, "code", None) == "insufficient_quota":
                raise QuotaExhausted("Insufficient quota/funds. Add funds and rerun — cache is preserved.") from None
            time.sleep(30 * (attempt + 1))
        except Exception as exc:
            print(f"\n  [API error attempt {attempt+1}/{max_retries}] {exc}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


def _entail_one(client: openai.OpenAI, model: str, premise: str, hypothesis: str) -> Optional[int]:
    raw = _call_llm(client, _ENTAIL_TEMPLATE.format(premise=premise, hypothesis=hypothesis), model)
    if raw is None:
        return None
    try:
        val = json.loads(raw).get("entailment_prediction")
        return int(val) if val in (0, 1, "0", "1") else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def ground_facts(
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
         "classification": "omitted_but_true" if e == 1 else "fabricated"}
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
    fabricated = omitted = 0

    for item in per_item_claims:
        idx = item.get("idx")
        hall_facts = item.get("hallucinated_pred_facts", [])
        if item.get("degenerate_pred_output") or "error" in item or not hall_facts or idx not in conv_lookup:
            continue

        print(f"  [{idx}] {len(hall_facts)} facts", end=" ", flush=True)
        results = ground_facts(
            conv_lookup[idx], hall_facts, cache_dir / f"{idx}.json", client, model, workers, force
        )
        n_fab = sum(1 for r in results if r["classification"] == "fabricated")
        n_om = sum(1 for r in results if r["classification"] == "omitted_but_true")
        fabricated += n_fab
        omitted += n_om
        per_item_out.append({"idx": idx, "groundings": results})
        print(f"-> fabricated={n_fab} omitted_but_true={n_om}")

    total = fabricated + omitted
    summary = {
        "total_hallucinated_checked": total,
        "fabricated": {"count": fabricated, "pct": round(fabricated / total, 3) if total else 0},
        "omitted_but_true": {"count": omitted, "pct": round(omitted / total, 3) if total else 0},
    }
    return {"summary": summary, "per_item": per_item_out}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Classify hallucinated facts as fabricated vs omitted-but-true.")
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
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".hallucination_grounding_cache"

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
            print(f"\n  fabricated={s['fabricated']['count']} ({s['fabricated']['pct']:.0%})  "
                  f"omitted_but_true={s['omitted_but_true']['count']} ({s['omitted_but_true']['pct']:.0%})")
    except QuotaExhausted as exc:
        print(f"\n*** STOPPED: {exc} ***")

    output = {
        "metadata": {
            "script": "hallucination_grounding_analysis.py",
            "method": "Binary entailment (FactEHR entailment_binary6.tmpl) of each hallucinated fact "
                      "against the full conversation transcript, adapting DocLens's citation-precision "
                      "metric (whole-transcript premise instead of per-citation targeting).",
            "judge_model": args.model,
        },
        "models": all_results,
    }
    out_path = output_dir / "13_hallucination_grounding.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
