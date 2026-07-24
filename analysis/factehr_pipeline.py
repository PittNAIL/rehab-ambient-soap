#!/usr/bin/env python3
"""
factehr_exact_claims.py

Faithful reproduction of the FactEHR (Munnangi et al. 2024, som-shahlab/factehr)
fact decomposition + entailment pipeline, applied to gold-vs-pred note comparison.

Source of truth (verified against the official repo, not paraphrased):
  - Decomposition prompt : data/prompt_templates/fact_decomposition_delimiter/prompt1_icl.tmpl
                           (the template actually invoked by scripts/init_all_datasets.sh
                           to build FactEHR's released dataset)
  - Entailment prompt    : data/prompt_templates/binary_entailment/entailment_binary6.tmpl
                           (binary 0/1 JSON output; GPT-4o was FactEHR's selected judge)
  - Precision/recall     : src/factehr/utils/parse_facts.py — these are SELF-CONSISTENCY
                           checks on the decomposition, not gold-vs-pred comparisons:
                             precision: note ⇒ fact        (is each fact grounded in its own note?)
                             recall   : fact-list ⇒ sentence (does the fact list cover every
                                                              sentence of its own note?)

Extension beyond FactEHR's own repo (needed for our actual goal — gold vs pred
comparison — which their repo does not do):
  - After verifying factuality of gold/pred facts via the precision self-check, the
    SAME binary entailment template is applied cross-document:
      recall    : pred_note ⇒ gold_fact   (does the prediction cover this verified gold fact?)
      precision : gold_note ⇒ pred_fact   (is this verified pred fact really true?)

Every entailment call is exactly one premise/hypothesis pair per FactEHR's template
(no batching of the prompt itself) — calls are simply parallelized with a thread pool
for wall-clock speed, which does not change the prompt or its semantics.

Usage
-----
python factehr_exact_claims.py \
    --soap_dir   metrics_test/soap_prompts \
    --output_dir metrics_test/gap_analysis \
    --keys       generation/keys.json \
    --model      gpt-4o \
    --subset     5 \
    --workers    8 \
    [--skip_recall_selfcheck] \
    [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ── FactEHR prompts (verbatim from som-shahlab/factehr) ────────────────────────

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


# ── LLM wrappers ───────────────────────────────────────────────────────────────
# FactEHR sends a single user message with no system prompt (see parse_facts.py:
# "messages": [{"role": "user", "content": prompt}]) — replicated here as-is.

class QuotaExhausted(Exception):
    """Raised immediately on insufficient_quota — never worth retrying."""


def _call_llm(
    client: openai.OpenAI,
    user_prompt: str,
    model: str,
    json_mode: bool,
    max_retries: int = 4,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.0,
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except openai.RateLimitError as exc:
            if getattr(exc, "code", None) == "insufficient_quota":
                raise QuotaExhausted(
                    "OpenAI account has insufficient quota/funds. Add funds and rerun "
                    "the same command — cached progress is preserved and will resume."
                ) from None
            wait = 30 * (attempt + 1)
            time.sleep(wait)
        except Exception as exc:
            print(f"\n  [API error attempt {attempt+1}/{max_retries}] {exc}", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return None


# ── Decomposition (delimiter parsing, exactly as FactEHR's pipeline parses it) ──

def decompose_facts(
    note: str,
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Optional[List[str]]:
    if not force and cache_path.exists():
        return load_json(cache_path)["facts"]

    raw = _call_llm(client, _DECOMP_TEMPLATE.format(text=note), model, json_mode=False)
    if raw is None:
        return None

    facts = [f.strip() for f in raw.split("//") if f.strip()]
    save_json({"facts": facts, "raw": raw}, cache_path)
    return facts


# ── Binary entailment (single premise/hypothesis pair, exactly as FactEHR) ─────

def _entail_one(client: openai.OpenAI, model: str, premise: str, hypothesis: str) -> Optional[int]:
    raw = _call_llm(
        client,
        _ENTAIL_TEMPLATE.format(premise=premise, hypothesis=hypothesis),
        model,
        json_mode=True,
    )
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
        val = parsed.get("entailment_prediction")
        return int(val) if val in (0, 1, "0", "1") else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def entail_list(
    premise: str,
    hypotheses: List[str],
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    workers: int,
    force: bool,
) -> List[Dict[str, Any]]:
    """Run one FactEHR binary-entailment call per hypothesis against a shared
    premise. Calls are fanned out across a thread pool purely for wall-clock
    speed — each call is still exactly one premise/hypothesis pair, unmodified
    from FactEHR's template."""
    if not force and cache_path.exists():
        return load_json(cache_path)

    if not hypotheses:
        save_json([], cache_path)
        return []

    results: List[Optional[int]] = [None] * len(hypotheses)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_entail_one, client, model, premise, h): i
            for i, h in enumerate(hypotheses)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    out = [
        {"hypothesis": h, "entailment": e}
        for h, e in zip(hypotheses, results)
    ]
    save_json(out, cache_path)
    return out


# ── Sentence splitting (practical stand-in for FactEHR's custom spaCy clinical
#    tokenizer in src/factehr/nlp/sbd.py — not reproduced verbatim) ────────────

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ── Invalid pred_note detection (heuristic gate, not part of FactEHR itself —
#    kept from the prior pipeline to avoid wasting calls on unusable text) ─────

_SPEAKER_TURN_RE = re.compile(r"\b(?:Doctor|Client|Patient|SPK_\d+)\s*:")


def is_degenerate(text: str, min_len: int = 50) -> Optional[str]:
    text = text.strip()
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


# ── Per-example pipeline ────────────────────────────────────────────────────────

def process_example(
    idx: str,
    gold_note: str,
    pred_note: str,
    cache: Dict[str, Path],
    client: openai.OpenAI,
    model: str,
    workers: int,
    skip_recall_selfcheck: bool,
    force: bool,
) -> Dict[str, Any]:
    degenerate_reason = is_degenerate(pred_note)
    if degenerate_reason:
        return {"idx": idx, "degenerate_pred_output": True, "degenerate_reason": degenerate_reason}

    # 1. Decompose gold and pred notes independently (FactEHR delimiter prompt)
    gold_facts = decompose_facts(gold_note, cache["decomp_gold"] / f"{idx}.json", client, model, force)
    pred_facts = decompose_facts(pred_note, cache["decomp_pred"] / f"{idx}.json", client, model, force)
    if gold_facts is None or pred_facts is None:
        return {"idx": idx, "error": "decomposition_failed"}

    # 2. Self-consistency PRECISION check: note ⇒ fact (is each fact grounded in
    #    the note it came from?) — this is the "ensure claims are factual" step.
    gold_self_precision = entail_list(
        gold_note, gold_facts, cache["self_prec_gold"] / f"{idx}.json", client, model, workers, force
    )
    pred_self_precision = entail_list(
        pred_note, pred_facts, cache["self_prec_pred"] / f"{idx}.json", client, model, workers, force
    )

    verified_gold_facts = [r["hypothesis"] for r in gold_self_precision if r["entailment"] == 1]
    verified_pred_facts = [r["hypothesis"] for r in pred_self_precision if r["entailment"] == 1]

    # 3. Self-consistency RECALL check: fact-list ⇒ sentence (did decomposition
    #    drop any information from the note?) — optional, for full fidelity.
    gold_completeness = pred_completeness = None
    if not skip_recall_selfcheck:
        gold_sentences = split_sentences(gold_note)
        pred_sentences = split_sentences(pred_note)
        gold_facts_joined = " // ".join(gold_facts)
        pred_facts_joined = " // ".join(pred_facts)

        gold_self_recall = entail_list(
            gold_facts_joined, gold_sentences, cache["self_rec_gold"] / f"{idx}.json", client, model, workers, force
        )
        pred_self_recall = entail_list(
            pred_facts_joined, pred_sentences, cache["self_rec_pred"] / f"{idx}.json", client, model, workers, force
        )
        gold_completeness = (
            sum(r["entailment"] == 1 for r in gold_self_recall) / len(gold_self_recall)
            if gold_self_recall else None
        )
        pred_completeness = (
            sum(r["entailment"] == 1 for r in pred_self_recall) / len(pred_self_recall)
            if pred_self_recall else None
        )

    # 4. Cross-document comparison on VERIFIED-FACTUAL facts only.
    cross_recall = entail_list(
        pred_note, verified_gold_facts, cache["cross_recall"] / f"{idx}.json", client, model, workers, force
    )
    cross_precision = entail_list(
        gold_note, verified_pred_facts, cache["cross_precision"] / f"{idx}.json", client, model, workers, force
    )

    n_covered = sum(r["entailment"] == 1 for r in cross_recall)
    n_grounded = sum(r["entailment"] == 1 for r in cross_precision)

    recall = n_covered / len(verified_gold_facts) if verified_gold_facts else 0.0
    precision = n_grounded / len(verified_pred_facts) if verified_pred_facts else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0

    return {
        "idx": idx,
        "degenerate_pred_output": False,
        "gold_facts_total": len(gold_facts),
        "pred_facts_total": len(pred_facts),
        "gold_facts_verified_factual": len(verified_gold_facts),
        "pred_facts_verified_factual": len(verified_pred_facts),
        "gold_decomposition_completeness": gold_completeness,
        "pred_decomposition_completeness": pred_completeness,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "gold_facts": gold_facts,
        "pred_facts": pred_facts,
        "verified_gold_facts": verified_gold_facts,
        "verified_pred_facts": verified_pred_facts,
        "missed_gold_facts": [r["hypothesis"] for r in cross_recall if r["entailment"] != 1],
        "hallucinated_pred_facts": [r["hypothesis"] for r in cross_precision if r["entailment"] != 1],
    }


# ── Per-model analysis ──────────────────────────────────────────────────────────

def analyse_model(
    soap_path: Path,
    cache_root: Path,
    client: openai.OpenAI,
    model: str,
    subset: int,
    workers: int,
    skip_recall_selfcheck: bool,
    force: bool,
) -> Dict[str, Any]:
    data = load_json(soap_path)[:subset]
    model_stem = soap_path.stem

    cache = {
        "decomp_gold":     cache_root / "decomp_gold",
        "decomp_pred":     cache_root / "decomp_pred" / model_stem,
        "self_prec_gold":  cache_root / "self_prec_gold",
        "self_prec_pred":  cache_root / "self_prec_pred" / model_stem,
        "self_rec_gold":   cache_root / "self_rec_gold",
        "self_rec_pred":   cache_root / "self_rec_pred" / model_stem,
        "cross_recall":    cache_root / "cross_recall" / model_stem,
        "cross_precision": cache_root / "cross_precision" / model_stem,
    }
    for d in cache.values():
        d.mkdir(parents=True, exist_ok=True)

    per_item: List[Dict] = []
    for i, item in enumerate(data):
        idx = item.get("idx", str(i))
        gold_note = (item.get("gold_note") or "").strip()
        pred_note = (item.get("pred_note") or "").strip()

        print(f"  [{i+1}/{len(data)}] {idx}", end=" ", flush=True)
        result = process_example(
            idx, gold_note, pred_note, cache, client, model, workers, skip_recall_selfcheck, force
        )
        per_item.append(result)

        if result.get("degenerate_pred_output"):
            print(f"[INVALID:{result['degenerate_reason']}]")
        elif "error" in result:
            print(f"SKIP ({result['error']})")
        else:
            print(
                f"R={result['recall']:.2f} P={result['precision']:.2f} F1={result['f1']:.2f}  "
                f"gold={result['gold_facts_verified_factual']}/{result['gold_facts_total']} "
                f"pred={result['pred_facts_verified_factual']}/{result['pred_facts_total']}"
            )

    ok_items = [x for x in per_item if not x.get("degenerate_pred_output") and "error" not in x]
    recalls = [x["recall"] for x in ok_items]
    precisions = [x["precision"] for x in ok_items]
    f1s = [x["f1"] for x in ok_items]

    summary = {
        "total_examples": len(data),
        "successful_examples": len(ok_items),
        "degenerate_pred_notes": sum(1 for x in per_item if x.get("degenerate_pred_output")),
        "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "mean_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "mean_gold_facts_total": round(
            sum(x["gold_facts_total"] for x in ok_items) / len(ok_items), 2
        ) if ok_items else None,
        "mean_pred_facts_total": round(
            sum(x["pred_facts_total"] for x in ok_items) / len(ok_items), 2
        ) if ok_items else None,
        "mean_gold_facts_verified_factual": round(
            sum(x["gold_facts_verified_factual"] for x in ok_items) / len(ok_items), 2
        ) if ok_items else None,
        "mean_pred_facts_verified_factual": round(
            sum(x["pred_facts_verified_factual"] for x in ok_items) / len(ok_items), 2
        ) if ok_items else None,
    }

    return {"summary": summary, "per_item": per_item}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exact FactEHR fact decomposition + entailment pipeline.")
    parser.add_argument("--soap_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys", default="generation/keys.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--subset", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_recall_selfcheck", action="store_true",
                        help="Skip the fact-list=>sentence completeness self-check to cut call volume.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    soap_dir = Path(args.soap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".factehr_exact_cache"

    keys = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"], base_url=keys.get("base_url"))

    soap_files = sorted(
        p for p in soap_dir.iterdir()
        if p.suffix.lower() in {".json", ".jsonl"} and not p.name.startswith(".")
    )
    print(f"SOAP files ({len(soap_files)}): {[p.name for p in soap_files]}")
    print(f"Judge model : {args.model}")
    print(f"Subset size : {args.subset} per model")
    print(f"Workers     : {args.workers}")
    print(f"Recall self-check: {'OFF' if args.skip_recall_selfcheck else 'ON'}")
    print(f"Cache dir   : {cache_root}\n")

    def build_output(results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "metadata": {
                "script": "factehr_exact_claims.py",
                "method": "FactEHR exact decomposition + binary entailment (delimiter prompt + entailment_binary6), "
                           "extended with cross-document gold/pred comparison",
                "judge_model": args.model,
                "recall_selfcheck": not args.skip_recall_selfcheck,
                "subset_per_model": args.subset,
                "generated_at": str(date.today()),
            },
            "models": results,
        }

    out_path = output_dir / "11_factehr_exact_claims.json"
    all_results: Dict[str, Any] = {}
    for soap_path in soap_files:
        print(f"\n{'='*60}\nModel: {soap_path.stem}\n{'='*60}")
        try:
            result = analyse_model(
                soap_path=soap_path,
                cache_root=cache_root,
                client=client,
                model=args.model,
                subset=args.subset,
                workers=args.workers,
                skip_recall_selfcheck=args.skip_recall_selfcheck,
                force=args.force,
            )
        except QuotaExhausted as exc:
            print(f"\n\n*** STOPPED: {exc} ***")
            save_json(build_output(all_results), out_path)
            print(f"Partial results for {len(all_results)} completed model(s) saved -> {out_path}")
            raise SystemExit(1)

        all_results[soap_path.stem] = result
        s = result["summary"]
        print(f"\n  Summary: R={s['mean_recall']} P={s['mean_precision']} F1={s['mean_f1']}")

    save_json(build_output(all_results), out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
