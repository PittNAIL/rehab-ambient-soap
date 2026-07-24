#!/usr/bin/env python3
"""
topic_model_uncategorized.py

Discovers what's actually inside the "uncategorized_by_creola" bucket from
14_creola_category_gaps.json using a focused adaptation of TopicGPT's
methodology (Pham et al., "TopicGPT: A Prompt-based Topic Modeling
Framework," NAACL 2024 -- https://github.com/chtmp223/topicgpt), rather than
manual keyword regexes.

TopicGPT's actual pipeline has 5 stages: generate level-1 topics, generate
level-2 (sub)topics, refine (merge duplicates), assign documents to topics
with supporting quotes, and correct any assignments that drift from the
final topic list. This script implements a single-level adaptation of the
first four stages (no level-2 sub-topic generation, no correction pass --
out of scope for this exploratory use), reusing this project's existing
LLM-call/retry pattern (creola_category_gap_analysis.py's _call_llm):

  1. Generate: iterate over batches of facts. Each batch sees the topic
     list accumulated so far and either reuses existing topics or proposes
     new ones -- topics must be generalizable (not fact-specific) and
     single-concept, mirroring TopicGPT's generation-prompt constraints.
  2. Refine: one pass over the accumulated topic list to merge near-
     duplicate/overlapping topics into a final clean set.
  3. Assign: batch facts again against the FIXED final topic list (no new
     topics allowed at this stage, per TopicGPT's assignment-prompt
     constraint), each with a supporting quote.

Operates on the 431 unique fact strings within the 1,687 gold-fact
occurrences classified as uncategorized_by_creola (deduplicated since the
same gold note is shared across all 7 models' per_item entries for a given
idx -- topic discovery only needs the unique content, not occurrence counts).

Usage
-----
python topic_model_uncategorized.py \
    --claims_json metrics_test/gap_analysis/14_creola_category_gaps.json \
    --output_dir  metrics_test/gap_analysis \
    --keys        generation/keys_foundry.json \
    --model       gpt-4o \
    [--batch_size 40]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import openai

# Reuse this project's existing retrying LLM-call helper rather than duplicating it.
from creola_category_gap_analysis import _call_llm, load_json, save_json


_GENERATION_SYS = (
    "You are performing topic modeling on clinical facts, following the TopicGPT "
    "methodology (Pham et al., NAACL 2024). Always return valid JSON exactly as "
    "specified -- no markdown, no explanation."
)

_GENERATION_USER = """\
These facts are from physical/occupational/speech-language therapy clinical notes. \
Identify GENERALIZABLE, high-level topics describing what KIND OF CONTENT each fact \
represents (not clinical coding, just natural content topics).

CURRENT TOPIC LIST (accumulated from earlier batches -- reuse these where they fit; \
do not invent a near-duplicate of an existing topic):
{current_topics}

Rules (per TopicGPT):
- Topics must be GENERALIZABLE, not specific to a single fact.
- Each topic must reflect a SINGLE concept, not a combination of concepts.
- If a fact fits an existing topic above, do not propose a new one for it.
- Only propose a NEW topic if no existing topic reasonably covers the fact.

FACTS (batch {batch_num} of {total_batches}):
{facts_json}

Return ONLY valid JSON listing every topic relevant to this batch (existing reused
topics AND any new ones), with a short label and one-sentence description each:
{{
  "topics": [
    {{"label": "<short topic label>", "description": "<one sentence>"}}
  ]
}}"""

_REFINE_USER = """\
This topic list was accumulated across many batches of clinical facts and likely \
contains near-duplicate or overlapping topics. Merge duplicates and overlapping \
topics into single clear topics, tighten vague labels, and drop anything that isn't \
a real recurring topic. Produce a final, clean, non-redundant topic list.

CURRENT TOPIC LIST:
{current_topics}

Return ONLY valid JSON:
{{
  "topics": [
    {{"label": "<final topic label>", "description": "<one sentence>"}}
  ]
}}"""

_ASSIGN_USER = """\
Assign each fact below to EXACTLY ONE topic from this FIXED list. Do not invent new \
topics -- if nothing fits well, use the closest one. Include a short supporting quote \
(the fact text itself, verbatim) as evidence, per TopicGPT's assignment methodology.

TOPIC LIST:
{topic_list}

FACTS ({n} total):
{facts_json}

Return ONLY valid JSON, one entry per fact, in the same order:
{{
  "assignments": [
    {{"fact": "<exact fact text>", "topic": "<exact topic label from the list above>"}}
  ]
}}"""


def _format_topics(topics: List[Dict[str, str]]) -> str:
    if not topics:
        return "(none yet)"
    return "\n".join(f"- {t['label']}: {t['description']}" for t in topics)


def _merge_topics(existing: List[Dict[str, str]], new: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_label = {t["label"].strip().lower(): t for t in existing}
    for t in new:
        key = t.get("label", "").strip().lower()
        if not key:
            continue
        by_label[key] = {"label": t["label"], "description": t.get("description", "")}
    return list(by_label.values())


def _call_json(client: openai.OpenAI, model: str, system: str, user: str) -> Dict[str, Any]:
    raw = _call_llm(client, user, model, system_prompt=system, temperature=0.0)
    if raw is None:
        raise RuntimeError("LLM call failed after retries")
    return json.loads(raw)


def generate_topics(facts: List[str], client: openai.OpenAI, model: str, batch_size: int) -> List[Dict[str, str]]:
    batches = [facts[i:i + batch_size] for i in range(0, len(facts), batch_size)]
    topics: List[Dict[str, str]] = []
    for i, batch in enumerate(batches, 1):
        print(f"  [generate {i}/{len(batches)}] topics so far: {len(topics)}", flush=True)
        prompt = _GENERATION_USER.format(
            current_topics=_format_topics(topics),
            batch_num=i, total_batches=len(batches),
            facts_json=json.dumps(batch, indent=2, ensure_ascii=False),
        )
        result = _call_json(client, model, _GENERATION_SYS, prompt)
        topics = _merge_topics(topics, result.get("topics", []))
    return topics


def refine_topics(topics: List[Dict[str, str]], client: openai.OpenAI, model: str) -> List[Dict[str, str]]:
    print(f"  [refine] {len(topics)} raw topics -> ", end="", flush=True)
    prompt = _REFINE_USER.format(current_topics=_format_topics(topics))
    result = _call_json(client, model, _GENERATION_SYS, prompt)
    final = result.get("topics", [])
    print(f"{len(final)} final topics")
    return final


def assign_topics(
    facts: List[str], final_topics: List[Dict[str, str]], client: openai.OpenAI, model: str, batch_size: int,
) -> Dict[str, str]:
    topic_list_str = _format_topics(final_topics)
    batches = [facts[i:i + batch_size] for i in range(0, len(facts), batch_size)]
    assignment: Dict[str, str] = {}
    for i, batch in enumerate(batches, 1):
        print(f"  [assign {i}/{len(batches)}]", flush=True)
        prompt = _ASSIGN_USER.format(
            topic_list=topic_list_str, n=len(batch),
            facts_json=json.dumps(batch, indent=2, ensure_ascii=False),
        )
        result = _call_json(client, model, _GENERATION_SYS, prompt)
        for a in result.get("assignments", []):
            if "fact" in a and "topic" in a:
                assignment[a["fact"]] = a["topic"]
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description="TopicGPT-style topic modeling of uncategorized_by_creola facts.")
    parser.add_argument("--claims_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--keys", default="generation/keys_foundry.json")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--batch_size", type=int, default=40)
    args = parser.parse_args()

    data = load_json(Path(args.claims_json))
    facts_with_counts: Dict[str, int] = {}
    for model_name, r in data["models"].items():
        for item in r["per_item"]:
            for c in item.get("gold_fact_classes", []):
                if c.get("category") == "uncategorized_by_creola":
                    facts_with_counts[c["fact"]] = facts_with_counts.get(c["fact"], 0) + 1

    unique_facts = sorted(facts_with_counts.keys())
    print(f"{len(unique_facts)} unique facts ({sum(facts_with_counts.values())} occurrences across models)")

    keys = load_json(Path(args.keys))
    client = openai.OpenAI(api_key=keys["api_key"], base_url=keys.get("base_url"))

    print("\n=== Stage 1: Generate ===")
    raw_topics = generate_topics(unique_facts, client, args.model, args.batch_size)

    print("\n=== Stage 2: Refine ===")
    final_topics = refine_topics(raw_topics, client, args.model)

    print("\n=== Stage 3: Assign ===")
    assignment = assign_topics(unique_facts, final_topics, client, args.model, args.batch_size)

    topic_counts: Dict[str, int] = {}
    topic_occurrence_counts: Dict[str, int] = {}
    for fact, topic in assignment.items():
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_occurrence_counts[topic] = topic_occurrence_counts.get(topic, 0) + facts_with_counts.get(fact, 0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "metadata": {
            "script": "topic_model_uncategorized.py",
            "method": "Single-level adaptation of TopicGPT (Pham et al., NAACL 2024) "
                       "generate->refine->assign stages; no level-2 sub-topics or "
                       "correction pass.",
            "source": str(Path(args.claims_json)),
            "judge_model": args.model,
            "unique_facts": len(unique_facts),
            "total_occurrences": sum(facts_with_counts.values()),
        },
        "final_topics": final_topics,
        "topic_counts_unique_facts": dict(sorted(topic_counts.items(), key=lambda kv: -kv[1])),
        "topic_counts_all_occurrences": dict(sorted(topic_occurrence_counts.items(), key=lambda kv: -kv[1])),
        "fact_to_topic": assignment,
    }
    out_path = output_dir / "15_uncategorized_topic_model.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
