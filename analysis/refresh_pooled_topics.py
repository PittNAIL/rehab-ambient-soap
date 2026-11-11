#!/usr/bin/env python3
"""
refresh_pooled_topics.py

Rebuild the pooled CREOLA input and refresh topic assignments for the
uncategorized missed facts that Table 4 breaks down.

Why this is usually free
------------------------
TopicGPT runs over facts from the GOLD note, and CREOLA classifies the gold
note once per encounter (the gold cache holds 153 files, one per encounter,
shared by every system). Adding a system therefore adds no new fact strings --
only new occurrences of facts already in the pool. Verified when JSL was added
to ENCOMPASS: 435 unique facts with it, 435 without.

So this script:
  1. rebuilds the pooled CREOLA file from both corpora,
  2. checks every uncategorized gold fact against the saved fact_to_topic map,
  3. assigns ONLY facts that lack a topic, against the FIXED existing topic
     list (no re-induction -- topics must stay stable across the paper),
  4. re-tallies occurrence counts, which is what actually changes.

If step 2 finds nothing missing, no LLM calls are made at all.

Usage:
  python refresh_pooled_topics.py --check          # report only
  python refresh_pooled_topics.py --write          # assign + re-tally
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

CORPORA = ["encompass", "adapts"]
POOLED = Path("metrics_test/results_pooled/14_creola_category_gaps_pooled.json")
TOPICS = Path("metrics_test/results_pooled/15_uncategorized_topic_model.json")
UNCAT = "uncategorized_by_creola"


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_pooled():
    out = {"metadata": {"script": "refresh_pooled_topics.py",
                        "note": "pooled CREOLA gaps; model keys are corpus-suffixed so "
                                "there are no collisions between corpora",
                        "sources": {}},
           "models": {}}
    for c in CORPORA:
        p = f"metrics_test/results_{c}/creola/14_creola_category_gaps.json"
        d = load(p)
        if not d:
            print(f"  MISSING {p} -- run CREOLA for {c} first")
            return None
        out["metadata"]["sources"][c] = {"path": p, "metadata": d.get("metadata", {})}
        for k, v in d["models"].items():
            assert k not in out["models"], f"key collision: {k}"
            out["models"][k] = v
    return out


def uncategorized_facts(pooled):
    """fact -> occurrence count across all system/encounter pairs."""
    counts = Counter()
    for m in pooled["models"].values():
        for item in m["per_item"]:
            for f in item.get("gold_fact_classes", []):
                if f.get("category") == UNCAT:
                    counts[f["fact"]] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = ap.parse_args()

    pooled = build_pooled()
    if pooled is None:
        return 1

    counts = uncategorized_facts(pooled)
    topics = load(TOPICS)
    if not topics:
        print(f"  {TOPICS} missing -- run topic_model_uncategorized.py first")
        return 1

    assigned = topics.get("fact_to_topic", {})
    missing = sorted(set(counts) - set(assigned))

    print(f"  systems pooled        : {len(pooled['models'])}")
    print(f"  uncategorized facts   : {len(counts)} unique, "
          f"{sum(counts.values())} occurrences")
    print(f"  already have a topic  : {len(counts) - len(missing)}")
    print(f"  NEED assignment       : {len(missing)}")
    for f in missing[:10]:
        print(f"     {f[:100]!r}")
    if len(missing) > 10:
        print(f"     ... and {len(missing) - 10} more")

    if args.check:
        print("\n  --check only; nothing written.")
        return 0

    POOLED.parent.mkdir(parents=True, exist_ok=True)
    POOLED.write_text(json.dumps(pooled, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  wrote {POOLED}")

    if missing:
        print(f"\n  {len(missing)} facts need assigning to the EXISTING topic list.")
        print("  Run topic_model_uncategorized.py --assign_only if available, or")
        print("  re-run it fully and diff the topic list before accepting new topics.")
        return 2

    # Re-tally: occurrence counts change when a system is added, topics do not.
    per_topic_occ = Counter()
    per_topic_uniq = Counter()
    for fact, n in counts.items():
        t = assigned[fact]
        per_topic_occ[t] += n
        per_topic_uniq[t] += 1

    topics["topic_counts_all_occurrences"] = dict(sorted(
        per_topic_occ.items(), key=lambda kv: -kv[1]))
    topics["topic_counts_unique_facts"] = dict(sorted(
        per_topic_uniq.items(), key=lambda kv: -kv[1]))
    topics.setdefault("metadata", {})["retallied_by"] = "refresh_pooled_topics.py"
    topics["metadata"]["unique_facts"] = len(counts)
    topics["metadata"]["total_occurrences"] = sum(counts.values())
    TOPICS.write_text(json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  re-tallied {TOPICS}")
    print(f"  topics unchanged: {len(topics.get('final_topics', []))}")
    print("\n  top 8 topics by occurrences:")
    for t, n in list(topics["topic_counts_all_occurrences"].items())[:8]:
        print(f"     {n:6d}  {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
