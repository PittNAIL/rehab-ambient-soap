#!/usr/bin/env python3
"""
creola_category_gap_analysis.py

Classifies the already-extracted, FactEHR-verified facts from
11_factehr_exact_claims.json using the taxonomies from:

  Asgari E, Montana-Brown N, Dubois M, Khalil S, Balloch J, Au Yeung J, Pimenta D.
  "A framework to assess clinical safety and hallucination rates of LLMs for medical
  text summarisation." npj Digital Medicine 8, 274 (2025).
  doi:10.1038/s41746-025-01670-7

This is the same paper category_severity_gap_analysis.py already cites for its severity
definition (major/minor, kept unchanged here). Its category taxonomy was not previously
adopted by this project; this script adopts it. The paper actually defines THREE axes,
applied across two error types:

  - category  : the paper's 4-label CONTENT-CATEGORY taxonomy (current_issues, pmfs,
                past medical/medication/family/social history, information_and_plan,
                plus uncategorized_by_creola, added here, see below) -- verbatim from
                the paper's omission-classification methods section. Applied to BOTH
                gold facts (missed_by_category) AND pred facts (hallucinated_by_category)
                so the two are directly comparable -- the paper itself only applied this
                axis to omissions, so cross-applying it to hallucinated pred facts is an
                explicit extension beyond the paper, not a literal replication.
  - type      : the paper's 5-label HALLUCINATION-MECHANISM taxonomy (fabrication,
                negation, causality, contextual, plus uncategorized_by_creola).
                Pred facts only -- a fact that was never generated has no "mechanism."
                Describes HOW a hallucination happened, not its content area -- a
                different axis from category, kept separate, not merged.
  - location  : the paper's hallucination-LOCATION axis (which note section a
                hallucination falls in). The paper used 6 labels tied to its own note
                template (Main history, Examination, Discussion, Symptoms, Assessment,
                Plan); this project instead reuses its OWN existing real (non-LLM)
                S/O/A/P header detector (gap_analysis.py, _split_sections, ported below)
                since two of our seven models (healthscribe_summaries, gpt5-pro_soap_preds)
                reliably emit real SOAP headers (confirmed via 2_soap_section_gaps.json:
                100% and 87% presence respectively). The other five essentially never do
                (0-13%), so location is "no_header" for their facts rather than an
                LLM-inferred guess -- zero added noise, an honest coverage gap instead of
                a fabricated one. Computed via lexical (word-Jaccard) matching against
                each detected section's text, NOT an LLM call.
  - severity  : major / minor, unchanged from category_severity_gap_analysis.py (already
                literature-sourced from this same paper). Travels with whichever
                classification stream produced it (category-stream severity feeds
                missed_by_severity/hallucinated_by_severity; type-stream severity feeds
                hallucinated_type_x_severity and pairs with location for
                hallucinated_location_x_severity).

The category and type axes add one label NOT in the paper, "uncategorized_by_creola",
for facts that clearly don't fit -- in particular functional status, ADL performance, or
mobility content, since this taxonomy is derived from primary-care consultation
transcripts and has no functional/ADL/mobility category. This keeps the official
categories literature-faithful and reports the domain-coverage gap as its own explicit
metric instead of force-fitting rehab-note content into the wrong bucket.

The paper's 5x5 clinical-risk-consequence matrix is intentionally NOT adopted -- out of
scope; the existing binary major/minor severity is kept.

Methodology note -- simulated dual-annotation: the paper's own classification process
was 100% manual ("we require annotation by at least two clinicians for each input-output
pair", consolidated by "a senior clinician with over 20 years of clinical experience"
when the two disagree). No LLM or automated step was used in the paper for category/type.
This script simulates that process with LLM calls standing in for clinicians, for EACH of
the three LLM-based classification streams (gold-content, pred-content, pred-type): two
independent classification passes per note at temperature=0.7 (high enough that the two
passes can genuinely disagree, unlike the deterministic temperature=0.0 used elsewhere in
this pipeline), and when they disagree on category/type and/or severity for a fact, a
third adjudication pass at temperature=0.0, explicitly framed as a senior clinical
adjudicator, resolves it. This is an explicit LLM proxy for the paper's manual annotation
process, not a literal replication -- disclosed here the same way this project discloses
its other taxonomy/methodology adaptations (see meeting_update.md). Location uses zero
LLM calls by design (see above).

No new decomposition or entailment calls are made -- this reuses the same FactEHR-verified
fact lists already in 11_factehr_exact_claims.json.

Usage
-----
python creola_category_gap_analysis.py \
    --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
    --soap_dir    metrics_test/soap_prompts \
    --output_dir  metrics_test/gap_analysis \
    --keys        generation/keys_foundry.json \
    --model       gpt-4o \
    [--cache_dir  metrics_test/gap_analysis/.creola_category_cache] \
    [--force]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import openai


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


_CONTENT_CATEGORIES = [
    "current_issues", "pmfs", "information_and_plan", "uncategorized_by_creola",
]
_HALLUCINATION_TYPES = [
    "fabrication", "negation", "causality", "contextual", "uncategorized_by_creola",
]
_ANNOTATOR_TEMPERATURE = 0.7


# ── Hallucination-location utilities (ported from gap_analysis.py lines 328-378) ────
# Non-LLM, regex-based real SOAP-header detection -- duplicated here (rather than
# imported) to keep this script free of gap_analysis.py's unrelated heavier deps
# (nltk, requests, UMLS-fetching code).

_SOAP_SPLIT = re.compile(
    r"(?=(?:subjective|objective|assessment(?:\s+(?:and\s+)?plan)?|plan)\s*[:\-])",
    re.IGNORECASE,
)
_SOAP_HEADER = re.compile(
    r"^(subjective|objective|assessment(?:\s+(?:and\s+)?plan)?|plan)\s*[:\-]\s*",
    re.IGNORECASE,
)
_SECTION_LABEL = {
    "subjective": "S",
    "objective": "O",
    "assessment": "A",
    "assessment and plan": "AP",
    "assessment plan": "AP",
    "plan": "P",
}
_LOC_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "is", "was", "with", "for",
    "on", "at", "by", "from", "that", "this", "it", "he", "she", "his",
    "her", "they", "their", "be", "been", "are", "were", "has", "have",
    "had", "not", "no", "as", "or", "but", "if", "so", "do", "did", "does",
    "also", "then", "than", "when", "while", "which", "who", "what",
    "where", "how", "all", "any", "each", "there", "up", "out", "about",
    "into", "through", "before", "after", "more", "other", "some", "such",
    "only", "both", "very", "well", "will", "can", "could", "would",
}


def _loc_word_set(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in _LOC_STOPWORDS}


def _loc_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _split_sections(text: str) -> Dict[str, str]:
    chunks = _SOAP_SPLIT.split(text)
    sections: Dict[str, str] = {}
    for chunk in chunks:
        m = _SOAP_HEADER.match(chunk.strip())
        if m:
            header = m.group(1).strip().lower()
            label = _SECTION_LABEL.get(header, header[0].upper())
            content = chunk[m.end():].strip()
            if content:
                sections[label] = content
    return sections


def _locate_fact(fact: str, sections: Dict[str, str]) -> str:
    """Real (non-LLM) location: best lexical match among detected SOAP sections,
    or "no_header" if the note has no parseable headers at all."""
    if not sections:
        return "no_header"
    fact_words = _loc_word_set(fact)
    best_label, best_score = "no_header", -1.0
    for label, content in sections.items():
        score = _loc_jaccard(fact_words, _loc_word_set(content))
        if score > best_score:
            best_label, best_score = label, score
    return best_label


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYS = (
    "You are a clinical NLP expert classifying pre-extracted clinical facts. "
    "Always return valid JSON exactly as specified — no markdown, no explanation."
)

_SYS_ADJUDICATE = (
    "You are a senior clinical adjudicator with over 20 years of clinical experience, "
    "resolving disagreements between two independent annotators. Always return valid "
    "JSON exactly as specified — no markdown, no explanation."
)

_CLASSIFY_GOLD_USER = """\
Each fact below was already extracted from the clinical note. For each fact, assign:

1. category — exactly ONE of (per Asgari et al. 2025, npj Digital Medicine, content-category
   taxonomy):
   current_issues          – details about the current presentation
                              (Asgari et al. 2025: "details about the current presentation")
   pmfs                     – past medical history, medications (including allergies), family
                              history, and social history including drinking and smoking
                              (Asgari et al. 2025: "details about the past medical history,
                              medications including allergies, family and social history,
                              including drinking and smoking")
   information_and_plan    – discussions and explanations of the condition and management plans
                              (Asgari et al. 2025: "discussions and explanations of the condition
                              and management plans")
   uncategorized_by_creola  – use when a fact clearly fits none of the above — in particular,
                              functional status, ADL performance, or mobility content (transfers,
                              gait, balance, assistive devices). This taxonomy is derived from
                              primary-care notes and has no functional/ADL/mobility category, so
                              do not force such facts into the 3 categories above.

2. severity — exactly ONE of:
   major – this fact would impact the diagnosis or management of the patient if it were
           incorrect or omitted (e.g. medications, safety risks, diagnoses, vital abnormalities,
           falls/safety incidents)
   minor – anything else (e.g. routine observations, pleasantries, minor procedural detail) that
           would NOT change clinical decision-making if incorrect or omitted

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 4 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}

CLINICAL NOTE (for context):
{note}

FACTS TO CLASSIFY ({n} total):
{facts_json}"""

_CLASSIFY_PRED_CONTENT_USER = """\
Each fact below was already extracted from the model's predicted clinical note. For each fact,
assign:

1. category — exactly ONE of (per Asgari et al. 2025, npj Digital Medicine, content-category
   taxonomy — the SAME categories used for the human/gold note, so the two are directly
   comparable):
   current_issues          – details about the current presentation
                              (Asgari et al. 2025: "details about the current presentation")
   pmfs                     – past medical history, medications (including allergies), family
                              history, and social history including drinking and smoking
                              (Asgari et al. 2025: "details about the past medical history,
                              medications including allergies, family and social history,
                              including drinking and smoking")
   information_and_plan    – discussions and explanations of the condition and management plans
                              (Asgari et al. 2025: "discussions and explanations of the condition
                              and management plans")
   uncategorized_by_creola  – use when a fact clearly fits none of the above — in particular,
                              functional status, ADL performance, or mobility content (transfers,
                              gait, balance, assistive devices). This taxonomy is derived from
                              primary-care notes and has no functional/ADL/mobility category, so
                              do not force such facts into the 3 categories above.

2. severity — exactly ONE of:
   major – this fact would impact the diagnosis or management of the patient if it were
           incorrect or omitted (e.g. medications, safety risks, diagnoses, vital abnormalities,
           falls/safety incidents)
   minor – anything else (e.g. routine observations, pleasantries, minor procedural detail) that
           would NOT change clinical decision-making if incorrect or omitted

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 4 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}

CLINICAL NOTE (for context):
{note}

FACTS TO CLASSIFY ({n} total):
{facts_json}"""

_CLASSIFY_PRED_TYPE_USER = """\
Each fact below was already extracted from the model's predicted clinical note. For each fact,
assign:

1. category — exactly ONE of (per Asgari et al. 2025, npj Digital Medicine, hallucination-type
   taxonomy — this describes the MECHANISM of a hallucination, not a clinical content area):
   fabrication    – information produced that was not evidenced in the text
                    (Asgari et al. 2025: "information... not evidenced in the text")
   negation       – output negates a clinically relevant fact
                    (Asgari et al. 2025: "model output negates a clinically relevant fact")
   causality      – speculates the cause of a given condition without explicit support
                    (Asgari et al. 2025: "model speculates the cause of a given condition
                    without explicit support")
   contextual     – mixes topics otherwise not related to the given context
                    (Asgari et al. 2025: "model mixes topics otherwise not related to the
                    given context")
   uncategorized_by_creola – use when none of the 4 types above clearly fit.

2. severity — exactly ONE of:
   major – this fact would impact the diagnosis or management of the patient if it were
           incorrect or omitted (e.g. medications, safety risks, diagnoses, vital abnormalities,
           falls/safety incidents)
   minor – anything else (e.g. routine observations, pleasantries, minor procedural detail) that
           would NOT change clinical decision-making if incorrect or omitted

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 5 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}

CLINICAL NOTE (for context):
{note}

FACTS TO CLASSIFY ({n} total):
{facts_json}"""

_ADJUDICATE_GOLD_USER = """\
Two independent annotators classified facts from the clinical note below into the content-category
taxonomy from Asgari et al. 2025 (npj Digital Medicine):
   current_issues          – details about the current presentation
   pmfs                     – past medical history, medications (including allergies), family
                              history, and social history including drinking and smoking
   information_and_plan    – discussions and explanations of the condition and management plans
   uncategorized_by_creola  – use for facts (e.g. functional/ADL/mobility status) this taxonomy
                              has no category for

They disagreed (on category and/or severity) for the facts listed below. As a senior clinical
adjudicator with over 20 years of clinical experience (mirroring the consolidation step in
Asgari et al. 2025), review each disagreement in the context of the note and decide the single
correct classification.

CLINICAL NOTE (for context):
{note}

DISAGREEMENTS ({n} total):
{disagreements_json}

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 4 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}"""

_ADJUDICATE_PRED_CONTENT_USER = """\
Two independent annotators classified facts from the model's predicted clinical note below into
the content-category taxonomy from Asgari et al. 2025 (npj Digital Medicine) — the SAME
categories used for the human/gold note:
   current_issues          – details about the current presentation
   pmfs                     – past medical history, medications (including allergies), family
                              history, and social history including drinking and smoking
   information_and_plan    – discussions and explanations of the condition and management plans
   uncategorized_by_creola  – use for facts (e.g. functional/ADL/mobility status) this taxonomy
                              has no category for

They disagreed (on category and/or severity) for the facts listed below. As a senior clinical
adjudicator with over 20 years of clinical experience (mirroring the consolidation step in
Asgari et al. 2025), review each disagreement in the context of the note and decide the single
correct classification.

CLINICAL NOTE (for context):
{note}

DISAGREEMENTS ({n} total):
{disagreements_json}

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 4 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}"""

_ADJUDICATE_PRED_TYPE_USER = """\
Two independent annotators classified facts from the model's predicted clinical note below into
the hallucination-type taxonomy from Asgari et al. 2025 (npj Digital Medicine):
   fabrication    – information produced that was not evidenced in the text
   negation       – output negates a clinically relevant fact
   causality      – speculates the cause of a given condition without explicit support
   contextual     – mixes topics otherwise not related to the given context
   uncategorized_by_creola – use when none of the 4 types above clearly fit

They disagreed (on category and/or severity) for the facts listed below. As a senior clinical
adjudicator with over 20 years of clinical experience (mirroring the consolidation step in
Asgari et al. 2025), review each disagreement in the context of the note and decide the single
correct classification.

CLINICAL NOTE (for context):
{note}

DISAGREEMENTS ({n} total):
{disagreements_json}

Return ONLY valid JSON with exactly one entry per fact, in the same order:
{{
  "classifications": [
    {{"fact": "<exact fact text>", "category": "<one of the 5 labels>",
      "severity": "major" or "minor", "severity_reason": "<one sentence>"}},
    ...
  ]
}}"""


# ── LLM wrapper ────────────────────────────────────────────────────────────────

def _call_llm(
    client: openai.OpenAI,
    user_prompt: str,
    model: str,
    system_prompt: str = _SYS,
    temperature: float = 0.0,
    max_retries: int = 4,
) -> Optional[str]:
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
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


def _classify_once(
    note: str,
    facts: List[str],
    prompt_template: str,
    client: openai.OpenAI,
    model: str,
    temperature: float,
) -> Optional[List[Dict[str, str]]]:
    prompt = prompt_template.format(
        note=note, n=len(facts),
        facts_json=json.dumps(facts, indent=2, ensure_ascii=False),
    )
    raw = _call_llm(client, prompt, model, system_prompt=_SYS, temperature=temperature)
    if raw is None:
        return None
    try:
        return json.loads(raw).get("classifications", [])
    except json.JSONDecodeError:
        return None


def _adjudicate(
    note: str,
    disputed: List[Dict[str, Any]],
    adjudicate_template: str,
    client: openai.OpenAI,
    model: str,
) -> Optional[List[Dict[str, str]]]:
    prompt = adjudicate_template.format(
        note=note, n=len(disputed),
        disagreements_json=json.dumps(disputed, indent=2, ensure_ascii=False),
    )
    raw = _call_llm(client, prompt, model, system_prompt=_SYS_ADJUDICATE, temperature=0.0)
    if raw is None:
        return None
    try:
        return json.loads(raw).get("classifications", [])
    except json.JSONDecodeError:
        return None


def _build_lookup(classifications: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {c["fact"]: c for c in classifications if "fact" in c}


def classify_facts_dual(
    note: str,
    facts: List[str],
    cache_path: Path,
    client: openai.OpenAI,
    model: str,
    classify_template: str,
    adjudicate_template: str,
    force: bool,
) -> Optional[Dict[str, Any]]:
    """Simulates Asgari et al. 2025's two-annotator + senior-adjudicator process.

    Two independent LLM passes at temperature=_ANNOTATOR_TEMPERATURE stand in for the
    two clinicians; a third pass at temperature=0.0, framed as the senior clinician,
    resolves any fact where the two passes disagree on category and/or severity.
    """
    if not force and cache_path.exists():
        return load_json(cache_path)
    if not facts:
        result = {"final": [], "annotator_1": [], "annotator_2": [], "disagreements": []}
        save_json(result, cache_path)
        return result

    ann1 = _classify_once(note, facts, classify_template, client, model, _ANNOTATOR_TEMPERATURE)
    ann2 = _classify_once(note, facts, classify_template, client, model, _ANNOTATOR_TEMPERATURE)
    if ann1 is None or ann2 is None:
        return None

    lookup1, lookup2 = _build_lookup(ann1), _build_lookup(ann2)
    final_lookup: Dict[str, Dict[str, str]] = {}
    disputed: List[Dict[str, Any]] = []
    for f in facts:
        c1, c2 = lookup1.get(f, {}), lookup2.get(f, {})
        if c1.get("category") == c2.get("category") and c1.get("severity") == c2.get("severity"):
            final_lookup[f] = c1
        else:
            disputed.append({
                "fact": f,
                "annotator_1": {"category": c1.get("category"), "severity": c1.get("severity"),
                                 "severity_reason": c1.get("severity_reason")},
                "annotator_2": {"category": c2.get("category"), "severity": c2.get("severity"),
                                 "severity_reason": c2.get("severity_reason")},
            })

    if disputed:
        adjudicated = _adjudicate(note, disputed, adjudicate_template, client, model)
        if adjudicated is None:
            return None
        for c in adjudicated:
            if "fact" in c:
                final_lookup[c["fact"]] = c

    final = [final_lookup.get(f, {"fact": f, "category": "other", "severity": "minor"}) for f in facts]

    result = {
        "final": final,
        "annotator_1": ann1,
        "annotator_2": ann2,
        "disagreements": [d["fact"] for d in disputed],
    }
    save_json(result, cache_path)
    return result


# ── Per-model analysis ───────────────────────────────────────────────────────────

def analyse_model(
    model_name: str,
    per_item_claims: List[Dict[str, Any]],
    soap_lookup: Dict[str, Dict[str, str]],
    gold_cache_dir: Path,
    pred_content_cache_dir: Path,
    pred_type_cache_dir: Path,
    client: openai.OpenAI,
    model: str,
    force: bool,
) -> Dict[str, Any]:
    pred_content_cache_dir.mkdir(parents=True, exist_ok=True)
    pred_type_cache_dir.mkdir(parents=True, exist_ok=True)

    missed_by_cat: Dict[str, int] = defaultdict(int)
    hall_by_cat: Dict[str, int] = defaultdict(int)
    hall_by_type: Dict[str, int] = defaultdict(int)
    hall_by_loc: Dict[str, int] = defaultdict(int)
    gold_total_by_cat: Dict[str, int] = defaultdict(int)
    pred_total_by_cat: Dict[str, int] = defaultdict(int)
    pred_total_by_type: Dict[str, int] = defaultdict(int)
    pred_total_by_loc: Dict[str, int] = defaultdict(int)
    missed_by_sev: Dict[str, int] = defaultdict(int)
    hall_by_sev: Dict[str, int] = defaultdict(int)
    missed_cat_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hall_cat_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hall_type_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    hall_loc_sev: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_item_out: List[Dict[str, Any]] = []
    agreement_count = 0
    agreement_total = 0

    for item in per_item_claims:
        idx = item.get("idx")
        if item.get("degenerate_pred_output") or "error" in item or idx not in soap_lookup:
            continue

        gold_note = soap_lookup[idx]["gold_note"]
        pred_note = soap_lookup[idx]["pred_note"]
        gold_facts = item.get("gold_facts", [])
        pred_facts = item.get("pred_facts", [])
        hallucinated_facts = item.get("hallucinated_pred_facts", [])

        print(f"  [{idx}]", end=" ", flush=True)

        gold_result = classify_facts_dual(
            gold_note, gold_facts, gold_cache_dir / f"{idx}.json",
            client, model, _CLASSIFY_GOLD_USER, _ADJUDICATE_GOLD_USER, force,
        )
        pred_content_result = classify_facts_dual(
            pred_note, pred_facts, pred_content_cache_dir / f"{idx}.json",
            client, model, _CLASSIFY_PRED_CONTENT_USER, _ADJUDICATE_PRED_CONTENT_USER, force,
        )
        # Type (hallucination mechanism) only makes sense for facts actually flagged as
        # hallucinated -- asking "which mechanism" about a true, grounded fact has no good
        # answer, which is why an earlier run scoped to ALL pred_facts produced a meaningless
        # majority "uncategorized_by_creola" (the LLM correctly recognizing it didn't apply).
        pred_type_result = classify_facts_dual(
            pred_note, hallucinated_facts, pred_type_cache_dir / f"{idx}.json",
            client, model, _CLASSIFY_PRED_TYPE_USER, _ADJUDICATE_PRED_TYPE_USER, force,
        )
        if gold_result is None or pred_content_result is None or pred_type_result is None:
            print("SKIP (classification failed)")
            continue

        gold_classes = gold_result["final"]
        pred_content_classes = pred_content_result["final"]
        pred_type_classes = pred_type_result["final"]
        agreement_total += len(gold_facts) + len(pred_facts) + len(hallucinated_facts)
        agreement_count += (
            (len(gold_facts) - len(gold_result["disagreements"]))
            + (len(pred_facts) - len(pred_content_result["disagreements"]))
            + (len(hallucinated_facts) - len(pred_type_result["disagreements"]))
        )

        gold_lookup = _build_lookup(gold_classes)
        pred_content_lookup = _build_lookup(pred_content_classes)
        pred_type_lookup = _build_lookup(pred_type_classes)

        sections = _split_sections(pred_note)
        loc_lookup = {f: _locate_fact(f, sections) for f in pred_facts}

        for f in gold_facts:
            c = gold_lookup.get(f, {})
            gold_total_by_cat[c.get("category", "other")] += 1
        for f in pred_facts:
            c = pred_content_lookup.get(f, {})
            pred_total_by_cat[c.get("category", "other")] += 1
            pred_total_by_loc[loc_lookup[f]] += 1

        for f in item.get("missed_gold_facts", []):
            c = gold_lookup.get(f, {})
            cat, sev = c.get("category", "other"), c.get("severity", "minor")
            missed_by_cat[cat] += 1
            missed_by_sev[sev] += 1
            missed_cat_sev[cat][sev] += 1

        for f in hallucinated_facts:
            c = pred_content_lookup.get(f, {})
            cat, sev = c.get("category", "other"), c.get("severity", "minor")
            hall_by_cat[cat] += 1
            hall_by_sev[sev] += 1
            hall_cat_sev[cat][sev] += 1

            t = pred_type_lookup.get(f, {})
            typ, type_sev = t.get("category", "other"), t.get("severity", "minor")
            hall_by_type[typ] += 1
            pred_total_by_type[typ] += 1
            hall_type_sev[typ][type_sev] += 1

            loc = loc_lookup.get(f, "no_header")
            hall_by_loc[loc] += 1
            hall_loc_sev[loc][type_sev] += 1

        per_item_out.append({
            "idx": idx,
            "gold_fact_classes": gold_classes,
            "pred_content_fact_classes": pred_content_classes,
            "pred_type_fact_classes": pred_type_classes,
            "fact_locations": loc_lookup,
            "gold_disagreements": gold_result["disagreements"],
            "pred_content_disagreements": pred_content_result["disagreements"],
            "pred_type_disagreements": pred_type_result["disagreements"],
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
        "hallucinated_by_type": dict(sorted(
            _cat_summary(hall_by_type, pred_total_by_type, total_hall).items(),
            key=lambda kv: -kv[1]["count"],
        )),
        "hallucinated_by_location": dict(sorted(
            _cat_summary(hall_by_loc, pred_total_by_loc, total_hall).items(),
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
        "hallucinated_type_x_severity": {typ: dict(sevs) for typ, sevs in hall_type_sev.items()},
        "hallucinated_location_x_severity": {loc: dict(sevs) for loc, sevs in hall_loc_sev.items()},
        "total_missed": total_missed,
        "total_hallucinated": total_hall,
        "agreement_rate": round(agreement_count / agreement_total, 3) if agreement_total else None,
    }

    return {"summary": summary, "per_item": per_item_out}


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CREOLA (Asgari et al. 2025) category/type/location + severity classification "
                     "of FactEHR-exact facts, via simulated dual-annotator + adjudication."
    )
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
    cache_root = Path(args.cache_dir) if args.cache_dir else output_dir / ".creola_category_cache"
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
            pred_content_cache_dir=pred_cache_root / model_name / "content",
            pred_type_cache_dir=pred_cache_root / model_name / "type",
            client=client,
            model=args.model,
            force=args.force,
        )
        all_results[model_name] = result
        s = result["summary"]
        print(f"\n  missed={s['total_missed']} hallucinated={s['total_hallucinated']} "
              f"agreement_rate={s['agreement_rate']}")

    output = {
        "metadata": {
            "script": "creola_category_gap_analysis.py",
            "source": str(Path(args.claims_json)),
            "category_taxonomy": (
                "Three axes from Asgari E, Montana-Brown N, Dubois M, Khalil S, Balloch J, "
                "Au Yeung J, Pimenta D. 'A framework to assess clinical safety and "
                "hallucination rates of LLMs for medical text summarisation.' npj Digital "
                "Medicine 8, 274 (2025). doi:10.1038/s41746-025-01670-7. "
                "missed_by_category AND hallucinated_by_category share the SAME 4-label "
                "content-category scheme (current_issues, pmfs, information_and_plan, "
                "uncategorized_by_creola) verbatim from the paper's omission-classification "
                "methods -- cross-applying it to hallucinated pred facts is an explicit "
                "extension beyond the paper, not a literal replication (the paper itself "
                "only applied this axis to omissions). hallucinated_by_type uses the "
                "paper's separate 5-label HALLUCINATION-MECHANISM taxonomy (fabrication, "
                "negation, causality, contextual, uncategorized_by_creola) -- pred facts "
                "only, describes how a hallucination happened, not its content area. "
                "hallucinated_by_location is derived NON-LLM via a real regex-based SOAP "
                "header detector (ported from gap_analysis.py's _split_sections), labeled "
                "'no_header' for the 5 of 7 models that don't emit parseable SOAP headers "
                "(only healthscribe_summaries and gpt5-pro_soap_preds reliably do, per "
                "2_soap_section_gaps.json) -- a deliberate substitute for the paper's own "
                "6-label template-specific location axis, using this project's actual "
                "available structure instead of LLM-inferring section boundaries. Both "
                "category and type axes add one label not in the paper, "
                "uncategorized_by_creola, for facts (notably functional/ADL/mobility "
                "status) that this primary-care-derived taxonomy has no category for. "
                "Methodology note: the paper's own category/type classification was 100% "
                "manual (two clinicians annotating each pair, with a senior clinician of "
                "20+ years experience adjudicating disagreements). This script simulates "
                "that process for each LLM-based stream (gold-content, pred-content, "
                "pred-type): two independent LLM annotation passes per note at "
                "temperature=0.7 (so they can genuinely disagree), with a third LLM "
                "adjudication pass at temperature=0.0 resolving any disagreement -- an "
                "explicit LLM proxy for the paper's manual annotation, not a literal "
                "replication of it. Location uses zero LLM calls by design. Finding, not "
                "a bug: hallucinated_by_type is dominated by uncategorized_by_creola "
                "(~75%+). This is consistent with hallucination_grounding_analysis.py's "
                "prior result that ~77% of facts this pipeline flags as 'hallucinated' "
                "(not entailed by gold_note) are actually grounded in the full "
                "conversation transcript -- omitted by the human note-writer, not "
                "fabricated by the model. The paper's 4 mechanism types (fabrication/"
                "negation/causality/contextual) presuppose genuine unfaithfulness, so the "
                "classifier correctly declines to assign one to most of these facts; this "
                "is itself a reportable result, not a classification failure."
            ),
            "severity_definition": "npj Digital Medicine 2025 framework: major = impacts diagnosis/management if uncorrected",
            "judge_model": args.model,
            "annotator_temperature": _ANNOTATOR_TEMPERATURE,
        },
        "models": all_results,
    }
    out_path = output_dir / "14_creola_category_gaps.json"
    save_json(output, out_path)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
