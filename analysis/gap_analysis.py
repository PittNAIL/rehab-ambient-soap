#!/usr/bin/env python3
"""
gap_analysis.py

Six analyses to identify what is missing or hallucinated in AI-generated
SOAP notes compared to human-written gold notes.

  1. Semantic-type stratification  (requires --umls_api_key; skipped otherwise)
  2. SOAP-section-level gap analysis
  3. Top-N most frequently missed CUIs per model + cross-model consensus
  4. Top-N most hallucinated CUIs per model + cross-model consensus
  5. Negation-aware gap analysis (lightweight NegEx-style, no external deps)
  6. Numerical / dosage / temporal gap analysis (regex-based)

Usage
-----
python gap_analysis.py \
    --soap_dir     metrics_test/soap_prompts \
    --coverage_dir metrics_test/metric_reports \
    --output_dir   metrics_test/gap_analysis \
    [--umls_api_key <key>] \
    [--top_n 20]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import nltk
import requests


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def coverage_stem(filename: str) -> str:
    """'gpt5-pro_soap_preds_cui_coverage.json' -> 'gpt5-pro_soap_preds'"""
    return re.sub(r"_cui_coverage\.(json|jsonl)$", "", filename)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 1 – UMLS semantic-type stratification
# ─────────────────────────────────────────────────────────────────────────────

UMLS_BASE = "https://uts-ws.nlm.nih.gov/rest"

# Maps UMLS semantic type full name -> one of 15 Semantic Groups
# Source: NLM Semantic Groups (McCray et al. 2001)
SEMTYPE_TO_GROUP: Dict[str, str] = {
    # Activities & Behaviors
    "Daily or Recreational Activity": "Activities & Behaviors",
    "Event": "Activities & Behaviors",
    "Governmental or Regulatory Activity": "Activities & Behaviors",
    "Individual Behavior": "Activities & Behaviors",
    "Machine Activity": "Activities & Behaviors",
    "Molecular Function": "Activities & Behaviors",
    "Occupational Activity": "Activities & Behaviors",
    "Social Behavior": "Activities & Behaviors",
    # Anatomy
    "Body Location or Region": "Anatomy",
    "Body Part, Organ, or Organ Component": "Anatomy",
    "Body Space or Junction": "Anatomy",
    "Body Substance": "Anatomy",
    "Body System": "Anatomy",
    "Cell": "Anatomy",
    "Cell Component": "Anatomy",
    "Embryonic Structure": "Anatomy",
    "Fully Formed Anatomical Structure": "Anatomy",
    "Tissue": "Anatomy",
    # Chemicals & Drugs
    "Antibiotic": "Chemicals & Drugs",
    "Biologically Active Substance": "Chemicals & Drugs",
    "Biomedical or Dental Material": "Chemicals & Drugs",
    "Chemical": "Chemicals & Drugs",
    "Chemical Viewed Functionally": "Chemicals & Drugs",
    "Chemical Viewed Structurally": "Chemicals & Drugs",
    "Clinical Drug": "Chemicals & Drugs",
    "Element, Ion, or Isotope": "Chemicals & Drugs",
    "Enzyme": "Chemicals & Drugs",
    "Hazardous or Poisonous Substance": "Chemicals & Drugs",
    "Hormone": "Chemicals & Drugs",
    "Immunologic Factor": "Chemicals & Drugs",
    "Indicator, Reagent, or Diagnostic Aid": "Chemicals & Drugs",
    "Inorganic Chemical": "Chemicals & Drugs",
    "Lipid": "Chemicals & Drugs",
    "Nucleic Acid, Nucleoside, or Nucleotide": "Chemicals & Drugs",
    "Organic Chemical": "Chemicals & Drugs",
    "Organophosphorus Compound": "Chemicals & Drugs",
    "Pharmacologic Substance": "Chemicals & Drugs",
    "Receptor": "Chemicals & Drugs",
    "Steroid": "Chemicals & Drugs",
    "Vitamin": "Chemicals & Drugs",
    # Devices
    "Drug Delivery Device": "Devices",
    "Medical Device": "Devices",
    "Research Device": "Devices",
    # Disorders
    "Acquired Abnormality": "Disorders",
    "Anatomical Abnormality": "Disorders",
    "Cell or Molecular Dysfunction": "Disorders",
    "Congenital Abnormality": "Disorders",
    "Disease or Syndrome": "Disorders",
    "Experimental Model of Disease": "Disorders",
    "Finding": "Disorders",
    "Injury or Poisoning": "Disorders",
    "Mental or Behavioral Dysfunction": "Disorders",
    "Neoplastic Process": "Disorders",
    "Pathologic Function": "Disorders",
    "Sign or Symptom": "Disorders",
    # Genes & Molecular Sequences
    "Amino Acid, Peptide, or Protein": "Genes & Molecular Sequences",
    "Nucleotide Sequence": "Genes & Molecular Sequences",
    "Gene or Genome": "Genes & Molecular Sequences",
    # Geographic Areas
    "Geographic Area": "Geographic Areas",
    # Living Beings
    "Age Group": "Living Beings",
    "Animal": "Living Beings",
    "Archaeon": "Living Beings",
    "Bacterium": "Living Beings",
    "Eukaryote": "Living Beings",
    "Fungus": "Living Beings",
    "Human": "Living Beings",
    "Organism": "Living Beings",
    "Patient or Disabled Group": "Living Beings",
    "Plant": "Living Beings",
    "Population Group": "Living Beings",
    "Professional or Occupational Group": "Living Beings",
    "Reptile": "Living Beings",
    "Virus": "Living Beings",
    # Objects
    "Constructed or Altered Territory": "Objects",
    "Food": "Objects",
    "Manufactured Object": "Objects",
    "Physical Object": "Objects",
    "Substance": "Objects",
    # Occupations
    "Occupation or Discipline": "Occupations",
    # Organizations
    "Health Care Related Organization": "Organizations",
    "Organization": "Organizations",
    "Professional Society": "Organizations",
    "Self-help or Relief Organization": "Organizations",
    # Phenomena
    "Biologic Function": "Phenomena",
    "Environmental Effect of Humans": "Phenomena",
    "Human-caused Phenomenon or Process": "Phenomena",
    "Laboratory or Test Result": "Phenomena",
    "Natural Phenomenon or Process": "Phenomena",
    "Phenomenon or Process": "Phenomena",
    # Physiology
    "Cell Function": "Physiology",
    "Genetic Function": "Physiology",
    "Mental Process": "Physiology",
    "Organ or Tissue Function": "Physiology",
    "Organism Function": "Physiology",
    "Physiologic Function": "Physiology",
    # Procedures
    "Diagnostic Procedure": "Procedures",
    "Educational Activity": "Procedures",
    "Health Care Activity": "Procedures",
    "Laboratory Procedure": "Procedures",
    "Molecular Biology Research Technique": "Procedures",
    "Research Activity": "Procedures",
    "Therapeutic or Preventive Procedure": "Procedures",
    # Concepts & Ideas
    "Classification": "Concepts & Ideas",
    "Conceptual Entity": "Concepts & Ideas",
    "Functional Concept": "Concepts & Ideas",
    "Group Attribute": "Concepts & Ideas",
    "Idea or Concept": "Concepts & Ideas",
    "Intellectual Product": "Concepts & Ideas",
    "Language": "Concepts & Ideas",
    "Qualitative Concept": "Concepts & Ideas",
    "Quantitative Concept": "Concepts & Ideas",
    "Regulation or Law": "Concepts & Ideas",
    "Spatial Concept": "Concepts & Ideas",
    "Temporal Concept": "Concepts & Ideas",
}


class UMLSLookup:
    """UMLS REST API wrapper with in-memory cache."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._cache: Dict[str, Optional[Dict]] = {}

    def _fetch(self, cui: str, retries: int = 3) -> Optional[Dict]:
        url = f"{UMLS_BASE}/content/current/CUI/{cui}"
        for attempt in range(retries):
            try:
                r = requests.get(url, params={"apiKey": self.api_key}, timeout=10)
                if r.status_code == 200:
                    return r.json().get("result")
                if r.status_code == 404:
                    return None
                time.sleep(0.5 * (attempt + 1))
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return None

    def concept(self, cui: str) -> Optional[Dict]:
        if cui not in self._cache:
            self._cache[cui] = self._fetch(cui)
            time.sleep(0.05)  # gentle rate-limit
        return self._cache[cui]

    def name(self, cui: str) -> str:
        c = self.concept(cui)
        return c.get("name", cui) if c else cui

    def semtypes(self, cui: str) -> List[str]:
        c = self.concept(cui)
        if not c:
            return ["unknown"]
        return [st.get("name", "unknown") for st in c.get("semanticTypes", [])] or ["unknown"]

    def semgroup(self, cui: str) -> str:
        sts = self.semtypes(cui)
        return SEMTYPE_TO_GROUP.get(sts[0], "Other") if sts else "Other"


def _missed_hallu(items: List[Dict]) -> tuple[Counter, Counter]:
    """Return (missed_counter, hallucinated_counter) from coverage items."""
    missed: Counter = Counter()
    hallu: Counter = Counter()
    for item in items:
        gold = set(item.get("gold_cuis", []))
        pred = set(item.get("pred_cuis", []))
        common = set(item.get("common_cuis", []))
        for c in gold - common:
            missed[c] += 1
        for c in pred - common:
            hallu[c] += 1
    return missed, hallu


def run_analysis_1(
    coverage_files: List[Path],
    umls: UMLSLookup,
    top_n: int,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    # cross-model counters
    global_missed: Counter = Counter()
    global_hallu: Counter = Counter()

    for cov_path in coverage_files:
        model = coverage_stem(cov_path.name)
        data = load_json(cov_path)
        items = data.get("per_item", [])
        missed, hallu = _missed_hallu(items)

        global_missed.update(missed)
        global_hallu.update(hallu)

        # fetch semtypes for unique CUIs (up to top_n * 5 to fill each group)
        n_fetch = top_n * 5
        print(f"    fetching semtypes for {min(n_fetch, len(missed))} missed CUIs in [{model}]...")

        missed_by_group: Dict[str, List] = defaultdict(list)
        for cui, freq in missed.most_common(n_fetch):
            entry = {
                "cui": cui,
                "name": umls.name(cui),
                "semtypes": umls.semtypes(cui),
                "freq_missed": freq,
                "pct_examples": round(freq / len(items), 3) if items else 0,
            }
            missed_by_group[umls.semgroup(cui)].append(entry)

        hallu_by_group: Dict[str, List] = defaultdict(list)
        for cui, freq in hallu.most_common(n_fetch):
            entry = {
                "cui": cui,
                "name": umls.name(cui),
                "semtypes": umls.semtypes(cui),
                "freq_hallucinated": freq,
                "pct_examples": round(freq / len(items), 3) if items else 0,
            }
            hallu_by_group[umls.semgroup(cui)].append(entry)

        results[model] = {
            "missed_group_counts": {g: len(v) for g, v in missed_by_group.items()},
            "missed_by_semantic_group": dict(missed_by_group),
            "hallucinated_group_counts": {g: len(v) for g, v in hallu_by_group.items()},
            "hallucinated_by_semantic_group": dict(hallu_by_group),
        }

    # Cross-model aggregate
    num_models = len(coverage_files)
    agg_missed_by_group: Dict[str, List] = defaultdict(list)
    for cui, freq in global_missed.most_common(top_n * 3):
        agg_missed_by_group[umls.semgroup(cui)].append(
            {"cui": cui, "name": umls.name(cui), "total_freq": freq,
             "semtypes": umls.semtypes(cui)}
        )

    results["_aggregate_across_models"] = {
        "missed_group_counts": {g: len(v) for g, v in agg_missed_by_group.items()},
        "missed_by_semantic_group": dict(agg_missed_by_group),
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2 – SOAP-section gap analysis
# ─────────────────────────────────────────────────────────────────────────────

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
_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "is", "was", "with", "for",
    "on", "at", "by", "from", "that", "this", "it", "he", "she", "his",
    "her", "they", "their", "be", "been", "are", "were", "has", "have",
    "had", "not", "no", "as", "or", "but", "if", "so", "do", "did", "does",
    "also", "then", "than", "when", "while", "which", "who", "what",
    "where", "how", "all", "any", "each", "there", "up", "out", "about",
    "into", "through", "before", "after", "more", "other", "some", "such",
    "only", "both", "very", "well", "will", "can", "could", "would",
}


def _word_set(text: str) -> Set[str]:
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {t for t in tokens if t not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
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


def run_analysis_2(soap_files: List[Path]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for soap_path in soap_files:
        model = soap_path.stem
        data = load_json(soap_path)

        # Running totals for presence + jaccard
        presence: Dict[str, int] = defaultdict(int)
        jaccard_acc: Dict[str, List[float]] = defaultdict(list)
        per_item = []

        for item in data:
            pred = item.get("pred_note", "") or ""
            gold = item.get("gold_note", "") or ""
            gold_words = _word_set(gold)

            sections = _split_sections(pred)
            found_labels = list(sections.keys())

            item_entry: Dict[str, Any] = {
                "idx": item.get("idx"),
                "sections_present": found_labels,
                "gold_word_count": len(gold.split()),
                "pred_word_count": len(pred.split()),
                "section_word_counts": {},
                "section_jaccard_vs_full_gold": {},
                "section_unique_words": {},
            }

            for label, content in sections.items():
                words = _word_set(content)
                j = _jaccard(words, gold_words)
                presence[label] += 1
                jaccard_acc[label].append(j)
                item_entry["section_word_counts"][label] = len(content.split())
                item_entry["section_jaccard_vs_full_gold"][label] = round(j, 4)
                item_entry["section_unique_words"][label] = len(words)

            per_item.append(item_entry)

        n = len(data)
        section_summary: Dict[str, Any] = {}
        for label in ["S", "O", "A", "P", "AP"]:
            jvals = jaccard_acc.get(label, [])
            section_summary[label] = {
                "presence_rate": round(presence.get(label, 0) / n, 3) if n else 0,
                "mean_jaccard_vs_full_gold": round(sum(jvals) / len(jvals), 4) if jvals else None,
                "n_with_section": presence.get(label, 0),
            }

        results[model] = {
            "total_examples": n,
            "section_summary": section_summary,
            "per_item": per_item,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3 – Top-N most frequently missed CUIs
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_3(
    coverage_files: List[Path],
    top_n: int,
    umls: Optional[UMLSLookup],
) -> Dict[str, Any]:
    per_model: Dict[str, Any] = {}
    # tracks how many models miss each CUI at least once
    models_missing_cui: Counter = Counter()

    for cov_path in coverage_files:
        model = coverage_stem(cov_path.name)
        data = load_json(cov_path)
        items = data.get("per_item", [])
        missed, _ = _missed_hallu(items)

        for cui in missed:
            models_missing_cui[cui] += 1

        top = []
        for cui, freq in missed.most_common(top_n):
            entry: Dict[str, Any] = {
                "cui": cui,
                "frequency_missed": freq,
                "pct_of_examples": round(freq / len(items), 3) if items else 0,
            }
            if umls:
                entry["name"] = umls.name(cui)
                entry["semtypes"] = umls.semtypes(cui)
                entry["semgroup"] = umls.semgroup(cui)
            top.append(entry)

        per_model[model] = {
            "total_examples": len(items),
            "total_unique_missed_cuis": len(missed),
            "top_missed_cuis": top,
        }

    # Consensus: CUIs missed by the majority of models
    num_models = len(coverage_files)
    threshold = max(2, num_models // 2)
    consensus: List[Dict] = []
    for cui, cnt in models_missing_cui.most_common(top_n):
        if cnt < threshold:
            break
        entry = {
            "cui": cui,
            "num_models_missing": cnt,
            "pct_models": round(cnt / num_models, 3),
        }
        if umls:
            entry["name"] = umls.name(cui)
            entry["semtypes"] = umls.semtypes(cui)
            entry["semgroup"] = umls.semgroup(cui)
        consensus.append(entry)

    return {
        "per_model": per_model,
        "consensus_missed_cuis": consensus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4 – Top-N hallucinated CUIs
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis_4(
    coverage_files: List[Path],
    top_n: int,
    umls: Optional[UMLSLookup],
) -> Dict[str, Any]:
    per_model: Dict[str, Any] = {}
    models_hallucinating_cui: Counter = Counter()

    for cov_path in coverage_files:
        model = coverage_stem(cov_path.name)
        data = load_json(cov_path)
        items = data.get("per_item", [])
        _, hallu = _missed_hallu(items)

        for cui in hallu:
            models_hallucinating_cui[cui] += 1

        top = []
        for cui, freq in hallu.most_common(top_n):
            entry: Dict[str, Any] = {
                "cui": cui,
                "frequency_hallucinated": freq,
                "pct_of_examples": round(freq / len(items), 3) if items else 0,
            }
            if umls:
                entry["name"] = umls.name(cui)
                entry["semtypes"] = umls.semtypes(cui)
                entry["semgroup"] = umls.semgroup(cui)
            top.append(entry)

        per_model[model] = {
            "total_examples": len(items),
            "total_unique_hallucinated_cuis": len(hallu),
            "top_hallucinated_cuis": top,
        }

    num_models = len(coverage_files)
    threshold = max(2, num_models // 2)
    consensus: List[Dict] = []
    for cui, cnt in models_hallucinating_cui.most_common(top_n):
        if cnt < threshold:
            break
        entry = {
            "cui": cui,
            "num_models_hallucinating": cnt,
            "pct_models": round(cnt / num_models, 3),
        }
        if umls:
            entry["name"] = umls.name(cui)
            entry["semtypes"] = umls.semtypes(cui)
            entry["semgroup"] = umls.semgroup(cui)
        consensus.append(entry)

    return {
        "per_model": per_model,
        "consensus_hallucinated_cuis": consensus,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 5 – Negation-aware gap analysis (lightweight NegEx-style)
# ─────────────────────────────────────────────────────────────────────────────

_PRE_NEG = re.compile(
    r"\b("
    r"no|not|never|without|absent|denies?|denied|"
    r"negative(?:\s+for)?|rules?\s+out|ruled\s+out|"
    r"no\s+evidence\s+of|no\s+sign\s+of|no\s+history\s+of|"
    r"no\s+complaint\s+of|no\s+signs?\s+of|failed\s+to|"
    r"unable\s+to|refus(?:ed?|ing)|unremarkable(?:\s+for)?|"
    r"clear\s+of|free\s+of|without\s+evidence\s+of"
    r")\b",
    re.IGNORECASE,
)
_PSEUDO_NEG = re.compile(
    r"\b(not\s+only|not\s+just|not\s+necessarily|no\s+wonder|"
    r"no\s+doubt|no\s+longer|not\s+yet|cannot\s+be\s+determined)\b",
    re.IGNORECASE,
)
_NEG_WINDOW = 6  # tokens to look at after a trigger


def _sent_tokenize(text: str) -> List[str]:
    try:
        return nltk.sent_tokenize(text)
    except Exception:
        # fallback: split on period/newline
        return [s.strip() for s in re.split(r"[.\n]", text) if s.strip()]


def _has_negation(sent: str) -> bool:
    clean = _PSEUDO_NEG.sub("", sent)
    return bool(_PRE_NEG.search(clean))


def _negated_terms(sent: str, window: int = _NEG_WINDOW) -> List[str]:
    """Extract content tokens in the window after any negation trigger."""
    tokens = sent.split()
    terms: List[str] = []
    i = 0
    while i < len(tokens):
        span = " ".join(tokens[i : i + 4])  # check multi-word triggers
        # multi-word trigger check
        multi_m = re.match(
            r"(no\s+evidence\s+of|no\s+sign(?:s)?\s+of|no\s+history\s+of|"
            r"no\s+complaint\s+of|ruled\s+out|rules\s+out|free\s+of|"
            r"clear\s+of|without\s+evidence\s+of)",
            span,
            re.IGNORECASE,
        )
        if multi_m:
            skip = len(multi_m.group(0).split())
            window_tokens = tokens[i + skip : i + skip + window]
            terms.extend(
                re.sub(r"[^\w]", "", t).lower()
                for t in window_tokens
                if len(re.sub(r"[^\w]", "", t)) > 3
            )
            i += skip
            continue
        # single-word trigger check
        if _PRE_NEG.match(tokens[i]) and not _PSEUDO_NEG.search(
            " ".join(tokens[max(0, i - 1) : i + 3])
        ):
            window_tokens = tokens[i + 1 : i + 1 + window]
            terms.extend(
                re.sub(r"[^\w]", "", t).lower()
                for t in window_tokens
                if len(re.sub(r"[^\w]", "", t)) > 3
            )
        i += 1
    return terms


def run_analysis_5(soap_files: List[Path]) -> Dict[str, Any]:
    # Download punkt if needed
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    results: Dict[str, Any] = {}

    for soap_path in soap_files:
        model = soap_path.stem
        data = load_json(soap_path)

        gold_neg_total = 0
        pred_neg_total = 0
        gold_sent_total = 0
        pred_sent_total = 0

        gold_term_ctr: Counter = Counter()
        pred_term_ctr: Counter = Counter()
        missed_term_ctr: Counter = Counter()  # negated in gold, not in pred
        extra_term_ctr: Counter = Counter()   # negated in pred, not in gold

        per_item = []

        for item in data:
            pred = item.get("pred_note", "") or ""
            gold = item.get("gold_note", "") or ""

            g_sents = _sent_tokenize(gold)
            p_sents = _sent_tokenize(pred)

            g_neg_sents = [s for s in g_sents if _has_negation(s)]
            p_neg_sents = [s for s in p_sents if _has_negation(s)]

            g_terms: List[str] = []
            for s in g_neg_sents:
                g_terms.extend(_negated_terms(s))

            p_terms: List[str] = []
            for s in p_neg_sents:
                p_terms.extend(_negated_terms(s))

            g_set = set(g_terms)
            p_set = set(p_terms)

            missed = g_set - p_set
            extra = p_set - g_set

            for t in g_terms:
                gold_term_ctr[t] += 1
            for t in p_terms:
                pred_term_ctr[t] += 1
            for t in missed:
                missed_term_ctr[t] += 1
            for t in extra:
                extra_term_ctr[t] += 1

            gold_neg_total += len(g_neg_sents)
            pred_neg_total += len(p_neg_sents)
            gold_sent_total += len(g_sents)
            pred_sent_total += len(p_sents)

            per_item.append({
                "idx": item.get("idx"),
                "gold_total_sentences": len(g_sents),
                "pred_total_sentences": len(p_sents),
                "gold_negated_sentences": len(g_neg_sents),
                "pred_negated_sentences": len(p_neg_sents),
                "gold_negated_terms": g_terms,
                "pred_negated_terms": p_terms,
                "missed_negated_terms": sorted(missed),
                "extra_negated_terms": sorted(extra),
            })

        results[model] = {
            "summary": {
                "total_examples": len(data),
                "gold_negation_rate": round(gold_neg_total / gold_sent_total, 4) if gold_sent_total else 0,
                "pred_negation_rate": round(pred_neg_total / pred_sent_total, 4) if pred_sent_total else 0,
                "gold_mean_negated_sents_per_note": round(gold_neg_total / len(data), 2) if data else 0,
                "pred_mean_negated_sents_per_note": round(pred_neg_total / len(data), 2) if data else 0,
            },
            "top_gold_negated_terms": [
                {"term": t, "freq": f} for t, f in gold_term_ctr.most_common(20)
            ],
            "top_pred_negated_terms": [
                {"term": t, "freq": f} for t, f in pred_term_ctr.most_common(20)
            ],
            "top_missed_negated_terms": [
                {"term": t, "freq": f} for t, f in missed_term_ctr.most_common(20)
            ],
            "top_extra_negated_terms": [
                {"term": t, "freq": f} for t, f in extra_term_ctr.most_common(20)
            ],
            "per_item": per_item,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 6 – Numerical / dosage / temporal gap analysis
# ─────────────────────────────────────────────────────────────────────────────

_NUM_PATTERNS: Dict[str, re.Pattern] = {
    "blood_pressure": re.compile(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b"),
    "heart_rate": re.compile(
        r"\b(?:HR|heart\s+rate|pulse)\s*[:\-]?\s*(\d{2,3})\b", re.IGNORECASE
    ),
    "spo2": re.compile(
        r"\b(?:SpO2?|O2\s+sat(?:uration)?|oxygen\s+sat(?:uration)?)\s*[:\-]?\s*(\d{2,3})\s*%?\b",
        re.IGNORECASE,
    ),
    "temperature": re.compile(
        r"\b(?:temp(?:erature)?)\s*[:\-]?\s*(\d{2,3}(?:\.\d)?)\s*(?:°?[FfCc])?\b",
        re.IGNORECASE,
    ),
    "weight": re.compile(r"\b(\d{2,4}(?:\.\d+)?)\s*(?:kg|lbs?|pounds?)\b", re.IGNORECASE),
    "dosage_mg": re.compile(r"\b(\d+(?:\.\d+)?)\s*mg\b", re.IGNORECASE),
    "dosage_mcg": re.compile(r"\b(\d+(?:\.\d+)?)\s*mcg\b", re.IGNORECASE),
    "dosage_ml": re.compile(r"\b(\d+(?:\.\d+)?)\s*m[lL]\b"),
    "dosage_units": re.compile(r"\b(\d+(?:\.\d+)?)\s*units?\b", re.IGNORECASE),
    "lab_value": re.compile(
        r"\b(\d+(?:\.\d+)?)\s*(?:mmol/L|mg/dL|mEq/L|g/dL|IU/L|U/L|ng/mL|mcg/mL)\b",
        re.IGNORECASE,
    ),
    "percentage": re.compile(r"\b(\d+(?:\.\d+)?)\s*%\b"),
    "frequency": re.compile(
        r"\b(BID|TID|QID|QD|QHS|PRN|q\.?d\.?|b\.?i\.?d\.?|t\.?i\.?d\.?|"
        r"once\s+daily|twice\s+daily|three\s+times\s+(?:a\s+)?daily|"
        r"every\s+\d+\s+hours?|q\d+h)\b",
        re.IGNORECASE,
    ),
    "duration_days": re.compile(r"\bfor\s+(\d+)\s+days?\b", re.IGNORECASE),
    "duration_weeks": re.compile(r"\bfor\s+(\d+)\s+weeks?\b", re.IGNORECASE),
    "duration_months": re.compile(r"\bfor\s+(\d+)\s+months?\b", re.IGNORECASE),
    "followup_days": re.compile(r"\bfollow.?up\s+(?:in\s+)?(\d+)\s+days?\b", re.IGNORECASE),
    "followup_weeks": re.compile(r"\bfollow.?up\s+(?:in\s+)?(\d+)\s+weeks?\b", re.IGNORECASE),
    "followup_months": re.compile(
        r"\bfollow.?up\s+(?:in\s+)?(\d+)\s+months?\b", re.IGNORECASE
    ),
    "return_visit": re.compile(
        r"\b(?:return|come\s+back|follow.?up|f/u|RTC)\b", re.IGNORECASE
    ),
    "range": re.compile(r"\b(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\b"),
}


def _extract_numerics(text: str) -> Dict[str, List[str]]:
    found: Dict[str, List[str]] = {}
    for cat, pat in _NUM_PATTERNS.items():
        matches = pat.findall(text)
        if not matches:
            continue
        flat: List[str] = []
        for m in matches:
            if isinstance(m, tuple):
                flat.append("/".join(str(x) for x in m if x))
            else:
                flat.append(str(m))
        found[cat] = flat
    return found


def run_analysis_6(soap_files: List[Path]) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    for soap_path in soap_files:
        model = soap_path.stem
        data = load_json(soap_path)

        gold_total: Dict[str, int] = defaultdict(int)
        pred_total: Dict[str, int] = defaultdict(int)
        # gap = gold_count - pred_count per example, positive means pred is missing values
        gap_acc: Dict[str, List[int]] = defaultdict(list)

        per_item = []

        for item in data:
            pred = item.get("pred_note", "") or ""
            gold = item.get("gold_note", "") or ""

            g_nums = _extract_numerics(gold)
            p_nums = _extract_numerics(pred)

            all_cats = set(g_nums) | set(p_nums)
            item_gaps: Dict[str, Any] = {}
            for cat in all_cats:
                g_cnt = len(g_nums.get(cat, []))
                p_cnt = len(p_nums.get(cat, []))
                gap = g_cnt - p_cnt
                item_gaps[cat] = {
                    "gold_count": g_cnt,
                    "pred_count": p_cnt,
                    "gap": gap,
                }
                gold_total[cat] += g_cnt
                pred_total[cat] += p_cnt
                gap_acc[cat].append(gap)

            per_item.append({
                "idx": item.get("idx"),
                "gold_numerics": g_nums,
                "pred_numerics": p_nums,
                "gap_by_category": item_gaps,
            })

        all_cats = set(list(gold_total.keys()) + list(pred_total.keys()))
        category_summary: Dict[str, Any] = {}
        for cat in sorted(all_cats):
            g = gold_total.get(cat, 0)
            p = pred_total.get(cat, 0)
            gaps = gap_acc.get(cat, [])
            category_summary[cat] = {
                "total_gold": g,
                "total_pred": p,
                "pred_coverage_rate": round(p / g, 3) if g else None,
                "mean_gap_per_example": round(sum(gaps) / len(gaps), 2) if gaps else 0,
                "pct_examples_with_gap": round(
                    sum(1 for x in gaps if x > 0) / len(gaps), 3
                ) if gaps else 0,
            }

        results[model] = {
            "category_summary": category_summary,
            "per_item": per_item,
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap analysis: what is missing in AI vs human SOAP notes."
    )
    parser.add_argument("--soap_dir", required=True, help="Folder with model prediction JSONs.")
    parser.add_argument("--coverage_dir", required=True, help="Folder with CUI coverage JSONs.")
    parser.add_argument("--output_dir", required=True, help="Output folder (will be created).")
    parser.add_argument("--umls_api_key", default=None, help="UMLS API key (enables analysis 1).")
    parser.add_argument("--top_n", type=int, default=20, help="Top-N CUIs to report per model.")
    args = parser.parse_args()

    soap_dir = Path(args.soap_dir)
    coverage_dir = Path(args.coverage_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    soap_files = sorted(
        p for p in soap_dir.iterdir()
        if p.suffix.lower() in {".json", ".jsonl"} and not p.name.startswith(".")
    )
    coverage_files = sorted(
        p for p in coverage_dir.iterdir()
        if p.suffix.lower() in {".json", ".jsonl"}
        and "cui_coverage" in p.name
        and "summary" not in p.name
        and not p.name.startswith(".")
        and p.is_file()
    )

    print(f"SOAP files ({len(soap_files)}):     {[p.name for p in soap_files]}")
    print(f"Coverage files ({len(coverage_files)}): {[p.name for p in coverage_files]}")

    umls = UMLSLookup(args.umls_api_key) if args.umls_api_key else None

    # ── 1: Semantic-type stratification ─────────────────────────────────────
    print("\n[1/6] Semantic-type stratification...")
    if umls:
        a1 = run_analysis_1(coverage_files, umls, top_n=args.top_n)
    else:
        a1 = {
            "status": "skipped",
            "reason": "No --umls_api_key provided. Re-run with a UMLS API key to enable this analysis.",
        }
    save_json(a1, output_dir / "1_semantic_type_stratification.json")
    print("  -> 1_semantic_type_stratification.json")

    # ── 2: SOAP-section gap analysis ─────────────────────────────────────────
    print("\n[2/6] SOAP-section gap analysis...")
    a2 = run_analysis_2(soap_files)
    save_json(a2, output_dir / "2_soap_section_gaps.json")
    print("  -> 2_soap_section_gaps.json")

    # ── 3: Top-N missed CUIs ─────────────────────────────────────────────────
    print("\n[3/6] Top-N missed CUIs...")
    a3 = run_analysis_3(coverage_files, top_n=args.top_n, umls=umls)
    save_json(a3, output_dir / "3_top_missed_cuis.json")
    print("  -> 3_top_missed_cuis.json")

    # ── 4: Top-N hallucinated CUIs ───────────────────────────────────────────
    print("\n[4/6] Top-N hallucinated CUIs...")
    a4 = run_analysis_4(coverage_files, top_n=args.top_n, umls=umls)
    save_json(a4, output_dir / "4_hallucinated_cuis.json")
    print("  -> 4_hallucinated_cuis.json")

    # ── 5: Negation-aware gap analysis ───────────────────────────────────────
    print("\n[5/6] Negation-aware gap analysis...")
    a5 = run_analysis_5(soap_files)
    save_json(a5, output_dir / "5_negation_gaps.json")
    print("  -> 5_negation_gaps.json")

    # ── 6: Numerical / dosage / temporal gap ─────────────────────────────────
    print("\n[6/6] Numerical / dosage / temporal gap analysis...")
    a6 = run_analysis_6(soap_files)
    save_json(a6, output_dir / "6_numerical_gaps.json")
    print("  -> 6_numerical_gaps.json")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "soap_files": [p.name for p in soap_files],
        "coverage_files": [p.name for p in coverage_files],
        "analyses": {
            "1_semantic_type_stratification": {
                "status": "complete" if umls else "skipped",
                "note": "Requires --umls_api_key" if not umls else None,
            },
            "2_soap_section_gaps": {"status": "complete"},
            "3_top_missed_cuis": {"status": "complete"},
            "4_hallucinated_cuis": {"status": "complete"},
            "5_negation_gaps": {"status": "complete"},
            "6_numerical_gaps": {"status": "complete"},
        },
        "output_files": [
            "1_semantic_type_stratification.json",
            "2_soap_section_gaps.json",
            "3_top_missed_cuis.json",
            "4_hallucinated_cuis.json",
            "5_negation_gaps.json",
            "6_numerical_gaps.json",
        ],
    }
    save_json(summary, output_dir / "summary.json")
    print("\n  -> summary.json")
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
