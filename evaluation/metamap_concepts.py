#!/usr/bin/env python3
"""
add_umls_concept_names.py

Add UMLS concept names to a JSON file that already contains CUIs.

This runs fully offline using MRCONSO.RRF from a local UMLS installation.

Example:
python metamap_concepts.py \
  --input metrics_test/metric_reports/gpt5-pro_soap_preds_cui_coverage.json \
  --mrconso /path/to/pymetamaplite/data/public_mm_lite/data/ivf/2022AB/meta_data/MRCONSO.RRF \
  --output metrics_test/metric_reports/complete_concept_data/gpt5-pro_soap_preds_cui_coverage.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


DEFAULT_CUI_KEYS = [
    "cuis",
    "pred_cuis",
    "gold_cuis",
    "common_cuis",
]


def collect_cuis(obj: Any, cui_keys: Set[str]) -> Set[str]:
    """
    Recursively collect CUIs from a nested JSON object.

    It only collects CUIs from keys listed in cui_keys.
    """
    cuis = set()

    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in cui_keys:
                if isinstance(value, list):
                    cuis.update(str(x) for x in value if str(x).startswith("C"))
                elif isinstance(value, str) and value.startswith("C"):
                    cuis.add(value)
            else:
                cuis.update(collect_cuis(value, cui_keys))

    elif isinstance(obj, list):
        for item in obj:
            cuis.update(collect_cuis(item, cui_keys))

    return cuis


def load_cui_to_name_from_mrconso(
    mrconso_path: str,
    target_cuis: Set[str],
    language: str = "ENG",
    preferred_sources: List[str] | None = None,
) -> Dict[str, str]:
    """
    Load CUI -> concept name mapping from MRCONSO.RRF.

    MRCONSO.RRF columns are pipe-separated. The key fields used here are:
    - CUI: column 0
    - LAT: column 1, language
    - TS: column 2, term status
    - STT: column 4, string type
    - ISPREF: column 6, atom preferred flag
    - SAB: column 11, source vocabulary
    - STR: column 14, concept string

    Preference logic:
    1. Prefer rows where LAT == ENG.
    2. Prefer source vocabularies listed in preferred_sources.
    3. Prefer rows marked as preferred: TS == P, STT == PF, ISPREF == Y.
    4. Otherwise keep the first usable English name.
    """
    preferred_sources = preferred_sources or []

    cui_to_best = {}

    def score_row(ts: str, stt: str, ispref: str, sab: str) -> int:
        score = 0

        if sab in preferred_sources:
            # Earlier source in preferred_sources gets slightly higher score.
            score += 1000 - preferred_sources.index(sab)

        if ts == "P":
            score += 100

        if stt == "PF":
            score += 50

        if ispref == "Y":
            score += 25

        return score

    with open(mrconso_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")

            # MRCONSO has many columns; malformed lines should be ignored.
            if len(parts) < 15:
                continue

            cui = parts[0]
            lat = parts[1]
            ts = parts[2]
            stt = parts[4]
            ispref = parts[6]
            sab = parts[11]
            concept_name = parts[14]

            if cui not in target_cuis:
                continue

            if lat != language:
                continue

            row_score = score_row(ts=ts, stt=stt, ispref=ispref, sab=sab)

            if cui not in cui_to_best or row_score > cui_to_best[cui]["score"]:
                cui_to_best[cui] = {
                    "name": concept_name,
                    "score": row_score,
                    "source": sab,
                }

    return {cui: row["name"] for cui, row in cui_to_best.items()}


def add_names_to_json(
    obj: Any,
    cui_to_name: Dict[str, str],
    cui_keys: Set[str],
) -> Any:
    """
    Recursively add name dictionaries next to CUI-list fields.

    Example:
    "pred_cuis": ["C0030193"]

    becomes:
    "pred_cuis": ["C0030193"],
    "pred_cui_names": {
        "C0030193": "Pain"
    }
    """
    if isinstance(obj, dict):
        new_obj = {}

        for key, value in obj.items():
            new_obj[key] = add_names_to_json(value, cui_to_name, cui_keys)

            if key in cui_keys:
                name_key = key[:-1] + "_names" if key.endswith("s") else key + "_names"

                if isinstance(value, list):
                    new_obj[name_key] = {
                        str(cui): cui_to_name.get(str(cui), None)
                        for cui in value
                    }

                elif isinstance(value, str):
                    new_obj[name_key] = {
                        value: cui_to_name.get(value, None)
                    }

        return new_obj

    if isinstance(obj, list):
        return [add_names_to_json(item, cui_to_name, cui_keys) for item in obj]

    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add UMLS concept names to CUIs in a JSON file using local MRCONSO.RRF."
    )
    parser.add_argument("--input", required=True, help="Input JSON file with CUIs.")
    parser.add_argument("--mrconso", required=True, help="Path to MRCONSO.RRF.")
    parser.add_argument("--output", required=True, help="Output JSON file.")
    parser.add_argument(
        "--cui_keys",
        nargs="*",
        default=DEFAULT_CUI_KEYS,
        help="JSON keys that contain CUIs.",
    )
    parser.add_argument(
        "--preferred_sources",
        nargs="*",
        default=["SNOMEDCT_US", "MSH", "LNC", "RXNORM", "ICD10CM", "CPT", "HCPCS"],
        help="Preferred UMLS source vocabularies when choosing concept names.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    mrconso_path = Path(args.mrconso)

    if not input_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_path}")

    if not mrconso_path.exists():
        raise FileNotFoundError(f"MRCONSO.RRF not found: {mrconso_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cui_keys = set(args.cui_keys)

    target_cuis = collect_cuis(data, cui_keys)

    print(f"Found {len(target_cuis)} unique CUIs in JSON.")

    cui_to_name = load_cui_to_name_from_mrconso(
        mrconso_path=str(mrconso_path),
        target_cuis=target_cuis,
        preferred_sources=args.preferred_sources,
    )

    print(f"Resolved names for {len(cui_to_name)} CUIs.")
    print(f"Unresolved CUIs: {len(target_cuis) - len(cui_to_name)}")

    output_data = add_names_to_json(
        obj=data,
        cui_to_name=cui_to_name,
        cui_keys=cui_keys,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()