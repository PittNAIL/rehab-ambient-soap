#!/usr/bin/env python3
"""
build_concept_tables.py

Aggregate MetaMapLite CUI output into ranked concept frequency tables for
Table 2 and Supplementary Table 6.

Table 2 reports TOTAL DETECTION counts (repeats within a note included);
Supplementary Table 6 reports ENCOUNTER-PRESENCE counts. Both rankings are
emitted in full -- every concept, not just the top 20 -- so the final table
can be assembled by slicing this file without re-running MetaMapLite.

Gold concepts are deduplicated by encounter: the gold note is shared across
all systems within a corpus, so counting it once per model would multiply it
by the number of systems.

Concept names come from the MetaMapLite IVF index (cuiconcept postings),
reformatted as MRCONSO-style rows; see --names.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict


def load_names(path: Path, targets: set) -> Dict[str, str]:
    out = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if len(parts) > 14 and parts[0] in targets:
                out.setdefault(parts[0], parts[14])
    return out


def rank(detections: Counter, presence: Counter, names, semtypes, n_enc: int):
    rows = []
    for cui, det in detections.items():
        rows.append({
            "cui": cui,
            "name": names.get(cui, ""),
            "semtypes": semtypes.get(cui, []),
            "total_detections": det,
            "encounters_present": presence[cui],
            "pct_of_encounters": round(presence[cui] / n_enc, 4) if n_enc else 0.0,
        })
    rows.sort(key=lambda r: (-r["total_detections"], -r["encounters_present"], r["cui"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_dirs", nargs="+", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out = {"metadata": {"script": "build_concept_tables.py",
                        "table_2_metric": "total_detections (repeats within a note included)",
                        "supp_table_6_metric": "encounters_present"},
           "corpora": {}}

    all_cuis = set()
    staged = {}

    for rd in args.report_dirs:
        rd = Path(rd)
        corpus = "adapts" if "adapts" in str(rd) else "encompass"
        gold_det, gold_pres = Counter(), Counter()
        gold_seen = set()
        semtypes: Dict[str, list] = {}
        models = {}

        for fp in sorted(rd.glob("*_cui_coverage.json")):
            d = json.loads(fp.read_text(encoding="utf-8"))
            semtypes.update(d.get("cui_semtypes", {}))
            name = fp.name.replace("_cui_coverage.json", "")
            det, pres = Counter(), Counter()
            for it in d["per_item"]:
                for c, n in it.get("pred_cui_counts", {}).items():
                    det[c] += n
                    pres[c] += 1
                # gold is shared across models -- count each encounter once
                key = (corpus, it["idx"])
                if key not in gold_seen:
                    gold_seen.add(key)
                    for c, n in it.get("gold_cui_counts", {}).items():
                        gold_det[c] += n
                        gold_pres[c] += 1
            models[name] = (det, pres, len(d["per_item"]))
            all_cuis |= set(det)
        all_cuis |= set(gold_det)
        staged[corpus] = (gold_det, gold_pres, len(gold_seen), models, semtypes)

    names = load_names(Path(args.names), all_cuis)

    for corpus, (gd, gp, n_enc, models, semtypes) in staged.items():
        node = {"n_encounters": n_enc,
                "gold": rank(gd, gp, names, semtypes, n_enc),
                "systems": {}}
        for m, (det, pres, n) in models.items():
            node["systems"][m] = rank(det, pres, names, semtypes, n)
        out["corpora"][corpus] = node

    Path(args.output).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for corpus, node in out["corpora"].items():
        print(f"{corpus}: {node['n_encounters']} encounters, gold {len(node['gold'])} concepts, "
              f"{len(node['systems'])} systems")
        for m, rows in node["systems"].items():
            print(f"   {m:34s} {len(rows)} concepts")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
