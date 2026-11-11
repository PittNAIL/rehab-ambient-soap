#!/usr/bin/env python3
"""
Build the Study 2 dataset (internal corpus key `adapts`) from the diarized XML
transcripts and the session-note CSV, in the same JSON schema as the Study 1
file (data/json_dataset.json), so every downstream script runs unchanged.

Multi-part sessions recorded the same day (suffix a/b/c) are merged in order and
joined to the therapist note for that session. The source files are
access-controlled (see the paper's Data availability statement); all IDs shown
below are synthetic.

Output schema (identical to data/json_dataset.json):
[
  {
    "idx": "900_1-2-03_OT",
    "conversation": "SPK_0: ...\nSPK_1: ...\n",
    "full_note": "Client seen for initial session. ..."
  },
  ...
]

Run:
    python data_prep/build_study2_dataset.py \
        --xml_dir /path/to/study2_xml_transcripts \
        --notes_csv /path/to/study2_session_notes.csv \
        --output data/adapts_dataset.json
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Turn inside <TEXT>:
#   SPK_0  OK. What time do you get up?  [00:39.42-00:42.12]
# The timestamp uses an en dash and is dropped.
TURN_RE = re.compile(r"^(SPK_\d+)\s+(.*?)\s*(?:\[[\d:.]+[–—-][\d:.]+\])?\s*$")

CDATA_RE = re.compile(r"<TEXT>\s*<!\[CDATA\[(.*?)\]\]>\s*</TEXT>", re.S)

# 900_1-2-03a.txt.xml_XYZ -> ("900", 1, 2, "03", "a")
STEM_RE = re.compile(r"^(\d+)[_-](\d{1,2})[_-](\d{1,2})[_-](\d{2,4})([a-z]?)$")

# Filenames in the delivered folder that do not follow the convention, mapped to
# the corrected stem, e.g. {"900_1-2a-03": "900_1-2-03a"} (part letter misplaced).
# The study's own correction is recorded with the access-controlled data, not
# here, because the stem encodes a participant ID and a session date.
FILENAME_FIXUPS: Dict[str, str] = {}


def normalize_stem(filename: str) -> Optional[Tuple[str, str]]:
    """
    Return (session_id, part) from a filename or a CSV ID.

    Both of these normalize to ("900_1-2-03", "a"):
        900_1-2-03a.txt.xml_XYZ
        900_01_02_2003a
    """
    stem = filename
    stem = re.sub(r"^new_", "", stem)
    stem = re.sub(r"\.txt.*$", "", stem)  # .txt.xml_<initials>, .txt_1.xml_<initials>, .txt.xml
    stem = FILENAME_FIXUPS.get(stem, stem)

    match = STEM_RE.match(stem)
    if not match:
        return None

    subject, month, day, year, part = match.groups()
    session_id = f"{subject}_{int(month)}-{int(day)}-{year[-2:]}"

    return session_id, part


def parse_transcript(path: Path) -> List[str]:
    """
    Extract speaker turns from one Study 2 XML file.

    Returns lines shaped like "SPK_0: text", matching the convention already used
    by the `conversation` field in data/json_dataset.json.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    match = CDATA_RE.search(raw)
    if not match:
        return []

    lines = []

    for raw_line in match.group(1).splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        turn = TURN_RE.match(raw_line)
        if not turn:
            # Continuation of the previous turn rather than a new speaker.
            if lines:
                lines[-1] = f"{lines[-1]} {raw_line}".strip()
            continue

        speaker, text = turn.group(1), turn.group(2).strip()
        if text:
            lines.append(f"{speaker}: {text}")

    return lines


def load_notes(notes_csv: Path) -> Dict[str, Dict[str, str]]:
    """
    Read the Study 2 session-note CSV.

    Row 1 is the header, row 2 maps columns to CMS Section GG codes and is skipped.
    """
    with notes_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    notes = {}

    for row in rows[2:]:
        if not row or not row[0].strip():
            continue

        normalized = normalize_stem(row[0].strip())
        if normalized is None:
            print(f"Skipping unparseable CSV id: {row[0]}")
            continue

        session_id, _ = normalized

        notes[session_id] = {
            "discipline": row[1].strip(),
            "duration": row[2].strip(),
            "time_generate": row[3].strip(),
            "note": row[4].strip(),
            "comments": row[5].strip(),
        }

    return notes


def build_dataset(xml_dir: Path, notes_csv: Path) -> List[Dict[str, str]]:
    notes = load_notes(notes_csv)

    # session_id -> [(part, path), ...]
    grouped: Dict[str, List[Tuple[str, Path]]] = defaultdict(list)

    for path in sorted(xml_dir.iterdir()):
        if not path.is_file():
            continue

        normalized = normalize_stem(path.name)
        if normalized is None:
            print(f"Skipping unparseable filename: {path.name}")
            continue

        session_id, part = normalized
        grouped[session_id].append((part, path))

    dataset = []

    for session_id in sorted(grouped):
        # Unsplit file first, then parts a, b, c... so the merged conversation
        # follows session order.
        parts = sorted(grouped[session_id], key=lambda item: item[0])

        lines: List[str] = []
        for _, path in parts:
            lines.extend(parse_transcript(path))

        if not lines:
            print(f"Skipping {session_id}: no transcript turns parsed")
            continue

        meta = notes.get(session_id)
        if meta is None:
            print(f"Skipping {session_id}: transcript has no matching CSV note")
            continue

        if not meta["note"]:
            print(f"Skipping {session_id}: CSV note is empty")
            continue

        discipline = meta["discipline"].upper() or "NA"

        dataset.append(
            {
                "idx": f"{session_id}_{discipline}",
                "conversation": "\n".join(lines) + "\n",
                "full_note": meta["note"],
            }
        )

    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Study 2 dataset from XML transcripts and the session-note CSV."
    )
    parser.add_argument("--xml_dir", type=str, required=True)
    parser.add_argument("--notes_csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    notes_csv = Path(args.notes_csv)
    output_path = Path(args.output)

    if not xml_dir.is_dir():
        raise NotADirectoryError(f"Not a folder: {xml_dir}")
    if not notes_csv.is_file():
        raise FileNotFoundError(f"Missing CSV: {notes_csv}")

    dataset = build_dataset(xml_dir=xml_dir, notes_csv=notes_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    turns = sum(row["conversation"].count("\n") for row in dataset)
    print(f"\nWrote {len(dataset)} sessions ({turns} turns) to {output_path}")


if __name__ == "__main__":
    main()
