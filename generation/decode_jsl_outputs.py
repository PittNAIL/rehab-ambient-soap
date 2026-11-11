#!/usr/bin/env python3
"""
decode_jsl_outputs.py

Undo JSL Text2SOAP's double-JSON encoding in generated-note files.

Text2SOAP returns each note as a quote-wrapped JSON string whose newlines are
two-character escape sequences:

    "S: The patient ... a \\"bird\\" sensation.\\nO: ..."

Left encoded, the wrapping quotes and literal \\n are scored as note content:
ROUGE tokenises them, MetaMapLite maps them, and the S/O/A/P sections read as
one unbroken line. Decoding recovers the prose the model actually produced.
Nothing else about the text is changed.

The decode is idempotent and only fires on strings that are both quote-wrapped
and valid JSON, so running it twice, or on an already-plain file, is a no-op.

Usage
-----
  python decode_jsl_outputs.py --check  FILE [FILE ...]   # report only
  python decode_jsl_outputs.py --write  FILE [FILE ...]   # rewrite in place
                                                          # (.pre_decode backup)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def decode_vendor_json(text):
    """Return `text` with one layer of JSON string encoding removed, if present."""
    if not isinstance(text, str):
        return text
    s = text.strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                return decoded
        except json.JSONDecodeError:
            pass
    return text


def process(path: Path, key: str, write: bool) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data if isinstance(data, list) else list(data.values())

    changed, newlines_before, newlines_after, wrapped = 0, 0, 0, 0
    for rec in records:
        if not isinstance(rec, dict) or key not in rec:
            continue
        original = rec[key] or ""
        newlines_before += original.count("\n")
        if isinstance(original, str) and original.strip().startswith('"'):
            wrapped += 1
        decoded = decode_vendor_json(original)
        newlines_after += decoded.count("\n")
        if decoded != original:
            changed += 1
            if write:
                rec[key] = decoded

    n = len(records)
    print(f"{path}")
    print(f"   records            : {n}")
    print(f"   quote-wrapped      : {wrapped}/{n}")
    print(f"   changed by decode  : {changed}/{n}")
    print(f"   newlines before/after: {newlines_before} -> {newlines_after}")

    if write and changed:
        backup = path.with_suffix(path.suffix + ".pre_decode")
        if not backup.exists():          # never clobber an existing backup
            shutil.copy2(path, backup)
            print(f"   backup             : {backup.name}")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"   REWRITTEN")
    elif write:
        print(f"   nothing to change")
    print()
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--key", default="pred_note")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report only, change nothing")
    mode.add_argument("--write", action="store_true", help="rewrite in place, keeping a backup")
    args = ap.parse_args()

    total = 0
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"{p}: MISSING\n")
            continue
        total += process(p, args.key, args.write)

    verb = "would change" if args.check else "changed"
    print(f"total notes {verb}: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
