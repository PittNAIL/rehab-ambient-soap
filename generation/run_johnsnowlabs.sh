#!/bin/bash
# JSL Text2SOAP (jsl_meds_text2soap_v1, Spark NLP AutoGGUFModel / llama.cpp).
# Requires a John Snow Labs license and a GPU node; run from the repository root.
#
# The GGUF artifact has a fixed 4,096-token context. setNCtx(16384) is accepted
# by the Python object and silently ignored by llama.cpp, so the usable prompt
# budget is derived from --real_n_ctx 4096 minus --n_predict, and longer
# transcripts are truncated BEFORE generation. Each output row records
# prompt_chars_original / prompt_chars_used / prompt_truncated /
# prompt_fraction_seen.
#
# The vendor returns each note double-JSON-encoded; decode_jsl_outputs.py
# recovers the plain text (idempotent). The paper scores decoded notes.
set -euo pipefail

# Study 2 (adapts): token-budgeted truncation -> 100/108 sessions truncated.
python generation/johnsnowlabs_eval.py \
  --input_path  data/adapts_dataset.json \
  --output_path outputs/jsl-text2soap_adapts.json \
  --n_ctx 16384 --real_n_ctx 4096 \
  --n_predict 1024 --batch_size 4 --partitions 1 --temperature 0.0
python generation/decode_jsl_outputs.py --write outputs/jsl-text2soap_adapts.json

# Study 1 (encompass): generated in the earlier run, which capped the prompt at
# 15,000 characters (just under the real 4,096-token ceiling) -> 28/45 truncated.
python generation/johnsnowlabs_eval.py \
  --input_path  data/json_dataset.json \
  --output_path outputs/jsl-text2soap_encompass.json \
  --n_ctx 16384 --max_prompt_chars 15000 \
  --n_predict 1024 --batch_size 4 --partitions 1 --temperature 0.0
python generation/decode_jsl_outputs.py --write outputs/jsl-text2soap_encompass.json
