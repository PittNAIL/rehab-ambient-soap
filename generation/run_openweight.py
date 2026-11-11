#!/usr/bin/env python3
"""
Generate SOAP notes with the four open-weight biomedical models, for either corpus
(Study 1: data/json_dataset.json; Study 2: data/adapts_dataset.json).

Paper configuration
-------------------
Every reported open-weight note was generated with --no-examples (the zero-shot
SOAP instruction: same sections and wording as the GPT-5 Pro prompt, worked
clinical examples removed), greedy decoding, --max-new-tokens 1024:

    key                   repo                              precision  max_ctx
    medgemma-27b          google/medgemma-27b-it            bf16       32,768
    med42-70b             m42-health/Llama3-Med42-70B       4-bit NF4   8,192
    openbiollm-8b         aaditya/Llama3-OpenBioLLM-8B      bf16        8,192
    asclepius-llama3-8b   starmpcc/Asclepius-Llama3-8B      bf16        8,192

MedGemma is NOT quantized: under NF4 it emits a constant <pad> token from the
first step (NaN logits). `asclepius-13b` is retained as an option for the pilot
comparison only and is not part of the paper.

Why --no-examples: in piloting, all four models reproduced the worked examples
("SPO2 92% on room air", "recalled 4/5 words", "walked 100 feet with min
assist") as clinical observations. Pilot generations were discarded; see the
paper's Methods and Supplementary Note 1.

Prompt construction
-------------------
    system : SYSTEM_INSTRUCTION
    user   : SOAP_PROMPT_PREFIX_NO_EXAMPLES + conversation

with SPK_0 -> "Doctor", SPK_1 -> "Client". Each model receives the pair through
its own chat template; Asclepius uses its published instruction format and
OpenBioLLM a plain completion prompt (it ships without a chat template).

Context limits
--------------
Transcripts over a model's budget (max_ctx - prompt scaffold - generation
reserve) are truncated head+tail around a marker rather than clipped at the
tail, and every output row records `truncated` and `n_prompt_tokens`. In the
reported run: Study 1, no session truncated for any model; Study 2, 30 of 108
sessions truncated for each 8,192-token model and none for MedGemma-27B-it.
A per-model truncation report is written next to the predictions.

Usage
-----
    python generation/run_openweight.py --model medgemma-27b --no-examples \
        --input-path data/adapts_dataset.json --output-dir outputs/adapts_noex \
        --max-new-tokens 1024
    python generation/run_openweight.py --model med42-70b --no-examples --dry-run

Output is resumable: rerunning skips sessions already in the .jsonl.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --------------------------------------------------------------------------
# Prompt strings — copied verbatim from openai_generations.py. Keep in sync.
# --------------------------------------------------------------------------

SOAP_PROMPT_PREFIX = (
    "You are a clinical assistant observing therapy sessions. Identify key components "
    "of the dialogue, and document it in a narrative SOAP format (paragraph-style). "
    "“Patients” are referred to as “Clients”.\n"
    "The key components of each SOAP part are as follows:\n"
    "**Subjective** covers "
    "Symptoms (e.g. Client reports fatigue.), "
    "Concerns (e.g. Client reports fear of falling.), and "
    "Pertinent notes about their presentation (e.g. Client alert and agreeable to therapy session).\n"
    "**Objective** covers "
    "Vital signs (e.g. Client SOB, SPO2 92% on room air), "
    "Assessment results (e.g. Client recalled 4/5 words given at beginning of the session), and "
    "Statement of activities (e.g. Client walked 100 feet with min assist and wheeled walker; Client completed laundry tasks).\n"
    "**Assessment** covers "
    "Interpretation of subjective and objective sections such as "
    "1. During ambulation, client required moderate guiding cues for safe hand placement, "
    "indicating continued deficits in safety awareness during functional transfers.\n"
    "2. Client with one LOB during laundry task. Upon reflection, client demonstrated poor "
    "insight into why she lost her balance and how to prevent future losses of balance.\n"
    "**Plan** covers "
    "Description of the plan moving forward, including goals, patient education, etc., "
    "e.g., Client provided education on falls prevention, will continue incorporating into future sessions, and "
    "Client left seated in room with call bell in reach, RN aware. Next session to focus "
    "on memory and problem solving.\n"
)

# NOTE: openai_generations.py ends this string with "Dialogue:\n", which made
# sense when the user turn was the bare transcript. With the SOAP prefix now in
# the user turn, that trailing header dangles — nothing follows it in the system
# message. OpenBioLLM-8B responded by echoing this text back verbatim and
# emitting EOS (25 tokens, no note). GPT-5-Pro tolerates it; an 8B does not.
# The trailing header is dropped here. Everything else is byte-identical.
# Ablation variant: identical section definitions and wording, with every
# concrete clinical instance removed. Used to test whether the worked examples
# are the source of the fabricated content seen in OpenBioLLM-8B and
# Asclepius-Llama3-8B (which reproduce "SPO2 92%", "4/5 words", "laundry task"
# in sessions where none of those occurred), while MedGemma-27B, Med42-70B and
# GPT-5 Pro do not. Nothing is added — no extra instruction — so the presence of
# the examples is the only variable.
SOAP_PROMPT_PREFIX_NO_EXAMPLES = (
    "You are a clinical assistant observing therapy sessions. Identify key components "
    "of the dialogue, and document it in a narrative SOAP format (paragraph-style). "
    "“Patients” are referred to as “Clients”.\n"
    "The key components of each SOAP part are as follows:\n"
    "**Subjective** covers "
    "Symptoms, Concerns, and "
    "Pertinent notes about their presentation.\n"
    "**Objective** covers "
    "Vital signs, Assessment results, and Statement of activities.\n"
    "**Assessment** covers "
    "Interpretation of the subjective and objective sections.\n"
    "**Plan** covers "
    "Description of the plan moving forward, including goals, patient education, etc.\n"
)

# Which prefix the prompt builders use. Switched by --no-examples in main().
_ACTIVE_SOAP_PREFIX = SOAP_PROMPT_PREFIX


def soap_prefix() -> str:
    return _ACTIVE_SOAP_PREFIX


def set_prompt_variant(no_examples: bool) -> None:
    global _ACTIVE_SOAP_PREFIX
    _ACTIVE_SOAP_PREFIX = (
        SOAP_PROMPT_PREFIX_NO_EXAMPLES if no_examples else SOAP_PROMPT_PREFIX
    )


SYSTEM_INSTRUCTION = (
    "You are a clinical assistant observing a doctor-patient conversation. "
    "Generate a concise clinical note summary from the dialogue.\n"
)

TRUNCATION_MARKER = "\n[... middle of session transcript omitted for context length ...]\n"


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

# prompt_style:
#   "chat"      use the tokenizer's own chat template
#   "llama3"    explicit Llama-3 header format (for Llama-3 models that ship
#               without a chat_template — verified true of OpenBioLLM-8B)
#   "asclepius" the published Asclepius instruction format
#
# fallback_style is used when prompt_style == "chat" but the tokenizer turns out
# to have no chat template. That is not hypothetical: apply_chat_template raises
# ValueError on aaditya/Llama3-OpenBioLLM-8B.

MODELS: Dict[str, Dict[str, Any]] = {
    "openbiollm-8b": {
        "repo": "aaditya/Llama3-OpenBioLLM-8B",
        "max_ctx": 8192,
        # Not instruct-tuned despite the Llama-3 base: no chat_template, and
        # eos_token_id is the base <|end_of_text|>. Llama-3 chat tokens make it
        # echo the instruction and stop. Use a plain completion prompt.
        "prompt_style": "plain",
        "fallback_style": "plain",
        "default_4bit": False,
        "gated": False,
    },
    "asclepius-13b": {
        "repo": "starmpcc/Asclepius-13B",
        # VERIFIED from the model config: max_position_embeddings = 2048, not the
        # 4096 a Llama-2 lineage would imply. Do not raise this — going over the
        # position-embedding limit degrades output silently rather than erroring.
        # Consequence: with the SOAP scaffold (~600 tok) and generation reserve,
        # only ~900 tokens of transcript fit, against a median session of ~4700.
        # Effectively every session is heavily truncated. See README.
        "max_ctx": 2048,
        "prompt_style": "asclepius",   # Llama-1/2 era, predates chat templates
        "fallback_style": "asclepius",
        "default_4bit": False,
        "gated": False,
    },
    # Newest model in the same Asclepius family, from the same authors, using the
    # identical instruction format — but Llama-3 based with an 8192 context, which
    # matches OpenBioLLM-8B and Med42-70B. Truncation drops from ~107/108 to
    # ~23/108, so running this alongside asclepius-13b separates "the model is
    # weak" from "the model only saw a fifth of each session".
    # License: CC-BY-NC-SA 4.0, research use only — state this in the methods.
    "asclepius-llama3-8b": {
        "repo": "starmpcc/Asclepius-Llama3-8B",
        "max_ctx": 8192,
        "prompt_style": "asclepius",
        "fallback_style": "asclepius",
        "default_4bit": False,
        "gated": False,
    },
    "medgemma-27b": {
        "repo": "google/medgemma-27b-it",
        # Native ctx is 128k; capped to bound KV cache. Longest ADAPTS session
        # is ~15.5k tokens, so nothing is truncated at 32k.
        "max_ctx": 32768,
        "prompt_style": "chat",
        # Gemma-family tokens, NOT llama3 — see build_gemma_prompt.
        "fallback_style": "gemma",
        # bf16, NOT 4-bit. Loaded in 4-bit this model emits a constant <pad>
        # token from the first step — the signature of NaN logits. Gemma-3 has
        # large activation outliers and does not survive nf4 quantization here.
        # bf16 needs ~54GB, so run it on rtx6k (96GB) or h200, not a 40GB a100.
        "default_4bit": False,
        "gated": True,
    },
    "med42-70b": {
        "repo": "m42-health/Llama3-Med42-70B",
        "max_ctx": 8192,
        "prompt_style": "chat",
        "fallback_style": "llama3",
        "default_4bit": True,
        "gated": True,
    },
}


def build_llama3_prompt(conversation: str) -> str:
    """Explicit Llama-3 instruct format, for repos that ship no chat template."""
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_INSTRUCTION}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{soap_prefix()}{conversation}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_plain_prompt(conversation: str) -> str:
    """
    Plain-text completion prompt, for checkpoints that are not instruct-tuned.

    OpenBioLLM-8B ships no chat template and its generation_config sets
    eos_token_id=128001 (<|end_of_text|>, the BASE Llama-3 EOS) rather than
    128009 (<|eot_id|>, used by Llama-3-Instruct). Given Llama-3 chat special
    tokens it goes out of distribution: it echoes the nearest instruction text
    verbatim and immediately emits EOS — 23 tokens, no note, on prompts of both
    764 and 7096 tokens, so this is format, not context length.

    The original mainstream_opensource_models.py also used a plain string, but
    ended at the transcript with no cue for where the answer starts, so the model
    simply continued the dialogue. The trailing "SOAP Note:" is that cue, and is
    the one addition beyond the shared prompt content.
    """
    return (
        f"{SYSTEM_INSTRUCTION}\n"
        f"{soap_prefix()}\n"
        "Dialogue:\n"
        f"{conversation}\n\n"
        "SOAP Note:\n"
    )


def build_gemma_prompt(conversation: str) -> str:
    """
    Gemma turn format, for MedGemma if its chat template is unavailable.

    Gemma has no system role; the convention is to fold the system text into the
    first user turn. Using the Llama-3 format here instead would emit special
    tokens Gemma does not recognise, so the fallbacks must not be interchanged.
    """
    return (
        "<bos><start_of_turn>user\n"
        f"{SYSTEM_INSTRUCTION}\n{soap_prefix()}{conversation}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def build_asclepius_prompt(conversation: str) -> str:
    """
    Asclepius' published instruction format, with the SOAP spec as a preamble.

    Ordering matters and matches the original asclepius_eval.py: the SOAP
    specification comes BEFORE [Conversation Begin], and only a short imperative
    sits in the instruction block at the end.

    An earlier version here placed SOAP_PROMPT_PREFIX inside [Instruction Begin],
    after the transcript. That put the worked examples ("Client reports fatigue",
    "one LOB during laundry task", "falls prevention") immediately before the
    generation point, and the model summarised the examples instead of the
    session — producing a note about a fictitious patient. Keep the spec first.
    """
    return (
        "You are an intelligent clinical languge model.\n"
        "Below is a snippet of doctor-patient conversation and a following instruction "
        "from healthcare professional.\n"
        "Write a response that appropriately completes the instruction.\n"
        "The response should provide the accurate answer to the instruction, while being concise.\n\n"
        f"{soap_prefix()}\n"
        "[Conversation Begin]\n"
        f"{conversation}\n"
        "[Conversation End]\n\n"
        "[Instruction Begin]\n"
        "Generate the corresponding clinical note as written by the therapist, "
        "in the narrative SOAP format described above. Use all the information from "
        "the dialogue, but do not add any new information. Do not describe the "
        "examples given above — document only the conversation.\n"
        "[Instruction End]\n\n"
        # Generation cue. Without it the model answers in prose ("Based on the
        # given conversation, the patient is a 33-year-old female...") and only
        # 36/108 notes carried any SOAP structure, with 25 echoing the
        # instruction back. This is the same cue that fixed OpenBioLLM-8B, and
        # is kept identical across the two so they stay comparable.
        "SOAP Note:\n"
    )


SPEAKER_RE = re.compile(r"SPK_(\d+)")


def normalize_speakers(conversation: str) -> str:
    """
    Map diarization labels to readable roles.

    SPK_0 -> Doctor and SPK_1 -> Client, matching the GPT-5-Pro run exactly.

    Sessions with a third or fourth speaker (family member, aide, second
    therapist) are handled too: the original scripts replaced only SPK_0/SPK_1,
    so raw "SPK_2:" tokens leaked into the prompt in 9 of the 45 sessions of the
    earlier corpus. Those become "Other Speaker 1", "Other Speaker 2", ...,
    which keeps distinct people distinct without inventing a clinical role.

    ADAPTS contains only SPK_0/SPK_1, so this is a no-op there and does not
    affect already-generated ADAPTS outputs.
    """
    def repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if n == 0:
            return "Doctor"
        if n == 1:
            return "Client"
        return f"Other Speaker {n - 1}"

    return SPEAKER_RE.sub(repl, conversation)


def truncate_to_budget(tokenizer, conversation: str,
                       budget_tokens: int) -> Tuple[str, bool]:
    """
    Fit `conversation` into budget_tokens, keeping the head and tail.

    Therapy notes draw Subjective content from the session opening and Plan
    content from the close, so dropping the middle preserves more of what the
    gold note covers than clipping the tail would.
    """
    ids = tokenizer(conversation, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget_tokens:
        return conversation, False

    marker_len = len(tokenizer(TRUNCATION_MARKER, add_special_tokens=False)["input_ids"])
    usable = max(budget_tokens - marker_len, 64)
    head_len = int(usable * 0.6)
    tail_len = usable - head_len

    head = tokenizer.decode(ids[:head_len], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-tail_len:], skip_special_tokens=True)
    return head + TRUNCATION_MARKER + tail, True


def resolve_style(tokenizer, model_key: str) -> str:
    """
    Pick the prompt format, degrading to the fallback if the tokenizer has no
    usable chat template. Called once per run; the choice is printed and
    recorded in the output so it is auditable after the fact.
    """
    spec = MODELS[model_key]
    style = spec["prompt_style"]

    if style != "chat":
        return style

    if getattr(tokenizer, "chat_template", None):
        return "chat"

    fallback = spec["fallback_style"]
    print(f"[warn] {spec['repo']} ships no chat_template; "
          f"falling back to '{fallback}' format")
    return fallback


def make_renderer(tokenizer, style: str):
    """Return a function mapping a transcript to the final prompt string."""
    if style == "chat":
        # Some templates (Gemma-family especially) reject a system role outright.
        # Probe once, then commit to whichever shape works, so the choice is made
        # deliberately instead of throwing mid-run on session 40.
        def render_with_system(convo: str) -> str:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": soap_prefix() + convo},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        def render_merged(convo: str) -> str:
            return tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": f"{SYSTEM_INSTRUCTION}\n{soap_prefix()}{convo}"}],
                tokenize=False,
                add_generation_prompt=True,
            )

        try:
            render_with_system("probe")
            return render_with_system
        except Exception as e:
            print(f"[warn] chat template rejected a system role ({type(e).__name__}); "
                  "folding the system instruction into the user turn")
            return render_merged

    if style == "llama3":
        return build_llama3_prompt
    if style == "plain":
        return build_plain_prompt
    if style == "gemma":
        return build_gemma_prompt
    if style == "asclepius":
        return build_asclepius_prompt
    raise ValueError(f"unknown prompt style: {style}")


def assert_prompt_integrity(prompt: str, style: str, model_key: str) -> None:
    """
    Fail loudly if the instruction did not survive into the final prompt.

    This exists because the previous generation script silently dropped the
    entire SOAP prompt for Med42-70B — bare string literals that were never
    concatenated — and the loss was invisible until the outputs were audited
    months later. A missing prompt must stop the run, not produce a plausible
    note from an unprompted model.
    """
    required = [
        "narrative SOAP format",
        "**Subjective** covers",
        "**Objective** covers",
        "**Assessment** covers",
        "**Plan** covers",
    ]
    # Asclepius' format carries its own preamble instead of a system turn.
    if style != "asclepius":
        required.append("clinical assistant observing a doctor-patient conversation")

    missing = [m for m in required if m not in prompt]
    if missing:
        raise RuntimeError(
            f"PROMPT INTEGRITY FAILURE for {model_key} (style={style}): "
            f"the rendered prompt is missing {missing!r}. "
            "Refusing to generate — this is the bug that silently made Med42-70B "
            "an unprompted baseline. Check the prompt strings and template."
        )


def build_model_input(tokenizer, model_key: str, conversation: str,
                      max_new_tokens: int, render=None,
                      style: Optional[str] = None) -> Tuple[str, bool, int]:
    """
    Return (prompt_text, was_truncated, n_prompt_tokens).

    Budget is computed against the scaffold (system message, template tokens,
    SOAP prefix) so only the transcript absorbs the truncation.
    """
    spec = MODELS[model_key]
    reserve = max_new_tokens + 128  # generation headroom + template slack

    if style is None:
        style = resolve_style(tokenizer, model_key)
    if render is None:
        render = make_renderer(tokenizer, style)

    scaffold_tokens = len(tokenizer(render(""), add_special_tokens=False)["input_ids"])
    budget = spec["max_ctx"] - scaffold_tokens - reserve

    if budget <= 0:
        raise RuntimeError(
            f"{model_key}: prompt scaffold ({scaffold_tokens} tokens) plus generation "
            f"reserve ({reserve}) exceeds the {spec['max_ctx']}-token context. "
            "Lower --max-new-tokens."
        )

    fitted, truncated = truncate_to_budget(tokenizer, conversation, budget)
    prompt = render(fitted)
    assert_prompt_integrity(prompt, style, model_key)
    n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    return prompt, truncated, n_tokens


def load_completed(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                done.add(json.loads(line)["idx"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(MODELS),
                        help="Which open-weight model to run.")
    parser.add_argument("--input-path", type=Path, default=Path("data/adapts_dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/adapts"))
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Generation budget, also reserved out of the context. "
                             "1024 is what every reported run used (Study 2 gold notes: "
                             "median 89 tokens, max 230). Lowering it frees context and "
                             "reduces truncation on the 8k models, but does not "
                             "reproduce the paper.")
    parser.add_argument("--load-4bit", action="store_true", default=None,
                        help="Force 4-bit. Defaults on for med42-70b only (MedGemma fails under NF4).")
    parser.add_argument("--no-4bit", dest="load_4bit", action="store_false",
                        help="Force full precision (needs 80GB+ for 27B, 160GB for 70B).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only run the first N sessions (smoke test).")
    parser.add_argument("--corpus-tag", default=None,
                        help="Corpus label used in output filenames. Inferred from "
                             "--input-path when omitted (encompass / adapts).")
    parser.add_argument("--no-examples", action="store_true",
                        help="Use SOAP_PROMPT_PREFIX_NO_EXAMPLES (same sections and wording, no "
                             "worked clinical instances). REQUIRED to reproduce the paper: "
                             "all reported open-weight notes used this variant. Output "
                             "filenames get a '_noex' suffix.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render and validate every prompt using the tokenizer only. "
                             "Downloads no weights and needs no GPU. Run this first — it "
                             "catches template and context problems in seconds instead of "
                             "after a 55GB download.")
    args = parser.parse_args()

    spec = MODELS[args.model]
    use_4bit = spec["default_4bit"] if args.load_4bit is None else args.load_4bit

    # Select the prompt variant before any prompt is built.
    set_prompt_variant(args.no_examples)
    variant = "no-examples" if args.no_examples else "with-examples"
    suffix = "_noex" if args.no_examples else ""

    # Corpus tag comes from the input filename, not hardcoded: running
    # --input-path data/json_dataset.json previously still wrote "*_adapts_preds",
    # which mislabelled every 45-session output as ADAPTS.
    corpus = args.corpus_tag or (
        "adapts" if "adapts" in args.input_path.stem.lower() else
        "encompass" if args.input_path.stem == "json_dataset" else
        args.input_path.stem
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / f"{args.model}{suffix}_{corpus}_preds.jsonl"
    json_path = args.output_dir / f"{args.model}{suffix}_{corpus}_preds.json"
    report_path = args.output_dir / f"{args.model}{suffix}_truncation_report.json"

    with args.input_path.open(encoding="utf-8") as f:
        rows: List[Dict[str, Any]] = json.load(f)
    if args.limit:
        rows = rows[: args.limit]

    print(f"model      : {args.model}  ({spec['repo']})")
    print(f"precision  : {'4-bit' if use_4bit else 'bf16'}")
    print(f"context    : {spec['max_ctx']} tokens")
    print(f"sessions   : {len(rows)}")
    print(f"output     : {jsonl_path}")

    tokenizer = AutoTokenizer.from_pretrained(spec["repo"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Resolve the prompt format once, up front, so the choice is visible in the
    # log rather than being re-derived silently on every session.
    style = resolve_style(tokenizer, args.model)
    render = make_renderer(tokenizer, style)
    print(f"prompt fmt : {style}")
    print(f"prompt var : {variant}")

    if args.dry_run:
        print("\n--- dry run: rendering and validating all prompts ---")
        n_trunc = 0
        token_counts = []
        for row in tqdm(rows, desc="validating"):
            prompt, was_truncated, n_tok = build_model_input(
                tokenizer, args.model, normalize_speakers(row["conversation"]),
                args.max_new_tokens, render=render, style=style)
            n_trunc += was_truncated
            token_counts.append(n_tok)
            if n_tok > spec["max_ctx"]:
                raise RuntimeError(
                    f"{row['idx']}: rendered prompt is {n_tok} tokens, over the "
                    f"{spec['max_ctx']} limit. Truncation budget is wrong.")

        sample, _, _ = build_model_input(
            tokenizer, args.model, normalize_speakers(rows[0]["conversation"]),
            args.max_new_tokens, render=render, style=style)
        print("\n--- first 700 chars of a rendered prompt ---")
        print(sample[:700])
        print("--- end sample ---\n")
        print(f"OK: {len(rows)} prompts rendered, all passed integrity checks")
        print(f"    prompt tokens: min {min(token_counts)}, "
              f"max {max(token_counts)}, limit {spec['max_ctx']}")
        print(f"    truncated    : {n_trunc} of {len(rows)}")
        print("\nDry run only — no weights loaded, nothing generated.")
        return

    # Without a GPU, device_map="auto" silently falls back to CPU and generation
    # crawls for hours instead of erroring. On SLURM this usually means the job
    # was allocated 0 GPUs (e.g. `crc-interactive --gpu` without `-u 1`).
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device visible — refusing to run on CPU, which would take "
            "days for 108 sessions.\n"
            f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "Request a GPU explicitly, e.g.:\n"
            "  crc-interactive --gpu -u 1 -t 4 -c 4 -b 32\n"
            "or check that your sbatch script passes --gres=gpu:N.\n"
            "(--dry-run needs no GPU and is unaffected by this check.)"
        )
    print(f"gpus       : {torch.cuda.device_count()} x "
          f"{torch.cuda.get_device_name(0)}")

    model_kwargs: Dict[str, Any] = {"device_map": "auto"}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    print("loading weights (first run also downloads them) ...")
    model = AutoModelForCausalLM.from_pretrained(spec["repo"], **model_kwargs)
    model.eval()
    model.config.use_cache = True

    completed = load_completed(jsonl_path)
    if completed:
        print(f"resuming: {len(completed)} sessions already done")

    truncated_ids: List[str] = []
    start = time.time()

    for row in tqdm(rows, desc=args.model):
        if row["idx"] in completed:
            continue

        conversation = normalize_speakers(row["conversation"])
        prompt, was_truncated, n_prompt_tokens = build_model_input(
            tokenizer, args.model, conversation, args.max_new_tokens,
            render=render, style=style)
        if was_truncated:
            truncated_ids.append(row["idx"])

        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,          # greedy, matches prior open-weight runs
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        pred_note = tokenizer.decode(generated, skip_special_tokens=True).strip()

        record = {
            "idx": row["idx"],
            "conversation": row["conversation"],
            "gold_note": row.get("full_note", ""),
            "prompt": prompt,
            "pred_note": pred_note,
            "model_provider": "open-weight-local",
            "model": spec["repo"],
            "prompt_style": style,
            "prompt_variant": variant,
            "quantization": "4bit-nf4" if use_4bit else "bf16",
            "n_prompt_tokens": n_prompt_tokens,
            "truncated": was_truncated,
        }
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Consolidate to .json in the same shape as the GPT-5-Pro outputs.
    records = [json.loads(l) for l in jsonl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    n_trunc = sum(1 for r in records if r.get("truncated"))
    with report_path.open("w", encoding="utf-8") as f:
        json.dump({
            "model": spec["repo"],
            "max_ctx": spec["max_ctx"],
            "max_new_tokens": args.max_new_tokens,
            "n_sessions": len(records),
            "n_truncated": n_trunc,
            "truncated_idxs": [r["idx"] for r in records if r.get("truncated")],
        }, f, indent=2)

    elapsed = (time.time() - start) / 60
    print(f"\ndone in {elapsed:.1f} min")
    print(f"  {len(records)} sessions -> {json_path}")
    print(f"  {n_trunc} truncated -> {report_path}")


if __name__ == "__main__":
    main()
