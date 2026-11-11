import os
import json
import argparse
from pathlib import Path

import pandas as pd
from johnsnowlabs import nlp, medical
from pyspark.ml import Pipeline


# recommended prompt https://nlp.johnsnowlabs.com/2025/04/09/jsl_meds_text2soap_v1_en.html
DEFAULT_SYSTEM_PROMPT = """You are an expert medical professor assisting in the creation of medically accurate SOAP summaries.
Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting.
Create a Medical SOAP note summary from the dialogue, following these guidelines:
    S (Subjective): Summarize the patient's reported symptoms, including chief complaint and relevant history. Rely on the patient's statements as the primary source and ensure standardized terminology.
    O (Objective): Highlight critical findings such as vital signs, lab results, and imaging, emphasizing important details like the side of the body affected and specific dosages. Include normal ranges where relevant.
    A (Assessment): Offer a concise assessment combining subjective and objective data. State the primary diagnosis and any differential diagnoses, noting potential complications and the prognostic outlook.
    P (Plan): Outline the management plan, covering medication, diet, consultations, and education. Ensure to mention necessary referrals to other specialties and address compliance challenges.
    Considerations: Compile the report based solely on the transcript provided. Maintain confidentiality and document sensitively. Use concise medical jargon and abbreviations for effective doctor communication.
    Please format the summary in a clean, simple list format without using markdown or bullet points. Use 'S:', 'O:', 'A:', 'P:' directly followed by the text. Avoid any styling or special characters."""
# Rough chars-per-token for English clinical prose; used only to size the
# prompt budget from the model context, so a conservative value is fine.
CHARS_PER_TOKEN = 3.5

BOS = "<|im_start|>"
EOS = "<|im_end|>"


def normalize_speakers(text: str) -> str:
    if text is None:
        return ""
    return (
        text.replace("SPK_0", "Doctor")
            .replace("SPK_1", "Patient")
            .replace("spk_0", "Doctor")
            .replace("spk_1", "Patient")
    )


def build_prompt(conversation: str, system_prompt: str) -> str:
    conversation = normalize_speakers(conversation).strip()
    return f"""{BOS}user
{system_prompt}
### Dialogue:
{conversation}
{EOS}
{BOS}assistant
"""


def decode_vendor_json(text):
    """Undo the vendor's double-JSON encoding of the generated note.

    Text2SOAP returns each note as a quote-wrapped JSON string whose newlines
    are two-character escape sequences, e.g.

        "S: The patient ... a \\"bird\\" sensation.\\nO: ..."

    Left as-is, the wrapping quotes and literal \\n sequences are scored as
    note content: ROUGE tokenises them, MetaMapLite maps them, and the S/O/A/P
    sections read as one unbroken line. Decoding recovers the prose the model
    actually produced, so scores reflect the note rather than a serialization
    artifact of the vendor API. No other change is made to the text.
    """
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


def extract_text2soap_output(value):
    """
    Handle common Spark NLP / JSL output shapes.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return decode_vendor_json(value.strip()).strip()

    # Spark annotation arrays sometimes come back as list[Row] or list[dict]
    if isinstance(value, list):
        texts = []
        for item in value:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                if "result" in item:
                    texts.append(str(item["result"]))
            else:
                # Row-like object
                result = getattr(item, "result", None)
                if result is not None:
                    texts.append(str(result))
        # Decode each element before joining: the vendor encodes per-note, so
        # joining first would leave inner quotes embedded mid-string.
        joined = "\n".join(decode_vendor_json(t.strip()).strip() for t in texts if t)
        return joined.strip()

    # Row-like object
    result = getattr(value, "result", None)
    if result is not None:
        return decode_vendor_json(str(result).strip()).strip()

    return decode_vendor_json(str(value).strip()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to input JSON file containing a list of examples.")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to output JSON file.")
    parser.add_argument("--text_col", type=str, default="conversation",
                        help="Input text column in the JSON records.")
    parser.add_argument("--id_col", type=str, default="idx",
                        help="ID column in the JSON records.")
    parser.add_argument("--reference_col", type=str, default="full_note",
                        help="Optional reference note column.")
    parser.add_argument("--model_name", type=str, default="jsl_meds_text2soap_v1",
                        help="JSL model name.")
    parser.add_argument("--model_class", choices=["autogguf", "medicalllm"],
                        default="autogguf",
                        help="Which annotator class to load the model with. The "
                             "vendor page documents both; they may differ in "
                             "whether setNCtx reaches the llama.cpp runtime.")
    parser.add_argument("--use_chat_template", type=int, default=1,
                        help="setUseChatTemplate. The annotator carries its own "
                             "systemPrompt and applies a chat wrapper, while "
                             "build_prompt already emits <|im_start|> markers, so "
                             "1 double-wraps. Try 0 to send the prompt verbatim.")
    parser.add_argument("--real_n_ctx", type=int, default=0,
                        help="Size the prompt budget against the context llama.cpp "
                             "ACTUALLY uses, when that differs from --n_ctx. "
                             "0 = trust --n_ctx.")
    parser.add_argument("--debug_raw", action="store_true",
                        help="Print the untouched annotation for row 0 before "
                             "extraction. Distinguishes empty generation from a "
                             "shape extract_text2soap_output does not handle.")
    parser.add_argument("--partitions", type=int, default=1,
                        help="Spark partitions. Keep at 1: each partition loads its "
                             "own copy of the model into VRAM.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only process the first N encounters. Use for a smoke "
                             "test before committing GPU hours; 0 = all.")
    parser.add_argument("--n_ctx", type=int, default=16384,
                        help="Model context window in tokens; must match setNCtx.")
    parser.add_argument("--max_prompt_chars", type=int, default=0,
                        help="Hard prompt cap in characters. 0 = derive from "
                             "(n_ctx - n_predict) * CHARS_PER_TOKEN.")
    parser.add_argument("--n_predict", type=int, default=2048,
                        help="Max generated tokens.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="JSL model batch size.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Generation temperature.")
    parser.add_argument("--use_gpu_layers", type=int, default=1000,
                        help="Number of layers offloaded to GPU for llama.cpp-backed model.")
    args = parser.parse_args()

    Path(os.path.dirname(args.output_path) or ".").mkdir(parents=True, exist_ok=True)

    # Start JSL/Spark session after `nlp.install()` has been run at least once.
    # spark = nlp.start()
    try:
        spark = nlp.start(hardware_target="gpu")
        # Pin parallelism to 1. repartition() alone is not enough: it shuffles,
        # but the annotator's transform stage still fans out to one task per
        # core, and each task loads its own copy of the 4.6GB model. That
        # exhausts host RAM and kills the JVM ("Answer from Java side is
        # empty"), which is survivable with --mem=48G in batch but not in a
        # default interactive session.
        spark.conf.set("spark.sql.shuffle.partitions", "1")
        try:
            spark.sparkContext.setLocalProperty("spark.default.parallelism", "1")
        except Exception:
            pass

        with open(args.input_path, "r") as f:
            rows = json.load(f)
        if args.limit:
            rows = rows[:args.limit]
            print(f"SMOKE TEST: limited to first {len(rows)} encounters")

        # Build prompts.
        #
        # The prompt budget is derived from the model's context rather than
        # hard-coded. setNCtx below is 16384 TOKENS; at ~4 chars/token that is
        # ~65k characters, of which n_predict must be reserved for the answer.
        # The previous fixed 15000-char cap used under a quarter of that and
        # truncated 76/108 ADAPTS transcripts (median 18.7k chars) and 7/45
        # ENCOMPASS ones -- silently discarding most of each session.
        # Budget against the context llama.cpp actually enforces. Measured on
        # ADAPTS: 25,595 chars -> 6,000 tokens, i.e. ~4.27 chars/token, so
        # CHARS_PER_TOKEN=3.5 is conservative in the safe direction.
        effective_ctx = args.real_n_ctx or args.n_ctx
        max_prompt_chars = args.max_prompt_chars or int(
            (effective_ctx - args.n_predict) * CHARS_PER_TOKEN
        )
        if args.real_n_ctx and args.real_n_ctx != args.n_ctx:
            print(f"NOTE: sizing prompts against real_n_ctx={args.real_n_ctx}, "
                  f"not n_ctx={args.n_ctx} (setNCtx does not reach llama.cpp)")
        prepared_rows = []
        truncated = []
        for row in rows:
            conversation = row.get(args.text_col, "")
            prompt = build_prompt(conversation, DEFAULT_SYSTEM_PROMPT)
            full_len = len(prompt)
            if len(prompt) > max_prompt_chars:
                truncated.append((row.get(args.id_col), len(prompt)))
                prompt = prompt[:max_prompt_chars]
            prepared_rows.append({
                "idx": row.get(args.id_col),
                "conversation": conversation,
                "gold_note": row.get(args.reference_col, ""),
                # Per-session truncation is recorded so downstream analysis can
                # separate "the model summarised badly" from "the model never
                # saw the second half of the session". llama.cpp enforces a
                # 4096-token context regardless of setNCtx, so most ADAPTS
                # transcripts are cut; without this the two are indistinguishable.
                "prompt_chars_original": full_len,
                "prompt_chars_used": len(prompt),
                "prompt_truncated": full_len > len(prompt),
                "prompt_fraction_seen": round(len(prompt) / full_len, 4) if full_len else 1.0,
                "text": prompt,
            })

        print(f"prompt budget: {max_prompt_chars} chars "
              f"(ctx={effective_ctx}, n_predict={args.n_predict})")
        if truncated:
            worst = max(t[1] for t in truncated)
            print(f"WARNING: {len(truncated)}/{len(rows)} prompts truncated "
                  f"(longest {worst} chars). Truncated sessions lose the tail of "
                  f"the transcript, which is where plan content usually sits.")
            for idx, n in truncated[:10]:
                print(f"   {idx}: {n} chars")
            if len(truncated) > 10:
                print(f"   ... and {len(truncated) - 10} more")
        else:
            print("no prompts truncated")

        pdf = pd.DataFrame(prepared_rows)
        # One partition, deliberately.
        #
        # Spark otherwise splits the rows across one task per core, and each
        # task instantiates its OWN copy of the GGUF model. On a 12-core
        # session that is 12 x 4.6GB against 40GB of VRAM: the JVM is killed
        # outright and py4j reports "Answer from Java side is empty".
        #
        # Batching still happens inside the annotator via setBatchSize, so a
        # single partition costs throughput only, not correctness.
        sdf = spark.createDataFrame(pdf).repartition(args.partitions)
        print(f"partitions: {sdf.rdd.getNumPartitions()}")

        document_assembler = nlp.DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        # Model page shows both MedicalLLM.pretrained(...) and AutoGGUFModel.pretrained(...).
        # This version follows the GGUF/llamacpp path shown on the model page.
        # outputcol: completions -> soap
        # setNCtx() is honoured by the Python object (getNCtx() returns what you
        # set) but NOT by the llama.cpp runtime, which reported n_ctx=4096 while
        # the object held 16384. The model card class may bind it differently,
        # so allow switching between the two classes the vendor documents.
        # NB: medical.AutoGGUFModel and medical.MedicalLLM are the SAME class in
        # sparknlp_jsl (AutoGGUFModel is an alias), so this switch cannot change
        # how nCtx binds. Kept only so the log records which name was requested.
        model_cls = (medical.MedicalLLM if args.model_class == "medicalllm"
                     else medical.AutoGGUFModel)
        print(f"model class: {model_cls.__name__}")

        medical_llm = model_cls.pretrained(
            args.model_name, "en", "clinical/models"
        ).setInputCols("document") \
        .setOutputCol("completions") \
        .setBatchSize(args.batch_size) \
        .setNPredict(args.n_predict) \
        .setUseChatTemplate(bool(args.use_chat_template)) \
        .setTemperature(args.temperature) \
        .setNCtx(args.n_ctx)

        # Some installations support GPU layer offload on this model type.
        try:
            medical_llm = medical_llm.setNGpuLayers(args.use_gpu_layers)
        except Exception:
            pass

        if args.debug_raw:
            # llama.cpp reported n_ctx=4096 despite setNCtx(16384), so confirm
            # whether the parameter is reaching the model at all.
            print("=== model params after configuration ===")
            try:
                # Unfiltered: setNCtx is honoured by the object but ignored by
                # llama.cpp, so list every param in case another one controls
                # the runtime context (rope scaling, n_ctx_train override, ...).
                for param, value in sorted(medical_llm.extractParamMap().items(),
                                           key=lambda kv: getattr(kv[0], "name", "")):
                    name = getattr(param, "name", str(param))
                    print(f"   {name} = {str(value)[:120]!r}")
                print("   --- params with no value set ---")
                shown = {getattr(p_, "name", "") for p_ in medical_llm.extractParamMap()}
                for p_ in medical_llm.params:
                    if getattr(p_, "name", "") not in shown:
                        print(f"   {p_.name}  (unset)  {p_.doc[:90]}")
            except Exception as exc:
                print(f"   could not read param map: {exc}")
            for getter in ("getNCtx", "getNPredict", "getBatchSize"):
                try:
                    print(f"   {getter}() -> {getattr(medical_llm, getter)()!r}")
                except Exception as exc:
                    print(f"   {getter}() unavailable: {type(exc).__name__}")
            print("=== end model params ===")

        pipeline = Pipeline(stages=[document_assembler, medical_llm])

        result_df = pipeline.fit(sdf).transform(sdf)

        # Pull only the needed columns back
        result_rows = result_df.select(
            "idx", "conversation", "gold_note", "completions",
            "prompt_chars_original", "prompt_chars_used",
            "prompt_truncated", "prompt_fraction_seen").collect()

        if args.debug_raw and result_rows:
            raw = result_rows[0]["completions"]
            print("=== RAW completions (row 0) ===")
            print(f"  python type: {type(raw)}")
            if isinstance(raw, list):
                print(f"  list length: {len(raw)}")
                for i, item in enumerate(raw[:3]):
                    print(f"    [{i}] type={type(item)}")
                    for attr in ("result", "annotatorType", "begin", "end"):
                        if hasattr(item, attr):
                            v = getattr(item, attr)
                            print(f"        .{attr} = {str(v)[:300]!r}")
                    if hasattr(item, "metadata"):
                        md = getattr(item, "metadata") or {}
                        for k, v in (md.items() if hasattr(md, "items") else []):
                            print(f"        .metadata[{k!r}] = {str(v)[:800]!r}")
            else:
                print(f"  repr: {repr(raw)[:500]}")
            print(f"  extract_text2soap_output -> "
                  f"{repr(extract_text2soap_output(raw))[:300]}")
            print("=== END RAW ===")

        outputs = []
        llama_errors = []
        for row in result_rows:
            # Spark NLP swallows llama.cpp failures into annotation metadata and
            # returns an empty result. Left unreported, that is indistinguishable
            # from a model that legitimately produced nothing -- which is exactly
            # how silent "degenerate outputs" get into a results table.
            for ann in (row["completions"] or []):
                md = getattr(ann, "metadata", None) or {}
                if hasattr(md, "get") and md.get("llamacpp_exception"):
                    llama_errors.append((row["idx"], str(md["llamacpp_exception"])))
                    break
            pred = extract_text2soap_output(row["completions"])
            outputs.append({
                "idx": row["idx"],
                "conversation": row["conversation"],
                "gold_note": row["gold_note"],
                "pred_note": pred,
                "prompt_chars_original": row["prompt_chars_original"],
                "prompt_chars_used": row["prompt_chars_used"],
                "prompt_truncated": row["prompt_truncated"],
                "prompt_fraction_seen": row["prompt_fraction_seen"],
            })

        if llama_errors:
            print(f"\nWARNING: llama.cpp raised on {len(llama_errors)}/{len(result_rows)} "
                  f"encounters; those notes are EMPTY, not merely short:")
            for idx, msg in llama_errors[:5]:
                print(f"   {idx}: {msg[:400]}")
            if len(llama_errors) > 5:
                print(f"   ... and {len(llama_errors) - 5} more")

        with open(args.output_path, "w") as f:
            json.dump(outputs, f, indent=2)

        print(f"Saved {len(outputs)} predictions to {args.output_path}")
    
    # close lingering spark sessions
    finally:
        if 'spark' in locals():
            spark.stop()
            print("✅ License checked back in successfully.")


if __name__ == "__main__":
    main() # run with "bash run_johnsnowlabs.sh" 
    
    # python johnsnowlabs_eval.py \
    # --input_path ./cache/json_dataset.json \
    # --output_path ./cache/outputs/jsl_text2soap_predictions.json \
    # --text_col conversation \
    # --id_col idx \
    # --reference_col full_note \
    # --n_predict 1024 \
    # --batch_size 1 \
    # --temperature 0.0