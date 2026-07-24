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


def extract_text2soap_output(value):
    """
    Handle common Spark NLP / JSL output shapes.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

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
        return "\n".join(t.strip() for t in texts if t).strip()

    # Row-like object
    result = getattr(value, "result", None)
    if result is not None:
        return str(result).strip()

    return str(value).strip()


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
    parser.add_argument("--n_predict", type=int, default=1024,
                        help="Max generated tokens.")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="JSL model batch size. Start with 1 unless you know memory is enough.")
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

        with open(args.input_path, "r") as f:
            rows = json.load(f)

        # Build prompts
        prepared_rows = []
        for row in rows:
            conversation = row.get(args.text_col, "")
            prompt = build_prompt(conversation, DEFAULT_SYSTEM_PROMPT)
            print(len(prompt))
            if len(prompt) > 15000: # truncate for context length
                prompt = prompt[:15000]
            prepared_rows.append({
                "idx": row.get(args.id_col),
                "conversation": conversation,
                "gold_note": row.get(args.reference_col, ""),
                "text": prompt,
            })

        pdf = pd.DataFrame(prepared_rows)
        sdf = spark.createDataFrame(pdf)

        document_assembler = nlp.DocumentAssembler() \
            .setInputCol("text") \
            .setOutputCol("document")

        # Model page shows both MedicalLLM.pretrained(...) and AutoGGUFModel.pretrained(...).
        # This version follows the GGUF/llamacpp path shown on the model page.
        # outputcol: completions -> soap
        medical_llm = medical.AutoGGUFModel.pretrained(
            args.model_name, "en", "clinical/models"
        ).setInputCols("document") \
        .setOutputCol("completions") \
        .setBatchSize(args.batch_size) \
        .setNPredict(args.n_predict) \
        .setUseChatTemplate(True) \
        .setTemperature(args.temperature) \
        .setNCtx(16384)  # Increase this to 8192 or 16384 for long clinical notes

        # Some installations support GPU layer offload on this model type.
        try:
            medical_llm = medical_llm.setNGpuLayers(args.use_gpu_layers)
        except Exception:
            pass

        pipeline = Pipeline(stages=[document_assembler, medical_llm])

        result_df = pipeline.fit(sdf).transform(sdf)

        # Pull only the needed columns back
        result_rows = result_df.select("idx", "conversation", "gold_note", "completions").collect()

        outputs = []
        for row in result_rows:
            pred = extract_text2soap_output(row["completions"])
            outputs.append({
                "idx": row["idx"],
                "conversation": row["conversation"],
                "gold_note": row["gold_note"],
                "pred_note": pred,
            })

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