import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


# Select the open-weight model to run. Override with the MODEL_PATH env var, e.g.
#   MODEL_PATH="m42-health/Llama3-Med42-70B" python generation/mainstream_opensource_models.py
# Options used in the paper:
#   google/medgemma-27b-it
#   m42-health/Llama3-Med42-70B
#   aaditya/Llama3-OpenBioLLM-8B
MODEL_PATH = os.environ.get("MODEL_PATH", "google/medgemma-27b-it")

INPUT_PATH = "data/json_dataset.json"
OUTPUT_PATH = f"outputs/{os.path.basename(MODEL_PATH)}.json"
USE_SOAP_PROMPT = True

if USE_SOAP_PROMPT:
    OUTPUT_PATH = OUTPUT_PATH.replace('outputs/', 'outputs/soap_prompts/')
    os.makedirs('outputs/soap_prompts/', exist_ok=True)

print(f"Saving output to {OUTPUT_PATH}")


if ('medgemma' in MODEL_PATH) or ('m42-health' in MODEL_PATH):
    QUANTIZED = True
else:
    QUANTIZED = False

BATCH_SIZE = 2  # adjust based on your GPU memory
if "7b" in MODEL_PATH:
    BATCH_SIZE = 4

os.makedirs("outputs", exist_ok=True)

def format_prompt(input_text: str) -> str:
    """
    Format prompt for inference with model-specific formatting.
    Models supported: meditron, llama, mistral.
    """
    if ('medgemma' in MODEL_PATH) or ('Llama3-OpenBioLLM-8B' in MODEL_PATH):
        prompt = (
            "Generate a SOAP note from the following dialogue.\n"
            "S:\n"
            "O:\n"
            "A:\n"
            "P:\n"
            "Dialogue:\n"
        )
        prompt += input_text
        
    elif 'm42-health' in MODEL_PATH: # 'Llama3-Med42-8B' in MODEL_PATH:
        # Med42 prefers chat-style prompting
        prompt = "<|system|>\n"
        "You are a clinical assistant.\n"
        "<|user|>\n"
        "Generate a SOAP note from this dialogue:\n"
        prompt += input_text
        prompt += "\n<|assistant|>\n"
        # import pdb;pdb.set_trace()

    return prompt


def format_soap_prompt(input_text: str) -> str:
    """
    Format custom SOAP prompt for inference adapting to model-specific formatting.
    Models supported: meditron, llama, mistral.
    """
    if ('medgemma' in MODEL_PATH) or ('Llama3-OpenBioLLM-8B' in MODEL_PATH):
        prompt = (
            "You are a clinical assistant observing therapy sessions. Identify key components of the dialogue, and document it in a narrative SOAP format (paragraph-style). “Patients” are referred to as “Clients”.\n"
            "The key components of each SOAP part are as follows:\n"
            "**Subjective**\n"
                "Symptoms (e.g. Client reports fatigue.)\n"
                "Concerns (e.g. Client reports fear of falling.)\n"
                "Pertinent notes about their presentation (e.g. Client alert and agreeable to therapy session).\n"
            "**Objective**\n"
                "Vital signs (e.g. Client SOB, SPO2 92% on room air).\n"
                "Assessment results (e.g. Client recalled 4/5 words given at beginning of the session).\n"
                "Statement of activities (e.g. Client walked 100 feet with min (A) and ww.; Client completed laundry tasks).\n"
            "**Assessment**\n"
                "*Interpretation of subjective and objective sections*\n"
                "During ambulation, client required moderate guiding cues for safe hand placement, indicating continued deficits in safety awareness during functional transfers.\n"
                "Client with one LOB during laundry task. Upon reflection, client demonstrated poor insight into why she lost her balance and how to prevent future losses of balance.\n"
            "**Plan**\n"
                "*Description of the plan moving forward, including goals, patient education, etc.*\n"
                "Client provided education on falls prevention, will continue incorporating into future sessions.\n"
                "Client left seated in room with call bell in reach, RN aware. Next session to focus on memory and problem solving.\n"
            "Dialogue:\n"
        )
        prompt += input_text
        prompt += "Do not hallucinate or fabricate new information\n"
        
        
    elif 'm42-health' in MODEL_PATH: # 'Llama3-Med42-8B' in MODEL_PATH:
        # Med42 prefers chat-style prompting
        prompt = "<|system|>\n"
        "You are a clinical assistant observing therapy sessions.\n"
        "<|user|>\n"
        "You are a clinical assistant observing therapy sessions. Identify key components of the dialogue, and document it in a narrative SOAP format (paragraph-style). “Patients” are referred to as “Clients”.\n"
        "The key components of each SOAP part are as follows:\n"
        "**Subjective**\n"
        "Symptoms (e.g. Client reports fatigue.)\n"
        "Concerns (e.g. Client reports fear of falling.)\n"
        "Pertinent notes about their presentation (e.g. Client alert and agreeable to therapy session).\n"
        "**Objective**\n"
        "Vital signs (e.g. Client SOB, SPO2 92% on room air).\n"
        "Assessment results (e.g. Client recalled 4/5 words given at beginning of the session).\n"
        "Statement of activities (e.g. Client walked 100 feet with min (A) and ww.; Client completed laundry tasks).\n"
        "**Assessment**\n"
        "*Interpretation of subjective and objective sections*\n"
        "During ambulation, client required moderate guiding cues for safe hand placement, indicating continued deficits in safety awareness during functional transfers.\n"
        "Client with one LOB during laundry task. Upon reflection, client demonstrated poor insight into why she lost her balance and how to prevent future losses of balance.\n"
        "**Plan**\n"
        "*Description of the plan moving forward, including goals, patient education, etc.*\n"
        "Client provided education on falls prevention, will continue incorporating into future sessions.\n"
        "Client left seated in room with call bell in reach, RN aware. Next session to focus on memory and problem solving.\n"
        "Dialogue:\n"
        prompt += input_text
        prompt += "Do not hallucinate or fabricate new information\n"
        prompt += "\n<|assistant|>\n"

    return prompt


def build_prompt(conversation: str, use_soap_prompt: bool=False) -> str:
    if use_soap_prompt:
        return format_soap_prompt(conversation)
    return format_prompt(conversation)


class ClinicalNotesDataset(Dataset):
    def __init__(self, input_path: str, use_soap_prompt: bool=False):
        with open(input_path, "r") as f:
            self.rows = json.load(f)
        self.use_soap_prompt = use_soap_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        if self.use_soap_prompt:
            conversation = (
                row["conversation"]
                .replace("SPK_0", "Doctor")
                .replace("SPK_1", "Client")
            )
        else:
            conversation = (
                row["conversation"]
                .replace("SPK_0", "Doctor")
                .replace("SPK_1", "Patient")
            )
        prompt = build_prompt(conversation=conversation, use_soap_prompt=self.use_soap_prompt)
        return {
            "idx": row["idx"],
            "conversation": row["conversation"],
            "gold_note": row.get("full_note", ""),
            "prompt": prompt,
        }


def make_collate_fn(tokenizer):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [item["prompt"] for item in batch]

        tokenized = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "idxs": [item["idx"] for item in batch],
            "conversations": [item["conversation"] for item in batch],
            "gold_notes": [item["gold_note"] for item in batch],
            "prompts": prompts,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    return collate_fn


def main():
    Path("outputs").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if QUANTIZED:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    model.config.use_cache = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Important for decoder-only batched generation
    tokenizer.padding_side = "left"

    dataset = ClinicalNotesDataset(INPUT_PATH, use_soap_prompt=USE_SOAP_PROMPT)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer),
    )

    records = []

    for batch in tqdm(dataloader, total=len(dataloader)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # Decode only generated continuation, not the original prompt
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        for i in range(outputs.size(0)):
            generated_ids = outputs[i, prompt_lengths[i]:]
            pred_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            record = {
                "idx": batch["idxs"][i],
                "conversation": batch["conversations"][i],
                "gold_note": batch["gold_notes"][i],
                "pred_note": pred_text,
                "full_pred": tokenizer.decode(outputs[i], skip_special_tokens=True),
            }
            records.append(record)

    with open(OUTPUT_PATH, "w") as out_f:
        json.dump(records, out_f, indent=2)

    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()