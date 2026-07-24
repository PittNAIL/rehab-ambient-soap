# prompt = """You are an intelligent clinical languge model.
# Below is a snippet of patient's discharge summary and a following instruction from healthcare professional.
# Write a response that appropriately completes the instruction.
# The response should provide the accurate answer to the instruction, while being concise.

# [Discharge Summary Begin]
# {note}
# [Discharge Summary End]

# [Instruction Begin]
# {question}
# [Instruction End] 
# """

# from transformers import AutoTokenizer, AutoModelForCausalLM
# tokenizer = AutoTokenizer.from_pretrained("starmpcc/Asclepius-7B", use_fast=False)
# model = AutoModelForCausalLM.from_pretrained("starmpcc/Asclepius-7B")

# note = "This is a sample note"
# question = "What is the diagnosis?"

# model_input = prompt.format(note=note, question=question)
# input_ids = tokenizer(model_input, return_tensors="pt").input_ids
# output = model.generate(input_ids)
# print(tokenizer.decode(output[0]))

import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

USE_SOAP_PROMPT = True
MODEL_PATH = "starmpcc/Asclepius-13B"
INPUT_PATH = "data/json_dataset.json"
# OUTPUT_PATH = "outputs/asclepius/asclepius_7b_predictions_new.json"
OUTPUT_PATH = "outputs/soap_prompts/asclepius_13b_predictions.json"
BATCH_SIZE = 2  # adjust based on your GPU memory
if "7b" in MODEL_PATH:
    BATCH_SIZE = 4

os.makedirs("outputs", exist_ok=True)


def format_prompt(input_text: str, role: str) -> str:
    """
    Format prompt for inference with model-specific formatting.
    """
    prompt = f"""You are an intelligent clinical languge model.
    Below is a snippet of doctor-patient conversation and a following instruction from healthcare professional.
    Write a response that appropriately completes the instruction.
    The response should provide the accurate answer to the instruction, while being concise.

    [Conversation Begin]
    {input_text}
    [Conversation Summary End]

    [Instruction Begin]
    Generate the corresponding clinical note as written by the physician. Make sure to use all the information from the dialogue into the note, but do not add any new information. Now, generate the corresponding clinical note: 
    [Instruction End] 
    """

    return prompt

# def format_soap_prompt(input_text: str) -> str:
    """
    Format custom SOAP prompt for inference adapting to model-specific formatting.
    """
    prompt = "You are a clinical assistant observing therapy sessions. Identify key components "
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
    "[Conversation Begin]"

    
    prompt += input_text
    
    prompt += "[Conversation Summary End]"
    "[Instruction Begin]"
    "Generate the corresponding clinical note as written by the physician. Make sure to use all the information from the dialogue into the note, but do not add any new information. Now, generate the corresponding clinical note:" 
    "[Instruction End]"
    import pdb;pdb.set_trace()    
    return prompt

def format_soap_prompt(input_text: str) -> str:
    prompt = """You are a clinical assistant observing therapy sessions. Identify key components of the dialogue, and document it in a narrative SOAP format (paragraph-style). “Patients” are referred to as “Clients”.
    The key components of each SOAP part are as follows:
    **Subjective** covers Symptoms (e.g. Client reports fatigue.), Concerns (e.g. Client reports fear of falling.), and Pertinent notes about their presentation (e.g. Client alert and agreeable to therapy session).
    **Objective** covers Vital signs (e.g. Client SOB, SPO2 92% on room air), Assessment results (e.g. Client recalled 4/5 words given at beginning of the session), and Statement of activities (e.g. Client walked 100 feet with min assist and wheeled walker; Client completed laundry tasks).
    **Assessment** covers Interpretation of subjective and objective sections such as 1. During ambulation, client required moderate guiding cues for safe hand placement, indicating continued deficits in safety awareness during functional transfers.
    2. Client with one LOB during laundry task. Upon reflection, client demonstrated poor insight into why she lost her balance and how to prevent future losses of balance.
    **Plan** covers Description of the plan moving forward, including goals, patient education, etc., e.g., Client provided education on falls prevention, will continue incorporating into future sessions, and Client left seated in room with call bell in reach, RN aware. Next session to focus on memory and problem solving.
    [Conversation Begin]""" + input_text + """[Conversation Summary End][Instruction Begin]Generate the corresponding clinical note as written by the physician. Make sure to use all the information from the dialogue into the note, but do not add any new information. Now, generate the corresponding clinical note:[Instruction End]"""
    import pdb;pdb.set_trace
    return prompt

# def build_prompt(conversation: str, role: str) -> str:
#     return format_prompt(conversation, role)

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
        conversation = (
            row["conversation"]
            .replace("SPK_0", "Doctor")
            .replace("SPK_1", "Patient")
        )
        if row['idx'].endswith('OT'): 
            role = "occupational"
        elif row['idx'].endswith('PT'):
            role = "physical"
        elif row['idx'].endswith('SLP'):
            role = "speech language pathology"
        # prompt = build_prompt(conversation, role)
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
                # max_new_tokens=2048,
                max_new_tokens=10000,
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