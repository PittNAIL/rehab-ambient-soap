#!/usr/bin/env python3
"""
Generate clinical note summaries from a PyTorch Dataset using:

1. GPT-5.5 via Azure OpenAI / Microsoft Foundry Responses API
2. Claude Opus 4.6 via Microsoft Foundry Claude Messages API

Input JSON expected:
[
  {
    "idx": 0,
    "conversation": "SPK_0: ... SPK_1: ...",
    "full_note": "optional gold/reference note"
  },
  ...
]

Output JSONL:
Each line contains:
{
  "idx": ...,
  "conversation": ...,
  "gold_note": ...,
  "prompt": ...,
  "pred_note": ...,
  "model_provider": ...,
  "model": ...
}
"""

# export AZURE_OPENAI_gpt5_DEPLOYMENT="your-gpt-5.5-deployment-name"

import argparse
import json
import os
# Configure these via environment variables (see .env.example) rather than
# hard-coding institutional endpoints. Example values:
#   AZURE_OPENAI_ENDPOINT   = https://<your-resource>.openai.azure.com/openai/v1/
#   AZURE_CLAUDE_BASE_URL   = https://<your-resource>.services.ai.azure.com/anthropic/
#   AZURE_OPENAI_gpt5_DEPLOYMENT = <your-deployment-name>
os.environ.setdefault('AZURE_OPENAI_gpt5_DEPLOYMENT', os.getenv('AZURE_OPENAI_gpt5_DEPLOYMENT', 'gpt-5-pro'))
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from openai import OpenAI
from anthropic import AnthropicFoundry


# SOAP_PROMPT_PREFIX = (
#     "You are a clinical assistant observing therapy sessions. Identify key components "
#     "of the dialogue, and document it in a narrative SOAP format (paragraph-style). "
#     "“Patients” are referred to as “Clients”.\n"
#     "The key components of each SOAP part are as follows:\n"
#     "**Subjective**\n"
#     "Symptoms (e.g. Client reports fatigue.)\n"
#     "Concerns (e.g. Client reports fear of falling.)\n"
#     "Pertinent notes about their presentation (e.g. Client alert and agreeable to therapy session).\n"
#     "**Objective**\n"
#     "Vital signs (e.g. Client SOB, SPO2 92% on room air).\n"
#     "Assessment results (e.g. Client recalled 4/5 words given at beginning of the session).\n"
#     "Statement of activities (e.g. Client walked 100 feet with min (A) and ww.; Client completed laundry tasks).\n"
#     "**Assessment**\n"
#     "*Interpretation of subjective and objective sections*\n"
#     "During ambulation, client required moderate guiding cues for safe hand placement, "
#     "indicating continued deficits in safety awareness during functional transfers.\n"
#     "Client with one LOB during laundry task. Upon reflection, client demonstrated poor "
#     "insight into why she lost her balance and how to prevent future losses of balance.\n"
#     "**Plan**\n"
#     "*Description of the plan moving forward, including goals, patient education, etc.*\n"
#     "Client provided education on falls prevention, will continue incorporating into future sessions.\n"
#     "Client left seated in room with call bell in reach, RN aware. Next session to focus "
#     "on memory and problem solving.\n"
#     # "Dialogue:\n"
# )

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


SYSTEM_INSTRUCTION = (
    "You are a clinical assistant observing a doctor-patient conversation. "
    "Generate a concise clinical note summary from the dialogue.\n"
    "Dialogue:\n"
)

class ClinicalNotesDataset(Dataset):
    def __init__(self, input_path: str, use_soap_prompt: bool = False):
        with open(input_path, "r", encoding="utf-8") as f:
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

        # prompt = build_prompt(
        #     conversation=conversation,
        #     use_soap_prompt=self.use_soap_prompt,
        # )
        
        prompt = conversation

        return {
            "idx": row["idx"],
            "conversation": row["conversation"],
            "gold_note": row.get("full_note", ""),
            "prompt": prompt,
        }


def simple_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep batch as a list of dictionaries.

    This avoids PyTorch trying to tensorize strings.
    """
    return batch


class FoundryNoteGenerator:
    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        claude_thinking: str = "disabled",
        claude_effort: str = "medium",
    ):
        """
        provider:
            - "azure-gpt5"
            - "azure-claude-opus-4-7"

        model:
            For Azure GPT-5.5: deployment name.
            For Claude: deployment name.
        """
        self.provider = provider
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.claude_thinking = claude_thinking
        self.claude_effort = claude_effort

        # import pdb;pdb.set_trace()
        if provider == "azure-gpt5":
            self.model = model or os.getenv("AZURE_OPENAI_gpt5_DEPLOYMENT", "gpt-5")
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            azure_key = os.getenv("OPENAI_API_KEY")

            if not azure_endpoint:
                raise EnvironmentError("Missing AZURE_OPENAI_ENDPOINT")
            if not azure_key:
                raise EnvironmentError("Missing AZURE_OPENAI_API_KEY")

            # Azure OpenAI Responses API via OpenAI SDK.
            # Endpoint should look like:
            # https://<resource-name>.openai.azure.com/openai/v1/
            self.client = OpenAI(
                base_url=f"{azure_endpoint}",
                api_key=azure_key,
            )

        elif provider == "azure-claude-opus-4-7":
            self.model = model or os.getenv("AZURE_CLAUDE_DEPLOYMENT", "claude-opus-4-7")
            claude_base_url = os.getenv("AZURE_CLAUDE_BASE_URL")
            claude_key = os.getenv("AZURE_API_KEY")

            if not claude_base_url:
                raise EnvironmentError("Missing AZURE_CLAUDE_BASE_URL")
            if not claude_key:
                raise EnvironmentError("Missing AZURE_CLAUDE_API_KEY")

            # Microsoft Foundry Claude endpoint via Anthropic SDK.
            # Base URL should look like:
            # https://<resource-name>.services.ai.azure.com/anthropic
            self.client = AnthropicFoundry(
                api_key=claude_key,
                base_url=claude_base_url,
            )

        else:
            raise ValueError(
                "Unsupported provider. Use one of: "
                "'azure-gpt5', 'azure-claude-opus-4-7'"
            )

    def generate(self, prompt: str) -> str:
        if self.provider == "azure-gpt5":
            return self._generate_gpt5(prompt)

        if self.provider == "azure-claude-opus-4-7":
            return self._generate_claude(prompt)

        raise ValueError(f"Unsupported provider: {self.provider}")

    def _generate_gpt5(self, prompt: str) -> str:
        """
        GPT-5.5 via Azure OpenAI Responses API.

        Note:
        - model should be your Azure deployment name.
        - Some GPT-5.5 deployments may support reasoning params;
          keep this minimal for broad compatibility.
        """
        # import pdb;pdb.set_trace()
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            # temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        # import pdb;pdb.set_trace()

        return response.output_text.strip()

    def _generate_claude(self, prompt: str) -> str:
        """
        Claude Opus 4.6 via Microsoft Foundry Claude Messages API.

        Microsoft docs note that Opus 4.6 supports thinking:
        disabled, enabled, adaptive.

        For clinical note generation, disabled or adaptive is usually preferable.
        """
        message = self.client.messages.create(
            model=self.model,
            system=SYSTEM_INSTRUCTION,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=self.max_output_tokens,
            temperature=self.temperature,
            thinking={"type": self.claude_thinking},
            output_config={"effort": self.claude_effort},
            stream=False,
        )

        return extract_claude_text(message)


def extract_claude_text(message: Any) -> str:
    """
    Extract text from Anthropic Messages API response.

    message.content is usually a list of content blocks.
    Text blocks have .text.
    """
    parts = []

    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)

    return "\n".join(parts).strip()


def load_completed_indices_from_jsonl(output_path: Path) -> set:
    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add(row["idx"])

    return completed


def append_jsonl(output_path: Path, row: Dict[str, Any]) -> None:
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_jsonl_to_json(jsonl_path: Path, json_path: Path) -> None:
    rows = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def generate_with_retries(
    generator: FoundryNoteGenerator,
    prompt: str,
    max_retries: int = 5,
    retry_sleep: float = 2.0,
) -> Optional[str]:
    last_error = None

    for attempt in range(max_retries):
        try:
            return generator.generate(prompt)

        except Exception as e:
            error_str = str(e)
            # Content filter errors will not resolve on retry — skip immediately.
            if "content_filter" in error_str or "content management policy" in error_str:
                print(f"Content filter triggered — skipping item.")
                return ""
            last_error = e
            sleep_time = retry_sleep * (2**attempt)
            print(f"API error on attempt {attempt + 1}/{max_retries}: {e}")
            print(f"Sleeping for {sleep_time:.1f} seconds before retrying...")
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed after {max_retries} retries. Last error: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate clinical note summaries from ClinicalNotesDataset using "
            "GPT-5.5 or Claude Opus 4.6 on Microsoft Azure Foundry."
        )
    )

    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to input JSON file.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to output JSONL file.",
    )

    parser.add_argument(
        "--final-json-path",
        type=str,
        default=None,
        help="Optional path to save final JSON list after generation.",
    )

    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["azure-gpt5", "azure-claude-opus-4-7"],
        help="Model provider/backend.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Deployment name. For Azure, this is usually your deployment name, "
            "not necessarily the raw model ID."
        ),
    )

    parser.add_argument(
        "--use-soap-prompt",
        action="store_true",
        help="Use SOAP prompt and replace SPK_1 with Client.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="DataLoader batch size. API calls are still made one item at a time.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1200,
        help="Maximum output tokens per generated note.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Maximum retries per API call.",
    )

    parser.add_argument(
        "--claude-thinking",
        type=str,
        default="disabled",
        choices=["disabled", "enabled", "adaptive"],
        help="Claude thinking mode. Opus 4.6 supports disabled, enabled, adaptive.",
    )

    parser.add_argument(
        "--claude-effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "max"],
        help="Claude effort setting.",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = ClinicalNotesDataset(
        input_path=args.input_path,
        use_soap_prompt=args.use_soap_prompt,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=simple_collate_fn,
    )

    generator = FoundryNoteGenerator(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        claude_thinking=args.claude_thinking,
        claude_effort=args.claude_effort,
    )

    completed_indices = load_completed_indices_from_jsonl(output_path)
    print(f"Loaded {len(completed_indices)} completed predictions from {output_path}")

    for batch in tqdm(dataloader, desc=f"Generating notes with {args.provider}"):
        for item in batch:
            idx = item["idx"]

            if idx in completed_indices:
                continue

            pred_note = generate_with_retries(
                generator=generator,
                prompt=item["prompt"],
                max_retries=args.max_retries,
            )

            output_row = {
                "idx": idx,
                "conversation": item["conversation"],
                "gold_note": item["gold_note"],
                "prompt": item["prompt"],
                "pred_note": pred_note,
                "model_provider": args.provider,
                "model": generator.model,
            }

            append_jsonl(output_path, output_row)
            completed_indices.add(idx)

    if args.final_json_path:
        convert_jsonl_to_json(
            jsonl_path=output_path,
            json_path=Path(args.final_json_path),
        )
        print(f"Saved final JSON to {args.final_json_path}")

    print(f"Saved JSONL predictions to {output_path}")


if __name__ == "__main__":
    main()
    
# python openai_generations.py   --input-path data/json_dataset.json   --output-path outputs/gpt5-pro/gpt5-pro_soap_preds.jsonl   --final-json-path outputs/gpt5-pro/gpt5-pro_soap_preds.json   --provider azure-gpt5-pro   --model gpt-5-pro  --use-soap-prompt   --temperature 0.2   --max-output-tokens 10000

# python openai_generations.py   --input-path data/json_dataset.json   --output-path outputs/gpt5-pro/gpt5.4-mini_soap_preds.jsonl   --final-json-path outputs/gpt5.4-mini/gpt5.4-mini_soap_preds.json   --provider azure-gpt5.4-mini   --model gpt-5.4-mini  --use-soap-prompt   --temperature 0.2   --max-output-tokens 10000

# python openai_generations.py   --input-path data/json_dataset.json   --output-path outputs/claude-opus-4-7/claude-opus-4-7_soap_preds.jsonl   --final-json-path outputs/claude-opus-4-7/claude-opus-4-7_soap_preds.json   --provider azure-claude-opus-4-7   --model claude-opus-4-7  --use-soap-prompt   --temperature 0.2   --max-output-tokens 10000