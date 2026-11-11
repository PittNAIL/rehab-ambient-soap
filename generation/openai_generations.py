#!/usr/bin/env python3
"""
Generate clinical note summaries from a PyTorch Dataset using:

1. GPT-5 Pro via the Azure OpenAI Responses API (the system evaluated in the paper)
2. A Claude model via the Microsoft Foundry Claude Messages API (optional; not
   part of the paper's evaluation)

Paper configuration (GPT-5 Pro): --provider azure-gpt5 --model <gpt-5-pro deployment>
--use-soap-prompt --max-output-tokens 10000. The few-shot SOAP_PROMPT_PREFIX below
(worked clinical examples) is the prompt GPT-5 Pro received; see Supplementary
Note 1. Temperature is not user-configurable for this reasoning model, so it is
not sent on the Responses API path.

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

# Endpoints and keys come from the environment or .env (see .env.example).

import argparse
import json
import os
import re

def _load_dotenv(path: str = ".env") -> None:
    """
    Populate os.environ from a .env file if present.

    Credentials live in .env but nothing was reading it, so the script only
    worked when the key happened to be exported in the shell. Values already
    set in the environment win, so an explicit export still overrides the file.
    No third-party dependency needed for KEY=value lines.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# setdefault, NOT assignment. This previously hard-assigned 'gpt-5.4-mini', which
# silently overrode the environment. It happened to be harmless because --model
# takes precedence, but running without --model would have sent every session to
# mini while the operator believed they were using pro.
os.environ.setdefault('AZURE_OPENAI_gpt5_DEPLOYMENT', 'gpt-5-pro')
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

_SPEAKER_RE = re.compile(r"SPK_(\d+)")


def normalize_speakers(conversation: str, second_speaker: str = "Client") -> str:
    """
    Map diarization labels to readable roles.

    Previously only SPK_0/SPK_1 were replaced, so a raw "SPK_2:" leaked into the
    prompt for the 9 of 45 sessions in the earlier corpus that have a third or
    fourth speaker (family member, aide, second therapist). Those now become
    "Other Speaker 1", "Other Speaker 2", ... matching the open-weight runner
    exactly, so every system sees identical transcripts.

    ADAPTS contains only SPK_0/SPK_1, so this is a no-op there and does not
    change the already-generated ADAPTS GPT-5 output.
    """
    def repl(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        if n == 0:
            return "Doctor"
        if n == 1:
            return second_speaker
        return f"Other Speaker {n - 1}"

    return _SPEAKER_RE.sub(repl, conversation)


def build_prompt(conversation: str, use_soap_prompt: bool = False) -> str:
    """
    Prepend the worked-example SOAP instruction to the transcript.

    SYSTEM_INSTRUCTION is sent separately as the system message and already ends
    with "Dialogue:\n", so the transcript follows the prefix directly.
    """
    if use_soap_prompt:
        return SOAP_PROMPT_PREFIX + conversation
    return conversation


class ClinicalNotesDataset(Dataset):
    def __init__(self, input_path: str, use_soap_prompt: bool = False):
        with open(input_path, "r", encoding="utf-8") as f:
            self.rows = json.load(f)
        self.use_soap_prompt = use_soap_prompt

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]

        second = "Client" if self.use_soap_prompt else "Patient"
        conversation = normalize_speakers(row["conversation"], second)

        prompt = build_prompt(
            conversation=conversation,
            use_soap_prompt=self.use_soap_prompt,
        )

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
            # .env defines AZURE_OPENAI_API_KEY; OPENAI_API_KEY kept as a fallback
            # so existing shell exports continue to work.
            azure_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

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
    
# Paper run (repeat with --input-path data/adapts_dataset.json for Study 2):
# python generation/openai_generations.py --input-path data/json_dataset.json #   --output-path outputs/encompass/gpt5-pro_encompass_preds.jsonl #   --final-json-path outputs/encompass/gpt5-pro_encompass_preds.json #   --provider azure-gpt5 --model <gpt-5-pro deployment> --use-soap-prompt #   --max-output-tokens 10000
