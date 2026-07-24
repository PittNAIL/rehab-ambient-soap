# Data formats

All inputs and outputs are JSON. Real study data (patient transcripts, therapist
reference notes, model generations) contains PHI and is **not** distributed with
this repository — see the *Data availability* statement in the paper. The
synthetic files in [`example_data/`](../example_data) demonstrate every schema so
the pipeline runs end to end without protected data.

## 1. Generation input — `data/json_dataset.json`

One object per encounter. This is the input to every generation script.

```json
[
  {
    "idx": "9001_PT",
    "conversation": "SPK_0: ... SPK_1: ...",
    "full_note": "therapist-authored reference (gold) note"
  }
]
```

| Field | Type | Notes |
|---|---|---|
| `idx` | string | Encounter ID. Convention: `<session>_<discipline>` (`PT`/`OT`/`SLP`). |
| `conversation` | string | Diarized transcript. `SPK_0` is normalized to *Doctor*, `SPK_1` to *Client* (`Patient` for the non-SOAP prompt) at load time. |
| `full_note` | string | Therapist reference note. Copied into `gold_note` downstream. |

## 2. Generation output / evaluation input — `metrics_test/soap_prompts/<model>.json`

Each generation script writes one file per model. Downstream evaluation reads
this folder. Minimum required keys:

```json
[
  {
    "idx": "9001_PT",
    "conversation": "SPK_0: ... SPK_1: ...",
    "gold_note": "therapist reference note",
    "pred_note": "model-generated SOAP note"
  }
]
```

`conversation` is required only by the transcript-grounding scripts. The API
generators also store `prompt`, `model`, and `model_provider`; these are ignored
by evaluation.

The model file stems used in the paper (referenced by the reporting scripts):

| Stem | System |
|---|---|
| `gpt5-pro_soap_preds` | GPT-5-Pro |
| `healthscribe_summaries` | AWS HealthScribe |
| `jsl_text2soap_predictions` | JSL Text2SOAP |
| `medgemma-27b-it` | MedGemma-27B |
| `Llama3-Med42-70B` | Med42-70B |
| `asclepius_13b_predictions` | Asclepius-13B |
| `Llama3-OpenBioLLM-8B` | OpenBioLLM-8B |

## 3. Key downstream artifacts (written under `metrics_test/`)

| Path | Produced by | Contents |
|---|---|---|
| `metric_reports/<model>_cui_coverage.json` | `metamap_coverage.py` | Per-note CUI recall/precision/F1 + CUI lists. |
| `quantitative_metrics/conventional_text_metrics*.json` | `conventional_text_metrics.py` | Per-note lexical/semantic metrics + BCa summaries. |
| `gap_analysis/11_factehr_exact_claims.json` | `factehr_pipeline.py` | Verified atomic claims + cross-entailment (claim recall/precision/F1). |
| `gap_analysis/18_transcript_cell_1_3.json` | `transcript_fact_extraction.py` | Verified transcript claim set + 2×2 coverage. |
| `gap_analysis/13_hallucination_grounding.json` | `hallucination_grounding_analysis.py` | AI-exclusive claims split fabricated vs encounter-supported. |
| `gap_analysis/16_missed_fact_grounding.json` | `missed_fact_grounding_analysis.py` | Missed claims split in-transcript vs beyond-transcript. |
| `gap_analysis/14_creola_category_gaps.json` | `creola_category_gap_analysis.py` | CREOLA content-category labels (two passes + adjudication). |
| `gap_analysis/15_uncategorized_topic_model.json` | `topic_model_uncategorized.py` | TopicGPT topics for uncategorized claims. |

## Credentials

Do not commit credentials. Provide them via environment variables (`.env`, see
[`.env.example`](../.env.example)) or a `keys.json` file (see
[`keys.example.json`](../keys.example.json)). Both patterns are git-ignored.
