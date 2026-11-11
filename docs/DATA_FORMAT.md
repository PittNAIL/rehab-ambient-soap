# Data formats

All inputs and outputs are JSON. The study data (diarized session transcripts,
therapist-authored notes, model generations, and every cache derived from them)
contain protected health information and are **not** distributed with this
repository; access is described in the README's *Datasets* section and the
paper's *Data availability* statement. The synthetic files in
[`example_data/`](../example_data) demonstrate every schema.

## 1. Datasets — generation input

| File | Corpus key | Paper name | Encounters | Disciplines |
|---|---|---|---|---|
| `data/json_dataset.json` | `encompass` | Study 1 | 45 (29 patients) | 19 OT, 19 PT, 7 SLP |
| `data/adapts_dataset.json` | `adapts` | Study 2 | 108 (16 patients) | 102 OT, 6 PT |

One object per encounter:

```json
[
  {
    "idx": "9001_PT",
    "conversation": "SPK_0: ...\nSPK_1: ...",
    "full_note": "therapist-authored note (gold standard)"
  }
]
```

| Field | Type | Notes |
|---|---|---|
| `idx` | string | Encounter ID ending in `_<discipline>` (`OT`, `PT`, `SLP`); the discipline suffix is what the discipline analyses read. Study 2 IDs also encode the session; treat every real `idx` as an identifier. |
| `conversation` | string | Cleaned, diarized AWS transcript. At prompt time `SPK_0` → *Doctor*, `SPK_1` → *Client*, `SPK_n` (n ≥ 2) → *Other Speaker n−1*. |
| `full_note` | string | Narrative note written by the treating therapist during routine documentation. Copied to `gold_note` downstream. |

Study 2 is built from the raw XML transcripts and the session-note CSV by
`data_prep/build_study2_dataset.py`.

## 2. Generated notes — `metrics_test/eval_<corpus>/<system>_<corpus>.json`

One file per system per corpus. Minimum keys read by evaluation:

```json
[
  {
    "idx": "9001_PT",
    "conversation": "SPK_0: ...",
    "gold_note": "therapist note",
    "pred_note": "model-generated SOAP note"
  }
]
```

`conversation` is needed only by the transcript-grounding scripts. Generators
also record provenance that evaluation ignores:

| Field | Written by | Meaning |
|---|---|---|
| `prompt`, `model`, `model_provider` | all text generators | Exact prompt sent and model identifier. |
| `prompt_variant`, `prompt_style`, `quantization`, `n_prompt_tokens`, `truncated` | `run_openweight.py` | `no-examples` for every reported open-weight note. |
| `prompt_chars_original`, `prompt_chars_used`, `prompt_truncated`, `prompt_fraction_seen` | `johnsnowlabs_eval.py` | How much of the transcript JSL Text2SOAP actually saw. |

System stems (`<system>`):

| Stem | System |
|---|---|
| `gpt5-pro` | GPT-5 Pro |
| `healthscribe` | AWS HealthScribe |
| `jsl-text2soap` | JSL Text2SOAP |
| `medgemma-27b` | MedGemma-27B-it |
| `med42-70b` | Llama3-Med42-70B |
| `asclepius-llama3-8b` | Asclepius-Llama3-8B |
| `openbiollm-8b` | Llama3-OpenBioLLM-8B |

## 3. Downstream artifacts (under `metrics_test/`)

| Path | Produced by | Contents |
|---|---|---|
| `results_<c>/conventional_text_metrics_emptyonly.json` | `conventional_text_metrics.py` | Per-note ROUGE / BERTScore / Jaccard / readability. |
| `results_<c>/metric_reports/<system>_<c>_cui_coverage.json` | `metamap_coverage.py` | Per-note CUI sets, detection counts, semantic types, F1. |
| `results_<c>/concept_iou/…_conceptual_iou.json` | `concept_iou.py` | Concept IoU. |
| `results_pooled/16_concept_frequency_tables.json` | `build_concept_tables.py` | Ranked concept frequencies (Table 2, Suppl. Table 6). |
| `results_pooled/ontology_artifact_ablation.json` | `ontology_artifact_ablation.py` | Artifact CUIs and filtered F1. |
| `results_<c>/11_factehr_exact_claims.json` | `factehr_pipeline.py` | Verified claims, cross-entailment, recall/precision/F1, degeneracy flag. |
| `results_<c>/transcript/18_transcript_cell_1_3.json` | `transcript_fact_extraction.py` | Verified transcript claims and four-way coverage. |
| `results_<c>/grounding/13_hallucination_grounding.json` | `hallucination_grounding_analysis.py` | AI-exclusive claims: encounter-supported vs fabricated. |
| `results_<c>/grounding/16_missed_fact_grounding.json` | `missed_fact_grounding_analysis.py` | Missed gold claims: in transcript vs beyond transcript. |
| `results_<c>/creola/14_creola_category_gaps.json` | `creola_category_gap_analysis.py` | CREOLA labels (two passes + adjudication). |
| `results_pooled/15_uncategorized_topic_model.json` | `topic_model_uncategorized.py` | TopicGPT topics and fact → topic map. |

The one derived artifact that contains no PHI,
[`resources/artifact_cuis.json`](../resources/artifact_cuis.json) (the 198
audited MetaMapLite artifact CUIs with rule attribution), is committed.

## Credentials

Never commit credentials. Use `.env` (see [`.env.example`](../.env.example)) or
`keys.json` (see [`keys.example.json`](../keys.example.json)); both are
git-ignored.
