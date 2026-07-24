# Pipeline and manuscript map

Every script and how it maps to the tables and figures in *Validation of Ambient
Generative AI Models for Clinical Documentation in Rehabilitation Settings*.

**Run all commands from the repository root.** Scripts resolve `data/`,
`metrics_test/soap_prompts/`, `metrics_test/metric_reports/`, and
`metrics_test/gap_analysis/` relative to the working directory (the two reporting
scripts resolve them relative to the repo root regardless of CWD).

> **Not included here.** The qualitative LLM-as-a-judge evaluation (main-text
> Table X and Supplementary Tables 5 & 7 — the 15-judge panel, the empty-note
> probe, Krippendorff's α, and the leaderboard) was conducted as a separate
> component and is maintained in its own repository; it is referenced in the
> paper's Code Availability statement rather than duplicated here.

---

## Stage 0 — Data

Assemble `data/json_dataset.json` (schema in [DATA_FORMAT.md](DATA_FORMAT.md)).
Real data is access-controlled; `example_data/json_dataset.example.json` is a
synthetic stand-in.

## Stage 1 — Note generation → `metrics_test/soap_prompts/<model>.json`

| System | Script | Notes |
|---|---|---|
| GPT-5-Pro | `generation/openai_generations.py` | Azure OpenAI Responses API; `--use-soap-prompt`, `--max-output-tokens 10000`. |
| MedGemma-27B, Med42-70B, OpenBioLLM-8B | `generation/mainstream_opensource_models.py` | Local HF Transformers; select with `MODEL_PATH=...`. 4-bit for MedGemma/Med42. |
| Asclepius-13B | `generation/asclepius_eval.py` | Local HF Transformers, narrative SOAP prompt. |
| JSL Text2SOAP | `generation/johnsnowlabs_eval.py` / `run_johnsnowlabs.sh` | Spark NLP AutoGGUF pipeline, vendor system prompt. |
| AWS HealthScribe | `generation/aws_healthscribe/*.py` | Audio-in → structured note → paragraph SOAP. No prompt exposed. |

Example (GPT-5-Pro):
```bash
python generation/openai_generations.py \
  --input-path data/json_dataset.json \
  --output-path outputs/gpt5-pro/gpt5-pro_soap_preds.jsonl \
  --final-json-path metrics_test/soap_prompts/gpt5-pro_soap_preds.json \
  --provider azure-gpt5 --model <deployment> --use-soap-prompt \
  --temperature 0.2 --max-output-tokens 10000
```

## Stage 2 — Conventional text metrics → **Table 1**, **Suppl. Tables 1–2**

`evaluation/conventional_text_metrics.py` — ROUGE-1/L, BERTScore, Jaccard, token
counts, lexical entropy, readability. Two filter conventions:

```bash
# Table 1 convention: score every non-empty output (degenerate included)
python evaluation/conventional_text_metrics.py \
  --input-folder metrics_test/soap_prompts \
  --output-path metrics_test/quantitative_metrics/conventional_text_metrics_emptyonly.json \
  --filter empty-only

# Non-degenerate subset (matches the "Valid notes" N)
python evaluation/conventional_text_metrics.py \
  --input-folder metrics_test/soap_prompts \
  --output-path metrics_test/quantitative_metrics/conventional_text_metrics.json \
  --filter degenerate
```
`evaluation/metrics.py` and `evaluation/quantitative_metrics.py` are the earlier,
narrower implementations these supersede (kept for provenance).

## Stage 3 — Concept extraction → **Table 2**, **Suppl. Table 6**, ontology F1 in Table 1

Requires MetaMapLite + a local UMLS 2022AB index (`pip install pymetamaplite`; set
the IVF index path).

```bash
python evaluation/metamap_coverage.py --input_dir metrics_test/soap_prompts \
  --output_dir metrics_test/metric_reports --ivf_dir /path/to/ivf/2022AB/USAbase
python evaluation/concept_iou.py --input_dir metrics_test/metric_reports \
  --pred_key pred_cuis --gold_key gold_cuis --output_dir metrics_test/metric_reports/concept_iou
python evaluation/metamap_concepts.py --input <coverage>.json --mrconso /path/to/MRCONSO.RRF --output <named>.json
python evaluation/metamap_concepts_histogram.py   # top-N concept frequency (extractor audit)
```

## Stage 4 — Atomic claim extraction → **Table 3** (claim recall/precision/F1)

`analysis/factehr_pipeline.py` — faithful FactEHR decomposition + binary
entailment (GPT-4o), extended to gold-vs-pred comparison. Writes
`gap_analysis/11_factehr_exact_claims.json`.

```bash
python analysis/factehr_pipeline.py \
  --soap_dir metrics_test/soap_prompts --output_dir metrics_test/gap_analysis \
  --keys keys.json --model gpt-4o --subset 45 \
  --cache_dir metrics_test/gap_analysis/.factehr_exact_cache
```
`analysis/claim_overlap.py` is an alternative in-house claim-gap analysis (10-label
scheme, `7_claim_decomposition_gaps.json`); `analysis/cui_claim_crosscheck.py`
re-derives its covered/missed/hallucinated labels from scispaCy CUI overlap as an
independent check.

## Stage 5 — Transcript grounding → **Tables 4, 5, 6**, **Suppl. Table 3**

```bash
# Table 5 — verified transcript claim set + 2x2 note coverage
python analysis/transcript_fact_extraction.py --soap_dir metrics_test/soap_prompts \
  --output_dir metrics_test/gap_analysis --keys keys.json
# Table 6 (AI-exclusive) — fabricated vs encounter-supported
python analysis/hallucination_grounding_analysis.py \
  --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
  --soap_dir metrics_test/soap_prompts --output_dir metrics_test/gap_analysis --keys keys.json
# Table 4 / Table 6 (human-exclusive) — missed claims: in- vs beyond-transcript
python analysis/missed_fact_grounding_analysis.py \
  --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
  --soap_dir metrics_test/soap_prompts --output_dir metrics_test/gap_analysis --keys keys.json
```

## Stage 6 — Content categories & topics → **Table 4**, **Suppl. Tables 3–4**

```bash
python analysis/creola_category_gap_analysis.py --claims_json metrics_test/gap_analysis/11_factehr_exact_claims.json \
  --soap_dir metrics_test/soap_prompts --output_dir metrics_test/gap_analysis --keys keys.json   # Suppl. Table 4 (inter-pass agreement)
python analysis/category_severity_gap_analysis.py --claims_json ... --output_dir metrics_test/gap_analysis --keys keys.json
python analysis/topic_model_uncategorized.py --input metrics_test/gap_analysis/14_creola_category_gaps.json \
  --output_dir metrics_test/gap_analysis --keys keys.json                                          # TopicGPT (Table 4 uncategorized)
python analysis/missed_facts_category_x_grounding.py \
  --grounding_json metrics_test/gap_analysis/16_missed_fact_grounding.json \
  --category_json  metrics_test/gap_analysis/14_creola_category_gaps.json \
  --topic_json     metrics_test/gap_analysis/15_uncategorized_topic_model.json \
  --output_dir     metrics_test/gap_analysis                                                       # Suppl. Table 3 (category x topic x grounding)
```
`analysis/gap_analysis.py` (CUI-level: semantic types, SOAP-section, most
missed/hallucinated CUIs, negation, numerical) and
`analysis/conversation_grounding.py` (phase-level coverage) are supporting analyses.

## Stage 7 — Statistics → Results significance tests

`analysis/stat_analysis_claim_metrics.py` — BCa CIs, GPT-5-Pro vs HealthScribe
(Wilcoxon signed-rank), closed- vs open-weight (Mann–Whitney U), Holm–Bonferroni
(k=2). Reads `gap_analysis/11_factehr_exact_claims.json`.

## Stage 8 — Tables & figures

| Script | Produces |
|---|---|
| `reporting/assemble_comprehensive_table.py` | Table 1 (Markdown + LaTeX, both filter conventions). |
| `reporting/generate_all_metrics_markdown.py` | Suppl. Tables 1–2 (per-model summary + per-session detail). |
| `reporting/generate_viz.py` | Transcript-grounding figure (Fig. 4). |
| `reporting/visualize_category_severity.py` | Category × severity figures. |
| `reporting/visualize_hallucination_grounding.py` | Fabricated vs omitted-but-true figures. |
