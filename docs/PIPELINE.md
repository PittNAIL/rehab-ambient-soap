# Pipeline and manuscript map

How every script maps to the tables, figures, and in-text statistics of
*A Comprehensive Evaluation of Ambient AI for Clinical Documentation in Adult
Rehabilitation* (submitted manuscript).

**Run everything from the repository root.** The final reporting scripts read
a fixed layout under `metrics_test/` (git-ignored, because it holds PHI):

```
data/json_dataset.json            Study 1 encounters (45)        corpus key: encompass
data/adapts_dataset.json          Study 2 encounters (108)       corpus key: adapts
metrics_test/
  eval_<corpus>/<system>_<corpus>.json      scored notes, one file per system (Stage 1)
  results_<corpus>/                         per-corpus outputs of Stages 2-6
  results_pooled/                           pooled artifacts and rendered tables
```

`<system>` is one of `gpt5-pro`, `healthscribe`, `jsl-text2soap`,
`medgemma-27b`, `med42-70b`, `asclepius-llama3-8b`, `openbiollm-8b`.
The corpus keys are the internal names of the two source studies; the paper
calls them Study 1 and Study 2.

Every stage after generation is deterministic given its cache. LLM calls
(GPT-4o) happen only in Stages 4–6, and all of them are cached by content key,
so re-running a stage with a warm cache makes no new calls.

> **Not in this repository.** The LLM-as-a-judge rubric evaluation (Tables 7–8,
> Supplementary Tables 5 and 7: six candidate judges, the 25-note degenerate
> probe, Krippendorff's α) was run with the MedJUDGE framework:
> <https://github.com/PittNAIL/MedJUDGE-Framework>.

---

## Stage 0 — Datasets

| File | Built by | Notes |
|---|---|---|
| `data/json_dataset.json` | provided with the access-controlled data | Study 1, 45 encounters (19 OT, 19 PT, 7 SLP). |
| `data/adapts_dataset.json` | `data_prep/build_study2_dataset.py` | Study 2, 108 encounters (102 OT, 6 PT). Merges same-day multi-part recordings (a/b/c) and joins each session to its therapist note. |

Schema: [DATA_FORMAT.md](DATA_FORMAT.md). Synthetic stand-in:
`example_data/json_dataset.example.json`.

## Stage 1 — Note generation → `metrics_test/eval_<corpus>/`

Run each system on both dataset files, then copy (or symlink) each final
`*_preds.json` to `metrics_test/eval_<corpus>/<system>_<corpus>.json`.

| System | Script | Paper configuration |
|---|---|---|
| GPT-5 Pro | `generation/openai_generations.py` | Azure OpenAI Responses API; `--provider azure-gpt5 --use-soap-prompt --max-output-tokens 10000`. **Few-shot** prompt (worked examples). Up to 5 retries, exponential backoff. |
| MedGemma-27B-it · Llama3-Med42-70B · Llama3-OpenBioLLM-8B · Asclepius-Llama3-8B | `generation/run_openweight.py` | `--no-examples` (**zero-shot**, required), greedy, `--max-new-tokens 1024`. MedGemma bf16 / 32,768 ctx; Med42 4-bit NF4 / 8,192; OpenBioLLM and Asclepius bf16 / 8,192. |
| JSL Text2SOAP | `generation/run_johnsnowlabs.sh` → `johnsnowlabs_eval.py`, `decode_jsl_outputs.py` | Vendor system prompt, temperature 0, `n_predict` 1024, batch 4. Real context 4,096 tokens (setNCtx is ignored); transcripts truncated before generation. Output decoded from double-JSON. |
| AWS HealthScribe | `generation/aws_healthscribe/*.py` | Original audio in, default configuration, no prompt; structured output converted to paragraph SOAP. |

```bash
# GPT-5 Pro
python generation/openai_generations.py --input-path data/json_dataset.json \
  --output-path outputs/encompass/gpt5-pro_encompass_preds.jsonl \
  --final-json-path outputs/encompass/gpt5-pro_encompass_preds.json \
  --provider azure-gpt5 --model <gpt-5-pro deployment> --use-soap-prompt --max-output-tokens 10000

# Open-weight (repeat per --model and per dataset file); dry-run first, no GPU needed
python generation/run_openweight.py --model medgemma-27b --no-examples --dry-run \
  --input-path data/adapts_dataset.json
python generation/run_openweight.py --model medgemma-27b --no-examples \
  --input-path data/adapts_dataset.json --output-dir outputs/adapts_noex
```

Truncation in the reported run (recorded per row as `truncated` /
`prompt_truncated`):

| System | Study 1 | Study 2 |
|---|---|---|
| JSL Text2SOAP (4,096) | 28 / 45 | 100 / 108 (median fraction seen 0.49 among truncated) |
| Med42-70B, OpenBioLLM-8B, Asclepius-Llama3-8B (8,192) | 0 / 45 | 30 / 108 each |
| MedGemma-27B-it (32,768), GPT-5 Pro | 0 / 45 | 0 / 108 |

Every reported generation is a single draw per encounter. The final 1,071 notes
contain no verbatim prompt-example fragment (Methods).

## Stage 2 — Conventional text metrics → **Table 1** (ROUGE, BERTScore), **Suppl. Tables 2, 8** (inputs)

```bash
for c in encompass adapts; do
  python evaluation/conventional_text_metrics.py \
    --input-folder metrics_test/eval_$c \
    --output-path  metrics_test/results_$c/conventional_text_metrics_emptyonly.json \
    --filter empty-only
done
```
Table 1 convention is `--filter empty-only`: every non-empty output is scored,
including the 5 later classified as degenerate (none was empty), so n = 153 for
every system. `--filter degenerate` gives the non-degenerate subset.
`evaluation/metrics.py` and `evaluation/quantitative_metrics.py` are earlier,
narrower implementations kept for provenance.

## Stage 3 — UMLS concepts → **Table 1** (ontology F1/IoU), **Table 2**, **Suppl. Table 6**

Requires MetaMapLite and a UMLS 2022AB index (licensed separately from the NLM).

```bash
for c in encompass adapts; do
  python evaluation/metamap_coverage.py --input_dir metrics_test/eval_$c \
    --output_dir metrics_test/results_$c/metric_reports \
    --ivf_dir /path/to/ivf/2022AB/USAbase --max-span-tokens 25
  python evaluation/concept_iou.py --input_dir metrics_test/results_$c/metric_reports \
    --pred_key pred_cuis --gold_key gold_cuis --output_dir metrics_test/results_$c/concept_iou
done
python evaluation/build_concept_tables.py \
  --report_dirs metrics_test/results_encompass/metric_reports metrics_test/results_adapts/metric_reports \
  --names /path/to/MRCONSO_from_ivf.RRF \
  --output metrics_test/results_pooled/16_concept_frequency_tables.json

# Extractor audit: 198 artifact CUIs (rules A-D), F1 with artifacts removed
python analysis/ontology_artifact_ablation.py --export resources/artifact_cuis.json
```
`--max-span-tokens` caps candidate spans for speed; it does not change the
matched concepts (longest-first order is preserved).
The audited artifact list is committed at
[`resources/artifact_cuis.json`](../resources/artifact_cuis.json):
169 gene/language collisions, 1 without a semantic type, 23 function-word /
discourse / dosing fragments, 5 confirmed shorthand collisions.

## Stage 4 — Atomic claims → **Table 3**, **Suppl. Table 1**

FactEHR decomposition + self-verification + binary entailment, GPT-4o.

```bash
for c in encompass adapts; do
  n=45; [ "$c" = adapts ] && n=108          # --subset defaults to 5; set it explicitly
  python analysis/factehr_pipeline.py --soap_dir metrics_test/eval_$c \
    --output_dir metrics_test/results_$c --keys keys.json --model gpt-4o \
    --subset $n --cache_dir metrics_test/claim_cache
done
```
Writes `results_<corpus>/11_factehr_exact_claims.json` (per-note recall,
precision, F1, missed and note-exclusive claims, `degenerate_pred_output`). Claim metrics use each system's
non-degenerate notes (MedGemma 150, JSL 151, others 153).

`analysis/claim_overlap.py` and `analysis/cui_claim_crosscheck.py` are earlier
supporting analyses; the reported statistics come from Stage 8.

## Stage 5 — Transcript grounding → **Tables 4, 5, 6**, **Fig. 2**

One entailment call per claim, with the full transcript as premise (55,368 calls
in total).

```bash
for c in encompass adapts; do
  python analysis/transcript_fact_extraction.py --soap_dir metrics_test/eval_$c \
    --output_dir metrics_test/results_$c/transcript --keys keys.json \
    --cache_dir metrics_test/transcript_cache_$c                        # Table 5
  for s in hallucination_grounding_analysis missed_fact_grounding_analysis; do
    python analysis/$s.py \
      --claims_json metrics_test/results_$c/11_factehr_exact_claims.json \
      --soap_dir metrics_test/eval_$c --output_dir metrics_test/results_$c/grounding \
      --cache_dir metrics_test/grounding_cache/$s --keys keys.json --model gpt-4o
  done                                                                  # Table 6
done
```

## Stage 6 — Content categories and topics → **Table 4**, **Suppl. Tables 3–4**

```bash
for c in encompass adapts; do
  python analysis/creola_category_gap_analysis.py \
    --claims_json metrics_test/results_$c/11_factehr_exact_claims.json \
    --soap_dir metrics_test/eval_$c --output_dir metrics_test/results_$c/creola \
    --cache_dir metrics_test/creola_cache --keys keys.json              # 2 passes @ T=0.7 + adjudicator
done
python analysis/refresh_pooled_topics.py --check    # writes the pooled CREOLA file
python analysis/topic_model_uncategorized.py \
  --claims_json metrics_test/results_pooled/14_creola_category_gaps_pooled.json \
  --output_dir metrics_test/results_pooled --keys keys.json             # TopicGPT, 30 topics
python analysis/refresh_pooled_topics.py --write    # re-tally counts (no LLM calls)
```
TopicGPT runs on the 854 unique uncategorized gold-claim strings (exact-string
deduplicated); counts in Table 4 are model–encounter instances.

## Stage 7 — Rendering the manuscript tables (no LLM calls)

| Script | Produces |
|---|---|
| `reporting/render_table1.py` | Table 1 (pooled + per corpus; percentile bootstrap, 10,000 resamples). |
| `reporting/render_table2.py` | Table 2 (top-20 concepts: gold, MedGemma-27B-it, OpenBioLLM-8B). |
| `reporting/render_table3.py` | Table 3 (claim recall / precision / F1). |
| `reporting/render_table4.py`, `render_table4_unique.py` | Table 4, incl. the distinct-claim columns. |
| `reporting/render_table5.py` | Table 5 (transcript coverage). |
| `reporting/render_table6.py` | Table 6 (fabricated / beyond transcript; bootstrap resamples encounters, not claims). |

The per-system and per-session supplementary tables (Suppl. Tables 2 and 8) are
assembled from the same per-corpus outputs:
`results_<corpus>/conventional_text_metrics_emptyonly.json` and the claim files.

`reporting/assemble_comprehensive_table.py`, `generate_all_metrics_markdown.py`,
`generate_viz.py`, `visualize_*.py` and `analysis/stat_analysis_claim_metrics.py`
belong to the earlier single-corpus (45-encounter) pipeline. They read the old
`metrics_test/soap_prompts/` layout and model stems, including the
Asclepius-13B pilot. They are kept for provenance and do not produce any
submitted table.

## Stage 8 — In-text statistics → Results and Discussion

```bash
python analysis/paper_statistics.py
```
Prints: GPT-5 Pro vs AWS HealthScribe claim recall (median paired difference
+0.182, Wilcoxon signed-rank p = 2.8×10⁻²², 153 pairs); median system recall
(0.286); closed-source vs open-weight recall (0.351 vs 0.221, descriptive);
Study 1 recall by discipline with Kruskal–Wallis (all systems p = 0.036;
GPT-5 Pro 0.042, HealthScribe 0.025, Med42 0.025); the share of claims without
transcript support by discipline (distinct missed claims, bootstrap over
encounters: PT 53.5%, OT 39.3%, SLP 31.3%; the submitted manuscript reports
52.2%, 38.7%, 30.7%, with the same ordering and interval pattern); and the
observability ceiling (773 / 2,438
distinct missed gold claims = 31.7%; Study 1 44.0%, Study 2 22.4%; 2,438 =
98.6% of 2,472 verified gold claims).
