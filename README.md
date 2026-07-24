# Ambient AI for Rehabilitation Documentation — SOAP Note Generation & Evaluation

Code to generate and evaluate ambient generative-AI SOAP notes for the study
**"Validation of Ambient Generative AI Models for Clinical Documentation in
Rehabilitation Settings"** (submitted to *npj Digital Medicine*).

The paper benchmarks seven systems (frontier, commercial-ambient, and open-weight
biomedical) across 45 physical-, occupational-, and speech-language-therapy
encounters. Its central methodological contribution is **fact-level, transcript-grounded**
evaluation: every AI-generated claim is checked against both the therapist
reference note *and* the original session transcript, which separates true
fabrication from clinically valid content that the human note simply omitted.
This repository contains the generation and evaluation pipeline behind those results.

---

## What's here

| Folder | Purpose |
|---|---|
| [`generation/`](generation) | Produce SOAP notes from transcripts/audio for each of the seven systems. |
| [`evaluation/`](evaluation) | Conventional text-similarity metrics (ROUGE, BERTScore, Jaccard, readability) and UMLS concept coverage. |
| [`analysis/`](analysis) | Atomic-claim decomposition, entailment, transcript grounding, CREOLA/TopicGPT categorization, and statistics. |
| [`reporting/`](reporting) | Assemble the manuscript tables and figures. |
| [`docs/`](docs) | [Pipeline & manuscript map](docs/PIPELINE.md) and [data formats](docs/DATA_FORMAT.md). |
| [`example_data/`](example_data) | **Synthetic** encounters and generations so the pipeline runs without protected data. |

A script-by-script mapping to every table and figure is in
**[docs/PIPELINE.md](docs/PIPELINE.md)**.

## Systems evaluated

GPT-5-Pro · AWS HealthScribe · JSL Text2SOAP · MedGemma-27B-it ·
Llama3-Med42-70B · Llama3-OpenBioLLM-8B · Asclepius-13B.

## Data availability & ethics

The study corpus (real diarized therapy transcripts and therapist reference
notes) contains protected health information and is **not** included in this
repository. Access is governed by the paper's *Data availability* statement.
Everything under `example_data/` is fully synthetic and contains no patient
information; it exists only to exercise the code paths and document the schemas.

> ⚠️ Do not place real transcripts, notes, generations, or LLM-call caches under
> version control. `data/`, `metrics_test/`, `outputs/`, `cache/`, and all
> `.*_cache/` directories are git-ignored for this reason. Credentials
> (`.env`, `keys*.json`) are git-ignored as well.

## Scope note

The qualitative **LLM-as-a-judge** evaluation (main-text Table X and
Supplementary Tables 5 & 7 — the 15-judge panel, empty-note probe, and
Krippendorff's α) was developed as a separate component and lives in its own
repository, cited in the paper's Code Availability statement. It is intentionally
not duplicated here.

---

## Installation

Python ≥ 3.11.

```bash
git clone https://github.com/<org>/rehab-ambient-soap.git
cd rehab-ambient-soap

python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt      # core analysis stack
# or, with uv:  uv sync
```

The core install covers the text-similarity, claim-level, and transcript-grounding
analyses (the bulk of the results). Concept extraction (MetaMapLite + a local UMLS
index), local 4-bit generation (bitsandbytes/accelerate), JSL Text2SOAP
(johnsnowlabs/pyspark), and AWS HealthScribe (boto3) are optional — see the
comments in [`requirements.txt`](requirements.txt).

### Credentials

Copy `.env.example` → `.env` (or `keys.example.json` → `keys.json`) and fill in
your endpoints/keys. LLM-based scripts read these; nothing is hard-coded.

## Quickstart (synthetic data, no credentials, no GPU)

Score the two synthetic generation files against their reference notes:

```bash
python evaluation/conventional_text_metrics.py \
  --input-folder example_data/soap_prompts \
  --output-path  example_data/conventional_text_metrics.json \
  --filter empty-only
```

This computes ROUGE-1/L, BERTScore, Jaccard, token counts, lexical entropy, and
readability for the coherent example model and demonstrates how the degeneracy
filter (`--filter degenerate`) drops the empty/soft-refusal/dialogue-leak outputs
in `example_degenerate_model.json`. (BERTScore downloads `roberta-large` on first
run.)

The claim-level and grounding stages additionally require an LLM API key; see
[docs/PIPELINE.md](docs/PIPELINE.md) for the full run order on real data.

## Reproducing the paper

1. Assemble `data/json_dataset.json` (schema: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md)).
2. Generate notes → `metrics_test/soap_prompts/<model>.json` (Stage 1).
3. Run Stages 2–7 (text metrics → concepts → claims → grounding → categories → stats).
4. Assemble tables/figures (Stage 8).

Each stage, with exact commands and its manuscript table/figure, is in
[docs/PIPELINE.md](docs/PIPELINE.md).

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the *npj Digital Medicine* paper
if you use this code.

## License

[MIT](LICENSE).

## Acknowledgements

The claim-decomposition/entailment pipeline reproduces and extends
**FactEHR** (Munnangi et al., 2024). Content categorization adapts the
**CREOLA** clinical-safety taxonomy (Asgari et al., *npj Digital Medicine*, 2025);
topic modeling adapts **TopicGPT** (Pham et al., NAACL 2024); grounding builds on
**DocLens** (Xie et al., ACL 2024) and the intrinsic/extrinsic hallucination
framing of Maynez et al. (ACL 2020). Concepts are extracted with **MetaMapLite**
(Demner-Fushman et al., 2017) against UMLS 2022AB.
