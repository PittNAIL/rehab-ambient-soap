# Ambient AI for Rehabilitation Documentation — SOAP Note Generation & Evaluation

Code for **"A Comprehensive Evaluation of Ambient AI for Clinical Documentation
in Adult Rehabilitation"** (Zhao et al., submitted manuscript).

The study evaluates seven ambient AI systems on **153 full-length, recorded
adult rehabilitation sessions** (Study 1: 45 encounters; Study 2: 108
encounters), each paired with the note the treating therapist wrote. Notes are
compared on lexical/semantic similarity, UMLS concept coverage, and atomic
clinical claims. Every claim is then grounded against the session transcript
as well as the therapist's note. This separates true fabrication from
encounter-supported content the therapist simply did not record, and true
omissions from documentation that was never spoken aloud.

Headline results the code reproduces:

- **Text similarity does not track factual recovery.** The system ranked first
  on ROUGE-1, ROUGE-L and BERTScore recovered 13.8% of therapist-documented
  claims; GPT-5 Pro recovered 56.7%.
- **The therapist note is not complete ground truth.** 78.8% of AI-exclusive
  claims were supported by the transcript; 21.2% were fabricated.
- **Some documentation is not observable from audio.** 31.7% of the gold claims
  that systems missed had no support anywhere in the recorded encounter.

---

## What's here

| Folder | Purpose |
|---|---|
| [`data_prep/`](data_prep) | Build the Study 2 dataset file from the raw transcripts and notes. |
| [`generation/`](generation) | Generate SOAP notes for each of the seven systems. |
| [`evaluation/`](evaluation) | ROUGE, BERTScore, readability; MetaMapLite/UMLS concept coverage. |
| [`analysis/`](analysis) | Atomic-claim decomposition and entailment, transcript grounding, CREOLA/TopicGPT categorization, artifact-CUI audit, and the in-text statistics. |
| [`reporting/`](reporting) | Render manuscript Tables 1–6 and supplementary tables. |
| [`resources/`](resources) | `artifact_cuis.json`: the 198 audited MetaMapLite artifact CUIs (Methods). |
| [`docs/`](docs) | [Pipeline & manuscript map](docs/PIPELINE.md) and [data formats](docs/DATA_FORMAT.md). |
| [`example_data/`](example_data) | **Synthetic** encounters and generations so the code runs without protected data. |

The script-by-script mapping to every table, figure, and in-text statistic is in
**[docs/PIPELINE.md](docs/PIPELINE.md)**.

## Systems evaluated

| System | Category | Input | Prompt |
|---|---|---|---|
| GPT-5 Pro | Frontier general-purpose LLM | Cleaned transcript | Few-shot SOAP instruction |
| AWS HealthScribe | Commercial ambient AI service | Original audio | None (no prompting interface) |
| JSL Text2SOAP | Commercial ambient AI service | Cleaned transcript | Vendor system prompt |
| MedGemma-27B-it | Biomedical open-weight LLM | Cleaned transcript | Zero-shot (examples removed) |
| Llama3-Med42-70B | Biomedical open-weight LLM | Cleaned transcript | Zero-shot |
| Llama3-OpenBioLLM-8B | Biomedical open-weight LLM | Cleaned transcript | Zero-shot |
| Asclepius-Llama3-8B | Clinical-note-specialized LLM | Cleaned transcript | Zero-shot |

Model revisions, quantization, and context limits are in the paper (Table 9)
and in [`generation/run_openweight.py`](generation/run_openweight.py).

## Datasets

| Corpus | Paper name | Code key | Encounters | Disciplines | Source study design |
|---|---|---|---|---|---|
| 1 | Study 1 | `encompass` | 45 sessions, 29 patients | 19 OT, 19 PT, 7 SLP | Implementation pilot embedded in usual care; 2–3 sessions/day of ~20 min |
| 2 | Study 2 | `adapts` | 108 sessions, 16 patients | 102 OT, 6 PT | Phase 3 efficacy trial in addition to usual care; 1 session/day of ~45–60 min |

Each encounter is a video-recorded inpatient rehabilitation session. Its audio
was transcribed with AWS speech recognition and speaker diarization, then
cleaned manually for readability. Each session is paired with the narrative
note the treating therapist wrote during routine documentation. No system
received the video channel.

**Access.** The recordings, transcripts, therapist notes, model generations,
and every cache derived from them contain protected health information and are
**not** in this repository. They are held by the study team under University of
Pittsburgh IRB STUDY25080111. Aggregated, non-identifiable data underlying the
reported results are available from the corresponding author on reasonable
request, subject to IRB approval and institutional data-sharing requirements.

Approved users place the two dataset files at `data/json_dataset.json` and
`data/adapts_dataset.json` (schema: [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md))
and run the pipeline from there. Everything under `example_data/` is fully
synthetic.

## Scope note

The LLM-as-a-judge rubric evaluation (six candidate judges screened with a
25-note degenerate-output probe; Krippendorff's α per dimension; Tables 7–8) was
implemented with the **MedJUDGE** framework,
<https://github.com/PittNAIL/MedJUDGE-Framework>, and is not duplicated here.

---

## Installation

Python ≥ 3.11 (analyses were run on Python 3.12).

```bash
git clone https://github.com/PittNAIL/rehab-ambient-soap.git
cd rehab-ambient-soap

python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt      # core analysis stack
# or, with uv:  uv sync
```

The core install covers text similarity, claim-level evaluation, transcript
grounding, table rendering, and statistics. The following are optional; see the
comments in [`requirements.txt`](requirements.txt):

- concept extraction (MetaMapLite plus a UMLS 2022AB index, licensed from the NLM);
- local open-weight generation (torch, transformers, bitsandbytes; A100-class GPUs);
- JSL Text2SOAP (a John Snow Labs license, Spark NLP);
- AWS HealthScribe (boto3).

### Credentials

Copy `.env.example` → `.env` (or `keys.example.json` → `keys.json`) and fill in
your endpoints and keys. Nothing is hard-coded.

## Quickstart (synthetic data, no credentials, no GPU)

```bash
python evaluation/conventional_text_metrics.py \
  --input-folder example_data/soap_prompts \
  --output-path  example_data/conventional_text_metrics.json \
  --filter empty-only
```

This scores the two synthetic generation files. `--filter degenerate` shows how
degenerate outputs are excluded from the non-degenerate subset. BERTScore
downloads `roberta-large` on first run.

## Reproducing the paper

1. Place the two dataset files under `data/` (Stage 0).
2. Generate notes for each system and corpus → `metrics_test/eval_<corpus>/` (Stage 1).
3. Text metrics → concepts → claims → grounding → categories (Stages 2–6).
4. Render the tables (Stage 7) and the in-text statistics (Stage 8):
   `python analysis/paper_statistics.py`.

Stages 7–8 make no LLM calls and run in seconds from the cached outputs. Exact
commands are in [docs/PIPELINE.md](docs/PIPELINE.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

[MIT](LICENSE). The model weights and services evaluated have their own licenses;
for example, Asclepius-Llama3-8B is CC-BY-NC-SA 4.0 (research use only), and
MedGemma and Llama3-Med42 are gated.

## Acknowledgements

Supported by NIH awards R01LM014588 and R01HD117897. Computation used the
University of Pittsburgh Center for Research Computing.

Claim decomposition and entailment follow **FactEHR** (Munnangi et al., 2025).
Content categorization adapts the **CREOLA** clinical-safety taxonomy (Asgari et
al., *npj Digital Medicine*, 2025). Topic modeling adapts **TopicGPT** (Pham et
al., NAACL 2024). Fabrication follows **DocLens** (Xie et al., ACL 2024) and the
extrinsic-hallucination framing of Maynez et al. (ACL 2020). Concepts are
extracted with **MetaMapLite** (Demner-Fushman et al., 2017) against UMLS 2022AB.
