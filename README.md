# When Can an LLM-Judge Be Trusted?


## Overview

This repository contains the code for a systematic study of LLM-as-judge reliability in information retrieval evaluation. The work analyses when an LLM relevance judge (Llama-3.1-8B-Instruct with the UMBRELA prompt) can replace human assessors for system ranking, and develops budget-allocation policies that correct evaluation error by selectively requesting human judgements.

The study covers TREC Deep Learning 2019--2023, spanning both the MS MARCO v1 passage corpus (2019--2020) and the MS MARCO v2 passage corpus (2021--2023).

## Judge Configuration

- **Model:** Meta Llama-3.1-8B-Instruct
- **Prompt:** UMBRELA (Upadhyay et al., 2024) -- 4-point relevance scale (0--3)
- **Decoding:** Constrained to tokens {0, 1, 2, 3}, temperature 0
- **Output:** Grade + per-token log-probabilities for the four grade tokens

## Data Access

This repository contains **code only**. The evaluation data requires:

1. **TREC Deep Learning Track data** -- Available from [NIST](https://trec.nist.gov/data/deep2019.html) (requires registration for qrels and topics)
2. **System runs** -- Downloaded from the TREC results repository (see `data_setup/download_trec_dl_runs.py`)
3. **MS MARCO v1 passage corpus** -- Available from [Microsoft](https://microsoft.github.io/msmarco/)
4. **MS MARCO v2 passage corpus** -- Available from [Microsoft](https://microsoft.github.io/msmarco/TREC-Deep-Learning.html) (requires agreement)

Data files (`.jsonl`, `.csv`, `.json`, `.txt` qrels/runs) are excluded under the TREC data dissemination agreement.

## Hardware Requirements

- **GPU scoring:** RunPod with NVIDIA RTX 4090 (24 GB VRAM)
- **Feature computation & analysis:** Standard CPU (tested on Windows 11, Python 3.11)

## Repository Structure

```
scoring/          LLM scoring pipeline (GPU, RunPod)
data_setup/       Data acquisition and preparation
features/         Feature computation (CPU)
analysis/         Structural analysis and diagnostics
triage/           Correction experiments and hardening studies
figures/          Figure generation scripts and plot style
exploratory/      Non-load-bearing exploration scripts
superseded/       Older script versions replaced by hardened t-numbered scripts
thesis_figures/   Compiled PDF figures from the thesis
```

## Reproduction Order

1. **Data setup:** Acquire TREC DL data, download system runs, deduplicate v2 qrels
   ```
   python data_setup/download_trec_dl_runs.py
   python data_setup/dedup_v2_qrels.py
   ```

2. **LLM scoring (GPU):** Score all judged passages with Llama-3.1-8B-Instruct
   ```
   python scoring/score_passages.py --year 2019 --passages <path>
   ```

3. **Perturbation generation (CPU):** Generate B1a (synonym) and B1b (sentence permutation) variants
   ```
   python scoring/generate_b1_perturbations.py
   ```

4. **Feature computation:** Compute evaluation features
   ```
   python features/compute_level2.py
   python features/compute_b1b_features.py
   ```

5. **Structural analysis:** Score bias, spectral decomposition, displacement variance
   ```
   python analysis/run_spectral_linear.py
   ```

6. **Correction experiments:** Budget-allocation policies with bootstrap confidence intervals
   ```
   python triage/run_t12_resampling.py
   ```

7. **Figures:** Generate thesis figures
   ```
   python figures/make_ch3_figures.py
   python figures/make_chapter4_figures.py
   # etc.
   ```

## Dependencies

See `requirements.txt`. Core dependencies:

- Python 3.11
- vLLM 0.6.6.post1 (GPU scoring only)
- transformers 4.48.0
- numpy, scipy, pandas, matplotlib, seaborn, scikit-learn
- spacy, nltk (perturbation generation)
