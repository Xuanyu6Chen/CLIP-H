# CLIP-H: discharge destination

This folder runs CLIP-H on one task: from an adult hospital admission as it is known at
admission, find a small library of readable clinical hypotheses that predict whether the patient
is discharged **Home** (a private residence without services) rather than to another destination,
then test how well they predict on patients they were not built on. Which admissions are in the
cohort, and how the label is set, is described in `mimic_cleaning/README.md` (section 8).

There are two arms, one notebook each. They run the same steps with the same settings and differ
only in the models they call.

| Arm | Notebook | Interprets | Refines and judges | Merges | Annotates | Second judge |
|---|---|---|---|---|---|---|
| GPT | `discharge_runner_gpt.ipynb` | `gpt-4.1` | `gpt-5` | `gpt-5` | `gpt-4.1-mini` | `claude-opus-4-8` |
| DeepSeek | `discharge_runner_deepseek.ipynb` | `deepseek-v4-pro` | `deepseek-v4-pro` | `deepseek-v4-flash` | `deepseek-v4-flash` | `claude-opus-4-8` |

Patient data is not included: MIMIC-IV needs credentialed access through PhysioNet. Everything
else needed is in this folder.

The same pipeline runs on in-hospital mortality in `clip_h/mortality/`. The `mimic_eval/`
and `myHypotheSAEs/` folders are identical in the two task folders; a task is defined by the
Settings cell of its runners and by its cohort and profile builders in `mimic_cleaning/src_2/`
(repository root), which `mimic_cleaning/run/run_discharge.py` uses to write `data/discharge/`.

## 1. Process: admissions → profiles → neurons → hypotheses → prediction

```
mimic_cleaning/mimic-iv-3.1/  MIMIC-IV v3.1 hosp/ and icu/ tables (you provide them)
      │
      │  mimic_cleaning/run/run_discharge.py   (mimic_cleaning/README.md, section 8)
      ▼
../../data/discharge/          cohort rows, split, profiles and labels, manifest.json
      │
      │  Data        read them; stops if they were written with other settings than Step 1
      ▼
profiles + labels
      │
      │  Phase 1     embed every profile (local model), train the SAE on the train embeddings
      ▼
sae/
      │
      │  Phase 2.1   select 64 neurons, 3 candidate names each, keep the name that best
      │              reproduces its neuron (fidelity F1 ≥ 0.30)
      │  Phase 2.2   split train 70 / 30 into discovery and confirmation; critic and rewriter
      │              refine the hypotheses on discovery, up to 3 rounds
      │  Phase 2.3   two judges, 5 votes each, score every hypothesis; drop leaky, invalid
      │              or incoherent ones
      │  Phase 2.4   merge hypotheses that fire on the same patients (discovery)
      │  Phase 2.5   certify on confirmation: prevalence floor, BH-FDR, bootstrap stability
      ▼
hypotheses_stable.csv        the certified hypotheses
      │
      │  Phase 3     annotate train and test, fit an L2 logistic regression on train,
      │              score test (AUROC, F1, 95% bootstrap intervals), descriptive refit
      │  Evaluation  TF-IDF + LR, structured LR and XGBoost on the same split
      ▼
heldout_metrics.csv, certified_library.csv, manifest.json
```

Words used in this file:

- **Profile**: one admission written as a paragraph, using information available at admission.
  It is the only thing the models read.
- **SAE** (sparse autoencoder): maps a profile's embedding to 1,024 **neurons**, of which 16 are
  active per profile. An active neuron is said to **fire**.
- **Hypothesis**: one sentence that is true or false for a patient, for example "patients aged
  75 years or older".
- **Annotation**: an LLM reads a profile and answers Yes or No for one hypothesis.
- **Fidelity**: how well a hypothesis reproduces the firing of its neuron, as an F1 score of the
  annotations against the neuron on profiles where it fires strongly and where it does not.
- **Discovery / confirmation**: the two parts of train (70% / 30%). Hypotheses are refined and
  deduplicated on discovery and certified on confirmation.
- **Arm**: one of the two model sets in the table above.
- **Run folder**: `clip_h_runs/<task>_<hash>_seed<seed>/`. The hash covers every setting that
  changes a result, so each configuration has its own folder.

## 2. Folder structure

```
clip_h/discharge/
├── discharge_runner_gpt.ipynb        GPT-4.1 arm
├── discharge_runner_deepseek.ipynb   DeepSeek-V4 arm
├── requirements.txt
├── mimic_eval/
│   ├── clip_h_lib.py                 curation, evaluation and bookkeeping
│   ├── self_refine.py                discovery / confirmation split, Phase 2.2 loop
│   └── baselines_ci.py               feature baselines and bootstrap intervals
├── myHypotheSAEs/                    modified copy of HypotheSAEs (Apache-2.0), see its README
│   └── hypothesaes/                  SAE, embeddings, neuron selection, interpretation,
│                                     annotation, LLM clients, prompts/
│
│   Created when you run (git ignores all of them):
├── clip_h_runs/<run>/                OUTPUT: one folder per configuration
├── emb_cache/                        cached embeddings
├── myHypotheSAEs/annotation_cache/   cached annotations
├── predictions/discharge/            test-set predictions per method and arm
└── ../../data/discharge/             INPUT: written by mimic_cleaning/run/run_discharge.py (mimic_cleaning/README.md, section 8)
```

The notebooks change into `mimic_eval/` and put `mimic_eval/` and `myHypotheSAEs/` on the import
path, so nothing needs to be installed as a package.

### Input files needed

`data/discharge/` at the repository root, written by `mimic_cleaning/run/run_discharge.py` from MIMIC-IV
**v3.1** (the raw files it needs, and where to put them, are listed in `mimic_cleaning/README.md`):

| File in `data/discharge/` | Used for |
|---|---|
| `train_cohort.parquet`, `test_cohort.parquet` | the train and test admissions with their IDs, label `y` and features (feature baselines, bootstrap groups, prediction IDs) |
| `train_profile.jsonl`, `test_profile.jsonl` | the text profiles |
| `train_labels.csv`, `test_labels.csv` | the labels, checked against the rows |
| `manifest.json` | the settings and column lists the files were written with; the Data cell stops if they differ from Step 1 |

`mimic_cleaning/src_2/` is the code these notebooks ran before it moved there: the files hold the
same patients and profiles as the reported runs, and the run folders keep their names.

## 3. How to run

```
cd mimic_cleaning
pip install -r requirements.txt
python run/preprocess_labevents.py        # optional: builds the lab-events cache faster
python run/run_discharge.py                # writes data/discharge/
cd "../clip_h/discharge"
pip install -r requirements.txt
export OPENAI_API_KEY=...                 # GPT arm
export DEEPSEEK_API_KEY=...               # DeepSeek arm
export ANTHROPIC_API_KEY=...              # both arms (Claude judge)
jupyter notebook discharge_runner_gpt.ipynb
```

Run all cells from the top. A key that is not in the environment is asked for once and kept only
for the session. The preflight cell sends one short request to every model and stops the run if
any of them fails. The default critique uses two judges and needs the Anthropic key; set
`CRITIC_PROVIDER = 'openai'` in Step 1 to use only the arm's own judge.

Tested with Python 3.11.5, numpy 2.4.6, pandas 3.0.3, scipy 1.17.1, scikit-learn 1.9.0,
statsmodels 0.14.6, xgboost 3.2.0, pyarrow 24.0.0, torch 2.7.0 (CPU), sentence-transformers
5.5.1, transformers 5.10.2, openai 2.41.0, anthropic 0.109.2 and tiktoken 0.13.0. A GPU is
optional; on CPU, embedding the 6,000 profiles is the slowest local step.

- **Quick check.** Set `N_PATIENTS_TRAIN, N_PATIENTS_TEST = 500, 100` and `CHEAP_MODELS = True`
  in Step 1 to run every step end to end at a small fraction of the cost.
- **Resume.** Each stage saves its output in the run folder and skips its work when that output
  exists (`RESUME = True`). The judges' votes are saved one by one in `screen_votes/`.
  Annotations and embeddings are cached across runs (section 7), so an interrupted run repeats
  only the calls that did not finish.
- **Cost.** Annotation is the largest API cost: every certified or candidate hypothesis is
  checked against thousands of profiles. `USE_BATCH_ANNOTATION = True` (GPT arm only) sends the
  same requests through the OpenAI Batch API at about half the price.

## 4. Notebook structure

The two runners have the same cells. They differ only in the Settings cell (models and worker
counts) and the API-key cell.

```
Steps 1-6    settings, compute check, paths, dependencies, API keys, seed and library import
Preflight    configuration checks and one test call per model
Data         read data/discharge/: rows, split, profiles and labels
             (written by mimic_cleaning/run/run_discharge.py)
Phase 1      embeddings and SAE training                              → sae/
Phase 2.1    neuron selection, interpretation, fidelity gate          → hypotheses_raw.csv
Phase 2.2    self-refinement on discovery                             → ckpt_refine.csv
Phase 2.3    clinical critique                                        → hypotheses_final.csv
Phase 2.4    deduplication by firing vectors                          → dedup_*.csv
Phase 2.5    significance gate on confirmation                        → hypotheses_stable.csv
Phase 3      logistic head, held-out metrics, descriptive refit,      → certified_library.csv,
             lineage, exports                                           test_predictions.csv
Evaluation   feature baselines on the same split                      → heldout_metrics.csv
Run record   manifest and summary                                     → manifest.json
```

## 5. Outputs and where the reported numbers are

Files in `clip_h_runs/<run>/`:

| Stage | Files |
|---|---|
| Phase 1 | `sae/` |
| Phase 2.1 | `hypotheses_raw.csv`, `stage_generate.csv` (fidelity per neuron, kept or dropped) |
| Phase 2.2 | `ckpt_refine.csv`, `stage_refine.csv` (critic action per hypothesis and round), `self_refine_trace.json` |
| Phase 2.3 | `critic_evidence.csv` (prevalence shown to the judges), `screen_votes/`, `stage_screen.csv` (verdict and reason), `hypotheses_final.csv` |
| Phase 2.4 | `dedup_pairs.csv`, `dedup_clusters.csv`, `dedup_complements.csv`, `dedup_map.csv`, `ckpt_dedup.csv`, `ckpt_dedup_map.json` |
| Phase 2.5 | `hypotheses_stable.csv`, `stage_gate.csv` (firing count, q-value, selection frequency, failed criterion) |
| Phase 3 | `test_predictions.csv`, `certified_library.csv`, `hypothesis_lineage.csv` |
| Evaluation | `heldout_metrics.csv` |
| Run record | `manifest.json`, `annotation_failures.json` |

Where the numbers in the paper come from:

- **Main results table**, CLIP-H and feature-baseline rows: `heldout_metrics.csv` (AUROC and F1
  with 95% intervals from 10,000 bootstrap resamples of the 1,000 test admissions; F1 at the
  prevalence-matched operating point).
- **Stage-by-stage counts**: the kept rows of `stage_generate.csv`, `ckpt_refine.csv`,
  `stage_screen.csv`, `dedup_map.csv` and `stage_gate.csv`; the BH-significant count of the
  descriptive refit is `significance.n_sig_bh` in `manifest.json`.
- **Certified-library tables**: `certified_library.csv`. `regression_coef` is the coefficient in
  the prediction model, `feature_prevalence` the share of training profiles where the hypothesis
  is true, `sig_bh` the significance marker from the descriptive refit, and `is_demographic` the
  demographic tag from the critique.
- **LLM baselines.** They are not run in this folder. `predictions/discharge/clip_h/<arm>.csv`
  holds CLIP-H's test-set scores and decisions. The comparators read the same patients from
  `data/discharge/` (`{train,test}.jsonl`, `_profile.jsonl`, `_labels.csv`), which
  `mimic_cleaning/run/run_discharge.py` writes before any CLIP-H run.

## 6. Settings

All settings are in Step 1 of the runner. The values below are the ones used in the paper; every
setting that changes a result is part of the run-folder hash.

| Setting | Value | Meaning |
|---|---|---|
| `SEED` | 42 | seeds the split, the subsample, the SAE, the bootstraps and the LLM calls that accept a seed |
| `N_PATIENTS_TRAIN`, `N_PATIENTS_TEST` | 5000, 1000 | admissions drawn after the patient-level split; must match `run_discharge.py` |
| `TRAIN_BALANCE`, `TEST_BALANCE` | 0.5, 0.5 | positive fraction after undersampling the majority class; must match `run_discharge.py` |
| `SAE_M`, `SAE_K` | 1024, 16 | SAE neurons and active neurons per profile |
| `SAE_EPOCHS`, `SAE_PATIENCE` | 1000, 50 | maximum epochs, early stopping on a 10% validation split |
| `N_HYPOTHESES` | 64 | neurons selected and interpreted |
| `CLEAN_COHORT` | True | the discharge-cohort exclusions; must match `run_discharge.py` |
| `SELECTION_METHOD`, `MMR_LAMBDA` | `'diverse'`, 0.7 | maximal-marginal-relevance selection and its weight |
| `NAMING` | `'faithful'` | name each neuron at the granularity of its firing |
| `MIN_FIDELITY` | 0.30 | minimum fidelity F1 |
| `CONTROL_LOOP`, `REFINE_ITERS`, `DISCOVERY_FRAC` | True, 3, 0.7 | self-refinement on, up to 3 rounds, 70% discovery |
| `CRITIC_PROVIDER`, `CRITIC_VOTES`, `CRITIC_VOTE_TEMP` | `'both'`, 5, 0.5 | two judges, 5 votes each, vote temperature |
| `MIN_VALIDITY`, `LEAKAGE_DROP` | 1, 2 | drop validity below 1 (0-3) or leakage risk 2 (0-2) |
| `CRITIC_EVIDENCE_N` | 300 | profiles used to show the judges each prevalence |
| `DEDUP_CORR_THR`, `DEDUP_AUTO_THR` | 0.90, 0.95 | firing correlation for adjudicated and automatic merges |
| `FDR_Q`, `STABILITY_PI`, `STABILITY_B` | 0.10, 0.6, 1000 | BH-FDR level, minimum bootstrap selection frequency, resamples |
| `ESTIMABILITY_FLOOR`, `MIN_SURVIVORS` | 15, 5 | minimum firing patients; provisional top-up if fewer pass |
| `EVAL_N_BOOT`, `EVAL_RESAMPLE_BY` | 10000, `'row'` | bootstrap resamples of the test set and their unit |
| `F1_THRESHOLD_MODE` | `'prevalence_matched'` | top-k decisions, k = number of true positives |

Settings that only change speed or cost: `RESUME`, `CHEAP_MODELS`, `USE_BATCH_ANNOTATION`,
`INTERP_WORKERS`, `ANNOT_WORKERS` and the environment variables in Step 5.

## 7. Rules

**Leakage.** The features use information available at admission. The excluded fields are
listed in `data/discharge/manifest.json` and in `mimic_cleaning/README.md` (section 8); the Data
cell stops if any of them is in the data.

**Split.** No patient appears in both train and test. Discovery and confirmation are disjoint
parts of train. The SAE and neuron selection use the whole training set. Self-refinement and
deduplication use only discovery, and none of the curation stages reads the outcome label; the
judges see each hypothesis's prevalence but not its association with the outcome. The gate then
tests significance once, on confirmation.

**Annotation.** The annotator answers Yes only when the profile directly and unambiguously
supports the hypothesis; missing, ambiguous, negated or weakly implied evidence is No. It runs at
temperature 0. A merged hypothesis is the OR of its members' annotations.

**Caches.** Annotations are cached in `myHypotheSAEs/annotation_cache/`, keyed by model,
hypothesis and the SHA-256 of the profile text, and embeddings in `emb_cache/`, keyed by profile
text. A cached answer is reused only for the same model, hypothesis and profile, so caches never change results.

**Failures.** LLM calls retry transient errors with backoff. A judge that loses the majority of
its votes stops the run, and the critique stops if it keeps no hypothesis. An
annotation that still fails after all retries enters the feature matrix as No; it is not cached,
and the number of such failures is written to `manifest.json` under `data_quality`.

**Determinism.** The split, subsample, SAE, statistics and logistic models are seeded.
Interpretation and annotation run at temperature 0; DeepSeek calls use non-thinking mode, which
honors temperature. GPT-5 is a reasoning model without a temperature setting and is used only in
curation roles. Hosted models are close to, but not exactly, deterministic, so a run from empty
caches can differ slightly from the reported one.

## 8. Source and attribution

- `myHypotheSAEs/` is a modified copy of HypotheSAEs (Movva et al., 2025, *Sparse Autoencoders
  for Hypothesis Generation*), released under the Apache License 2.0. Its README lists the
  changes; its LICENSE applies to that folder.
- Self-refinement follows Self-Refine (Madaan et al., 2023).
- Profiles are embedded with Qwen3-Embedding-0.6B.
- MIMIC-IV (Johnson et al., 2023) is available to credentialed users through PhysioNet under its
  data use agreement. Cohorts, profiles, caches and run outputs contain patient-level data; keep
  them out of version control (the repository's `.gitignore` excludes them) and do not share them.
