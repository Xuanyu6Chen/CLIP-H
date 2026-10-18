# CLIP-H: ICU length of stay

This folder runs CLIP-H on one task: from the first 24 hours of an ICU stay, find a small set
of readable clinical hypotheses that predict an ICU stay longer than 4 days, then test how
well they predict on patients they were not built on.

There are two arms, one notebook each. They run the same steps and differ in the LLM they call.

| Arm | Notebook | Interprets | Annotates | Refines | Confirms merges | Results folder |
|---|---|---|---|---|---|---|
| GPT | `icu_los_runner_gpt.ipynb` | `gpt-4.1` | `gpt-4.1-mini` | `gpt-5` | `gpt-5` | `results/<run>/icu_los_gpt/` |
| DeepSeek | `icu_los_runner_deepseek.ipynb` | `deepseek-v4-pro` | `deepseek-v4-flash` | `deepseek-v4-pro` | `deepseek-v4-flash` | `results/<run>/icu_los_deepseek/` |

The patient data is not included (MIMIC-IV needs credentialed access). Everything else needed is
in this folder.

The same pipeline runs on two other tasks, each in its own folder: `clip_h/readmission/`
and `clip_h/hosp_los/`. `code/` is the same in the three folders. A task is defined
by its Settings cell (section 6) and its prompt files in `prompts/`.

## 1. Process: profiles → neurons → hypotheses → prediction

```
../../data/icu_los4/         patient profiles (text) and labels, train and test
      │
      │  Phase 0     load, draw 5,000 train and 1,000 test, split train 70 / 30
      │              into discovery and confirmation
      ▼
profiles + labels
      │
      │  Phase 1     embed every profile (local model), train the SAE on the train
      │              embeddings, select 96 neurons, split them into strong and weak
      ▼
phase1_sae/                  embeddings.npz, sae.pkl
      │
      │  Phase 2A    strong neurons: 3 candidate hypotheses each, keep the one that best
      │              reproduces its neuron (fidelity)
      │  Phase 2A+   weak neurons: name each, group related names into OR-hypotheses
      │  gate        drop low fidelity and exact duplicates
      ▼
phase2a_interpret/           the candidate hypotheses
      │
      │  Phase 2B    self-refine on the discovery sample: critic and refiner, up to 3 rounds
      ▼
phase2b_self_refine/
      │
      │  Phase 2C    write the review prompt          ← MANUAL: paste it into a chat model,
      │  Phase 2C+   read the reply, keep or drop        save the reply as a file
      ▼
phase2c_review/
      │
      │  Phase 2D    merge hypotheses that are true for the same patients
      │  Phase 2E    confirmation gate on the confirmation split
      ▼
phase2d_dedup/  phase2e_confirm/      the certified hypotheses
      │
      │  Phase 3A    annotate train and test, fit a logistic regression on train, score test
      │  Phase 3B    check the files, write the summary, print the cost
      ▼
phase3_prediction/           the hypotheses with their statistics, AUROC / AUPRC / F1
```

Words used in this file:

- **Profile**: the first 24 hours of one patient's ICU stay written as a paragraph. It is the
  only thing the models read.
- **SAE** (sparse autoencoder): turns a profile's embedding into 1,024 **neurons**, of which 16
  are active per profile. A neuron that is active is said to **fire**.
- **Hypothesis**: one sentence that is true or false for a patient, for example "Patient
  required mechanical ventilation within the first 24 hours of ICU admission".
- **Annotation**: an LLM reads a profile and answers 1 or 0 for each hypothesis.
- **Fidelity**: how well a hypothesis reproduces the firing of its neuron, as an F1 score on 30
  profiles (15 where the neuron fires most, 15 where it does not fire).
- **Discovery / confirmation**: the two parts of train. Hypotheses are refined, reviewed and
  de-duplicated on discovery, and tested on confirmation.
- **Arm**: one of the two model sets in the table above.
- **Run**: one value of `RUN_NAME`. Each run has its own results folder.

## 2. Folder structure

```
clip_h/icu_los/
├── icu_los_runner_gpt.ipynb         what you run: the GPT arm
├── icu_los_runner_deepseek.ipynb    what you run: the DeepSeek arm (differs in one line)
├── code/                            functions only, nothing runs here
│   ├── code_paths.py                makes every folder below importable
│   ├── sae/                         Phase 1
│   │   ├── hypothesaes/                 sae.py, select_neurons.py of HypotheSAEs v0.2.0, unchanged; LICENSE
│   │   ├── embed_profiles.py            local embeddings, cached in chunks
│   │   └── clip_h_sae.py                train / load the SAE, activations, neuron selection, strong / weak split
│   ├── hypotheses/
│   │   ├── interpret/          Phase 2A, 2A+, fidelity gate
│   │   │   ├── interpret_neurons.py     candidates per neuron, fidelity scoring, best candidate
│   │   │   ├── weak_groups.py           name the weak neurons, group them
│   │   │   └── fidelity_gate.py         drop low fidelity and exact duplicates
│   │   ├── self_refine/        Phase 2B, and the discovery / confirmation split
│   │   │   └── self_refine.py
│   │   ├── review/             Phase 2C, 2C+
│   │   │   └── conversational_review.py build the review prompt, check and apply the reply
│   │   ├── dedup/              Phase 2D
│   │   │   └── dedup_lib.py
│   │   └── confirm/            Phase 2E
│   │       └── confirmation_gate.py
│   ├── prediction/                  Phase 3A
│   │   └── evaluation.py                per-hypothesis statistics, association tests, the prediction model
│   └── common/                      used by several phases
│       ├── llm_client.py                every LLM call: retries, call cache, usage log, Batch API, progress lines
│       ├── annotation.py                profile x hypothesis labels, label caches
│       ├── prompts.py                   reads prompts/
│       ├── phase_tables.py              the table shown at the end of each phase, the `from` ids, the metrics table
│       └── clip_h_data.py               load profiles and labels, subsample, JSON helpers
├── prompts/                         every prompt, one .txt file per step; README.md there lists the sections
├── results/                         OUTPUT: one folder per run
│   ├── <RUN_NAME>/
│   │   ├── phase1_sae/                  embeddings.npz, sae.pkl, sae_checkpoint/   (shared by both arms)
│   │   ├── icu_los_gpt/                 one arm
│   │   │   ├── phase2a_interpret/           hypotheses_pool.csv, pooled_group.json, hypotheses_curated_candidates.csv, curation_summary.json
│   │   │   ├── phase2b_self_refine/         self_refine_hypotheses.csv, self_refine_trace.json
│   │   │   ├── phase2c_review/              CONVO_01_critique_prompt.md, CONVO_02_critique_response.json, hypotheses.csv, hypotheses_conversational_merged.csv, conversational_curation_summary.json
│   │   │   ├── phase2d_dedup/               dedup_pairs.csv, dedup_clusters.csv
│   │   │   ├── phase2e_confirm/             confirmation_gate_tests.csv, confirmation_gate_summary.json
│   │   │   ├── phase3_prediction/           annotation_matrix.npz, hypothesis_evaluation.csv, annotation_metrics.json, hypothesis_association_tests.csv, hypothesis_association_summary.json, heldout_metrics.json, model_metrics.csv
│   │   │   ├── caches/                      every cache and the usage log (section 7.3)
│   │   │   └── run_manifest.json, task_summary.json, results_index.json
│   │   └── icu_los_deepseek/            the other arm, same layout
│   └── icu_los_emb_cache/           embedding chunks, shared by all runs (made when profiles are embedded)
├── requirements.txt
└── README.md
```

The notebook puts `code` on `sys.path` and imports `code_paths`; after that each file is
imported by its file name, for example `from clip_h_sae import train_sae`.

Where a function goes: used by one phase → that phase's folder; used by several → `common/`.

### Input files needed

The notebook reads `data/icu_los4/` at the repository root (setting `TASK_DATA_REL`). The files
are written there by `mimic_cleaning/run/run_icu_los.py`.

| File in `data/icu_los4/` | Used for | Phase |
|---|---|---|
| `train_profile.jsonl`, `test_profile.jsonl` | the profiles | 0 |
| `train_labels.csv`, `test_labels.csv` | the labels (1 = ICU stay longer than 4 days) | 0 |
| `train.jsonl`, `test.jsonl` | profile and label together; loaded when present, else built from the four files above | 0 |
| `train_features.csv` | raw first-24h values shown to the reviewer next to each hypothesis | 2C |

The cleaned set has 5,000 train patients (2,500 with a long ICU stay) and 1,000 test patients
(500). There is one row per patient: the last ICU stay.

The raw values the reviewer sees (`icu_raw_facts`) are the flags that are set (ICU from the
column `was_in_icu`, mechanical ventilation, dialysis, oliguria) and then the admission type,
service, lactate, creatinine, hemoglobin, WBC and the count of abnormal labs. A column that is
not in `train_features.csv` is left out. The file written by `run_icu_los.py` has no
`was_in_icu` (every patient is in the ICU), so the ICU flag is not shown.

## 3. How to run

```bash
pip install -r requirements.txt
jupyter notebook icu_los_runner_gpt.ipynb        # or icu_los_runner_deepseek.ipynb
```

- Tested with Python 3.14.8, numpy 2.4.0, pandas 3.0.1, torch 2.12.1, scikit-learn 1.9.0 and
  openai 2.44.0.
- Start Jupyter from inside this folder; the notebook finds the folder from the working
  directory. Started anywhere else (another directory, a hosted notebook service), set
  `RUN_FOLDER` in Settings to the path of this folder.
- Run the cells top to bottom. The Setup cell asks for the API key once and keeps it only in
  the session (`OPENAI_KEY_SAE` or `DEEPSEEK_API_KEY` in the environment skip the question).
- **The run stops once, on purpose, at Phase 2C+.** Phase 2C writes
  `phase2c_review/CONVO_01_critique_prompt.md`. Paste the whole file into a chat model, save its
  JSON-only reply as `CONVO_02_critique_response.json` in the same folder, and run the cell again.
- **A step that finds its saved file loads it and makes no call.** With a complete results
  folder, the whole notebook runs without any API call. `FORCE_RERUN` (Settings) redoes a step.
- **An interrupted cell can simply be run again.** Every reply and every label is saved the
  moment it arrives, so nothing is paid twice.
- Phase 1 needs the embedding model (`Qwen/Qwen3-Embedding-0.6B`, downloaded on first use) only
  when `embeddings.npz` is missing. It runs on CPU; a GPU only makes it faster.
- On the GPT arm, the annotation of Phases 2B to 2E and 3A goes through the OpenAI Batch API:
  half the price, and each annotation step waits for its batch (up to 24 hours; Phase 2B
  annotates once per round). A step with fewer than 100 calls is called live.
  `batch_api=False` in the `'gpt'` line of `_MODEL_PROFILES` calls live.
- While a cell with many calls runs, it prints a line every 15 seconds:
  `[Phase 2E annotation] 320 of 1500 (21%) | 1:12 elapsed, about 4:30 left | requests 960, from cache 0, retries 2, failed 0`.
- Phase 3B prints what the arm has cost so far, from `caches/usage_log.jsonl`.

### What a cell shows when it ends

One line with the count in and out, then one table with a row per hypothesis:

```
Phase 2E: 53 in -> 37 out. tested on 1500 confirmation patients. Certified = true for at least 15, q at most 0.1, stability at least 0.6.
 #  hypothesis                        from       true for  long ICU stay if true / if not  odds ratio (95% CI)  q       stability  status     reason
 1  Patient required mechanical ...   N359+N718  565       64% / 42%                       2.42 (1.96-3.01)     <0.001  1.00       certified
38  Patient was an inter-hospital ..  ?          13        92% / 50%                       12.18 (1.58-93.90)   0.005   0.94       dropped    true for only 13 (needs 15)
```

- `#`, `hypothesis`, `from`, `status` and `reason` are in every table; the columns between
  them are the numbers of that phase. Dropped rows are grey and at the bottom.
- `from` is where the hypothesis came from:

  | `from` | Meaning |
  |---|---|
  | `N339` | neuron 339 |
  | `G2` | weak-neuron group 2 |
  | `R1` | added by the reviewers in Phase 2C |
  | `N12+N88` | merged from two |
  | `~N753` | neuron 753 as a best guess |
  | `?` | written in Phase 2B, or not traceable |

- **The link through Phase 2B is recovered from the wording.** The refiner returns a plain
  list of hypotheses, not which one each came from. An identical or nearly identical text
  keeps its id; a text reworded a lot gets `~`; a split, a merge or a new idea gets `?`.
  The id is for reading the tables. No step of the pipeline uses it.
- Numbers are printed one way everywhere: shares as `28.7%`, outcome rates as
  `72% / 47%` (when the hypothesis is true / when it is not), odds ratios with their 95%
  interval, q-values with three decimals (`<0.001` below that).
- Phase 3A and 3B show the held-out result as a table: value, 95% interval (10,000 bootstrap
  resamples of the test set, the model is not refit), and what to compare it with. Phase 3B
  also shows the counts of every phase in one table and, when both arms of the run exist,
  the two arms side by side.

### A second run that reuses the first

A new `RUN_NAME` starts a new, empty folder in `results/`. It does not pay again for what
any earlier run already holds:

```python
RUN_NAME = 'my_ablation'
PHASE1_FROM = 'clip_full'       # use that run's embeddings and SAE
REUSE_CACHES_FROM = 'all'       # the default: read the saved replies and labels of every other run
```

- `REUSE_CACHES_FROM = 'all'`: before a call is made, the run looks in its own `caches/`, then
  in the `caches/` of every other run folder in `results/`. So the first ablation reads
  `clip_full`, the second reads `clip_full` and the first ablation, and so on. A list of run
  names reads only those; `[]` reads none.
- What is found is used as it is. A label is also copied into the new run's own `caches/`.
  The other runs are never written to. Only the same arm is read, and a reply is reused only
  for the same model and the same prompt.
- `PHASE1_FROM`: the embeddings and the SAE are read from the named run. Same SAE = same
  neurons, so the two runs can be compared hypothesis by hypothesis.
- Result files are not reused. To start an ablation from a later phase, copy the phase folders
  you want to keep into the new run's arm folder; each step that finds its file loads it.

## 4. Notebook structure

Each notebook has 14 code cells: the settings, the setup, then one cell per phase. The two
notebooks differ in one line, `MODEL_PROFILE`.

```
icu_los_runner_gpt.ipynb · icu_los_runner_deepseek.ipynb
│
├── 1  SETTINGS   (everything that can be changed)
│   ├── run:      RUN_FOLDER, RUN_NAME, PHASE1_FROM, REUSE_CACHES_FROM, SEED
│   ├── task:     TASK_NAME, TASK_DATA_REL, TASK_DESCRIPTION, POS_LABEL, POS_NAME, NEG_NAME, RAW_FACTS
│   ├── models:   MODEL_PROFILE → models, reply style, parallel calls, timeout, Batch API
│   ├── phases:   one block of settings per phase (section 6)
│   └── FORCE_RERUN   which steps to redo
│
├── 2  SETUP      find the folder, make the result folders, API key, call cache, usage log
│
└── PHASES  (fixed)
    ├── 3   Phase 0     prepare_split → stratified_subsample → discovery_confirmation_split     → run_manifest.json
    ├── 4   Phase 1     get_local_embeddings → train_sae → select_neurons → split_weak_neurons  → phase1_sae/
    ├── 5   Phase 2A    interpret_neurons → score_interpretations → keep_best_interpretations   → hypotheses_pool.csv
    ├── 6   Phase 2A+   interpret_weak_groups                                                   → pooled_group.json, rows added to the pool
    ├── 7   gate        curate                                                                  → hypotheses_curated_candidates.csv
    ├── 8   Phase 2B    agentic_refine (critic, refiner)                                        → phase2b_self_refine/
    ├── 9   Phase 2C    build_concept_exemplars → build_review_prompt                           → CONVO_01_critique_prompt.md
    ├── 10  Phase 2C+   ingest_review_response                                                  → hypotheses.csv
    ├── 11  Phase 2D    predictive_dedup                                                        → phase2d_dedup/
    ├── 12  Phase 2E    build_annotation_matrix → confirmation_gate                             → phase2e_confirm/
    ├── 13  Phase 3A    build_annotation_matrix → fit_and_score_heldout → association tests     → phase3_prediction/
    └── 14  Phase 3B    file check, summary, usage_report                                       → task_summary.json, results_index.json
```

The first cell of each notebook holds the same map as a table, with what every cell reads and
what it does when its file is missing.

## 5. Inputs and outputs of every phase

Paths are inside `results/<RUN_NAME>/<arm>/` unless they start with `phase1_sae/` (one
level up, shared by both arms).

| Phase | Reads | Writes | Calls |
|---|---|---|---|
| 0 | `data/icu_los4/` | `run_manifest.json` | none |
| 1 | `phase1_sae/embeddings.npz`, `sae.pkl` when present | the same two files | none (local model) |
| 2A | `hypotheses_pool.csv` when present | `phase2a_interpret/hypotheses_pool.csv` | interpreter, annotator |
| 2A+ | `pooled_group.json` when present | `pooled_group.json`, rows with a negative `neuron_idx` in the pool | interpreter |
| gate | the pool | `hypotheses_curated_candidates.csv`, `curation_summary.json` | none |
| 2B | `self_refine_hypotheses.csv` when present | `phase2b_self_refine/self_refine_hypotheses.csv`, `self_refine_trace.json` | annotator (Batch API on GPT), refiner |
| 2C | `train_features.csv`, `prompts/review.txt` | `phase2c_review/CONVO_01_critique_prompt.md` (rewritten every time) | annotator (Batch API on GPT), only for labels not cached |
| 2C+ | `CONVO_02_critique_response.json` (**you write this**) | `hypotheses.csv`, `hypotheses_conversational_merged.csv`, `conversational_curation_summary.json` | none |
| 2D | labels from the cache | `phase2d_dedup/dedup_pairs.csv`, `dedup_clusters.csv` | annotator (Batch API on GPT); refiner for borderline pairs |
| 2E | labels from the cache | `phase2e_confirm/confirmation_gate_tests.csv`, `confirmation_gate_summary.json` | annotator (Batch API on GPT) |
| 3A | `annotation_matrix.npz`, `hypothesis_evaluation.csv`, `annotation_metrics.json` when present | those, and `heldout_metrics.json`, `hypothesis_association_tests.csv`, `hypothesis_association_summary.json`, `model_metrics.csv` | annotator (Batch API on GPT) |
| 3B | every file above | `task_summary.json`, `results_index.json` | none |

Where the numbers are:

| You want | File |
|---|---|
| AUROC, AUPRC, F1 of the run, with 95% intervals (`ci95`) | `phase3_prediction/heldout_metrics.json` (fit on train, scored on test) |
| The final hypotheses | `phase3_prediction/hypothesis_evaluation.csv` |
| Odds ratio, interval and q-value of each hypothesis on test | `phase3_prediction/hypothesis_association_tests.csv` |
| Why a hypothesis was certified or dropped | `phase2e_confirm/confirmation_gate_tests.csv`, `confirmation_gate_summary.json` |
| What the reviewers said | `phase2c_review/hypotheses_conversational_merged.csv` |
| What the critic changed | `phase2b_self_refine/self_refine_trace.json` |
| How many labels could not be obtained and count as 0 | `caches/annotation_failures.json` (per step), repeated in `task_summary.json` |
| Which neuron a hypothesis came from, and its fidelity | `phase2a_interpret/hypotheses_pool.csv` |

`annotation_metrics.json` and `hypothesis_evaluation.csv` hold statistics fit on the test set
itself. They describe the hypotheses; they are not the prediction result.

## 6. What you can change, what is fixed

**Can be changed**: the Settings cell, and the prompt files in `prompts/`.

| Setting | Meaning | Phase |
|---|---|---|
| `RUN_FOLDER` | path of this folder; `None` = found from the working directory | all |
| `RUN_NAME` | name of the run's results folder | all |
| `PHASE1_FROM` | run whose embeddings and SAE to use; `None` = this run makes its own | 1 |
| `REUSE_CACHES_FROM` | runs whose saved replies and labels may be read: `'all'`, a list of run names, or `[]` | all |
| `TASK_NAME`, `TASK_DATA_REL` | name of the task in folder and file names; folder with the input files | all |
| `TASK_DESCRIPTION`, `POS_LABEL` | the outcome in words, as told to the 2B critic and the 2D confirmer | 2B, 2D |
| `POS_NAME`, `NEG_NAME` | the two outcome classes as named in printed lines and table columns | display |
| `RAW_FACTS` | function of `conversational_review.py` that prints a patient's raw values for the reviewer | 2C |
| `MODEL_PROFILE` | `'gpt'` or `'deepseek'`; picks the line of `_MODEL_PROFILES` with the models and how they are called | all |
| `SEED` | fixes the subsample, the SAE, the 70 / 30 split, the 2B and 2D calls and the bootstraps | all |
| `N_PATIENTS_TRAIN`, `N_PATIENTS_TEST` | patients drawn, label-stratified; `None` = all | 0 |
| `DISCOVERY_FRAC` | share of train used for discovery | 0 |
| `STRATIFY_SPLIT` | `True` splits each outcome class separately, so both parts keep the outcome rate | 0 |
| `SAE_M`, `SAE_K`, `SAE_EPOCHS`, `SAE_PATIENCE` | neurons, active neurons per profile, training length | 1 |
| `N_HYPOTHESIS_POOL` | neurons selected. The only count setting: there is no quota on the final set | 1 |
| `SELECTION_METHOD`, `MMR_LAMBDA` | how neurons are selected; `'diverse'` = related to the outcome and not to each other | 1 |
| `SPARSE_MAX_PROFILES` | a selected neuron firing on fewer train profiles than this is weak | 1 |
| `N_CANDIDATES` | candidate interpretations per strong neuron (3); the best by fidelity F1 is kept | 2A |
| `MIN_FIDELITY` | fidelity below this drops the hypothesis | gate |
| `CONTROL_LOOP`, `REFINE_ITERS`, `SR_MIN_PREV`, `SR_SAMPLE_N` | run self-refine or not, rounds, "too rare" floor, discovery profiles annotated | 2B |
| `N_REVIEWERS`, `MIN_CLINICAL_VALIDITY`, `LEAKAGE_DROP`, `WITHHOLD_RAW_FIELDS` | reviewers the chat model plays, keep and drop thresholds, raw values hidden from them | 2C |
| `DEDUP_CORR_THR`, `DEDUP_AUTO_THR` | overlap at which a pair is considered, and at which it is merged without asking | 2D |
| `RUN_TRAIN_GATE`, `GATE_FDR_ALPHA`, `GATE_BOOTSTRAP`, `GATE_STABILITY`, `ESTIMABILITY_FLOOR`, `MIN_SURVIVORS`, `GATE_TOP_K`, `GATE_SAMPLE_N` | the confirmation gate (section 7.1) | 2E |
| `GATE_TRAIN_N` | train profiles annotated to fit the prediction model | 3A |
| `FORCE_RERUN` | steps to redo even when their file exists | all |

**Fixed**: the phase cells and everything in `code/`.

Things to know before changing a setting:

- A setting of Phase 0 or 1 changes which patients or which neurons the run works on, so every
  later file changes with it. Give the run a new `RUN_NAME`.
- A changed setting does not redo a step whose file already exists. Use a new `RUN_NAME`, or
  set the step in `FORCE_RERUN`, or the notebook keeps loading the earlier file.
- Phases 2C, 2D and 2E are recomputed on every run from the cached labels; they have no
  `FORCE_RERUN` entry.
- A changed prompt is a different call: saved replies for the earlier wording are not used, and
  the calls are paid again. The file name `interpret-neuron-binary` is part of the key of the
  interpretation cache; renaming the file has the same effect.
- A changed annotator model needs new labels for every profile and hypothesis.
- `STRATIFY_SPLIT`, `DISCOVERY_FRAC` and `SEED` change who is in the confirmation split, so the
  saved confirmation labels no longer cover it.
- Neuron selection uses the labels of all train profiles, the confirmation split included.
- `gpt-5` is pinned to `gpt-5-2025-08-07` in `llm_client.py` (`model_abbrev_to_id`). When a
  provider retires or reroutes a model name, results can change with no change in this folder;
  the model that actually answered is recorded in `caches/usage_log.jsonl`.
- The prices in `llm_client.py` (`PRICES`) are typed in by hand. Check them against the
  provider before trusting the printed cost.
- `timeout` in `_MODEL_PROFILES` limits one whole request of an OpenAI model. DeepSeek calls are
  streamed and have no total limit: a connection with no new chunk for 90 seconds is dropped
  and the call is made again.
- Every DeepSeek call is sent with thinking off, so it follows the temperature. A model name
  ending in `-think` (for example `gen='deepseek-v4-pro-think'` in `_MODEL_PROFILES`) keeps the
  thinking pass for that role: the model then ignores the temperature, and the client raises
  its token limit to at least 16,000 (it reasons inside the limit).
- A call that sets no temperature runs at 0 on every model that accepts one. `gpt-5` accepts
  none.
- `gpt-5` (refiner, critic, confirmer on the GPT arm) runs at reasoning effort `medium`
  (`GPT5_REASONING_EFFORT` in `llm_client.py`). Its reasoning is counted inside the token limit,
  so the client raises the limit of these calls to at least 16,000.
- `SELECTION_METHOD = 'lasso'` uses the `penalty` argument of scikit-learn's
  `LogisticRegression` (in the unchanged HypotheSAEs file), which scikit-learn announces it
  will remove in version 1.10. The default `'diverse'` does not use it.

## 7. Rules

### 7.1 Per phase

| Phase | Rule |
|---|---|
| 0 | Subsample: label-stratified, seeded. Split: each outcome class 70% discovery, 30% confirmation, seeded |
| 1 | SAE: Top-K, 1,024 neurons, 16 active, trained on train embeddings only. Selection: 96 neurons by `'diverse'`. Strong = fires on at least 15 train profiles |
| 2A | Per strong neuron: `N_CANDIDATES` (3) candidates at temperature 0, each written from 30 example profiles. Each is annotated on 30 profiles and scored against the neuron's firing; the best F1 is kept |
| 2A+ | Per weak neuron: one call gives it a short name. One call sorts the names into clinically related groups and states each group as one OR-hypothesis. No outcome label is used. A neuron related to no other is left out |
| gate | Drop fidelity below 0.30; drop exact text duplicates. Grouped hypotheses have no fidelity score and pass |
| 2B | On the 3,500 discovery profiles. The critic never sees the outcome. A hypothesis true for under 2% of the sample is flagged as too rare. At most 3 rounds |
| 2C | The chat model acts as 3 reviewers and sees the hypothesis text and raw values of up to 3 example patients, not the profiles or labels. It may reword, merge and add hypotheses (the prompt allows 3 additions; the reply check accepts up to 10) |
| 2C+ | The reply must be strict JSON, cover every candidate exactly once, keep scores in range, and name no outcome. Kept: majority of valid votes, clinical validity ≥ 1, leakage risk < 2 |
| 2D | Two hypotheses true for nearly the same discovery patients (phi ≥ 0.90) with the same expected direction are a candidate pair. phi ≥ 0.95 merges; below that one LLM call confirms or refuses. The one with the higher clinical validity is kept (then the one true for fewer patients) |
| 2E | On the 1,500 confirmation profiles. Certified only if: true for at least 15 profiles, significant at Benjamini-Hochberg FDR 0.10, and significant with the same sign in at least 60% of 1,000 bootstrap resamples |
| 3A | Logistic regression (L2, balanced class weights) fit on the train annotations of the certified hypotheses, scored on test. F1 is taken at the cut that predicts as many positives as there are in test |

Leakage: no hypothesis may name the ICU or hospital length of stay, discharge, death, or
anything after hour 24 of the ICU stay. The interpretation prompt forbids it and the review
scores it (`leakage_risk`; a score of 2 drops the hypothesis). Phase 2C+ also rejects a reply in
which a hypothesis contains one of a fixed list of words (`FORBIDDEN_HYPOTHESIS_TEXT` in
`conversational_review.py`: readmission, death, mortality, code status and the like). That list
is the same for every task and does not contain "length of stay", so for this task the leakage
score is what removes a hypothesis that names the stay length.

### 7.2 Annotation

- One call labels up to 30 hypotheses for one profile. A label missing from the reply is asked
  for again, then asked alone.
- The two arms use different reply formats (`annotate_style`: `checklist` for GPT, `labels` for
  DeepSeek); both are in `prompts/annotation.txt`.
- The same profile and hypothesis asked twice of the same model can get different answers.
  A saved label is therefore never asked for again: the label caches, not the model, decide.
- Each phase keeps its labels in its own cache file (section 7.3). The confirmation patients
  are annotated twice in a run: by Phase 2E with every hypothesis the gate tests, and by
  Phase 3A, as part of train, with the certified ones.

### 7.3 Caches

All in `<arm>/caches/`. Nothing here needs to be edited by hand. Deleting a cache only means
its calls are made, and paid, again.

| File | Holds | Keyed on |
|---|---|---|
| `llm_call_cache.jsonl` | every reply, one line each, appended as it arrives | model, system text, prompt, call settings, attempt number (the candidate number in Phase 2A; the try number where a step asks again) |
| `interpretation_cache.json` | Phase 2A candidates | model, prompt file, instructions, neuron, candidate number |
| `fidelity_annotation_cache.json` | Phase 2A labels | model, hypothesis, profile text |
| `selfrefine_annotation_cache.json` | labels on the discovery sample (2B, 2C, 2D) | model, hypothesis, profile text |
| `binary_confirmation_annotation_cache.json` | labels on the confirmation split (2E) | the same |
| `binary_train_annotation_cache.json` | labels on train (3A) | the same |
| `binary_annotation_cache.json` | labels on test (3A) | the same |
| `llm_call_cache.jsonl.batches.json` | IDs of submitted batches, saved before waiting, so a stopped cell picks its batch up again | |
| `annotation_failures.json` | per annotation step, how many labels could not be obtained and count as 0; a step that runs again replaces its entry | |
| `usage_log.jsonl` | one row per request: model asked, model that answered, tokens, retry, batch | |

`interpretation_cache.json` is not read across runs: neuron numbers mean something only within
one SAE, and the example profiles shown to the interpreter are drawn afresh in each run, so the
call cache of another run does not cover Phase 2A either. A second run that should keep the same Phase 2A
hypotheses copies `phase2a_interpret/` from the first (section 3).

### 7.4 When a call fails

| What happens | What the code does | What you see |
|---|---|---|
| Rate limit, server error, timeout | asks again, up to 8 requests, waiting longer each time (or as long as the provider says) | `[llm]` line with the request count |
| Empty or cut-off reply | asks again with a doubled token limit | `[llm]` line |
| An annotation reply that cannot be read | asks again, up to 3 tries; then asks for each of its labels alone | `[annotation]` line (the first 12), then the `NOTE:` line |
| A label missing from a reply | asks for it again, then alone | `NOTE:` line with the count |
| A label still missing at the end of a step | waits 60 seconds and asks again, up to 3 rescue rounds; a round that recovers nothing ends it | `[<step>] ... rescue round` lines |
| A label that could not be obtained at all (also: reply withheld by a content filter) | counted as 0 in this run, not saved, asked again on the next run | `WARNING:` line with the count; the count is saved in `caches/annotation_failures.json` and `task_summary.json` |
| A critic or refiner reply that cannot be used after 3 tries | the round goes on without it: every hypothesis counts as accepted, or the hypotheses stay unchanged | `[self-refine]` line; `critic_failed` / `refine_failed` in `self_refine_trace.json` |
| Wrong key, no balance, unknown model, malformed request | the cell stops at once; what was received is already saved | `FatalAPIError` |

Repeated `[llm]` lines of one kind are shown three times, then every 50th. The totals are in
every progress line (`retries`, `failed`).

## 8. Source

`sae/hypothesaes/` holds `sae.py` and `select_neurons.py` of HypotheSAEs v0.2.0, unchanged.
`embed_profiles.py`, `train_sae` in `clip_h_sae.py`, and parts of `interpret_neurons.py`,
`annotation.py` and `evaluation.py` are adapted from the same release (Rajiv Movva, Kenny Peng
et al., https://github.com/rmovva/HypotheSAEs). HypotheSAEs is distributed under the Apache
License 2.0; its text is in `sae/hypothesaes/LICENSE`, and `THIRD_PARTY_NOTICES.md` at the
repository root lists every use. Self-refine follows Madaan et al. 2023 (arXiv:2303.17651).
