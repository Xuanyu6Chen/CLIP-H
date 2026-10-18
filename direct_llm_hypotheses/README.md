# LLM direct hypotheses

A comparator for CLIP-H: an LLM writes clinical hypotheses straight from labeled train profiles
(no SAE, no curation beyond exact de-duplication). The patients are then annotated against the
hypotheses with CLIP-H's annotation code and prompts, a logistic regression is fit on the train
annotations, and the test patients are scored. Annotation and the held-out evaluation follow
CLIP-H Phase 3A, so the two methods differ only in where the hypotheses come from.

| Arm | Notebook | Generates | Annotates | Results folder |
|---|---|---|---|---|
| GPT | `direct_hypotheses_gpt.ipynb` | `gpt-4.1` | `gpt-4.1-mini` | `results/<run>/<task>_gpt/` |
| DeepSeek | `direct_hypotheses_deepseek.ipynb` | `deepseek-v4-pro` | `deepseek-v4-flash` | `results/<run>/<task>_deepseek/` |

The generator and annotator are the models CLIP-H uses as interpreter and annotator, called the same
way (dated snapshots, DeepSeek with thinking off). The two notebooks differ in one line, `MODEL_PROFILE`. `TASK` in Settings picks one of the five
tasks.

## 1. Process

```
../data/<task>/              patient profiles (text) and labels, train and test
      │
      │  Phase 0     load, draw 5,000 train and 1,000 test (label-stratified, seeded)
      │  Phase G1    4 class-balanced batches of 20 labeled train profiles; one call per batch
      │              proposes 20 hypotheses
      │  Phase G2    exact-text dedup, one hypothesis from each batch in turn, first 15
      ▼
phaseG_generate/             generation_pool.json, generated_hypotheses.csv
      │
      │  Phase 3A    annotate test and train against the 15 hypotheses, fit a cross-validated
      │              logistic regression on train, score test with 95% bootstrap intervals
      │  Phase 3B    check the files, write run_config.json, print the cost
      ▼
phase3_prediction/           annotation_matrix.npz, heldout_metrics.json, hypothesis_lr_coefs.csv, model_metrics.csv
```

The logistic regression is fit on 5,000 train patients for `readmission`, `hosp_los3` and
`icu_los4`, and on 1,000 for `mortality` and `discharge` (`gate_train_n` in `_TASKS`), as in the
reported runs.

## 2. Folder structure

```
direct_llm_hypotheses/
├── direct_hypotheses_gpt.ipynb        what you run: the GPT arm
├── direct_hypotheses_deepseek.ipynb   what you run: the DeepSeek arm (differs in one line)
├── code/                              functions only, nothing runs here
│   ├── code_paths.py                  makes the folders below importable
│   ├── generate/                      Phases G1, G2: direct_generation.py; fidelity_gate.py (its dedup key)
│   ├── prediction/                    Phase 3A: direct_fit.py (the cross-validated fit), evaluation.py (bootstrap intervals)
│   └── common/                        copied from clip_h/readmission/code/common/; annotation.py adds use_prompt_file
├── prompts/                           generation.txt and annotation_<task>.txt; README.md there lists the sections
├── results/                           OUTPUT (not in git): results/<RUN_NAME>/<task>_<arm>/
├── requirements.txt
└── README.md
```

## 3. Input files needed

`data/<task>/` at the repository root (setting `TASK_DATA_REL`), the folder every method reads:
`train_profile.jsonl`, `test_profile.jsonl`, `train_labels.csv`, `test_labels.csv`, and
`train.jsonl` / `test.jsonl` when present (built from the other four otherwise).

| `TASK` | Written by |
|---|---|
| `readmission`, `hosp_los3`, `icu_los4` | `mimic_cleaning/run/run_readmission.py`, `run_hosp_los.py`, `run_icu_los.py` |
| `mortality`, `discharge` | `mimic_cleaning/run/run_mortality.py`, `run_discharge.py` |

## 4. How to run

```bash
pip install -r requirements.txt
jupyter notebook direct_hypotheses_gpt.ipynb        # or direct_hypotheses_deepseek.ipynb
```

- Set `TASK` in Settings, then run the cells top to bottom.
- Start Jupyter from inside this folder; the notebook finds the folder from the working
  directory. Started anywhere else, set `MODULE_FOLDER` in Settings to the path of this folder.
- The Setup cell asks for the API key once and keeps it only in the session
  (`OPENAI_API_KEY` / `OPENAI_KEY_SAE` or `DEEPSEEK_API_KEY` in the environment skip the
  question).
- If `data/<task>/` is missing, Setup stops and names the folder it looked in.
- **A step that finds its saved file loads it and makes no call.** Every reply is saved in
  `caches/llm_call_cache.jsonl` and every label in a `caches/*_annotation_cache.json` file as
  soon as it arrives, so an interrupted cell can simply be run again. `FORCE_RERUN` (Settings)
  redoes a step.
- A label that could not be obtained stops Phase 3A; running the cell again asks for it.
- Phase 3B prints what the arm has cost so far, from `caches/usage_log.jsonl`.

## 5. Settings

Everything that can be changed is in the Settings cell.

| Setting | Meaning |
|---|---|
| `TASK` | `'readmission'`, `'hosp_los3'`, `'icu_los4'`, `'mortality'` or `'discharge'`; the row of `_TASKS` with its class names, train size and generation wording |
| `TASK_DATA_REL` | folder with the input files, relative to this folder |
| `MODULE_FOLDER` | path of this folder; `None` = found from the working directory |
| `RUN_NAME` | name of the run's results folder; a new name starts a new, empty run |
| `REUSE_CACHES_FROM` | runs whose saved replies and labels may be read: `'all'`, a list of run names, or `[]` |
| `SEED` | the subsample, the generation batches, the cross-validation, the intervals |
| `MODEL_PROFILE` | `'gpt'` or `'deepseek'`: the line of `_MODEL_PROFILES` with the models and how they are called |
| `N_PATIENTS_TRAIN`, `N_PATIENTS_TEST` | patients drawn, label-stratified; `None` = all |
| `GEN_BATCHES`, `GEN_BATCH_SIZE`, `GEN_PER_BATCH`, `GEN_TEMPERATURE` | generation calls, profiles per call, hypotheses asked for per call, temperature |
| `MIN_PER_BATCH`, `MAX_GEN_ATTEMPTS` | a reply with fewer hypotheses is asked again, up to this many calls |
| `K_HYPOTHESES` | library size |
| `FORCE_RERUN` | steps to redo even when their file exists |

A changed prompt or model is a different call: saved replies are not reused and the calls are
paid again. The saved generation pool is reused only when its settings are the same.

## 6. Prompts and code shared with CLIP-H

- `prompts/annotation_<task>.txt` is the same text as the `annotation.txt` of that task's
  CLIP-H folder (`readmission`, `hosp_los3` = `clip_h/hosp_los`, `icu_los4` = `clip_h/icu_los`),
  so both methods label patients with the same prompt. `mortality` and `discharge` share one.
- `code/common/` and `code/prediction/evaluation.py` are copies of the CLIP-H files; see
  `clip_h/readmission/README.md` for the LLM client, the caches and what happens when a call
  fails. `common/annotation.py` and `prediction/evaluation.py` include code adapted from
  HypotheSAEs (Apache License 2.0; see `THIRD_PARTY_NOTICES.md` at the repository root).
