# LLM direct prediction

A comparator for CLIP-H: an LLM reads each test patient's profile and answers with the label and
its probability that the label is the positive one. No hypotheses and no downstream model. The
probability is the score for AUROC; the label is the decision for F1.

| Arm | Model | Key read from |
|---|---|---|
| `gpt` | `gpt-4.1-mini` | `OPENAI_API_KEY` (or `OPENAI_KEY_SAE`) |
| `deepseek` | `deepseek-v4-flash` | `DEEPSEEK_API_KEY` |

These are the models CLIP-H uses to annotate each patient, so the per-patient judge is the same
across methods. A key that is not set is asked for (hidden, not stored).

## Input

`data/<task>/` at the repository root, the folder every method reads:

| File | Used for |
|---|---|
| `test_profile.jsonl` | the profiles, one `{"idx", "profile"}` per line |
| `test_labels.csv` | the labels, one column named after the task's label (below) |

| `--task` | Written by | Label column |
|---|---|---|
| `readmission` | `mimic_cleaning/run/run_readmission.py` | `readmission` |
| `hosp_los3` | `mimic_cleaning/run/run_hosp_los.py` | `los_gt3` |
| `icu_los4` | `mimic_cleaning/run/run_icu_los.py` | `icu_los_gt4` |
| `mortality` | `mimic_cleaning/run/run_mortality.py` | `mortality` |
| `discharge` | `mimic_cleaning/run/run_discharge.py` | `discharge` |

To read another folder, change `DATA_DIR` at the top of the script.

## How to run

```bash
pip install -r requirements.txt
python direct_llm_baseline.py --arm gpt --task readmission --limit 10    # small test first
python direct_llm_baseline.py --arm gpt --task readmission                # the first 1,000 test patients
```

- `--arm` is required; `--task` defaults to `TASK` at the top of the script; `--limit` takes the
  first N test patients (default `N_TEST` = 1,000).
- Before any paid call the script says how many calls it will make and asks to proceed; `--yes`
  skips the question.
- Every reply is saved the moment it arrives. A rerun only calls for patients without a saved
  reply, so a stopped run resumes and a finished one costs nothing. A failed call is not saved:
  the run stops, says how many failed, and the next run asks for them again.

## Output

`results/<task>/<arm>/confidence/` in this folder (not in git):

| File | Holds |
|---|---|
| `all_instances.jsonl` | one line per patient: `patient_id`, `true_label`, `predicted_label`, `prob_pos`, `model_output` |
| `run_config.json` | model, settings, prompt hash, counts of replies without a readable answer or probability |
| `cache.jsonl` | every reply, keyed by model, endpoint, settings and prompt |

The AUROC printed at the end is a check only. The paper scores AUROC from `prob_pos` and F1 from
`predicted_label` in `all_instances.jsonl`.

## What you can change

The settings block at the top of `direct_llm_baseline.py`: `TASK`, `N_TEST`, `N_WORKERS`
(parallel calls), `TIMEOUT`, `RETRIES`, `DATA_DIR`, the models in `ARMS`, the task wording in
`TASKS`, and the prompt text (`PROMPT_SYSTEM`, `PROMPT_USER`). Calls run at temperature 0.
A changed prompt or setting is a new call: saved replies for the old wording are not reused.
