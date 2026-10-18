"""LLM direct prediction: ask an LLM for the label of each test patient, with no hypotheses
and no downstream model.

For each of the 1,000 test patients, one prompt holds the task, the label definitions and the
full profile text. The model answers with a label and its probability (0-100) that the label is
the positive one. The probability is the score for AUROC; the label is the decision for F1.

Steps (one function each, run in order by main()):

    1. load      data/<task>/test_profile.jsonl + test_labels.csv (repository root)
    2. prompt    one prompt per patient (PROMPT_SYSTEM, PROMPT_USER below)
    3. call      one API call per patient, each reply saved to the cache as it arrives
    4. parse     the "Answer:" and "Probability:" lines of each reply
    5. save      results/<task>/<arm>/confidence/: all_instances.jsonl, run_config.json

A rerun reads the cache and only calls for patients without a saved reply, so a stopped run
resumes and a finished one costs nothing. A failed call is not cached: the run stops and says
how many failed, and the next run retries them.

Data: not included (MIMIC-IV needs credentialed access). Each task reads data/<task>/ at the
repository root (DATA_DIR), the folder every method reads, written by mimic_cleaning/run/
for all five tasks.
Arms: gpt-4.1-mini and deepseek-v4-flash (ARMS below), the models CLIP-H uses to annotate each
patient, so the per-patient judge is the same across methods. The API key is read from
OPENAI_API_KEY / DEEPSEEK_API_KEY, or asked for (hidden, not stored).
Scoring: the AUROC printed at the end is a check only. The paper scores AUROC from prob_pos
and F1 from predicted_label, in all_instances.jsonl.

Usage:
    pip install -r requirements.txt
    python direct_llm_baseline.py --arm gpt
    python direct_llm_baseline.py --arm deepseek --limit 10     # small test first
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import threading
import time
from datetime import datetime
from getpass import getpass
from pathlib import Path

import requests
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# =============================================================================================
# Settings
# =============================================================================================
TASK = "readmission"          # one of TASKS below
N_TEST = 1000                 # test patients to predict (the first N rows of the test split)
N_WORKERS = 8                 # parallel API calls
TIMEOUT = 300                 # seconds per call
RETRIES = 5                   # attempts per call before it counts as failed

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"   # <repo>/data/<task>/, the folder every method reads
RESULTS_DIR = BASE_DIR / "results"

# One entry per arm. "api_model" is the dated snapshot the API is asked for, the one CLIP-H's
# annotator uses ("model" names the arm and stays in the cache key, so saved replies keep
# matching). max_tokens caps reasoning + answer together for a reasoning model, so
# deepseek-v4-flash needs room to think before it writes the answer.
ARMS = {
    "gpt": {
        "model": "gpt-4.1-mini",
        "api_model": "gpt-4.1-mini-2025-04-14",
        "api_url": "https://api.openai.com/v1/chat/completions",
        "key_env": ["OPENAI_API_KEY", "OPENAI_KEY_SAE"],
        "max_tokens": 256,
    },
    "deepseek": {
        "model": "deepseek-v4-flash",
        "api_url": "https://api.deepseek.com/v1/chat/completions",
        "key_env": ["DEEPSEEK_API_KEY"],
        "max_tokens": 8000,
    },
}
TEMPERATURE = 0.0
TOP_P = 1.0

# One entry per task (the folder name in data/): the words the prompt uses, and the label column in test_labels.csv.
TASKS = {
    "readmission": {
        "task_phrase": "30-day hospital readmission",
        "pos": "Yes", "neg": "No",
        "label_definitions": (
            "Label definitions:\n"
            "- Yes: patient is readmitted to the hospital within 30 days of discharge\n"
            "- No: patient is not readmitted within 30 days of discharge\n"
        ),
        "label_column": "readmission",
    },
    "hosp_los3": {
        "task_phrase": "hospital length of stay (more than 3 days)",
        "pos": "Yes", "neg": "No",
        "label_definitions": (
            "Label definitions:\n"
            "- Yes: the hospital stay lasts MORE than 3 days\n"
            "- No: the hospital stay lasts 3 days or less\n"
        ),
        "label_column": "los_gt3",
    },
    "icu_los4": {
        "task_phrase": "ICU length of stay (more than 4 days)",
        "pos": "Yes", "neg": "No",
        "label_definitions": (
            "Label definitions:\n"
            "- Yes: the ICU stay lasts MORE than 4 days\n"
            "- No: the ICU stay lasts 4 days or less\n"
        ),
        "label_column": "icu_los_gt4",
    },
    "mortality": {
        "task_phrase": "in-hospital mortality",
        "pos": "Yes", "neg": "No",
        "label_definitions": (
            "Label definitions:\n"
            "- Yes: patient died during the indexed hospitalization\n"
            "- No: patient survived to discharge\n"
        ),
        "label_column": "mortality",
    },
    "discharge": {
        "task_phrase": "discharge destination",
        "pos": "Home", "neg": "Other",
        "label_definitions": (
            "Label definitions:\n"
            "- Home: discharged to a private home WITHOUT services\n"
            "- Other: any non-home discharge (facility-based care, home with services, "
            "hospice/deceased, etc.)\n"
        ),
        "label_column": "discharge",
    },
}

# =============================================================================================
# Prompts: the text sent to the model. {braces} are filled in by build_prompt().
# =============================================================================================
PROMPT_SYSTEM = (
    "You are a board-certified physician with extensive experience in inpatient hospital "
    "medicine, making a prediction from a de-identified patient admission record. "
    "Follow the label definitions and judge only from the record. "
    "You MAY include brief reasoning, but you MUST put the answer on the first line "
    "and a probability on the second line."
)

PROMPT_USER = """\
You will be given a structured patient admission record.

Task: {task_phrase} prediction.

{label_definitions}
Output format (strict):
Answer: <{pos}|{neg}>
Probability: <0-100>   # your probability (percent, 0-100) that the answer is {pos}
Thought:
- 2–5 short bullets (keep it brief)

Record:
{record}
"""


# =============================================================================================
# 1. Load
# =============================================================================================
def find_data_dir(task):
    """data/<task>/ at the repository root (DATA_DIR / task)."""
    d = DATA_DIR / task
    if not d.is_dir():
        raise SystemExit(f"no {d}: write it first with mimic_cleaning/run/ (see mimic_cleaning/README.md), or set DATA_DIR")
    return d


def load_test(task, n):
    """Return (ids, profiles, labels) for the first n test patients, rows aligned by position."""
    data_dir = find_data_dir(task)
    profile_path = data_dir / "test_profile.jsonl"
    label_path = data_dir / "test_labels.csv"
    for p in (profile_path, label_path):
        if not p.exists():
            raise SystemExit(f"missing {p}: run the data pipeline for {task} first")

    rows = [json.loads(line) for line in open(profile_path, encoding="utf-8") if line.strip()]
    with open(label_path, encoding="utf-8") as f:
        labels = [int(float(r[TASKS[task]["label_column"]])) for r in csv.DictReader(f)]
    if len(rows) != len(labels):
        raise SystemExit(f"{profile_path.name} has {len(rows)} rows, {label_path.name} has {len(labels)}")
    if len(rows) < n:
        raise SystemExit(f"asked for {n} test patients, the split has {len(rows)}")

    rows, labels = rows[:n], labels[:n]
    ids = [str(r["idx"]) for r in rows]
    profiles = [r["profile"].strip() for r in rows]
    return ids, profiles, labels


# =============================================================================================
# 2. Prompt
# =============================================================================================
def build_prompt(task, record):
    """Return (system, user) for one patient."""
    t = TASKS[task]
    user = PROMPT_USER.format(task_phrase=t["task_phrase"], label_definitions=t["label_definitions"],
                              pos=t["pos"], neg=t["neg"], record=record)
    return PROMPT_SYSTEM, user


def prompt_sha1(task):
    """Fingerprint of the prompt wording (profile left as a placeholder), saved in run_config."""
    system, user = build_prompt(task, "{record}")
    return hashlib.sha1((system + "\n---\n" + user).encode("utf-8")).hexdigest()


# =============================================================================================
# 3. Call, with a cache
# =============================================================================================
def cache_key(arm, system, user):
    """Everything that changes the reply: model, endpoint, settings, prompt."""
    a = ARMS[arm]
    parts = [a["model"], a["api_url"], str(a["max_tokens"]), str(TEMPERATURE), str(TOP_P), system, user]
    return hashlib.sha1("\n\x1f".join(parts).encode("utf-8")).hexdigest()


def load_cache(path):
    cache = {}
    if path.exists():
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                cache[r["key"]] = r["reply"]
    return cache


def call_llm(arm, api_key, system, user):
    """One chat-completions call. Returns the reply text, or None if every attempt failed."""
    a = ARMS[arm]
    payload = {
        "model": a.get("api_model", a["model"]),
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": TEMPERATURE, "top_p": TOP_P, "max_tokens": a["max_tokens"],
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    for attempt in range(RETRIES):
        try:
            r = requests.post(a["api_url"], json=payload, headers=headers, timeout=TIMEOUT)
            if r.status_code == 429:  # rate limited: wait as told, else back off
                time.sleep(float(r.headers.get("Retry-After") or 5 * (attempt + 1)))
                continue
            r.raise_for_status()
            content = r.json()["choices"][0]["message"].get("content")
            if isinstance(content, str) and content.strip():
                return content
            # Empty content: a reasoning model used up max_tokens before answering. The same
            # call would do it again, so do not retry; raise max_tokens for this arm instead.
            print("  empty reply (max_tokens too small?)")
            return None
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            print(f"  call failed (attempt {attempt + 1}/{RETRIES}): {exc}")
            if attempt < RETRIES - 1:
                time.sleep(3 * 2 ** attempt)
    return None


def get_replies(arm, api_key, prompts, cache_path):
    """Reply for every prompt: from the cache, or by calling. New replies are cached as they arrive."""
    cache = load_cache(cache_path)
    keys = [cache_key(arm, s, u) for s, u in prompts]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"{len(prompts) - len(todo)} replies cached, {len(todo)} calls to make")

    lock = threading.Lock()
    with open(cache_path, "a", encoding="utf-8") as out, \
            concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(call_llm, arm, api_key, *prompts[i]): i for i in todo}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            reply = fut.result()
            if reply is None:
                continue
            i = futures[fut]
            with lock:
                cache[keys[i]] = reply
                out.write(json.dumps({"key": keys[i], "reply": reply}, ensure_ascii=False) + "\n")
                out.flush()

    failed = [i for i, k in enumerate(keys) if k not in cache]
    if failed:
        raise SystemExit(f"{len(failed)} calls failed and were not cached; run again to retry them")
    return [cache[k] for k in keys]


# =============================================================================================
# 4. Parse
# =============================================================================================
def strip_think(text):
    """Drop <think> blocks a reasoning model may put in its reply (closing tag may be missing)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.replace("<think>", "").strip()


def parse_answer(task, reply):
    """The label on the last 'Answer:' line, lowercased ('yes' / 'no'), or None."""
    t = TASKS[task]
    lines = [ln for ln in strip_think(reply).splitlines() if ln.strip().lower().startswith("answer:")]
    if not lines:
        return None
    raw = re.sub(r"[<>\[\]\"'`*]", "", lines[-1].split(":", 1)[1]).strip().rstrip(".")
    token = re.split(r"[\s,;]", raw, maxsplit=1)[0].lower()
    return token if token in (t["pos"].lower(), t["neg"].lower()) else None


def parse_probability(reply):
    """The number on the last 'Probability:' line, as a fraction in [0, 1], or None."""
    m = re.findall(r"(?im)^\s*probability\s*[:=]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%?", strip_think(reply))
    if not m:
        return None
    val = float(m[-1])
    return min(max(val / 100.0 if val > 1.0 else val, 0.0), 1.0)


# =============================================================================================
# 5. Save
# =============================================================================================
def save(out_dir, ids, labels, answers, probs, replies, run_config, task):
    pos, neg = TASKS[task]["pos"].lower(), TASKS[task]["neg"].lower()
    with open(out_dir / "all_instances.jsonl", "w", encoding="utf-8") as f:
        for i in range(len(ids)):
            f.write(json.dumps({
                "patient_id": ids[i],
                "true_label": pos if labels[i] == 1 else neg,
                "predicted_label": answers[i],
                "prob_pos": probs[i],
                "model_output": replies[i],
            }, ensure_ascii=False) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


# =============================================================================================
# Run
# =============================================================================================
def get_api_key(arm):
    for env in ARMS[arm]["key_env"]:
        if os.environ.get(env):
            return os.environ[env]
    key = getpass(f"API key for {arm} ({ARMS[arm]['key_env'][0]} is not set): ").strip()
    if not key:
        raise SystemExit("no API key")
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=sorted(ARMS))
    parser.add_argument("--task", default=TASK, choices=sorted(TASKS))
    parser.add_argument("--limit", type=int, default=N_TEST, help="test patients to predict")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation before calling")
    args = parser.parse_args()
    task, arm = args.task, args.arm

    out_dir = RESULTS_DIR / task / arm / "confidence"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load
    ids, profiles, labels = load_test(task, args.limit)
    print(f"task {task} | arm {arm} ({ARMS[arm]['model']}) | {len(ids)} patients, {sum(labels)} positive")

    # 2. prompt
    prompts = [build_prompt(task, p) for p in profiles]

    # 3. call
    n_cached = len(set(cache_key(arm, s, u) for s, u in prompts) & set(load_cache(out_dir / "cache.jsonl")))
    if n_cached < len(prompts) and not args.yes:
        if input(f"{len(prompts) - n_cached} paid API calls to make. Proceed? [y/N] ").strip().lower() != "y":
            raise SystemExit("stopped, nothing called")
    api_key = get_api_key(arm) if n_cached < len(prompts) else None
    replies = get_replies(arm, api_key, prompts, out_dir / "cache.jsonl")

    # 4. parse
    answers = [parse_answer(task, r) for r in replies]
    probs = [parse_probability(r) for r in replies]
    n_no_answer = sum(a is None for a in answers)
    n_no_prob = sum(p is None for p in probs)
    if n_no_answer or n_no_prob:
        print(f"WARNING: {n_no_answer} replies without a readable answer, {n_no_prob} without a "
              f"probability; saved as null (see model_output)")

    # 5. save
    run_config = {
        "task": task, "arm": arm, **{k: v for k, v in ARMS[arm].items() if k != "key_env"},
        "temperature": TEMPERATURE, "top_p": TOP_P, "n_rows": len(ids),
        "prompt_sha1": prompt_sha1(task),
        "n_no_answer": n_no_answer, "n_no_probability": n_no_prob,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save(out_dir, ids, labels, answers, probs, replies, run_config, task)

    scored = [(y, p) for y, p in zip(labels, probs) if p is not None]
    auroc = roc_auc_score([y for y, _ in scored], [p for _, p in scored])
    print(f"saved to {out_dir}\nAUROC {auroc:.4f} on {len(scored)} patients (check only)")


if __name__ == "__main__":
    main()
