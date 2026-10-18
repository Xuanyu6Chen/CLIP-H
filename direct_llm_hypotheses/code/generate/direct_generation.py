"""Phases G1 and G2: hypotheses written by an LLM straight from labeled train profiles.

    generation_batches   seeded, class-balanced, non-overlapping batches of train rows
    generation_prompt    the user prompt for one batch (prompts/generation.txt, section user)
    parse_hypotheses     the numbered 'Patient ...' lines of a reply
    generate_pool        one call per batch, asked again while a reply holds too few hypotheses
    saved_pool           the pool of an earlier run, reused only when its settings are the same
    pick_library         exact-text dedup over the batches taken in turn, first K

Every call goes through llm_client.get_completion, so with the call cache installed a rerun
pays nothing.
"""
import json
import re

import numpy as np

import llm_client
from fidelity_gate import norm_hypothesis
from prompts import load_prompt


def generation_batches(y, n_batches, batch_size, seed):
    """Row numbers of ``n_batches`` batches, each half positive and half negative."""
    rng = np.random.default_rng(seed)
    pos = [i for i, v in enumerate(y) if v == 1]
    neg = [i for i, v in enumerate(y) if v == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    half, batches = batch_size // 2, []
    for b in range(n_batches):
        idxs = pos[b * half:(b + 1) * half] + neg[b * half:(b + 1) * half]
        rng.shuffle(idxs)
        batches.append(idxs)
    if not all(len(b) == batch_size for b in batches):
        raise ValueError("not enough positive or negative train patients for the batch plan")
    return batches


def generation_prompt(profiles, y, idxs, outcome, extra_rules, k):
    """User prompt for one batch: the labeled profiles and the task wording."""
    block = "\n\n".join(f"[Outcome: {'Yes' if y[i] == 1 else 'No'}]\n{profiles[i]}" for i in idxs)
    return load_prompt("generation", "user").format(
        n=len(idxs), outcome=outcome, k=k, extra_rules=(extra_rules + "\n") if extra_rules else "",
        profiles_block=block)


def parse_hypotheses(reply):
    """The numbered 'Patient ...' lines of a reply (a <think> block is ignored)."""
    text = re.sub(r"<think>.*?</think>", "", reply or "", flags=re.S)
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-•\d.)\s]*", "", re.sub(r"[*_`#]", "", line)).strip()
        if line.lower().startswith("patient"):
            out.append(line)
    return out


def generate_pool(prompts, model, temperature, max_tokens, min_per_batch, max_attempts, timeout=600.0):
    """Hypotheses per batch.

    A reply with fewer than ``min_per_batch`` hypotheses is asked again as a new call
    (``attempt`` 1, 2, ...); after ``max_attempts`` calls the run stops.
    """
    system = load_prompt("generation", "system")
    pool = []
    for b, prompt in enumerate(prompts):
        for attempt in range(max_attempts):
            reply = llm_client.get_completion(prompt=prompt, model=model, system=system, attempt=attempt,
                                              temperature=temperature, max_tokens=max_tokens, timeout=timeout)
            hyps = parse_hypotheses(reply)
            if len(hyps) >= min_per_batch:
                break
            print(f"  batch {b + 1}: only {len(hyps)} hypotheses in the reply, asking again")
        else:
            raise RuntimeError(f"batch {b + 1}: fewer than {min_per_batch} hypotheses after {max_attempts} calls")
        print(f"batch {b + 1}/{len(prompts)}: {len(hyps)} hypotheses")
        pool.append(hyps)
    return pool


def saved_pool(pool_path, config_path, settings):
    """The pool saved by an earlier run, or None if there is none.

    Reused only when that run's config records the same generation settings; otherwise the
    run stops, so a pool made with other settings is never reused by mistake.
    """
    if not pool_path.exists():
        return None
    saved = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    differ = {k: (saved.get(k), v) for k, v in settings.items() if saved.get(k) != v}
    if differ:
        raise RuntimeError(f"{pool_path} was made with other settings (saved, now): {differ}. "
                           "Use a new RUN_NAME, or set FORCE_RERUN['generation'] = True.")
    return json.loads(pool_path.read_text(encoding="utf-8"))


def pick_library(pool, k):
    """Exact-text dedup over the batches taken in turn (b1[0], b2[0], ..., b1[1], ...); first k."""
    seen, out = set(), []
    for j in range(max(map(len, pool), default=0)):
        for batch in pool:
            if j < len(batch):
                key = norm_hypothesis(batch[j])
                if key and key not in seen:
                    seen.add(key); out.append(batch[j])
    return out[:k]
