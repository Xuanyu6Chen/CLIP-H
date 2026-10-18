"""Phase 2A+: name the weak neurons as grouped OR-hypotheses.

A weak neuron fires on too few profiles to be interpreted alone (see
``clip_h_sae.split_weak_neurons``). The LLM groups them by clinical meaning; no outcome labels
are used.

    name_weak_neuron        one LLM call -> a short name for one weak neuron
    group_weak_neurons      one LLM call -> the names sorted into clinically related groups,
                            each group stated as one hypothesis
    interpret_weak_groups   name every neuron, then group
"""

import concurrent.futures
import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from llm_client import Progress, get_completion
from prompts import load_prompt
from interpret_neurons import truncate_text

PARSE_TRIES = 3   # how often a reply that cannot be used is asked for again


def _examples(texts, activations, neuron, top_per_neuron, max_chars_per_example):
    """The top profiles of one neuron, each cut to ``max_chars_per_example`` characters."""
    col = activations[:, neuron]
    top = [int(i) for i in np.argsort(col)[::-1] if col[i] > 0][:top_per_neuron]
    return "\n--\n".join(truncate_text(texts[i], max_chars=max_chars_per_example) for i in top)


def _negatives(texts, activations, neurons, n_negatives, max_chars_per_example, seed):
    """Random profiles on which none of ``neurons`` fire."""
    pool = np.where(~(activations[:, neurons] > 0).any(axis=1))[0]
    rng = np.random.default_rng(seed)
    idx = [int(i) for i in rng.choice(pool, size=min(n_negatives, len(pool)), replace=False)]
    return "\n---\n".join(truncate_text(texts[i], max_chars=max_chars_per_example) for i in idx)


def _one_line(reply) -> str:
    return (reply or "").strip().lstrip('-').strip().strip('"').strip()


def name_weak_neuron(
    texts: List[str],
    activations: np.ndarray,
    neuron: int,
    *,
    model: str,
    max_tokens: int,
    timeout: float,
    top_per_neuron: int,
    n_negatives: int,
    max_chars_per_example: int,
    negatives_seed: int,
) -> Optional[str]:
    """A short name for what one weak neuron marks (one LLM call), or None if no usable reply."""
    prompt = load_prompt('weak_groups', 'name_user').format(
        positive_texts=_examples(texts, activations, neuron, top_per_neuron, max_chars_per_example),
        negative_texts=_negatives(texts, activations, [neuron], n_negatives, max_chars_per_example,
                                  negatives_seed))
    for attempt in range(PARSE_TRIES):
        try:
            name = _one_line(get_completion(prompt, model=model, max_tokens=max_tokens, timeout=timeout,
                                            system=load_prompt('weak_groups', 'system'),
                                            attempt=attempt))
        except Exception:
            continue
        if name:
            return name
    return None


def group_weak_neurons(
    names: Dict[int, str],
    *,
    model: str,
    max_tokens: int,
    timeout: float,
) -> List[dict]:
    """Sort named weak neurons into clinically related groups and state each as one hypothesis
    (one LLM call).

    Returns a list of {'theme': str, 'neurons': [neuron indices], 'hypothesis': str}; every
    group has at least two neurons and no neuron is in two groups. A neuron the model leaves
    ungrouped is in no group. Raises RuntimeError when no reply can be read.
    """
    order = list(names)
    listing = "\n".join(f"{k}: {names[n]}" for k, n in enumerate(order, 1))
    prompt = load_prompt('weak_groups', 'group_user').format(conditions=listing)
    problem = ""
    for attempt in range(PARSE_TRIES):
        try:
            raw = get_completion(prompt, model=model, max_tokens=max_tokens, timeout=timeout,
                                 system=load_prompt('weak_groups', 'system'), attempt=attempt)
            text = re.sub(r"^```(?:json)?|```$", "", str(raw).strip(), flags=re.MULTILINE).strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            reply_groups = json.loads(match.group(0) if match else text)["groups"]
        except Exception as exc:
            problem = type(exc).__name__
            continue
        seen, groups = set(), []
        for g in reply_groups:
            members = []
            for k in g.get("members", []):
                if isinstance(k, int) and 1 <= k <= len(order) and k not in seen:
                    seen.add(k)
                    members.append(order[k - 1])
            hypothesis = _one_line(str(g.get("hypothesis", "")))
            if len(members) >= 2 and hypothesis:
                groups.append({'theme': str(g.get("theme", "")).strip(), 'neurons': members,
                               'hypothesis': hypothesis})
        return groups
    raise RuntimeError(f"weak-neuron grouping: no readable reply in {PARSE_TRIES} tries ({problem})")


def interpret_weak_groups(
    texts: List[str],
    activations: np.ndarray,
    weak_neurons: List[Tuple[int, int]],
    *,
    model: str,
    max_tokens: int,
    timeout: float,
    top_per_neuron: int = 5,
    n_negatives: int = 15,
    max_chars_per_example: int = 256,
    negatives_seed: int = 42,
    n_workers: int = 4,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Name each weak neuron from its profiles, then let the LLM sort the names into related
    groups and state each group as one OR-hypothesis.

    Returns:
        rows: one {'neuron_idx': group id, 'interpretation': text} per group; group ids are
            -1, -2, ... so that they cannot be mistaken for a neuron index
        groups: group id (as a string) -> {'theme', 'neuron_idxs', 'neuron_names',
            'activation_frequency', 'n_positives'}, as saved in pooled_group.json
    """
    neurons = [int(n) for n, _ in weak_neurons]
    kw = dict(model=model, max_tokens=max_tokens, timeout=timeout)
    shape = dict(top_per_neuron=top_per_neuron, n_negatives=n_negatives,
                 max_chars_per_example=max_chars_per_example, negatives_seed=negatives_seed)
    progress = Progress("Phase 2A+ naming weak neurons", len(neurons))

    def name_one(n):
        name = name_weak_neuron(texts, activations, n, **kw, **shape)
        if name is None:
            print(f"WARNING: weak neuron {n}: no usable name in {PARSE_TRIES} tries; left out", flush=True)
        progress.step()
        return name

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        named = list(ex.map(name_one, neurons))
    progress.close()
    names = {n: name for n, name in zip(neurons, named) if name}
    print(f"  weak neurons named: {len(names)} of {len(neurons)}")
    print(f"  [Phase 2A+] grouping {len(names)} names in one call...", flush=True)
    found = group_weak_neurons(names, **kw) if len(names) >= 2 else []
    n_grouped = sum(len(g['neurons']) for g in found)
    print(f"  sorted into {len(found)} group(s); {len(names) - n_grouped} neuron(s) left ungrouped")

    rows, groups = [], {}
    for g in found:
        members = g['neurons']
        fires = (activations[:, members] > 0).any(axis=1)
        group_id = -(len(rows) + 1)
        print(f"  [{g['theme']}] {len(members)} neurons, union {fires.mean() * 100:.2f}% "
              f"({int(fires.sum())} pos) -> {g['hypothesis']}")
        rows.append({'neuron_idx': group_id, 'interpretation': g['hypothesis']})
        groups[str(group_id)] = {'theme': g['theme'], 'neuron_idxs': members,
                                 'neuron_names': [names[n] for n in members],
                                 'activation_frequency': float(fires.mean()),
                                 'n_positives': int(fires.sum())}
    return rows, groups
