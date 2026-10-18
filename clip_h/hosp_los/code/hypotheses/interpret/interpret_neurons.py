# Portions adapted from HypotheSAEs v0.2.0 (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See THIRD_PARTY_NOTICES.md at the repository root.
"""Phase 2A: turn each selected SAE neuron into one natural-language hypothesis.

Functions are in the order the pipeline calls them:
    sample_top_zero            pick the example profiles shown for one neuron
    interpret_neuron           one LLM call -> one candidate interpretation
    interpret_neurons          every neuron x every candidate, with a resume cache
    score_interpretations      fidelity of every candidate on held-in examples
    keep_best_interpretations  one row per neuron: the best-scoring candidate
"""

import concurrent.futures
import hashlib
import json
import os
import threading
import tiktoken
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from llm_client import Progress, get_completion
from prompts import load_prompt
from annotation import annotate_fidelity


def truncate_text(
    text: str,
    max_words: Optional[int] = None,
    max_chars: Optional[int] = None,
    max_tokens: Optional[int] = None,
    truncation_message: str = "[... rest of text is truncated]"
) -> str:
    """
    Truncate text based on words, characters, or tokens.
    
    Args:
        text: Input text to truncate
        max_words: Maximum number of words
        max_chars: Maximum number of characters
        max_tokens: Maximum number of tokens (using tiktoken)
    
    Returns:
        Truncated text with indicator if truncated
    """
    if all(x is None for x in [max_words, max_chars, max_tokens]):
        return text
    
    if text.endswith(truncation_message):
        return text
    truncated = text
    
    if max_words is not None:
        words = text.split()
        if len(words) > max_words:
            truncated = ' '.join(words[:max_words])
            
    if max_chars is not None:
        if len(truncated) > max_chars:
            truncated = truncated[:max_chars]
            
    if max_tokens is not None:
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(truncated)
        if len(tokens) > max_tokens:
            truncated = enc.decode(tokens[:max_tokens])
            
    if truncated != text:
        truncated += truncation_message
        
    return truncated


def sample_top_zero(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample top activating examples and random zero-activation examples for a given neuron."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    neuron_acts = activations[:, neuron_idx]
    n_per_class = n_examples // 2
    
    # Get indices of positive activations and take top n_per_class (or fewer if not enough positive)
    count_positive_activating = np.sum(neuron_acts > 0)
    if count_positive_activating < n_per_class:
        print(f"Warning: Only found {count_positive_activating} examples with positive activation, using all available")
        top_indices = np.argsort(neuron_acts)[-count_positive_activating:]
    else:
        top_indices = np.argsort(neuron_acts)[-n_per_class:]
    
    # Get zero activation examples
    zero_indices = np.where(neuron_acts == 0)[0]
    if len(zero_indices) >= n_per_class:
        random_indices = np.random.choice(zero_indices, size=n_per_class, replace=False)
    else:
        print(f"Warning: Only found {len(zero_indices)} examples with zero activation, using all available")
        random_indices = zero_indices
    
    pos_texts = [texts[i] for i in top_indices]
    neg_texts = [texts[i] for i in random_indices]
    
    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words_per_example) for text in neg_texts]
    
    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[top_indices].tolist(),
        "negative_activations": neuron_acts[random_indices].tolist()
    }


def _parse_interpretation(response: str) -> str:
    """Parse raw LLM response into clean interpretation string."""
    response = response.strip()
    # DeepSeek v4 often wraps the line in ESCAPED quotes and can leave a trailing
    # backslash, which the plain-quote strip below misses; those characters would then stay
    # in the hypothesis text and in every prompt and table that shows it.
    response = response.replace('\\"', '"').strip().strip('\\').strip()
    if response.startswith('- '):
        response = response[2:]
    if response.startswith('"-'):
        response = response[2:]
    if response.startswith('" -'):
        response = response[3:]
    return response.strip('"').strip()


def interpret_neuron(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    *,
    model: str,
    prompt_name: str,
    task_specific_instructions: str,
    n_examples: int,
    max_words_per_example: Optional[int],
    temperature: float,
    max_tokens: int,
    timeout: float,
    random_seed: Optional[int] = None,
    candidate: int = 0,
) -> Optional[str]:
    """Generate one candidate interpretation for a single neuron (one LLM call).

    The prompt is the template ``prompt_name`` filled with ``task_specific_instructions`` and
    the examples from ``sample_top_zero``. Returns None for a neuron that never fires.
    ``candidate`` is the candidate's number for this neuron; it keeps two candidates that
    happen to get the same prompt from sharing one saved reply in the call cache.
    """
    if np.all(activations[:, neuron_idx] <= 0):
        print(f"Warning: All activations for neuron {neuron_idx} are <= 0. This neuron may be dead. Skipping interpretation.")
        return None

    formatted_examples = sample_top_zero(
        texts=texts,
        activations=activations,
        neuron_idx=neuron_idx,
        n_examples=n_examples,
        max_words_per_example=max_words_per_example,
        random_seed=random_seed,
    )

    prompt_template = load_prompt(prompt_name)
    try:
        prompt = prompt_template.format(
            task_specific_instructions=task_specific_instructions,
            **formatted_examples
        )
    except KeyError as e:
        raise KeyError(f"Missing required key {e} in the interpretation prompt template. Please ensure all required keys are provided in formatted_examples.")

    response = get_completion(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        attempt=candidate,
    )
    return _parse_interpretation(response)


def interpret_neurons(
    texts: List[str],
    activations: np.ndarray,
    neuron_indices: List[int],
    *,
    model: str,
    prompt_name: str,
    task_specific_instructions: str,
    n_candidates: int,
    n_examples: int,
    max_words_per_example: Optional[int],
    temperature: float,
    max_tokens: int,
    timeout: float,
    n_workers: int,
    random_seed: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> Dict[int, List[str]]:
    """Generate ``n_candidates`` interpretations for each neuron.

    With ``cache_path``, every finished call is written to that JSON file at once, so an
    interrupted run resumes without paying for the same call twice. The key is
    (model, prompt name, task instructions, neuron index, candidate slot); slots are given
    out per neuron in the order the calls start. The key does not include the prompt text,
    because the zero-activation examples are drawn afresh on each run.

    A call that fails is reported and left out, so a neuron can end with fewer candidates.
    """
    lock = threading.Lock()
    slots: Dict[int, int] = {}
    cache = {}
    if cache_path is not None:
        cache_path = str(cache_path)
        try:
            with open(cache_path, encoding="utf-8") as fh:
                cache = json.load(fh)
        except (OSError, ValueError):
            cache = {}
    signature = hashlib.sha1(
        f"{model}|{prompt_name}|{task_specific_instructions}".encode("utf-8")).hexdigest()[:12]

    def one_candidate(neuron_idx: int) -> Optional[str]:
        with lock:
            slot = slots.get(neuron_idx, 0)
            slots[neuron_idx] = slot + 1
            key = f"{signature}|{neuron_idx}|{slot}"
            if cache_path is not None and key in cache:
                return cache[key]
        value = interpret_neuron(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            model=model,
            prompt_name=prompt_name,
            task_specific_instructions=task_specific_instructions,
            n_examples=n_examples,
            max_words_per_example=max_words_per_example,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            random_seed=random_seed,
            candidate=slot,
        )
        if cache_path is not None:
            with lock:
                cache[key] = value
                tmp = cache_path + ".tmp"
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(cache, fh)
                os.replace(tmp, cache_path)   # atomic: a kill mid-write cannot corrupt the cache
        return value

    interpretation_tasks = [
        (neuron_idx, candidate_idx)
        for neuron_idx in neuron_indices
        for candidate_idx in range(n_candidates)
    ]

    interpretations = {idx: [] for idx in neuron_indices}
    progress = Progress(f"Phase 2A interpretation, {n_candidates} candidates per neuron",
                        len(interpretation_tasks))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {
            executor.submit(one_candidate, neuron_idx): (neuron_idx, candidate_idx)
            for neuron_idx, candidate_idx in interpretation_tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            neuron_idx, candidate_idx = future_to_task[future]
            try:
                interpretation = future.result()
                interpretations[neuron_idx].append(interpretation)
            except Exception as e:
                print(f"WARNING: no interpretation {candidate_idx} for neuron {neuron_idx}: "
                      f"{type(e).__name__}: {e}", flush=True)
            progress.step()
        progress.close()

    return interpretations


def _compute_metrics(
    annotations: np.ndarray,
    labels: np.ndarray,
    activations: np.ndarray
) -> Dict[str, float]:
    """Compute evaluation metrics for a single interpretation.
    
    Args:
        annotations: Annotations computed by an LLM by applying a neuron's natural language interpretation to a set of examples
        labels: Binarized neuron activations (e.g. by setting the top-N activations to 1 and the zero-activations to 0) for the scored examples
        activations: Continuous neuron activations for the scored examples
        
    Returns:
        A dictionary with, for the interpretation:
        recall       share of the firing examples (label 1) that are annotated 1
        precision    1 - the share of the non-firing examples (label 0) that are annotated 1.
                     This is the HypotheSAEs definition; in the usual terms it is the specificity.
        f1           harmonic mean of the two; the fidelity score of the pipeline
        correlation  Pearson correlation of the annotations with the continuous activations
    """
    if not (1 in labels and 0 in labels):
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0, "correlation": 0.0}
        
    true_pos = np.mean(annotations[labels == 1])
    false_pos = np.mean(annotations[labels == 0])
    precision = 1 - false_pos
    
    f1 = 2 * true_pos * precision / (true_pos + precision) if (true_pos + precision) > 0 else 0.0
    correlation = np.corrcoef(activations, annotations)[0,1] if len(np.unique(annotations)) > 1 else 0.0
    
    return {
        "recall": true_pos,
        "precision": precision,
        "f1": f1,
        "correlation": correlation
    }


def score_interpretations(
    texts: List[str],
    activations: np.ndarray,
    interpretations: Dict[int, List[str]],
    *,
    annotator_model: str,
    n_examples: int,
    max_words_per_example: Optional[int],
    n_workers: int,
    annotation_style: str,
    cache_path: Optional[str] = None,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Score all interpretations for all neurons.

    For each neuron, ``n_examples`` profiles are drawn (half top-activating, half
    zero-activating, seeded by the neuron index); ``annotation.annotate_fidelity`` labels
    every (profile, interpretation) pair in the ``annotation_style`` reply format, and the
    labels are compared with the neuron's firing.
    """
    tasks = []
    scoring_info = {}

    for neuron_idx, neuron_interps in interpretations.items():
        formatted_examples = sample_top_zero(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=n_examples,
            max_words_per_example=max_words_per_example,
            random_seed=neuron_idx,  # Deterministic seed based on neuron_idx
        )
        
        eval_texts = formatted_examples["positive_texts"] + formatted_examples["negative_texts"]
        scoring_info[neuron_idx] = {
            'texts': eval_texts,
            'activations': formatted_examples["positive_activations"] + formatted_examples["negative_activations"],
            'binarized_activations': np.concatenate([
                np.ones(len(formatted_examples["positive_texts"])),
                np.zeros(len(formatted_examples["negative_texts"]))
            ])
        }

        for interp in neuron_interps:
            for text in eval_texts:
                tasks.append((text, interp))

    # Annotate all tasks
    progress_desc = f"Phase 2A fidelity scoring, {len(interpretations)} neurons x {n_examples} profiles"
    
    annotations = annotate_fidelity(
        tasks=tasks,
        cache_path=cache_path,
        n_workers=n_workers,
        show_progress=True,
        model=annotator_model,
        progress_desc=progress_desc,
        style=annotation_style,
    )

    # Compute metrics for all interpretations
    all_metrics = {}
    for neuron_idx, neuron_interps in interpretations.items():
        all_metrics[neuron_idx] = {}
        neuron_scoring_info = scoring_info[neuron_idx]

        for interp in neuron_interps:
            annot = [annotations[interp][text] for text in neuron_scoring_info['texts']]
            all_metrics[neuron_idx][interp] = _compute_metrics(
                annotations=np.array(annot),
                labels=neuron_scoring_info['binarized_activations'],
                activations=neuron_scoring_info['activations']
            )

    return all_metrics


def keep_best_interpretations(
    neuron_indices: List[int],
    selection_scores: List[float],
    interpretations: Dict[int, List[str]],
    metrics: Dict[int, Dict[str, Dict[str, float]]],
    neuron_source_sae_info: list,
    *,
    selection_method: str,
    scoring_metric: str,
) -> pd.DataFrame:
    """One row per neuron: its best-scoring candidate interpretation.

    Neurons whose candidates all failed are reported and left out. Columns: neuron_idx,
    source_sae, target_{selection_method}, interpretation, {scoring_metric}_fidelity_score.
    """
    skipped = [idx for idx in neuron_indices if not interpretations.get(idx)]
    if skipped:
        print(f"  [warn] {len(skipped)}/{len(neuron_indices)} neurons produced no "
              f"interpretation (all candidates failed); dropped from pool: {skipped}")

    results = []
    for idx, score in zip(neuron_indices, selection_scores):
        if not interpretations.get(idx):
            continue
        # Find best interpretation and its score using max()
        best_interp = max(
            interpretations[idx],
            key=lambda interp: metrics[idx][interp][scoring_metric]
        )
        best_score = metrics[idx][best_interp][scoring_metric]

        results.append({
            'neuron_idx': idx,
            'source_sae': neuron_source_sae_info[idx],
            f'target_{selection_method}': score,
            'interpretation': best_interp,
            f'{scoring_metric}_fidelity_score': best_score
        })

    return pd.DataFrame(results)
