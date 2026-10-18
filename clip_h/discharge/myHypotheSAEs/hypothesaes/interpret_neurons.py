# Modified for CLIP-H from HypotheSAEs (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See ../README.md for the changes.
"""Methods for interpreting SAE neurons using LLMs."""

import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Any
from tqdm.auto import tqdm
import concurrent.futures
import os
from dataclasses import dataclass, field

from .llm_api import get_completion
from .utils import load_prompt, truncate_text
from .annotate import annotate, CACHE_DIR

DEFAULT_TASK_SPECIFIC_INSTRUCTIONS = """An example feature could be:
- "uses multiple adjectives to describe colors"
- "describes a patient experiencing seizures or epilepsy"
- "contains multiple single-digit numbers\""""

# System prompt for the interpreter agent. It is written for clinical admission profiles
# (task-specific steering is injected via task_specific_instructions) and fixes the role, the
# contrastive objective, and the output contract so every interpreter call is unambiguous.
#
# The `naming` mode controls requirements 2-3 (the granularity policy):
#   'faithful' (default): name each neuron at the granularity the evidence actually supports, so
#       distinct neurons keep distinct labels. A downstream step can still merge two features that
#       turn out to fire on the same patients, but it can never recover a distinction erased by an
#       over-broad name, so naming errs toward faithful granularity.
#   'broad': prefer a broadly applicable label; kept as an alternative for ablation.
_INTERPRETER_HEAD = (
    "You are a senior clinical-informatics scientist interpreting a pattern detector trained on "
    "hospital admission profiles. It fires strongly on one set of patients (the positive samples) "
    "and not on another set from the same population (the negative samples). Name the single "
    "clinical feature that is present in essentially all of the positive patients and absent from "
    "the negative ones. This name becomes a hypothesis a physician will review, and it is later "
    "tested against held-out data, so it must read like a clinical finding and be precise and "
    "generalizable.\n"
    "Follow these requirements:\n"
    "1. Name a concrete, chart-observable patient feature a clinician would recognize -- a "
    "laboratory abnormality and its direction, a vital-sign or acuity pattern, an organ "
    "dysfunction, a comorbidity, or an admission or care-need characteristic -- not an abstract "
    "theme. A demographic or administrative attribute (age, sex, race, insurance, language) is an "
    "admissible feature when it is genuinely what separates the positive patients: do not avoid it, "
    "and do not force a clinical framing onto a signal that is truly demographic.\n"
)
_INTERPRETER_TAIL = (
    "4. Phrase it as a patient-level clinical finding ('patients with ...') that another clinician "
    "could confirm Yes or No from a single patient's profile, worded clearly enough that two "
    "reviewers would annotate it the same way.\n"
    "5. Output only the feature in the exact format requested, with no preamble, reasoning, or "
    "alternatives."
)
_INTERPRETER_REQS = {
    "faithful": (
        "2. Name one coherent clinical concept. A compound pattern is welcome when the evidence "
        "supports one recognizable clinical picture (for example 'anaemia with renal impairment' or "
        "'elderly patient admitted from a skilled-nursing facility'); a richer faithful pattern is "
        "more informative than a single broad diagnosis. Do not, however, stack unrelated attributes "
        "that merely co-occur in a handful of samples (for example 'female and middle-aged and low "
        "BMI and emergency'), because such conjunctions overfit and do not generalize. If several "
        "unrelated attributes co-vary, name the single most informative one.\n"
        "3. Name the feature at the granularity the evidence actually supports: give the most specific "
        "description that still holds across essentially all the positive samples, using a clinically "
        "meaningful range or direction (for example 'low BMI' or 'age 80 or older') rather than a "
        "single point-value read off one record (for example 'BMI 18.2' or 'age 91'). Match the "
        "neuron's real granularity -- do not widen distinct patient groups into one broad umbrella "
        "label, and do not invent a precision the samples do not show. A later step can merge two "
        "features that fire on the same patients, but it can never recover a distinction lost to an "
        "over-broad name, so keep genuinely different neurons distinct.\n"
    ),
    "broad": (
        "2. Name one coherent clinical concept; a compound pattern is welcome when the evidence "
        "supports a single recognizable clinical picture. Do not stack unrelated conditions that "
        "merely co-occur in a few samples (for example 'female and middle-aged and low BMI and "
        "emergency'), since such conjunctions overfit and do not generalize. If several unrelated "
        "attributes co-vary, name the one dominant, generalizable concept.\n"
        "3. Prefer a broadly applicable feature over a narrow one: it should plausibly hold for a "
        "meaningful share of the population, not an exception of an exception. Use clinically "
        "meaningful ranges or directions such as 'low BMI' or 'advanced age' rather than specific "
        "numeric point-values such as 'BMI 18.2' or 'age 91'.\n"
    ),
}

def build_interpreter_system_prompt(naming: str = "faithful") -> str:
    """Assemble the interpreter system prompt for the given `naming` granularity policy
    ('faithful' default | 'broad' ablation). See the module comment above for the rationale."""
    if naming not in _INTERPRETER_REQS:
        raise ValueError(f"naming must be one of {sorted(_INTERPRETER_REQS)}, got {naming!r}")
    return _INTERPRETER_HEAD + _INTERPRETER_REQS[naming] + _INTERPRETER_TAIL

# Module-level constant holding the default ('faithful') interpreter system prompt.
INTERPRETER_SYSTEM_PROMPT = build_interpreter_system_prompt("faithful")

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

def sample_percentile_bins(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    n_examples: int,
    max_words_per_example: Optional[int] = None,
    high_percentile: Tuple[float, float] = (90, 100),
    low_percentile: Optional[Tuple[float, float]] = None,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample examples from high activation percentile bins and either low percentile or zero activations."""
    if random_seed is not None:
        np.random.seed(random_seed)
        
    neuron_acts = activations[:, neuron_idx]
    n_per_class = n_examples // 2
    
    pos_mask = neuron_acts > 0
    pos_vals = neuron_acts[pos_mask]
    pos_indices = np.where(pos_mask)[0]
    
    high_mask = (pos_vals >= np.percentile(pos_vals, high_percentile[0])) & \
               (pos_vals <= np.percentile(pos_vals, high_percentile[1]))
    high_indices = pos_indices[high_mask]
    if len(high_indices) >= n_per_class:
        high_sample_indices = np.random.choice(high_indices, size=n_per_class, replace=False)
    else:
        print(f"Warning: There are less than {n_per_class} examples in bin {high_percentile} for neuron {neuron_idx}; using {len(high_indices)} instead")
        high_sample_indices = high_indices
    
    if low_percentile is not None:
        low_mask = (pos_vals >= np.percentile(pos_vals, low_percentile[0])) & \
                  (pos_vals <= np.percentile(pos_vals, low_percentile[1]))
        low_indices = pos_indices[low_mask]
    else: # Use examples with zero activation as the negative examples
        low_indices = np.where(neuron_acts == 0)[0]

    if len(low_indices) >= n_per_class:
        low_sample_indices = np.random.choice(low_indices, size=n_per_class, replace=False)
    else:
        print(f"Warning: There are less than {n_per_class} examples in bin {low_percentile} for neuron {neuron_idx}; using {len(low_indices)} instead")
        low_sample_indices = low_indices
    
    pos_texts = [texts[i] for i in high_sample_indices]
    neg_texts = [texts[i] for i in low_sample_indices]
    
    if max_words_per_example:
        pos_texts = [truncate_text(text, max_words_per_example) for text in pos_texts]
        neg_texts = [truncate_text(text, max_words_per_example) for text in neg_texts]
    
    return {
        "positive_texts": pos_texts,
        "negative_texts": neg_texts,
        "positive_activations": neuron_acts[high_sample_indices].tolist(),
        "negative_activations": neuron_acts[low_sample_indices].tolist()
    }

def sample_custom(
    texts: List[str],
    activations: np.ndarray,
    neuron_idx: int,
    random_seed: Optional[int] = None
) -> Dict[str, List[str]]:
    """Sample examples using a custom function.
    
    This function should return a dictionary with keys that correspond to your prompt template.
    The default prompt template is "interpret-neuron-binary.txt", which expects two keys: positive_samples and negative_samples.

    For example, you can write a custom sampling function that outputs:
    - only top-activating examples
    - top-activating, medium-activating, and non-activating examples
    - etc.

    Note that if you change the sampling setup, you will also need to write a new prompt template.
    Ensure that the keys in your output dictionary match the keys in your prompt template.
    
    Args:
        texts: List of all text examples
        activations: Neuron activation matrix (n_samples, n_neurons)
        neuron_idx: Index of neuron to sample examples for
        [any other arguments]
    """
    pass

@dataclass
class SamplingConfig:
    function: Callable = sample_top_zero
    n_examples: int = 20 # Number of examples to sample to prompt the interpreter
    random_seed: Optional[int] = 0 # Random seed for example sampling (fixed for reproducibility)
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function

@dataclass
class LLMConfig:
    temperature: float = 0.0 # Temperature for the interpreter model (0.0 for reproducibility)
    max_interpretation_tokens: int = 250 # Maximum number of tokens for each generated interpretation
    timeout: float = 300.0 # Timeout for the interpreter model (in seconds)
    seed: Optional[int] = None # Seed forwarded to the OpenAI API for reproducibility

@dataclass
class InterpretConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    n_candidates: int = 1 # Number of candidate interpretations per neuron
    prompt_name: str = "interpret-neuron-binary" # Name of the prompt template file to use
    task_specific_instructions: str = DEFAULT_TASK_SPECIFIC_INSTRUCTIONS # Task-specific instructions for the interpreter model
    naming: str = "faithful" # Granularity policy for the system prompt: 'faithful' (default) | 'broad' (ablation)

@dataclass
class ScoringConfig:
    n_examples: int = 100 # Number of examples to score interpretation fidelity (half top-activating, half zero-activating)
    max_words_per_example: Optional[int] = 256 # Maximum number of words per text example, truncated if necessary
    sampling_function: Callable = sample_top_zero # Function to sample examples for scoring
    sampling_kwargs: Dict[str, Any] = field(default_factory=dict) # Extra keyword arguments for the sampling function
    use_batch: bool = False # Route the (large, one-shot) fidelity-scoring annotation through the OpenAI Batch API (about half the cost, same requests as the synchronous path; synchronous fallback for unresolved tasks)

class NeuronInterpreter:
    def __init__(
        self,
        interpreter_model: str = "gpt-4o",
        annotator_model: str = "gpt-4o-mini",
        n_workers_interpretation: int = 10,
        n_workers_annotation: int = 30,
        cache_name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize a NeuronInterpreter."""
        self.interpreter_model = interpreter_model
        self.annotator_model = annotator_model
        self.n_workers_interpretation = n_workers_interpretation
        self.n_workers_annotation = n_workers_annotation
        self.cache_name = cache_name
        self.seed = seed  # forwarded to scoring annotations for reproducibility
        
    def _get_interpretation_completion(
        self,
        prompt_template: str,
        formatted_examples: dict,
        config: InterpretConfig,
    ) -> str:
        """Get and parse interpretation completion from LLM."""
        try:
            prompt = prompt_template.format(
                task_specific_instructions=config.task_specific_instructions,
                **formatted_examples
            )
        except KeyError as e:
            raise KeyError(f"Missing required key {e} in the interpretation prompt template. Please ensure all required keys are provided in formatted_examples.")
        if 'qwen' in self.interpreter_model.lower():
            prompt += ' /no_think'  # Disable chain-of-thought for Qwen models (e.g. served by Ollama)
        if self.interpreter_model.startswith(('o', 'gpt-5')):
            # Reasoning models (o-series, gpt-5*): no temperature; pass the budget as
            # max_completion_tokens. get_completion floors this high so hidden reasoning
            # never starves the (short) interpretation output.
            response = get_completion(
                prompt=prompt,
                model=self.interpreter_model,
                system=build_interpreter_system_prompt(config.naming),
                max_completion_tokens=config.llm.max_interpretation_tokens,
                timeout=config.llm.timeout,   # reasoning is slow; use the configured timeout
                seed=config.llm.seed,
                reasoning_effort='low'
            )
        else:
            response = get_completion(
                prompt=prompt,
                model=self.interpreter_model,
                system=build_interpreter_system_prompt(config.naming),
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_interpretation_tokens,
                timeout=config.llm.timeout,
                seed=config.llm.seed
            )

        return self._parse_interpretation(response)
    
    def _parse_interpretation(self, response: str) -> str:
        """Parse raw LLM response into clean interpretation string."""
        import re
        # Strip <think>...</think> blocks produced by reasoning models (e.g. qwen3 without /no_think)
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        if response.startswith('- '):
            response = response[2:]
        if response.startswith('"-'):
            response = response[2:]
        if response.startswith('" -'):
            response = response[3:]
        return response.strip('"')

    def interpret_neuron(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_idx: int,
        config: InterpretConfig
    ) -> str:
        """Generate interpretation for a single neuron."""
        if np.all(activations[:, neuron_idx] <= 0):
            print(f"Warning: All activations for neuron {neuron_idx} are <= 0. This neuron may be dead. Skipping interpretation.")
            return None

        formatted_examples = config.sampling.function(
            texts=texts,
            activations=activations,
            neuron_idx=neuron_idx,
            n_examples=config.sampling.n_examples,
            max_words_per_example=config.sampling.max_words_per_example,
            random_seed=config.sampling.random_seed,
            **config.sampling.sampling_kwargs
        )
        
        prompt_template = load_prompt(config.prompt_name)
        return self._get_interpretation_completion(
            prompt_template=prompt_template,
            formatted_examples=formatted_examples,
            config=config
        )

    def interpret_neurons(
        self,
        texts: List[str],
        activations: np.ndarray,
        neuron_indices: List[int],
        config: Optional[InterpretConfig] = None
    ) -> Dict[int, List[str]]:
        """Generate interpretations for multiple neurons with multiple candidates each."""
        config = config or InterpretConfig()
        
        interpretation_tasks = [
            (neuron_idx, candidate_idx)
            for neuron_idx in neuron_indices
            for candidate_idx in range(config.n_candidates)
        ]
        
        interpretations = {idx: [] for idx in neuron_indices}
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers_interpretation) as executor:
            future_to_task = {
                executor.submit(
                    self.interpret_neuron,
                    texts=texts,
                    activations=activations,
                    neuron_idx=neuron_idx,
                    config=config
                ): (neuron_idx, candidate_idx)
                for neuron_idx, candidate_idx in interpretation_tasks
            }
            
            iterator = concurrent.futures.as_completed(future_to_task)
            iterator = tqdm(iterator, total=len(interpretation_tasks), 
                           desc=f"Generating {config.n_candidates} interpretation(s) per neuron")
            
            for future in iterator:
                neuron_idx, candidate_idx = future_to_task[future]
                try:
                    interpretation = future.result()
                    interpretations[neuron_idx].append(interpretation)
                except Exception as e:
                    print(f"Failed to generate interpretation {candidate_idx} for neuron {neuron_idx}: {e}")
        
        return interpretations

    def _compute_metrics(
        self,
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
            A dictionary containing the recall, precision, F1 score, and correlation for the interpretation
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
        self,
        texts: List[str],
        activations: np.ndarray,
        interpretations: Dict[int, List[str]],
        config: Optional[ScoringConfig] = None
    ) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Score all interpretations for all neurons."""
        config = config or ScoringConfig()
        tasks = []
        scoring_info = {}

        for neuron_idx, neuron_interps in interpretations.items():
            formatted_examples = config.sampling_function(
                texts=texts,
                activations=activations,
                neuron_idx=neuron_idx,
                n_examples=config.n_examples,
                max_words_per_example=config.max_words_per_example,
                random_seed=neuron_idx,  # Deterministic seed based on neuron_idx
                **config.sampling_kwargs
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
        progress_desc = f"Scoring neuron interpretation fidelity ({len(interpretations)} neurons; {len(next(iter(interpretations.values())))} candidate interps per neuron; {config.n_examples} examples to score each interp)"
        
        cache_path = None if self.cache_name is None else os.path.join(CACHE_DIR, f"{self.cache_name}_interp-scoring.json")
        annotations = annotate(
            tasks=tasks,
            cache_path=cache_path,
            n_workers=self.n_workers_annotation,
            show_progress=True,
            model=self.annotator_model,
            seed=self.seed,
            progress_desc=progress_desc,
            use_batch=config.use_batch,
        )

        # Compute metrics for all interpretations. A few annotations can fail permanently
        # (rate limits, or a reasoning model such as o-series or gpt-5* returning an empty
        # or timed-out completion on a trivial yes/no). annotate() then omits the (interp, text)
        # pair from its result dict, so a direct index would raise KeyError and abort the
        # entire hypothesis-generation phase. A missing pair is treated as 0 (concept absent),
        # consistent with the annotator's "when uncertain, answer No" policy and with the same
        # guard in annotate_texts_with_concepts, and the number defaulted is reported so a
        # degraded run is visible rather than silent.
        all_metrics = {}
        n_missing = 0
        for neuron_idx, neuron_interps in interpretations.items():
            all_metrics[neuron_idx] = {}
            neuron_scoring_info = scoring_info[neuron_idx]

            for interp in neuron_interps:
                col = annotations.get(interp, {})
                annot = []
                for text in neuron_scoring_info['texts']:
                    v = col.get(text)
                    if v is None:
                        v = 0
                        n_missing += 1
                    annot.append(v)
                all_metrics[neuron_idx][interp] = self._compute_metrics(
                    annotations=np.array(annot),
                    labels=neuron_scoring_info['binarized_activations'],
                    activations=neuron_scoring_info['activations']
                )

        if n_missing:
            # Add to the shared run-level counter as well as printing, so the run manifest
            # also records failures from this path (Phase 2.1 interpretation and fidelity
            # scoring). Missing annotations here lower fidelity F1 and can push hypotheses
            # below the fidelity threshold, so they must be visible in the run record.
            from . import annotate as _annotate_mod
            _annotate_mod.ANNOTATION_FAILURES += int(n_missing)
            print(f"Warning: {n_missing} interpretation-scoring annotation(s) failed after "
                  f"retries; defaulted to 0 (absent). Run total: "
                  f"{_annotate_mod.ANNOTATION_FAILURES}.")

        return all_metrics