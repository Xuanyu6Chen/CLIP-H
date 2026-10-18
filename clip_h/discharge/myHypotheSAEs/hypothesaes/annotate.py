# Modified for CLIP-H from HypotheSAEs (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See ../README.md for the changes.
"""Text annotation using LLM-based concept checking."""

import numpy as np
from typing import List, Optional, Dict, Tuple
import concurrent.futures
from tqdm.auto import tqdm
import hashlib
import os
import json
from pathlib import Path
import time

from .llm_api import get_completion
from .utils import load_prompt, truncate_text

CACHE_DIR = os.path.join(Path(__file__).parent.parent, 'annotation_cache')
DEFAULT_N_WORKERS = 30

# Running total of annotations in this process that failed permanently and were defaulted to 0
# (absent). See annotate_texts_with_concepts; the runner notebooks record this count in the run
# manifest.
ANNOTATION_FAILURES = 0

# Strict-grounding annotation contract, shared by the synchronous path (annotate_single_text)
# and the Batch API path (_annotation_request_body) so that both backends send identical
# requests. The Batch API runs the same model on the same prompt; it only resolves the
# requests asynchronously and at a discount.
ANNOTATE_SYSTEM_PROMPT = (
    "You are a meticulous binary annotator. You decide whether a single PROPERTY "
    "is explicitly supported by a given TEXT, using only the TEXT itself.\n"
    "Rules:\n"
    "1. Answer Yes only when the TEXT contains direct, unambiguous evidence for the PROPERTY.\n"
    "2. Do not infer, assume, or rely on outside knowledge or base rates.\n"
    "3. If the evidence is missing, ambiguous, negated, or only weakly implied, answer No.\n"
    "4. Judge the PROPERTY exactly as written; partial or related matches are No.\n"
    "Respond with exactly one word: yes or no."
)


def _build_annotation_prompt(
    text: str,
    concept: str,
    model: str,
    max_words_per_example: Optional[int] = None,
) -> Tuple[str, bool]:
    """Build the user prompt for one (text, concept) annotation and report whether the
    annotator is a reasoning model. Single source of truth for the sync + batch paths."""
    if max_words_per_example:
        text = truncate_text(text, max_words_per_example)
    prompt = load_prompt("annotate").format(hypothesis=concept, text=text)
    prompt += ' Answer with exactly one word: yes or no.'
    if 'qwen' in model.lower():
        prompt += ' /no_think'  # disable chain-of-thought for Qwen/Ollama models only
    _m = model.lower()
    is_reasoning = _m.startswith('o') or _m.startswith('gpt-5')
    return prompt, is_reasoning


def _parse_annotation_text(response_text: Optional[str]) -> Optional[int]:
    """Map a raw one-word completion to 1 (yes) / 0 (no) / None (unparseable)."""
    if response_text is None:
        return None
    t = str(response_text).strip().lower()
    if 'yes' in t:
        return 1
    if 'no' in t:
        return 0
    return None


def get_annotation_cache(cache_path: str) -> dict:
    """Load cached annotations from JSON file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: Failed to parse cache file {cache_path}, starting fresh cache")
            os.remove(cache_path)
    return {}

def save_annotation_cache(cache_path: str, cache: dict) -> None:
    """Save annotations to JSON cache file, atomically.

    Writes to a temp file and then calls ``os.replace`` (atomic on Windows and POSIX), so an
    interrupted write (a crash or an out-of-memory kill) never truncates the existing cache into
    unparseable JSON, which ``get_annotation_cache`` would delete along with all prior progress.
    This makes the frequent incremental flushes in ``_parallel_annotate`` safe."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp = f"{cache_path}.tmp{os.getpid()}"
    with open(tmp, 'w') as f:
        json.dump(cache, f)
    os.replace(tmp, cache_path)

def generate_cache_key(concept: str, text: str, model: str = "") -> str:
    """Generate a cache key for a (model, concept, text) triple.

    The model is part of the key so that annotations produced by different annotators (e.g. a
    cheap dry-run model and the annotator used for reported results) never collide: reusing a
    weaker model's yes/no for a run that declares a stronger annotator would silently contaminate
    results and break reproducibility. Same-model reruns still hit the cache; cross-model runs
    stay isolated.

    The text is hashed in full with SHA-256. A key built from only part of the text, such as its
    first and last 100 characters, would ignore the clinical middle of a patient profile (labs,
    vitals, medications, service): two patients who agree on the leading demographics and the
    trailing history would share one cache slot, the second would inherit the first's yes/no,
    and which patient wrote the slot would depend on thread completion order. Hashing the whole
    text gives every distinct text its own key.
    """
    text_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    return f"{model}|||{concept}|||sha256:{text_digest}"

def annotate_single_text(
    text: str,
    concept: str,
    model: str = "gpt-4o-mini",
    max_words_per_example: Optional[int] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
    timeout: float = 60.0,
    seed: Optional[int] = None
) -> Tuple[Optional[int], float]:  # Return tuple of (result, api_time)
    """
    Annotate a single text with given concept using LLM.
    Returns (annotation, api_time) where annotation is 1 (present), 0 (absent), or None (failed).
    """
    # Shared builder (see _build_annotation_prompt): the few-shot template teaches the
    # decision style; ANNOTATE_SYSTEM_PROMPT states the policy explicitly so the model answers
    # with one word instead of explaining or guessing. Using the same builder as the batch path
    # keeps requests identical across backends.
    prompt, is_reasoning = _build_annotation_prompt(text, concept, model, max_words_per_example)
    sys_prompt = ANNOTATE_SYSTEM_PROMPT
    # Reasoning models (o-series, gpt-5*) spend the completion budget on hidden reasoning
    # before emitting the visible answer. On a trivial yes/no that wastes latency and cost
    # and can starve the answer entirely (an empty completion, i.e. a failed annotation that
    # is then dropped from the result dict). Request minimal reasoning effort so the one-word
    # verdict is produced promptly and reliably. Non-reasoning models reject this parameter
    # with a 400 error, so it is added only for reasoning model ids.
    extra = {}
    if is_reasoning:
        extra['reasoning_effort'] = os.environ.get('ANNOTATE_REASONING_EFFORT', 'minimal')
    total_api_time = 0.0
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response_text = get_completion(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=10,
                timeout=timeout,
                system=sys_prompt,
                seed=seed,
                **extra
            ).strip().lower()
            total_api_time += time.time() - start_time

            parsed = _parse_annotation_text(response_text)
            if parsed is not None:
                return parsed, total_api_time

        except Exception as e:
            if attempt == max_retries - 1:
                # On the last attempt, re-raise API and network errors instead of returning
                # None, so that _parallel_annotate records the task as failed and retries it in
                # its sequential rescue rounds once the connection has had time to recover.
                # A parse failure is different: the model answered but the answer could not
                # be read, and retrying later will not help, so it falls through to return None.
                print(f"Failed to annotate after {max_retries} attempts: {e}")
                raise
            # Back off between outer attempts (5 s, then 10 s by default, capped at 30 s) so
            # the retries are spread out rather than spent within seconds during an outage.
            # ANNOT_RETRY_WAIT_S sets the base wait; 0 disables it.
            _base = float(os.environ.get("ANNOT_RETRY_WAIT_S", "5"))
            if _base > 0:
                time.sleep(min(_base * (2.0 ** attempt), 30.0))
            continue

    return None, total_api_time

def _parallel_annotate(
    tasks: List[Tuple[str, str]],
    n_workers: int,
    cache: dict,
    cache_path: Optional[str],
    results: Dict[str, Dict[str, int]],
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    **kwargs
) -> None:
    # Bounded-memory synchronous annotation. Submitting an entire (text, concept) matrix as
    # futures at once would hold every prompt and Future object in memory simultaneously and
    # persist nothing until the very end, which both spikes memory and loses all progress on a
    # crash. This path always handles annotators without a Batch API (such as DeepSeek), so
    # tasks are processed in chunks instead: the number of futures in flight is capped, the
    # cache is flushed after each chunk (so an interrupted run resumes), and each chunk is
    # released before the next. Outputs do not depend on chunking, because annotations are
    # cached per (model, concept, text) and are order-independent. ANNOT_CHUNK sets the chunk
    # size (default 4000). Per-item failures are counted, not printed, so a large matrix cannot
    # flood a notebook's output buffer.
    import gc
    _model = str(kwargs.get("model", ""))   # cache keys are model-scoped (see generate_cache_key)
    try:
        chunk = max(int(os.environ.get("ANNOT_CHUNK", "4000")), int(n_workers))
    except (ValueError, TypeError):
        chunk = 4000
    total = len(tasks)
    n_chunks = max(1, (total + chunk - 1) // chunk)
    retry_tasks = []
    n_failed = 0

    for ci, start in enumerate(range(0, total, chunk), 1):
        block = tasks[start:start + chunk]
        desc = progress_desc if n_chunks == 1 else f"{progress_desc} [{ci}/{n_chunks}]"
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {
                executor.submit(annotate_single_text, text=text, concept=concept, **kwargs):
                (text, concept)
                for text, concept in block
            }
            iterator = tqdm(concurrent.futures.as_completed(future_to_task),
                            total=len(block), desc=desc, disable=not show_progress,
                            mininterval=1.0)
            for future in iterator:
                text, concept = future_to_task[future]
                try:
                    annotation, _ = future.result()
                    if annotation is not None:
                        results.setdefault(concept, {})[text] = annotation
                        if cache_path:
                            cache[generate_cache_key(concept, text, _model)] = annotation
                except Exception:
                    retry_tasks.append((text, concept))
                    n_failed += 1
        # Persist progress + free this chunk's futures/prompts before the next one.
        if cache_path:
            save_annotation_cache(cache_path, cache)
        future_to_task = None
        gc.collect()

    # Retry failed tasks sequentially, in rounds, printing one summary line per round rather
    # than one line per item. A network or provider outage usually outlasts the parallel pass,
    # so a single sequential pass straight afterwards would mostly fail again; waiting between
    # rounds lets an outage of a few minutes cost a delay instead of lost annotations. This
    # matters because an unresolved task is later filled as 0 (absent), which is
    # indistinguishable from a real negative downstream. ANNOT_RESCUE_ROUNDS (default 4) and
    # ANNOT_RESCUE_WAIT_S (default 60) set the number of rounds and the wait between them.
    _rounds = max(1, int(os.environ.get("ANNOT_RESCUE_ROUNDS", "4")))
    _wait = float(os.environ.get("ANNOT_RESCUE_WAIT_S", "60"))
    for _round in range(_rounds):
        if not retry_tasks:
            break
        if _round:
            print(f"  [annotate] {len(retry_tasks)} still unresolved; waiting {_wait:.0f}s for the "
                  f"connection before round {_round+1}/{_rounds}...")
            time.sleep(_wait)
        print(f"  [annotate] {len(retry_tasks)} call(s) failed; retrying sequentially "
              f"(round {_round+1}/{_rounds})...")
        next_round = []
        for text, concept in retry_tasks:
            try:
                annotation, _ = annotate_single_text(text=text, concept=concept, **kwargs)
                if annotation is not None:
                    results.setdefault(concept, {})[text] = annotation
                    if cache_path:
                        cache[generate_cache_key(concept, text, _model)] = annotation
                else:
                    next_round.append((text, concept))   # parse failure: carry to the next round
            except Exception:
                next_round.append((text, concept))
        retry_tasks = next_round
        if cache_path:
            save_annotation_cache(cache_path, cache)
    if retry_tasks:
        print(f"  [annotate] {len(retry_tasks)} task(s) still unresolved after "
              f"{_rounds} rescue round(s) (filled downstream and counted as failures).")

# --------------------------------------------------------------------------- #
# OpenAI Batch API backend (bulk, order-free annotation at ~half the cost)
# --------------------------------------------------------------------------- #
# The Batch API runs the same model on the same request bodies as the synchronous path
# (both built by _build_annotation_prompt and ANNOTATE_SYSTEM_PROMPT); the only difference is
# asynchronous, discounted execution. Callers opt in per call with use_batch=True, which suits
# large, one-shot, latency-tolerant annotation matrices. Any request the batch does not
# resolve (partial failure, timeout, expiry) is returned to the caller for synchronous retry,
# so a batch failure never silently drops or weakens a feature.

def _annotation_request_body(
    text: str,
    concept: str,
    model: str,
    seed: Optional[int] = None,
    temperature: float = 0.0,
    max_words_per_example: Optional[int] = None,
) -> dict:
    """Build the /v1/chat/completions request body for one annotation, mirroring the
    synchronous call in annotate_single_text (same prompt, system, and params)."""
    from .llm_api import model_abbrev_to_id
    prompt, is_reasoning = _build_annotation_prompt(text, concept, model, max_words_per_example)
    model_id = model_abbrev_to_id.get(model, model)
    body = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": ANNOTATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    if is_reasoning:
        # Match the synchronous reasoning path: high output floor (billed on tokens used,
        # not the cap) + minimal effort so the one-word verdict is never starved.
        body["max_completion_tokens"] = int(os.environ.get('OPENAI_REASONING_MAX_TOKENS', '64000'))
        body["reasoning_effort"] = os.environ.get('ANNOTATE_REASONING_EFFORT', 'minimal')
    else:
        body["max_completion_tokens"] = 10
        body["temperature"] = temperature
        if seed is not None:
            body["seed"] = seed
    return body


def _parse_batch_answer(record: dict) -> Optional[int]:
    """Extract a 0/1 annotation from one Batch output record (or None if unparseable)."""
    resp = record.get("response") or {}
    status = resp.get("status_code")
    if status not in (None, 200):
        return None
    body = resp.get("body") or {}
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return _parse_annotation_text(content)


def _harvest_batch(client, batch, idx2task: List[Tuple[str, str]], resolved: dict) -> int:
    """Parse a terminal batch's output + error files into ``resolved[idx] = 0/1``.
    Returns the count of newly resolved tasks."""
    n0 = len(resolved)
    for fid in (getattr(batch, "output_file_id", None), getattr(batch, "error_file_id", None)):
        if not fid:
            continue
        try:
            content = client.files.content(fid)
            text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
        except Exception as e:
            print(f"  [batch] could not read result file {fid}: {e}")
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = str(obj.get("custom_id", ""))
            if not cid.startswith("t"):
                continue
            try:
                idx = int(cid[1:])
            except ValueError:
                continue
            if idx in resolved or not (0 <= idx < len(idx2task)):
                continue
            val = _parse_batch_answer(obj)
            if val is not None:
                resolved[idx] = val
    return len(resolved) - n0


def _annotate_via_batch(
    tasks: List[Tuple[str, str]],
    cache: dict,
    cache_path: Optional[str],
    results: Dict[str, Dict[str, int]],
    model: str,
    seed: Optional[int] = None,
    temperature: float = 0.0,
    max_words_per_example: Optional[int] = None,
    show_progress: bool = True,
    progress_desc: str = "Annotating (batch)",
    client=None,
) -> List[Tuple[str, str]]:
    """Resolve as many (text, concept) tasks as possible via the OpenAI Batch API, writing
    hits into ``results`` (and ``cache``). Returns the tasks it could not resolve, for the
    caller to finish synchronously. Batch is OpenAI-only; a non-OpenAI annotator raises so
    the caller falls back cleanly.

    Env knobs: OPENAI_BATCH_MAX_REQUESTS (chunk size, default 30000), OPENAI_BATCH_POLL_INTERVAL
    (seconds, default 20), OPENAI_BATCH_MAX_WAIT (seconds before giving up + syncing, default 86400).
    """
    from .llm_api import get_client, _provider_of, model_abbrev_to_id
    model_id = model_abbrev_to_id.get(str(model), str(model))
    if _provider_of(model_id) != "openai":
        raise RuntimeError(f"Batch annotation is OpenAI-only; {model_id!r} routes elsewhere.")
    if client is None:
        client = get_client()

    chunk_size = max(1, int(os.environ.get("OPENAI_BATCH_MAX_REQUESTS", "30000")))
    poll_interval = max(0.0, float(os.environ.get("OPENAI_BATCH_POLL_INTERVAL", "20")))
    max_wait = float(os.environ.get("OPENAI_BATCH_MAX_WAIT", "86400"))
    _model_key = str(model)
    idx2task = list(tasks)

    # ---- submit chunks (per-chunk try/except: a failed chunk just falls back to sync) ----
    pending: List[str] = []
    for start in range(0, len(idx2task), chunk_size):
        stop = min(start + chunk_size, len(idx2task))
        lines = []
        for i in range(start, stop):
            text, concept = idx2task[i]
            body = _annotation_request_body(text, concept, model, seed=seed,
                                            temperature=temperature,
                                            max_words_per_example=max_words_per_example)
            lines.append(json.dumps({"custom_id": f"t{i}", "method": "POST",
                                     "url": "/v1/chat/completions", "body": body}))
        data = ("\n".join(lines)).encode("utf-8")
        try:
            up = client.files.create(file=("clip_h_annotate.jsonl", data), purpose="batch")
            b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                      completion_window="24h", metadata={"clip_h": progress_desc})
            pending.append(b.id)
            print(f"  [batch] submitted {b.id}: {stop - start} requests "
                  f"(#{start}-{stop - 1} of {len(idx2task)})")
        except Exception as e:
            print(f"  [batch] chunk #{start}-{stop - 1} submit failed ({e}); "
                  f"those tasks will be annotated synchronously.")

    # ---- poll to completion ----
    resolved: dict = {}
    t0 = time.time()
    while pending and (time.time() - t0) < max_wait:
        if poll_interval:
            time.sleep(poll_interval)
        still = []
        for bid in pending:
            try:
                b = client.batches.retrieve(bid)
            except Exception as e:
                print(f"  [batch] retrieve {bid} failed ({e}); will retry.")
                still.append(bid)
                continue
            st = getattr(b, "status", "")
            if st in ("completed", "failed", "expired", "cancelled"):
                n_new = _harvest_batch(client, b, idx2task, resolved)
                print(f"  [batch] {bid} -> {st}: +{n_new} resolved "
                      f"({len(resolved)}/{len(idx2task)} total)")
            else:
                still.append(bid)
        pending = still
        if pending and show_progress:
            print(f"  [batch] waiting on {len(pending)} batch(es); "
                  f"{len(resolved)}/{len(idx2task)} resolved...")

    # ---- timeout: stop billing on stragglers, finish the rest synchronously ----
    if pending:
        print(f"  [batch] max wait {max_wait:.0f}s exceeded; cancelling {len(pending)} "
              f"pending batch(es) and finishing the remainder synchronously.")
        for bid in pending:
            try:
                client.batches.cancel(bid)
            except Exception:
                pass

    # ---- commit resolved annotations to results + cache ----
    for idx, val in resolved.items():
        text, concept = idx2task[idx]
        results.setdefault(concept, {})[text] = val
        if cache_path:
            cache[generate_cache_key(concept, text, _model_key)] = val
    if cache_path and resolved:
        save_annotation_cache(cache_path, cache)   # persist partial progress before any sync retry

    remaining = [idx2task[i] for i in range(len(idx2task)) if i not in resolved]
    if remaining:
        print(f"  [batch] {len(remaining)}/{len(idx2task)} task(s) unresolved by batch; "
              f"handing to the synchronous annotator.")
    return remaining


def annotate(
    tasks: List[Tuple[str, str]],
    cache_path: Optional[str] = None,
    n_workers: int = DEFAULT_N_WORKERS,
    show_progress: bool = True,
    use_batch: bool = False,
    **kwargs
) -> Dict[Tuple[str, str], int]:
    """
    Annotate a list of (text, concept) tasks.
    
    Args:
        tasks: List of (text, concept) tuples to annotate
        cache_path: Path to cache file
        n_workers: Number of workers for parallel processing
        show_progress: Whether to show progress bar
        **kwargs: Additional arguments passed to annotate_single_text
    
    Returns:
        Dictionary mapping (text, concept) to annotation result
    """
    # Load existing cache
    cache = get_annotation_cache(cache_path) if cache_path else {}
    results = {}
    uncached_tasks = []

    # Check cache and prepare uncached tasks (keys are scoped to the annotator model)
    _model = str(kwargs.get("model", ""))
    for text, concept in tasks:
        if concept not in results:
            results[concept] = {}
        cache_key = generate_cache_key(concept, text, _model)
        if cache_key in cache:
            results[concept][text] = cache[cache_key]
        else:
            uncached_tasks.append((text, concept))

    # Print cache statistics
    print(f"Found {len(tasks) - len(uncached_tasks)} cached items; annotating {len(uncached_tasks)} uncached items"
          + (" via Batch API (bulk; sync fallback for any stragglers)" if use_batch and uncached_tasks else ""))

    # Annotate uncached tasks. When use_batch is set, the bulk is resolved through the OpenAI
    # Batch API first; anything the batch cannot resolve (partial failure, timeout, non-OpenAI
    # annotator) falls through to the synchronous parallel path. Both paths send the same
    # requests, so the choice trades latency for cost.
    if uncached_tasks:
        remaining = uncached_tasks
        if use_batch:
            try:
                remaining = _annotate_via_batch(
                    uncached_tasks, cache, cache_path, results,
                    model=kwargs.get("model", ""), seed=kwargs.get("seed"),
                    temperature=kwargs.get("temperature", 0.0),
                    max_words_per_example=kwargs.get("max_words_per_example"),
                    show_progress=show_progress,
                    progress_desc=kwargs.get("progress_desc", "Annotating (batch)"),
                )
            except Exception as e:
                print(f"[batch] unavailable ({e}); using synchronous annotation for all "
                      f"{len(uncached_tasks)} uncached tasks.")
                remaining = uncached_tasks
        if remaining:
            _parallel_annotate(
                tasks=remaining,
                n_workers=n_workers,
                cache=cache,
                cache_path=cache_path,
                results=results,
                show_progress=show_progress,
                **kwargs
            )

    # Save cache if path provided
    if cache_path:
        save_annotation_cache(cache_path, cache)

    return results

def annotate_texts_with_concepts(
    texts: List[str],
    concepts: List[str],
    cache_name: Optional[str] = None,
    progress_desc: str = "Annotating",
    show_progress: bool = True,
    use_batch: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Annotate all texts in a list with all concepts in a list.

    Args:
        use_batch: route the uncached (text, concept) pairs through the OpenAI Batch API
            (about half the cost, asynchronous) with a synchronous fallback for anything it does
            not resolve. Requests are identical to the synchronous path (same model, same
            request), so the option suits large, one-shot, latency-tolerant annotation matrices.
    Returns:
        Dictionary mapping each concept to an array of annotation results, with the texts in the order they were passed in.
    """
    # Create tasks for each text-concept pair
    tasks = [(text, concept) for text in texts for concept in concepts]

    # Use the annotate function to process tasks
    results = annotate(
        tasks=tasks,
        cache_path=os.path.join(CACHE_DIR, f"{cache_name}_hypothesis-eval.json") if cache_name else None,
        n_workers=kwargs.pop('n_workers', DEFAULT_N_WORKERS),
        show_progress=show_progress,
        progress_desc=progress_desc,
        use_batch=use_batch,
        **kwargs
    )
    
    # Reorganize results into arrays per concept. A few annotations can fail permanently
    # (for example under sustained rate limiting); rather than raising KeyError, a missing
    # (concept, text) is treated as 0 (concept absent), consistent with the annotator's
    # "when uncertain, output No" policy, and the number defaulted is reported.
    concept_arrays = {}
    n_missing = 0
    for concept in concepts:
        col = results.get(concept, {})
        vals = []
        for text in texts:
            v = col.get(text)
            if v is None:
                v = 0
                n_missing += 1
            vals.append(v)
        concept_arrays[concept] = np.array(vals)
    if n_missing:
        # Accumulate across every call in a run. A defaulted annotation is recorded as
        # "concept absent", which is indistinguishable downstream from a real negative: it
        # shifts prevalence, effect sizes and gate decisions without leaving a trace in the
        # feature matrix. The runner notebooks read ANNOTATION_FAILURES when writing the run
        # manifest, record the count there, and flag the run's data quality when it is non-zero.
        global ANNOTATION_FAILURES
        ANNOTATION_FAILURES += int(n_missing)
        print(f"Warning: {n_missing} annotation(s) failed after retries; defaulted to 0 (absent). "
              f"Run total: {ANNOTATION_FAILURES}.")

    return concept_arrays
