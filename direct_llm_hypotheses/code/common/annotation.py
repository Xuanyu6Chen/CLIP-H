# Portions adapted from HypotheSAEs v0.2.0 (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See THIRD_PARTY_NOTICES.md at the repository root.
"""Annotation: does a hypothesis hold for a profile? Every answer is 0 or 1.

    build_annotation_matrix   profiles x hypotheses table (Phases 2B, 2C, 2D, 2E, 3A)
    annotate_fidelity         (profile, hypothesis) pairs for fidelity scoring (Phase 2A)
    score_pair                one profile, one hypothesis: the fallback call
    set_reuse_cache_dirs      also read the label caches of other runs (never written to)
    failed_labels             how many labels of each step could not be obtained

One call labels all hypotheses for one profile, 30 hypotheses at a time. A reply is read label by
label, so a cut-off reply still gives the labels it contains; labels that are still missing
after the retries are asked one pair at a time with ``score_pair``, and what is missing after
that is asked again in rescue rounds at the end of the step. A label that cannot be obtained
at all is counted as 0 for this run, reported, and NOT written to the cache, so the next run
asks for it again. The count of each step is saved in ``annotation_failures.json``.

Both caches are JSON files keyed by content. A file is replaced only after the new version is
fully written, and it is written every few profiles, so an interrupted run loses at most the
profiles in flight.
"""
import concurrent.futures
import hashlib
import json
import re
import threading
import time
from pathlib import Path

import numpy as np

import llm_client
from llm_client import Progress, get_completion
from prompts import load_prompt


# The prompts are read from prompts/<_PROMPT_FILE>.txt each time a prompt is built. CLIP-H uses
# annotation.txt; direct_llm_hypotheses keeps one file per task and picks it with use_prompt_file().
# The rubric header is the same for every task; the worked example is the task's own.
_PROMPT_FILE = 'annotation'


def use_prompt_file(name):
    """Read the annotation prompts from ``prompts/<name>.txt`` from now on."""
    global _PROMPT_FILE
    load_prompt(name, 'rubric_header')   # fails here, not at the first call, if the file is missing
    _PROMPT_FILE = name


def _batch_prompt(text, concepts):
    props = "; ".join(f'c{i}: "{c}"' for i, c in enumerate(concepts))
    patient = json.dumps({"id": "p0", "text": text})
    label_template = ",".join(f'"c{i}":0' for i in range(len(concepts)))
    return load_prompt(_PROMPT_FILE, 'labels').format(
        rubric_header=load_prompt(_PROMPT_FILE, 'rubric_header'),
        worked_example=load_prompt(_PROMPT_FILE, 'worked_example'), properties=props,
        patient=patient, n_properties=len(concepts), label_template=label_template)


def _extract_json_object(reply):
    """Return the first decodable JSON object, tolerating fences or brief preamble."""
    if not reply:
        return None
    cleaned = str(reply).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for start, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _binary_value(value):
    if value in (0, 1, "0", "1"):
        return float(int(value))
    return None


def _parse_labels_partial(reply, n_expected):
    """Labels of a 'labels' style reply, ``None`` for each label the reply does not hold."""
    obj = _extract_json_object(reply)
    if obj is None or "labels" not in obj:
        return None
    payload = obj["labels"]
    if isinstance(payload, dict):
        return [_binary_value(payload.get(f"c{i}")) for i in range(n_expected)]
    # A model may answer with a plain array instead of keyed labels. Only a complete array is
    # used: a short one does not say which property was left out.
    if isinstance(payload, list) and len(payload) == n_expected:
        labels = [_binary_value(value) for value in payload]
        return labels if all(value is not None for value in labels) else None
    return None


# --- 'checklist' reply style: one key per property, answered one property at a time ----------
# On 8 hand-labelled patients the GPT annotator agreed with the gold labels on 0.945 of the
# cells with this style and on 0.914 with the 'labels' style (which misread flag lines and OR
# lists), so the GPT arm uses 'checklist'. Both styles key every label (c0, c1, ...), so a
# reply that is cut off still gives the labels it contains, each for the right property.
def _checklist_prompt(text, concepts):
    props = "\n".join(f'c{i}: "{c}"' for i, c in enumerate(concepts))
    return load_prompt(_PROMPT_FILE, 'checklist').format(
        rubric_header=load_prompt(_PROMPT_FILE, 'rubric_header'),
        checklist_rules=load_prompt(_PROMPT_FILE, 'checklist_rules'), properties=props,
        profile=text, last_index=len(concepts) - 1)


def _parse_checklist_partial(reply, n_expected):
    obj = _extract_json_object(reply)
    if obj is None:
        return None
    return [_binary_value(obj.get(f"c{i}")) for i in range(n_expected)]

# Hypotheses per call. DeepSeek with thinking disabled handles this width reliably, while one
# reply for 148 hypotheses commonly leaves out one or two labels. The profile is sent again with
# every chunk, but that input is mostly a prefix-cache hit and the reply stays small.
BATCH_MAX_CONCEPTS = 30
# The cache is written after the first finished patient and then every few patients: often
# enough that an interruption loses little, without rewriting a file of many megabytes each time.
CACHE_CHECKPOINT_PATIENTS = 5
FIDELITY_CHECKPOINT_TEXTS = 25
_DIAGNOSTIC_LIMIT = 12
_diagnostic_count = 0
_diagnostic_lock = threading.Lock()


def _report_retry_problem(message):
    """Print a bounded, profile-free diagnostic instead of silently swallowing errors."""
    global _diagnostic_count
    with _diagnostic_lock:
        if _diagnostic_count >= _DIAGNOSTIC_LIMIT:
            return
        _diagnostic_count += 1
        number = _diagnostic_count
    print(f"  [annotation] {message} (note {number} of at most {_DIAGNOSTIC_LIMIT})", flush=True)


def _chunk_call(text, concepts, model, style):
    """The call that labels one chunk of concepts for one profile, and the parser of its reply."""
    if style == "checklist":
        prompt, parse = _checklist_prompt(text, concepts), _parse_checklist_partial
    else:
        prompt, parse = _batch_prompt(text, concepts), _parse_labels_partial
    return dict(prompt=prompt, model=model, temperature=0.0, max_tokens=96 + 10 * len(concepts)), parse


def _annotate_chunk(text, concepts, model, max_retries, timeout, style="labels"):
    """Annotate a chunk, merging any safely keyed labels across retries."""
    call, parse = _chunk_call(text, concepts, model, style)
    merged = [None] * len(concepts)
    for attempt in range(max_retries):
        try:
            reply = get_completion(timeout=timeout, attempt=attempt, **call)
        except Exception as exc:
            _report_retry_problem(f"API call raised {type(exc).__name__}; retrying")
            continue
        parsed = parse(reply, len(concepts))
        if parsed is None:
            _report_retry_problem(
                f"unparseable reply ({len(str(reply))} characters); retrying")
            continue
        for i, value in enumerate(parsed):
            if value is not None:
                merged[i] = value
        if all(value is not None for value in merged):
            break
    return merged


def _annotate_patient_partial(text, concepts, model, max_retries, timeout, style="labels"):
    """One label per concept, ``None`` where the reply gave no valid label.

    The concepts are sent ``BATCH_MAX_CONCEPTS`` at a time; the labels of the chunks that
    succeeded are kept when another chunk fails.
    """
    out = []
    for i in range(0, len(concepts), BATCH_MAX_CONCEPTS):
        chunk = concepts[i:i + BATCH_MAX_CONCEPTS]
        part = _annotate_chunk(text, chunk, model, max_retries, timeout, style=style)
        out.extend(part)
    return out


# --- OpenAI Batch API -----------------------------------------------------------------------
# ``build_annotation_matrix(batch_api=True)`` sends the first call of every (profile, chunk of
# hypotheses) through the OpenAI Batch API. The requests are the same ones that are otherwise
# sent live: all hypotheses for one profile in one request. The batch answers them within 24
# hours at half price and its replies go into the call cache; annotation then runs as usual and
# finds them there. Labels the batch did not deliver are asked live. Needs an OpenAI endpoint
# and an installed call cache.
BATCH_MIN_CALLS = 100   # fewer calls than this are not worth waiting in a batch queue


def _prefetch_by_batch(profile_concepts, model, style):
    """Answer the first-round calls for ``[(text, concepts), ...]`` through the Batch API."""
    calls = [_chunk_call(text, concepts[i:i + BATCH_MAX_CONCEPTS], model, style)[0]
             for text, concepts in profile_concepts
             for i in range(0, len(concepts), BATCH_MAX_CONCEPTS)]
    if not calls:
        return
    problem = llm_client.batch_available()
    if problem:
        print(f"[batch] not used: {problem}; calling live")
    elif len(calls) < BATCH_MIN_CALLS:
        print(f"[batch] not used: only {len(calls)} calls; calling live")
    else:
        llm_client.batch_complete(calls)


# --- labels that could not be obtained ------------------------------------------------------
# A label fails when every call for it fails. At the end of a step the labels that failed get
# more chances: wait, then ask again. A call that never got a reply (the connection was down)
# is sent again; a call whose reply could not be read is answered from the call cache and stays
# failed. A round that recovers nothing ends the rescue.
RESCUE_ROUNDS = 3
RESCUE_WAIT_SECONDS = 60.0
# A label that is still missing counts as 0 in the returned labels, which nothing downstream
# can tell from a real "no". Each step therefore writes its count to this file next to its
# cache: {step: {"failed": n, "of": labels}}. A step that runs again replaces its entry.
FAILURES_FILE = "annotation_failures.json"


def _rescue(step, count_missing, ask_again):
    """Rescue rounds for ``step``. ``count_missing()`` is the number of labels still missing and
    ``ask_again()`` asks once more for all of them. Returns the number missing at the end."""
    n_failed = count_missing()
    for round_no in range(1, RESCUE_ROUNDS + 1):
        if not n_failed:
            break
        print(f"[{step}] {n_failed} labels could not be obtained; waiting {RESCUE_WAIT_SECONDS:.0f} s, "
              f"then asking again (rescue round {round_no} of {RESCUE_ROUNDS})", flush=True)
        time.sleep(RESCUE_WAIT_SECONDS)
        ask_again()
        still = count_missing()
        print(f"[{step}] rescue round {round_no}: {n_failed - still} labels recovered, {still} still missing",
              flush=True)
        if still == n_failed:
            break
        n_failed = still
    return n_failed


def _note_failures(cache_path, step, n_failed, n_labels):
    """Write how many of the ``n_labels`` labels of ``step`` could not be obtained."""
    path = Path(cache_path).parent / FAILURES_FILE
    record = _load_cache(path)
    record[step] = {"failed": int(n_failed), "of": int(n_labels)}
    _save_cache_atomic(path, record)


def failed_labels(cache_dir):
    """``{step: {"failed": n, "of": labels}}`` for the annotation steps that wrote to ``cache_dir``."""
    return _load_cache(Path(cache_dir) / FAILURES_FILE)


# --- caches ---------------------------------------------------------------------------------
def _load_cache(cache_path):
    """Read a JSON cache. A file that cannot be read stops the run; it is never deleted."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Cache file {cache_path} is not valid JSON ({exc}). Move it away "
                           "or repair it; it was left untouched.") from exc


# Labels of other runs, read once by ``set_reuse_cache_dirs``. A label is the answer of one
# model to one hypothesis on one text, and that is exactly what its key says, so it holds in any
# run. A borrowed label that is used is copied into the run's own cache. The same pair can be in
# two cache files with different answers (it was asked twice), so a cache first borrows from the
# file of the same name and only then from the others.
_BORROWED = {"matrix": {}, "fidelity": {}}   # "matrix": file name -> labels; "fidelity": labels


def set_reuse_cache_dirs(dirs, verbose=True):
    """Make the label caches in ``dirs`` (``caches/`` folders of other runs) available for reading."""
    matrix, fidelity = {}, {}
    for folder in dirs:
        for path in sorted(Path(folder).glob("*_annotation_cache.json")):
            if path.name.startswith("fidelity"):
                fidelity.update(_load_cache(path))
            else:
                matrix.setdefault(path.name, {}).update(_load_cache(path))
    _BORROWED.update(matrix=matrix, fidelity=fidelity)
    if verbose and dirs:
        print(f"Labels readable from other runs: {sum(len(v) for v in matrix.values())} annotation "
              f"in {len(matrix)} files, {len(fidelity)} fidelity")


def _borrow(cache, keys, cache_name):
    """Copy into ``cache`` the labels for ``keys`` that other runs have. Returns how many."""
    files = _BORROWED["matrix"]
    sources = ([files[cache_name]] if cache_name in files else []) + \
              [labels for name, labels in files.items() if name != cache_name]
    n = 0
    for k in keys:
        if k not in cache:
            for labels in sources:
                if k in labels:
                    cache[k] = labels[k]
                    n += 1
                    break
    return n


def _save_cache_atomic(cache_path, cache):
    """Replace a cache only after its complete JSON payload has reached disk."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache))
    tmp_path.replace(cache_path)


def _matrix_key(mode, model, concept, text):
    """Key of the matrix caches. The model is part of it, so another annotator never reuses these labels."""
    return hashlib.sha1(f"{mode}|||{model}|||{concept}|||{text}".encode("utf-8")).hexdigest()


def fidelity_cache_key(concept, text, model):
    """Key of the fidelity cache: the model, the hypothesis and a hash of the whole text."""
    return f"{model}|||{concept}|||sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _truncate(text, max_words):
    if not max_words:
        return text
    words = str(text).split()
    return ' '.join(words[:max_words]) if len(words) > max_words else text


# --- one profile, one hypothesis ------------------------------------------------------------
def score_pair(text, concept, model):
    """One Yes/No call for one (profile, hypothesis) pair. Returns 1.0, 0.0, or None after 3 failed tries."""
    prompt = (load_prompt(_PROMPT_FILE, 'pair').format(hypothesis=concept, text=text)
              + load_prompt(_PROMPT_FILE, 'pair_suffix'))
    for attempt in range(3):
        try:
            reply = get_completion(prompt=prompt, model=model, temperature=0.0, max_tokens=1000,
                                   timeout=30.0, system=load_prompt(_PROMPT_FILE, 'pair_system'),
                                   attempt=attempt).strip().lower()
            match = re.match(r'(yes|no)\b', reply)  # first word only
            if match:
                return 1.0 if match.group(1) == 'yes' else 0.0
        except Exception:
            continue
    return None


def _labels_for_profile(text, concepts, model, max_retries, timeout, style):
    """All labels for one profile: batched calls first, then one call per missing label.

    Returns (labels, n_fallback): a label is None only when the pair call failed too.
    """
    labels = _annotate_patient_partial(text, concepts, model, max_retries, timeout, style=style)
    n_fallback = 0
    for j, value in enumerate(labels):
        if value is None:
            n_fallback += 1
            labels[j] = score_pair(text, concepts[j], model)
    return labels, n_fallback


# --- profiles x hypotheses ------------------------------------------------------------------
def build_annotation_matrix(texts, concepts, mode, model, cache_path, n_workers=10, max_words=None,
                            fill_failures=0.0, *, style, max_retries=3, timeout=60.0,
                            batch_api=False, label=None):
    """Matrix (n_profiles, n_hypotheses) of 0/1 labels, cached by content in ``cache_path``.

    ``style`` is the reply format: 'checklist' or 'labels'. ``max_words=None`` annotates the
    full profile. A label that could not be obtained, also not in the rescue rounds, is
    ``fill_failures`` in the returned matrix and is not cached. ``batch_api=True`` delivers the
    first round of calls through the OpenAI Batch API (half price, up to 24 hours); see
    ``_prefetch_by_batch``. ``label`` names the step in the printed progress lines and in
    ``annotation_failures.json``.
    """
    assert mode == "binary", mode
    texts = [_truncate(t, max_words) for t in texts]
    concepts = list(concepts)
    cache = _load_cache(cache_path)
    keys = [[_matrix_key(mode, model, c, t) for c in concepts] for t in texts]
    borrowed = _borrow(cache, [k for row in keys for k in row], Path(cache_path).name)
    todo = [ti for ti in range(len(texts)) if any(k not in cache for k in keys[ti])]
    n_missing = sum(sum(1 for k in keys[ti] if k not in cache) for ti in todo)
    label = label or "annotation"
    print(f"[{label}] {len(texts)} profiles x {len(concepts)} hypotheses: {len(todo)} profiles to "
          f"annotate, {len(concepts) * len(texts) - n_missing} labels already in {Path(cache_path).name}")

    def work(ti):
        missing = [ci for ci in range(len(concepts)) if keys[ti][ci] not in cache]
        labels, n_fallback = _labels_for_profile(
            texts[ti], [concepts[ci] for ci in missing], model, max_retries, timeout, style)
        return ti, missing, labels, n_fallback

    if batch_api:
        _prefetch_by_batch([(texts[ti], [c for c, k in zip(concepts, keys[ti]) if k not in cache])
                            for ti in todo], model, style)
    done = n_fallback_profiles = n_fallback_cells = 0
    progress = Progress(label, len(todo))   # before the workers start, so that their first requests are counted
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(work, ti) for ti in todo]
        for f in concurrent.futures.as_completed(futs):
            ti, missing, labels, n_fallback = f.result()
            progress.step()
            n_fallback_profiles += bool(n_fallback)
            n_fallback_cells += n_fallback
            for ci, value in zip(missing, labels):
                if value is not None:
                    cache[keys[ti][ci]] = value
            done += 1
            if done == 1 or done % CACHE_CHECKPOINT_PATIENTS == 0:
                _save_cache_atomic(cache_path, cache)
        progress.close()
    if todo or borrowed:
        _save_cache_atomic(cache_path, cache)

    def ask_again():
        failed = [ti for ti in todo if any(k not in cache for k in keys[ti])]
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            for ti, missing, labels, _ in ex.map(work, failed):
                for ci, value in zip(missing, labels):
                    if value is not None:
                        cache[keys[ti][ci]] = value
        _save_cache_atomic(cache_path, cache)

    n_failed = _rescue(label, lambda: sum(k not in cache for ti in todo for k in keys[ti]), ask_again)
    _note_failures(cache_path, label, n_failed, len(texts) * len(concepts))
    if borrowed:
        print(f"[{label}] {borrowed} labels taken from other runs' caches")
    if n_fallback_profiles:
        print(f"NOTE: {n_fallback_profiles} patients needed partial fallback: "
              f"{n_fallback_cells} cells asked one pair at a time.")
    if n_failed:
        print(f"WARNING: {n_failed} cells could not be annotated; counted as {fill_failures} "
              f"in this run and not cached, so the next run asks again. Count saved in {FAILURES_FILE}.")
    M = np.array([[cache.get(keys[ti][ci], fill_failures) for ci in range(len(concepts))]
                  for ti in range(len(texts))], dtype=float)
    return M.astype(int)


# --- (profile, hypothesis) pairs for fidelity scoring ---------------------------------------
def annotate_fidelity(tasks, cache_path=None, n_workers=10, show_progress=True, model=None,
                      progress_desc=None, *, style, max_retries=3, timeout=60.0):
    """Label (text, concept) pairs; one call per text carries all of its concepts.

    Returns ``{concept: {text: 0 or 1}}``. A pair that could not be labelled, also not in the
    rescue rounds, is 0 in this run and is not cached.
    """
    cache = _load_cache(cache_path) if cache_path else {}
    results, by_text = {}, {}
    n_borrowed = 0
    for text, concept in tasks:
        results.setdefault(concept, {})
        key = fidelity_cache_key(concept, text, model)
        if key not in cache and key in _BORROWED["fidelity"]:
            cache[key] = _BORROWED["fidelity"][key]
            n_borrowed += 1
        if key in cache:
            results[concept][text] = cache[key]
        elif concept not in by_text.setdefault(text, []):
            by_text[text].append(concept)
    by_text = {t: cs for t, cs in by_text.items() if cs}
    n_cells = sum(len(v) for v in by_text.values())
    print(f"[{progress_desc or 'fidelity scoring'}] {len(tasks)} (profile, hypothesis) pairs: {n_cells} to "
          f"label in {len(by_text)} calls, {len(tasks) - n_cells} already known")
    done = n_fallback = 0
    progress = Progress(progress_desc or "fidelity scoring", len(by_text) if show_progress else 0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_labels_for_profile, t, cs, model, max_retries, timeout, style): (t, cs)
                for t, cs in by_text.items()}
        for f in concurrent.futures.as_completed(futs):
            t, cs = futs[f]
            labels, n_fb = f.result()
            progress.step()
            n_fallback += n_fb
            for c, value in zip(cs, labels):
                if value is None:
                    results[c][t] = 0
                else:
                    results[c][t] = cache[fidelity_cache_key(c, t, model)] = int(value)
            done += 1
            if cache_path and (done == 1 or done % FIDELITY_CHECKPOINT_TEXTS == 0):
                _save_cache_atomic(cache_path, cache)
        progress.close()
    if cache_path and (by_text or n_borrowed):
        _save_cache_atomic(cache_path, cache)
    step = (progress_desc or "fidelity scoring").split(",")[0]   # the name without the counts that follow it

    def missing_pairs():
        return {t: [c for c in cs if fidelity_cache_key(c, t, model) not in cache] for t, cs in by_text.items()}

    def ask_again():
        failed = {t: cs for t, cs in missing_pairs().items() if cs}
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as ex:
            asked = ex.map(lambda t: _labels_for_profile(t, failed[t], model, max_retries, timeout, style), failed)
            for t, (labels, _) in zip(failed, asked):
                for c, value in zip(failed[t], labels):
                    if value is not None:
                        results[c][t] = cache[fidelity_cache_key(c, t, model)] = int(value)
        if cache_path:
            _save_cache_atomic(cache_path, cache)

    n_failed = _rescue(step, lambda: sum(len(cs) for cs in missing_pairs().values()), ask_again)
    if cache_path:
        _note_failures(cache_path, step, n_failed, len(tasks))
    if n_borrowed:
        print(f"[{progress_desc or 'fidelity scoring'}] {n_borrowed} labels taken from other runs' caches")
    if n_fallback:
        print(f"NOTE: {n_fallback} pairs asked one at a time")
    if n_failed:
        print(f"WARNING: {n_failed} pairs could not be annotated; counted as 0 in this run and "
              "not cached, so the next run asks again.")
    return results
