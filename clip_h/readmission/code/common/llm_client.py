"""The LLM client: every API call of CLIP-H goes through ``get_completion``.

    get_completion         one chat completion with retries (OpenAI and DeepSeek endpoints)
    install_call_cache     keep every reply on disk, so a call that was paid for is never repeated
    batch_complete         answer many calls through the OpenAI Batch API (half price) into that cache
    Progress               printed progress lines for a step that makes many calls
    FatalAPIError          raised when no retry can help (bad key, no balance): the run stops
    set_default_timeout    request timeout used when a caller gives none
    install_usage_meter    log the billed tokens of every call to a JSONL file
    usage_report           cost table from that log

Environment: OPENAI_KEY_SAE (API key), OPENAI_BASE_URL_SAE (endpoint; default OpenAI).
"""
import hashlib
import json
import os
import random
import re
import threading
import time

import httpx
import openai

# Short model names -> the dated model the API is asked for.
model_abbrev_to_id = {
    'gpt-5-nano': 'gpt-5-nano',
    'gpt5-nano': 'gpt-5-nano',
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt-5': 'gpt-5-2025-08-07',

    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
}

DEFAULT_TIMEOUT = 60.0


def set_default_timeout(seconds):
    """Request timeout (seconds) for calls that do not pass their own.

    It limits a whole request of a model that is not streamed. DeepSeek calls are streamed and
    have no total limit: the connection is dropped after 90 s without a new chunk (``_call_once``).
    """
    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = float(seconds)


# --- Billed-token meter -----------------------------------------------------------------
# The meter records the token counts the provider reports for each request. An estimate from
# prompt sizes comes out several times too low: reasoning tokens are billed as output but never
# appear in the reply, and every retried request is billed in full.
# USD per 1M tokens: (cached input, uncached input, output). The DeepSeek rows are the off-peak
# prices; DeepSeek bills twice as much in its peak hours (01:00-04:00 and 06:00-10:00 UTC,
# Monday to Friday), so a run made then costs up to twice what usage_report prints.
PRICES = {
    "deepseek-v4-pro":   (0.022,    0.66,  1.98),
    "deepseek-v4-flash": (0.003,    0.15,  0.60),
    "gpt-4.1":           (0.50,     2.00,  8.00),
    "gpt-4.1-mini":      (0.10,     0.40,  1.60),
    "gpt-5":             (0.125,    1.25, 10.00),
}
_USAGE_LOG = None
_USAGE_LOCK = threading.Lock()

# DeepSeek's own defaults are thinking=enabled + reasoning_effort=high, which spends about
# 2,100 output tokens on one annotation call and ignores the temperature. Every DeepSeek call
# is therefore sent with thinking off. A model name that ends in THINK_SUFFIX
# ("deepseek-v4-pro-think") keeps the thinking pass for the role it is assigned to.
THINK_SUFFIX = "-think"
DEEPSEEK_NO_THINK_BODY = {"thinking": {"type": "disabled"}}
DEEPSEEK_THINK_BODY = {"thinking": {"type": "enabled"}}

# OpenAI reasoning models (gpt-5, o-series) reason inside max_completion_tokens before they write
# the reply, so a limit sized for the reply is raised to REASONING_MIN_TOKENS (a limit, not a spend).
GPT5_REASONING_EFFORT = "medium"   # "minimal" | "low" | "medium" | "high"
REASONING_MIN_TOKENS = 16000

# The annotator tier. Its callers read a reply that was cut at the token limit label by label,
# so its calls keep their tight limit (see ``get_completion``).
_ANNOTATOR_RE = re.compile(r"deepseek.*flash")


def install_usage_meter(path, verbose=True):
    """Append one JSONL row per billed API call to ``path``. Call before a phase."""
    global _USAGE_LOG
    _USAGE_LOG = str(path)
    if verbose:
        print(f"Usage meter -> {_USAGE_LOG} (report with llm_client.usage_report)")


def _record(model_id, usage, attempt, failed=False, served=None, batch=False):
    """One usage row. ``served`` is the model name the API answered with; ``batch`` marks a
    Batch API reply, which is billed at half price."""
    if _USAGE_LOG is None:
        return
    d = getattr(usage, "prompt_tokens_details", None)
    row = {
        "model": model_id, "attempt": attempt, "failed": failed,
        "in": getattr(usage, "prompt_tokens", 0) or 0,
        "cached_in": (getattr(d, "cached_tokens", 0) or 0) if d else 0,
        "out": getattr(usage, "completion_tokens", 0) or 0,
    }
    if served:
        row["served"] = served
    if batch:
        row["batch"] = True
    try:
        with _USAGE_LOCK, open(_USAGE_LOG, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except OSError:
        pass  # metering must never break a run


def usage_report(path, fx=7.15):
    """Print billed tokens and cost per model from a usage log. Returns the total in USD.

    ``retry x`` is requests per usable reply (1.00 = no request had to be repeated). The cost
    uses ``PRICES``; ``fx`` is the CNY per USD rate of the CNY column.
    """
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []
    if not rows:
        print("no usage recorded yet")
        return 0.0
    total = 0.0
    print(f"{'model':<20}{'calls':>8}{'retry x':>9}{'in (M)':>10}{'cached%':>9}"
          f"{'out (M)':>10}{'USD':>9}{'CNY':>9}")
    for m in sorted({r["model"] for r in rows}):
        rs = [r for r in rows if r["model"] == m]
        hit, tin, out = (sum(r["cached_in"] for r in rs), sum(r["in"] for r in rs),
                         sum(r["out"] for r in rs))
        # model ids carry date suffixes (gpt-4.1-mini-2025-04-14): match the longest
        # price key that prefixes the id, so a dated snapshot is not silently mispriced.
        key = max((k for k in PRICES if m.startswith(k)), key=len, default=None)
        if key is None:
            print(f"  [warn] no price for {m}; excluded from the total")
            continue
        p = PRICES[key]
        usd = sum((r["cached_in"] * p[0] + (r["in"] - r["cached_in"]) * p[1] + r["out"] * p[2])
                  * (0.5 if r.get("batch") else 1.0) for r in rs) / 1e6
        total += usd
        n_ok = sum(1 for r in rs if not r["failed"]) or 1
        print(f"{m:<20}{len(rs):>8}{len(rs)/n_ok:>9.2f}{tin/1e6:>10.2f}"
              f"{100*hit/max(tin,1):>8.0f}%{out/1e6:>10.2f}{usd:>9.2f}{usd*fx:>9.2f}")
    print(f"{'TOTAL':<20}{'':>46}{total:>9.2f}{total*fx:>9.2f}")
    return total


# --- Call cache -----------------------------------------------------------------------------
# One line per finished call: {"key": ..., "model": ..., "reply": ...}, plus "served" when
# the API answered with another model name than the one asked for. The file is only ever
# appended to, so nothing in it is overwritten and a run that is interrupted keeps every call
# that had finished. The key is a hash of the model, the system and user prompt, the sampling
# settings and the caller's attempt number.
_CALL_CACHE = None
_CALL_CACHE_PATH = None
_CALL_CACHE_LOCK = threading.Lock()
CALL_CACHE_STATS = {"hits": 0, "new": 0}


def _read_call_cache(path):
    entries = {}
    if not os.path.exists(path):
        return entries
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
                entries[row["key"]] = row["reply"]
            except (ValueError, KeyError, TypeError):
                continue  # a line cut off by an interrupted write: that one call is asked again
    return entries


def install_call_cache(path, reuse=(), verbose=True):
    """Answer repeated calls from ``path`` and append every new reply to it.

    ``reuse`` lists call-cache files of other runs to read from (never written to). A reply is
    reused only for the same model, prompts and settings. ``path=None`` switches the cache off.
    """
    global _CALL_CACHE, _CALL_CACHE_PATH
    if path is None:
        _CALL_CACHE = _CALL_CACHE_PATH = None
        return
    entries = {}
    for other in reuse:
        entries.update(_read_call_cache(str(other)))
    n_reused = len(entries)
    entries.update(_read_call_cache(str(path)))
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    _CALL_CACHE, _CALL_CACHE_PATH = entries, str(path)
    CALL_CACHE_STATS.update(hits=0, new=0)
    if verbose:
        print(f"Call cache -> {_CALL_CACHE_PATH} ({len(entries)} replies on file"
              + (f", {n_reused} from reused caches" if reuse else "") + ")")


def _call_key(model_id, system, prompt, kwargs, attempt):
    payload = json.dumps([model_id, system, prompt, sorted((k, repr(v)) for k, v in kwargs.items()),
                          attempt], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _call_cache_put(key, model_id, reply, served=None):
    row = {"key": key, "model": model_id, "reply": reply}
    if served and served != model_id:
        row["served"] = served
    with _CALL_CACHE_LOCK:
        _CALL_CACHE[key] = reply
        CALL_CACHE_STATS["new"] += 1
        with open(_CALL_CACHE_PATH, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- Progress and warnings ------------------------------------------------------------------
# Counters since the start of the session. Progress lines report how they moved during a step.
CALL_STATS = {"requests": 0, "retries": 0, "failed": 0}
_STATS_LOCK = threading.Lock()
_WARN_COUNTS = {}


def _count(name):
    with _STATS_LOCK:
        CALL_STATS[name] += 1


def _warn(kind, text):
    """Print a call problem: the first three of each kind, then every 50th, so that an outage
    hitting twenty workers at once does not flood the output."""
    with _STATS_LOCK:
        n = _WARN_COUNTS[kind] = _WARN_COUNTS.get(kind, 0) + 1
    if n <= 3 or n % 50 == 0:
        print(f"  [llm] {text}" + (f" ({n} so far of this kind)" if n > 3 else ""), flush=True)


def _clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


class Progress:
    """Printed progress for a step of ``total`` items, one line every ``every`` seconds
    (``total=0`` prints nothing).

    Plain ``print`` lines, so they show in any notebook front end and stay in the saved output:
        [Phase 2E annotation] 320 of 1500 (21%) | 1:12 elapsed, about 4:30 left | requests 960, from cache 0, retries 2, failed 0
    """

    def __init__(self, label, total, every=15.0):
        self.label, self.total, self.every = label, int(total), every
        self.done = 0
        self.start = self.last = time.time()
        self.base = dict(CALL_STATS, cache=CALL_CACHE_STATS["hits"])
        self.lock = threading.Lock()
        if self.total:
            print(f"[{self.label}] starting: {self.total} to do", flush=True)

    def _line(self):
        elapsed = time.time() - self.start
        left = f", about {_clock(elapsed / self.done * (self.total - self.done))} left" \
            if 0 < self.done < self.total else ""
        moved = {k: CALL_STATS[k] - self.base[k] for k in CALL_STATS}
        cached = CALL_CACHE_STATS["hits"] - self.base["cache"]
        return (f"[{self.label}] {self.done} of {self.total} ({100 * self.done // max(self.total, 1)}%) | "
                f"{_clock(elapsed)} elapsed{left} | requests {moved['requests']}, from cache {cached}, "
                f"retries {moved['retries']}, failed {moved['failed']}")

    def step(self, n=1):
        with self.lock:
            self.done += n
            if self.total and (self.done == 1 or time.time() - self.last >= self.every):
                self.last = time.time()
                print(self._line(), flush=True)

    def close(self):
        if self.total:
            print(self._line() + " | done", flush=True)


_CLIENTS = {}
_CLIENTS_LOCK = threading.Lock()


def get_client():
    """OpenAI client built from the OPENAI_KEY_SAE / OPENAI_BASE_URL_SAE env vars.

    One client per (key, base URL) is kept and shared by all worker threads: every client owns a
    connection pool, so building one per call leaves a socket open per call.
    """
    key = (os.environ["OPENAI_KEY_SAE"],
           os.environ.get("OPENAI_BASE_URL_SAE", "https://api.openai.com/v1"))
    with _CLIENTS_LOCK:
        client = _CLIENTS.get(key)
        if client is None:
            client = openai.OpenAI(api_key=key[0], base_url=key[1])
            _CLIENTS[key] = client
        return client


# --- Errors ---------------------------------------------------------------------------------
class FatalAPIError(BaseException):
    """The API refuses every call (bad key, no balance, unknown model, malformed request).

    Asking again cannot help, so the run has to stop. It derives from ``BaseException`` so that
    the ``except Exception`` blocks of the callers, which turn a failed call into a missing
    label or a safe default, do not swallow it.
    """


class ReplyBlockedError(RuntimeError):
    """The provider's content filter withheld the reply. Asking again gives the same result."""


class _ServerBusy(Exception):
    """The provider stopped the reply itself (DeepSeek: insufficient_system_resource, aborted)."""


# After a fatal error every call fails at once, without contacting the API, for this long.
# Calls that are already queued in worker threads then stop within seconds; a run started
# after the cause is fixed (key replaced, balance topped up) works again.
FATAL_HOLD_SECONDS = 60.0
_FATAL = None   # (time, message)
# 400 codes that are about one prompt, not about the account or the request format.
_ONE_PROMPT_CODES = ("context_length_exceeded", "content_filter", "content_policy_violation")
_BUSY_FINISH = ("insufficient_system_resource", "aborted")
_SERVED_NOTED = set()


def _fatal_reason(exc):
    """Why no retry can make this call succeed, or None when a retry can."""
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", None) or "")
    text = str(exc)[:300]
    if status == 401:
        return f"the API key was rejected (401): {text}"
    if status == 402:
        return f"the account has no balance left (402): {text}"
    if status == 403:
        return f"the key is not allowed to make this call (403): {text}"
    if status == 404:
        return f"the model or endpoint does not exist (404): {text}"
    if status == 429 and (code in ("insufficient_quota", "billing_not_active")
                          or "exceeded your current quota" in text):
        return f"the account's credit or spend limit is used up (429 {code}): {text}"
    if status in (400, 422) and code not in _ONE_PROMPT_CODES:
        return f"the API rejected the request as malformed ({status}): {text}"
    return None


def _raise_if_fatal():
    if _FATAL is not None and time.time() - _FATAL[0] < FATAL_HOLD_SECONDS:
        raise FatalAPIError(_FATAL[1])


def _retry_after(exc):
    """Seconds the provider asks to wait (``Retry-After`` header), or None."""
    try:
        return float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None


def _note_served(model_id, served):
    """Say once when the API answers with another model than the one asked for."""
    if not served or served == model_id or (model_id, served) in _SERVED_NOTED:
        return
    _SERVED_NOTED.add((model_id, served))
    if not served.startswith(model_id):   # 'gpt-5' answered by 'gpt-5-2025-08-07' is the alias, not a swap
        print(f"NOTE: asked for model '{model_id}', the API answered with '{served}'.")


# --- Request settings -----------------------------------------------------------------------
def _thinks(model_id):
    """True for a DeepSeek model that keeps its thinking pass (the name ends in THINK_SUFFIX)."""
    return "deepseek" in model_id and model_id.endswith(THINK_SUFFIX)


def _api_model(model_id):
    """The model name the API is asked for: ``model_id`` without THINK_SUFFIX."""
    return model_id[:-len(THINK_SUFFIX)] if _thinks(model_id) else model_id


def _keeps_limit(model_id):
    return bool(_ANNOTATOR_RE.search(model_id)) and not _thinks(model_id)


def _shape_request(model_id, kwargs):
    """The settings sent to the API for a caller's ``kwargs`` (the same for live and batch calls)."""
    is_reasoning = model_id.startswith("o") or model_id.startswith("gpt-5")
    # Only OpenAI reasoning models take max_completion_tokens; DeepSeek ignores that name,
    # which would leave its reasoning without a cap.
    api_kwargs = ({("max_completion_tokens" if k == "max_tokens" else k): v for k, v in kwargs.items()}
                  if is_reasoning else dict(kwargs))
    if "deepseek" in model_id:
        if _thinks(model_id):
            # A thinking model reasons inside max_tokens, so a cap sized for the reply is used
            # up before any reply is written. The cap is a limit, not a spend.
            api_kwargs["extra_body"] = {**api_kwargs.get("extra_body", {}), **DEEPSEEK_THINK_BODY}
            api_kwargs["max_tokens"] = max(api_kwargs.get("max_tokens", 0) or 0, 16000)
        else:
            api_kwargs["extra_body"] = {**api_kwargs.get("extra_body", {}), **DEEPSEEK_NO_THINK_BODY}
    if is_reasoning:
        api_kwargs.pop("temperature", None)
        api_kwargs.pop("top_p", None)
        api_kwargs["max_completion_tokens"] = max(api_kwargs.get("max_completion_tokens", 0) or 0, REASONING_MIN_TOKENS)
        api_kwargs.setdefault("reasoning_effort", GPT5_REASONING_EFFORT if model_id.startswith("gpt-5") else "low")
    elif not _thinks(model_id):
        # A model that accepts a temperature runs at 0 unless the caller sets one.
        api_kwargs.setdefault("temperature", 0.0)
    return api_kwargs


def _messages(system, prompt):
    return ([{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            if system is not None else [{"role": "user", "content": prompt}])


def _budget(api_kwargs):
    return int(api_kwargs.get("max_tokens", api_kwargs.get("max_completion_tokens", 0)) or 0)


def _call_once(client, model_id, messages, api_kwargs, timeout):
    """One request. Returns (reply text, usage, finish_reason, model that answered)."""
    if "deepseek" in model_id:
        # DeepSeek calls are streamed: a thinking model is silent for minutes otherwise, and
        # a silent connection gets dropped on the way. read=90 is the longest gap allowed
        # between two chunks, so a dead link fails fast and is retried. include_usage makes
        # the last chunk carry the billed tokens.
        parts, usage, finish, served = [], None, None, None
        for chunk in client.chat.completions.create(
                model=_api_model(model_id), messages=messages,
                timeout=httpx.Timeout(connect=15.0, read=90.0, write=30.0, pool=30.0),
                stream=True, stream_options={"include_usage": True}, **api_kwargs):
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            served = getattr(chunk, "model", None) or served
            if chunk.choices:
                finish = getattr(chunk.choices[0], "finish_reason", None) or finish
                if getattr(chunk.choices[0].delta, "content", None):
                    parts.append(chunk.choices[0].delta.content)
        return "".join(parts), usage, finish, served
    response = client.chat.completions.create(
        model=model_id, messages=messages, timeout=timeout, **api_kwargs)
    choice = response.choices[0]
    return (choice.message.content, getattr(response, "usage", None),
            getattr(choice, "finish_reason", None), getattr(response, "model", None))


def get_completion(prompt, model="gpt-5-nano", timeout=None, max_retries=8,
                   backoff_factor=2.0, system=None, attempt=0, **kwargs):
    """One chat completion, returned as text. Every LLM call of the pipeline goes through here.

    Asked again (up to ``max_retries`` requests):
        rate limit, timeout, dropped connection, 500/503   wait, then ask again; the wait grows
                                                           and follows the Retry-After header
        empty reply, or a reply the provider cut short     ask again with twice the token limit
    Not asked again:
        bad key, no balance, unknown model, bad request    FatalAPIError: the run stops
        reply withheld by the content filter               ReplyBlockedError

    Each request is written to the usage log when one is installed, as one row. ``timeout=None``
    uses ``DEFAULT_TIMEOUT`` (see ``set_default_timeout``). DeepSeek calls are streamed:
    ``timeout`` does not apply to them, the limit is 90 s between two chunks.

    With a call cache installed, a call that was already answered returns the saved reply
    without contacting the API. A caller that asks again because it could not use a reply
    passes ``attempt=1, 2, ...`` so that the repeat is a new call with its own saved reply.

    Sampling: a call that sets no ``temperature`` runs at 0 on every model that accepts one
    (OpenAI reasoning models accept none). DeepSeek calls run with thinking off; a model name
    ending in ``THINK_SUFFIX`` keeps the thinking pass, which ignores the temperature.
    """
    global _FATAL
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    model_id = model_abbrev_to_id.get(model, model)
    cache_key = None
    if _CALL_CACHE is not None:
        cache_key = _call_key(model_id, system, prompt, kwargs, attempt)
        with _CALL_CACHE_LOCK:
            if cache_key in _CALL_CACHE:
                CALL_CACHE_STATS["hits"] += 1
                return _CALL_CACHE[cache_key]
    _raise_if_fatal()
    client = get_client()
    keeps_limit = _keeps_limit(model_id)
    kwargs = dict(kwargs)
    n_longer = 0
    # 8 requests with waits of up to 60 s ride out about four minutes without a connection.
    for retry in range(max_retries):
        last = retry == max_retries - 1
        logged = False   # one usage row per request
        try:
            api_kwargs = _shape_request(model_id, kwargs)
            _count("requests")
            content, usage, finish, served = _call_once(
                client, model_id, _messages(system, prompt), api_kwargs, timeout)
            empty = content is None or str(content).strip() == ""
            cut_off = finish == "length" and not keeps_limit and n_longer < 2 and not last
            if usage is not None:   # billed even when the reply cannot be used
                _record(model_id, usage, retry, served=served,
                        failed=empty or cut_off or finish in _BUSY_FINISH)
                logged = True
            _note_served(_api_model(model_id), served)
            if finish == "content_filter" and empty:
                _count("failed")
                _warn("blocked", f"{model_id}: the provider's content filter withheld a reply")
                raise ReplyBlockedError("the provider's content filter withheld the reply")
            if finish in _BUSY_FINISH:
                raise _ServerBusy(finish)
            if empty:
                raise ValueError("Empty completion; completion budget may have been too small")
            if cut_off:
                # Cut short at the token limit: ask again with twice the room. The annotator
                # tier keeps its tight limit; its callers read a cut-off reply label by label.
                n_longer += 1
                _count("retries")
                _warn("cut off", f"{model_id}: reply cut off at the token limit; asking again with more room")
                if not logged:
                    _record(model_id, None, retry, failed=True)
                kwargs["max_tokens"] = max(_budget(api_kwargs) * 2, 4000)
                continue
            if cache_key is not None:
                _call_cache_put(cache_key, model_id, content, served=served)
            return content
        except (openai.APIError, httpx.HTTPError, ValueError, _ServerBusy) as e:
            reason = _fatal_reason(e)
            if reason:
                _FATAL = (time.time(), reason)
                _count("failed")
                raise FatalAPIError(reason) from e
            retryable = isinstance(e, (openai.RateLimitError, openai.APITimeoutError,
                                       openai.APIConnectionError, openai.InternalServerError,
                                       httpx.HTTPError, ValueError, _ServerBusy))
            if not retryable:
                _count("failed")
                raise
            # A request that failed before any token count arrived is logged with zeros, so
            # that usage_report shows the true number of requests.
            if not logged:
                _record(model_id, None, retry, failed=True)
            if last:
                _count("failed")
                _warn("gave up", f"{model_id}: gave up after {max_retries} requests ({type(e).__name__})")
                raise
            _count("retries")
            what = "empty reply" if isinstance(e, ValueError) else type(e).__name__
            _warn(what, f"{model_id}: {what}; request {retry + 2} of {max_retries} follows")
            if isinstance(e, ValueError):
                if not keeps_limit:
                    kwargs["max_tokens"] = max(_budget(api_kwargs) * 2, 4000)
                time.sleep(1.0)
            else:
                # The wait does not depend on the request timeout: a rate limit or a dropped
                # connection needs seconds. The random part keeps parallel workers from all
                # asking again at the same moment.
                wait = min(4.0 * (backoff_factor ** retry), 60.0)
                asked = _retry_after(e)
                if asked is not None:
                    wait = max(wait, min(asked, 300.0))
                time.sleep(wait + random.uniform(0.0, 0.25 * wait))


# --- OpenAI Batch API -----------------------------------------------------------------------
# The Batch API answers a file of requests within 24 hours at half the price. It is used here
# only to fill the call cache: the replies are saved under the same keys as live calls, so the
# caller then runs as usual and finds its calls answered. Whatever the batch did not answer
# (expired, failed, cut short) is asked live by that normal run.
#
# The id of every submitted batch is written to '<call cache>.batches.json' before waiting, so
# a run that is interrupted while waiting picks the same batch up again instead of paying twice.
BATCH_MAX_REQUESTS = 50_000          # OpenAI limit per file
BATCH_MAX_BYTES = 190 * 1024 * 1024  # OpenAI limit is 200 MB per file
_BATCH_DONE = ("completed", "failed", "expired", "cancelled")


def batch_available():
    """Why the Batch API cannot be used right now, or '' when it can."""
    if _CALL_CACHE is None:
        return "no call cache is installed (install_call_cache)"
    if "api.openai.com" not in os.environ.get("OPENAI_BASE_URL_SAE", "https://api.openai.com/v1"):
        return "the endpoint is not OpenAI (DeepSeek has no batch endpoint)"
    return ""


def _batch_state_path():
    return _CALL_CACHE_PATH + ".batches.json"


def _read_batch_state():
    try:
        with open(_batch_state_path(), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return []


def _write_batch_state(state):
    tmp = _batch_state_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh)
    os.replace(tmp, _batch_state_path())


def _harvest_batch(client, batch, model_id):
    """Write the replies of a finished batch to the call cache. Returns how many were usable."""
    n = 0
    if getattr(batch, "output_file_id", None):
        for line in client.files.content(batch.output_file_id).text.splitlines():
            try:
                row = json.loads(line)
                body = row["response"]["body"]
                choice = body["choices"][0]
                content = choice["message"]["content"]
            except (ValueError, KeyError, TypeError, IndexError):
                continue   # a request that failed inside the batch: asked live later
            usage = body.get("usage") or {}
            _record(model_id, _Usage(usage), 0, served=body.get("model"), batch=True)
            if (row["response"].get("status_code") == 200 and content and str(content).strip()
                    and choice.get("finish_reason") != "length"):
                with _CALL_CACHE_LOCK:
                    known = row["custom_id"] in _CALL_CACHE
                if not known:
                    _call_cache_put(row["custom_id"], model_id, content, served=body.get("model"))
                n += 1
    return n


class _Usage:
    """Token counts of a batch reply, in the shape ``_record`` reads."""

    def __init__(self, usage):
        self.prompt_tokens = usage.get("prompt_tokens", 0)
        self.completion_tokens = usage.get("completion_tokens", 0)
        details = usage.get("prompt_tokens_details") or {}
        self.prompt_tokens_details = type("Details", (), {"cached_tokens": details.get("cached_tokens", 0)})()


def batch_complete(calls, poll_seconds=60.0, verbose=True):
    """Answer ``calls`` through the OpenAI Batch API and save the replies in the call cache.

    ``calls`` is a list of dicts holding the arguments of ``get_completion`` (prompt, model, and
    optionally system, attempt and the sampling settings). Calls that are already in the call
    cache, or in a batch that is still waiting, are not submitted again. Blocks until every
    batch has finished (OpenAI allows itself 24 hours); interrupting is safe, the next call
    resumes the wait. Returns {'submitted', 'answered', 'left'}.
    """
    problem = batch_available()
    if problem:
        raise RuntimeError(f"Batch API not available: {problem}")
    _raise_if_fatal()
    client = get_client()
    state = _read_batch_state()
    waiting = {k for b in state for k in b["keys"]}

    todo = {}   # model id -> {key: request line}
    for call in calls:
        call = dict(call)
        call.pop("timeout", None)
        prompt, system = call.pop("prompt"), call.pop("system", None)
        model = call.pop("model")
        model_id = model_abbrev_to_id.get(model, model)
        attempt = call.pop("attempt", 0)
        key = _call_key(model_id, system, prompt, call, attempt)
        if key in _CALL_CACHE or key in waiting:
            continue
        body = {"model": model_id, "messages": _messages(system, prompt), **_shape_request(model_id, call)}
        todo.setdefault(model_id, {})[key] = json.dumps(
            {"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": body},
            ensure_ascii=False)

    n_submitted = 0
    for model_id, lines in todo.items():   # one model per file
        chunk, size = [], 0
        for key, line in list(lines.items()) + [(None, None)]:
            n_bytes = len(line.encode("utf-8")) + 1 if line else 0
            if chunk and (key is None or len(chunk) >= BATCH_MAX_REQUESTS or size + n_bytes > BATCH_MAX_BYTES):
                payload = ("\n".join(l for _, l in chunk) + "\n").encode("utf-8")
                try:
                    upload = client.files.create(file=("clip_h_batch.jsonl", payload), purpose="batch")
                    batch = client.batches.create(input_file_id=upload.id, endpoint="/v1/chat/completions",
                                                  completion_window="24h")
                except openai.APIError as e:
                    reason = _fatal_reason(e)
                    if reason:
                        raise FatalAPIError(reason) from e
                    raise
                state.append({"id": batch.id, "model": model_id, "keys": [k for k, _ in chunk]})
                _write_batch_state(state)   # on disk before waiting: an interrupted run resumes this batch
                n_submitted += len(chunk)
                if verbose:
                    print(f"[batch] submitted {len(chunk)} calls for {model_id} as {batch.id}")
                chunk, size = [], 0
            if key is not None:
                chunk.append((key, line))
                size += n_bytes

    n_answered = n_left = 0
    for entry in list(state):
        shown = None
        while True:
            batch = client.batches.retrieve(entry["id"])
            counts = getattr(batch, "request_counts", None)
            line = (f"[batch] {entry['id']}: {batch.status}"
                    + (f", {counts.completed} of {counts.total} answered" if counts else ""))
            if verbose and line != shown:
                print(line)
                shown = line
            if batch.status in _BATCH_DONE:
                break
            time.sleep(poll_seconds)
        got = _harvest_batch(client, batch, entry["model"])
        n_answered += got
        n_left += len(entry["keys"]) - got
        state.remove(entry)
        _write_batch_state(state)
    if verbose and (n_answered or n_left):
        print(f"[batch] {n_answered} calls answered at batch price; {n_left} left for live calls")
    return {"submitted": n_submitted, "answered": n_answered, "left": n_left}
