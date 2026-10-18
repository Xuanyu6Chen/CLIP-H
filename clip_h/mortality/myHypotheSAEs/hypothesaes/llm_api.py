# Modified for CLIP-H from HypotheSAEs (https://github.com/rmovva/HypotheSAEs),
# Apache License 2.0. See ../README.md for the changes.
"""LLM API utilities for HypotheSAEs."""

import os
import time
import threading
from collections import deque
from typing import Optional, Tuple, Union
import openai

# Errors worth retrying: the request never produced an answer, and trying again may.
#
# Besides rate limits and timeouts, this covers connection errors (a dropped or interrupted
# network connection) and provider 5xx errors. Both usually clear within seconds to minutes;
# without retrying them, a run lasting hours could fail on a brief interruption, and in
# annotation the failed call would end up recorded as 0 ("concept absent"), indistinguishable
# from a real negative. Retrying is safe because a retry resends the identical request, so it
# affects only whether an answer arrives.
_TRANSIENT_ERRORS: Tuple[type, ...] = tuple(
    e for e in (
        getattr(openai, "RateLimitError", None),
        getattr(openai, "APITimeoutError", None),
        getattr(openai, "APIConnectionError", None),      # connection dropped or interrupted
        getattr(openai, "InternalServerError", None),     # provider 5xx
    ) if isinstance(e, type)
)
try:  # a mid-stream drop can surface as a raw httpx error, not an openai wrapper
    import httpx as _httpx
    _TRANSIENT_ERRORS = _TRANSIENT_ERRORS + (_httpx.TransportError,)
except Exception:  # pragma: no cover - httpx ships with openai, but never hard-fail on it
    pass

# Authentication-type errors, retried a few times with a long wait. When a network login page
# intercepts the connection, the request is not dropped but answered with an HTML page or a
# 302/401/403 status instead of JSON. The SDK maps that to AuthenticationError,
# PermissionDeniedError or APIResponseValidationError, none of which is transient, so without
# this the call would fail on the first attempt with no backoff, and with many parallel
# annotation workers a large number of annotations would be recorded as "concept absent" within
# minutes. A genuinely invalid key fails identically on every attempt and is caught by the
# runner's preflight check before the run starts, so retrying these a few times costs only a
# delay. LLM_PORTAL_RETRIES and LLM_PORTAL_WAIT_S set the number of retries and the wait.
_PORTAL_ERRORS: Tuple[type, ...] = tuple(
    e for e in (
        getattr(openai, "AuthenticationError", None),
        getattr(openai, "PermissionDeniedError", None),
        getattr(openai, "APIResponseValidationError", None),
    ) if isinstance(e, type)
)
_PORTAL_MAX_RETRIES = int(os.environ.get("LLM_PORTAL_RETRIES", "3"))
_PORTAL_WAIT_S = float(os.environ.get("LLM_PORTAL_WAIT_S", "30"))


class _TokenRateLimiter:
    """Process-wide token/request-per-minute throttle shared by every API call.

    Reactive 429 backoff alone causes retry storms: when many parallel workers
    hit the org TPM ceiling at once they all fail, back off together, and retry
    together. This limiter paces requests *proactively* against a rolling 60-second
    budget so the ceiling is approached smoothly and 429s become rare (the backoff
    in get_completion remains as a safety net for the residual cases and for usage
    from other processes sharing the same org limit).

    Configure via environment (read once at construction):
      OPENAI_TPM_LIMIT  tokens/minute budget   (default 180000; set it just under your
                        plan's TPM limit to leave headroom)
      OPENAI_RPM_LIMIT  requests/minute budget (default 0 = disabled; TPM usually binds)
    Set OPENAI_TPM_LIMIT=0 to disable throttling entirely (pure reactive behaviour).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tok = deque()  # (timestamp, tokens)
        self._req = deque()  # timestamps
        self._refresh_limits()

    def _refresh_limits(self) -> None:
        try:
            self.tpm = float(os.environ.get("OPENAI_TPM_LIMIT", "180000"))
        except ValueError:
            self.tpm = 180000.0
        try:
            self.rpm = float(os.environ.get("OPENAI_RPM_LIMIT", "0"))
        except ValueError:
            self.rpm = 0.0

    def _prune(self, now: float) -> None:
        while self._tok and now - self._tok[0][0] >= 60.0:
            self._tok.popleft()
        while self._req and now - self._req[0] >= 60.0:
            self._req.popleft()

    def acquire(self, tokens: int) -> None:
        # Re-read env each call so a notebook can retune the limit live without reimport.
        self._refresh_limits()
        if self.tpm <= 0 and self.rpm <= 0:
            return
        # A single request larger than the whole budget would never fit. It is admitted once
        # the window is empty (the `not self._tok` condition below), so it waits at most one
        # window rather than spinning forever.
        tokens = int(tokens)
        while True:
            with self._lock:
                now = time.time()
                self._prune(now)
                used_tok = sum(t for _, t in self._tok)
                used_req = len(self._req)
                ok_tok = self.tpm <= 0 or used_tok + tokens <= self.tpm or not self._tok
                ok_req = self.rpm <= 0 or used_req + 1 <= self.rpm or not self._req
                if ok_tok and ok_req:
                    self._tok.append((now, tokens))
                    self._req.append(now)
                    return
                waits = []
                if not ok_tok and self._tok:
                    waits.append(60.0 - (now - self._tok[0][0]))
                if not ok_req and self._req:
                    waits.append(60.0 - (now - self._req[0]))
                wait = max(0.05, min(waits)) if waits else 0.25
            time.sleep(wait)


_RATE_LIMITER = _TokenRateLimiter()


def _estimate_tokens(messages, max_output_tokens) -> int:
    """Rough token estimate for throttling: ~4 chars/token input + the output budget.

    Deliberately conservative (counts the requested completion budget, up to 8,000 tokens,
    rather than the tokens actually generated) so the limiter under- rather than
    over-shoots the real TPM ceiling.
    """
    chars = sum(len((m.get("content") or "")) for m in messages)
    # Reserve a realistic output budget rather than the full safety cap: reasoning models get
    # a 64k-token cap (see get_completion), which would make the token bucket over-reserve and
    # throttle them needlessly. Outputs here are short (yes/no, one-line names, small JSON);
    # the rare larger output that triggers a 429 is handled by the reactive backoff.
    return int(chars / 4) + int(min(int(max_output_tokens or 0), 8000)) + 20

"""
These model IDs point to the latest versions of the models as of 2025-05-04.
We point to a specific version for reproducibility, but feel free to update them as necessary.
Note that o-series models (o1, o1-mini, o3-mini) are also supported by get_completion().
We don't point these models to a specific version, so passing in these model names will use the latest version.

CLIP-H:
- gpt-5 pinned to gpt-5-2025-08-07, the snapshot the clip_h readmission, hosp_los and icu_los
  runners ask for (their llm_client.py), so every CLIP-H task calls the same GPT-5.

2025-05-04:
- Removed gpt-4 (deprecated by gpt-4o, will be removed from API soon)
- Added gpt-4.1 models (not used by HypotheSAEs paper, but potentially of interest)

2025-03-12:
- First version of this file: supports gpt-4o, gpt-4o-mini, gpt-4
"""
model_abbrev_to_id = {
    'gpt4o': 'gpt-4o-2024-11-20',
    'gpt-4o': 'gpt-4o-2024-11-20',
    'gpt4o-mini': 'gpt-4o-mini-2024-07-18',
    'gpt-4o-mini': 'gpt-4o-mini-2024-07-18',

    "gpt4.1": "gpt-4.1-2025-04-14",
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "gpt4.1-nano": "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",

    "gpt5": "gpt-5-2025-08-07",
    "gpt-5": "gpt-5-2025-08-07",
}

def get_client():
    """Get the OpenAI client, initializing it if necessary."""
    api_key = os.environ.get('OPENAI_KEY_SAE')
    if api_key is None or '...' in api_key:
        raise ValueError("Please set the OPENAI_KEY_SAE environment variable before using functions which require the OpenAI API.")
    
    base_url = os.environ.get("OPENAI_BASE_URL_SAE", "https://api.openai.com/v1")
    return openai.OpenAI(api_key=api_key, base_url=base_url)


# Provider routing for the generator sweep. Every non-Anthropic provider below speaks the
# OpenAI Chat Completions API, so one client class serves all of them; only the base URL and
# API-key env var differ. Prefixes are matched case-insensitively on the (abbrev-resolved)
# model id. Endpoints/keys are overridable via env for local vLLM or proxy setups.
#   gemini*    -> Google Generative Language OpenAI-compat endpoint (GEMINI_API_KEY)
#   deepseek*  -> DeepSeek platform (DEEPSEEK_API_KEY)
#   qwen*/other open-weight -> a self-hosted vLLM / OpenAI-compatible server
#                             (OPENWEIGHT_BASE_URL + OPENWEIGHT_API_KEY, e.g. a local Qwen3)
def _provider_of(model_id: str) -> str:
    m = model_id.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "gemini"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("qwen") or os.environ.get("OPENWEIGHT_BASE_URL"):
        # qwen* always routes to the open-weight server; for any other id we only divert to a
        # self-hosted server when one is explicitly configured, else fall through to OpenAI.
        if m.startswith("qwen") or m.startswith(("mistral", "llama", "gemma", "phi")):
            return "openweight"
    return "openai"


def _resolve_deepseek_thinking(model_id: str, thinking_env: Optional[str] = None) -> Tuple[str, bool]:
    """Map a DeepSeek model id to (clean_model_id, thinking_enabled).

    DeepSeek v4 (deepseek-v4-flash / deepseek-v4-pro) enables thinking mode by default at
    the server. For reproducibility, cost and latency the mode is set explicitly, and the
    default is non-thinking (fast and temperature-honoring, which suits bulk binary
    annotation and the interpretation and vote calls). A step opts into thinking mode either
    by suffixing its model id with ``-think`` (e.g. ``deepseek-v4-pro-think``) or by setting
    ``DEEPSEEK_THINKING=enabled``; ``thinking_env``, when given, is used in place of that
    environment variable. The ``-think`` suffix is stripped from the returned id so the real
    model name is sent to the API.
    """
    env = thinking_env if thinking_env is not None else os.environ.get("DEEPSEEK_THINKING", "")
    think = model_id.endswith("-think") or env == "enabled"
    if model_id.endswith("-think"):
        model_id = model_id[: -len("-think")]
    return model_id, think


def _openai_compatible_client(model_id: str):
    """Return an OpenAI-compatible client for any non-Anthropic provider, resolving the
    base URL and API key from provider-specific env vars (with sensible public defaults)."""
    provider = _provider_of(model_id)
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key or "..." in key:
            raise ValueError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use a Gemini model.")
        base = os.environ.get("GEMINI_BASE_URL",
                              "https://generativelanguage.googleapis.com/v1beta/openai/")
        return openai.OpenAI(api_key=key, base_url=base), provider
    if provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key or "..." in key:
            raise ValueError("Set DEEPSEEK_API_KEY to use a DeepSeek model.")
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        return openai.OpenAI(api_key=key, base_url=base), provider
    if provider == "openweight":
        base = os.environ.get("OPENWEIGHT_BASE_URL") or os.environ.get("VLLM_BASE_URL")
        if not base:
            raise ValueError("Set OPENWEIGHT_BASE_URL (or VLLM_BASE_URL) to the OpenAI-compatible "
                             "endpoint of your self-hosted open-weight model (e.g. a local Qwen3 vLLM).")
        # vLLM/Ollama servers usually accept any key; allow an override for gated proxies.
        key = os.environ.get("OPENWEIGHT_API_KEY", "EMPTY")
        return openai.OpenAI(api_key=key, base_url=base), provider
    return get_client(), "openai"


def _get_anthropic_client():
    """Anthropic client for the Claude judge (ANTHROPIC_API_KEY). Lazy import so the
    package is only required when a `claude*` model is actually requested."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or "..." in key:
        raise ValueError("Set ANTHROPIC_API_KEY to use a Claude model (e.g. the ensemble judge).")
    try:
        import anthropic
    except ImportError as e:
        raise ValueError("The `anthropic` package is required for Claude models: pip install anthropic") from e
    return anthropic.Anthropic(api_key=key)


def _anthropic_completion(prompt, model_id, system, timeout, max_retries, backoff_factor,
                          return_meta, **kwargs):
    """Claude path for get_completion. Anthropic has no `seed`; determinism comes from
    temperature=0 (default here). Returns the text (or (text, None) with return_meta)."""
    client = _get_anthropic_client()   # validates key + package before we import error types
    import anthropic
    # Newer reasoning Claude models (e.g. Opus 4.8) deprecate `temperature` and return a 400
    # error if it is sent; older models accept it. It is sent whenever the caller provides it
    # and stripped on the deprecation error, so both work without per-model name lists.
    # (Without temperature, Anthropic's own default still gives the sampling variation the
    # self-consistency vote relies on.)
    # Anthropic requires max_tokens and caps output at it; billing is on tokens actually
    # generated, so a high cap is free insurance against a truncated JSON answer. The cap is
    # floored (ANTHROPIC_MAX_TOKENS_FLOOR, default 64000) so structured-vote outputs are not
    # cut off.
    _req = int(kwargs.get("max_tokens", kwargs.get("max_completion_tokens", 1024)) or 1024)
    max_tokens = max(_req, int(os.environ.get("ANTHROPIC_MAX_TOKENS_FLOOR", "64000")))
    msg_kwargs = dict(model=model_id, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
    if "temperature" in kwargs:
        msg_kwargs["temperature"] = float(kwargs["temperature"])
    if system is not None:
        msg_kwargs["system"] = system
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(timeout=timeout, **msg_kwargs)
            text = "".join(getattr(b, "text", "") for b in resp.content).strip()
            if not text:
                if attempt == max_retries - 1:
                    raise RuntimeError("Empty Claude completion after retries")
                time.sleep(1.0); continue
            return (text, None) if return_meta else text
        except anthropic.BadRequestError as e:
            # An unsupported param for this model (temperature/top_p on reasoning models):
            # drop the offending param and retry immediately rather than failing the call.
            msg = str(e).lower()
            dropped = False
            for p in ("temperature", "top_p"):
                if p in msg and p in msg_kwargs:
                    msg_kwargs.pop(p, None); dropped = True
            if dropped and attempt < max_retries - 1:
                continue
            raise
        # Retry the same transient error classes as the OpenAI path (rate limit, timeout,
        # connection, 5xx). anthropic is imported lazily, so the tuple is built at call time.
        except tuple(e for e in (getattr(anthropic, "RateLimitError", None),
                                 getattr(anthropic, "APITimeoutError", None),
                                 getattr(anthropic, "APIConnectionError", None),
                                 getattr(anthropic, "InternalServerError", None))
                     if isinstance(e, type)) as e:
            if attempt == max_retries - 1:
                raise e
            wait = min(2.0 * (backoff_factor ** attempt), 30.0)
            print(f"Anthropic error: {e}; retrying in {wait:.1f}s... ({attempt+1}/{max_retries})")
            time.sleep(wait)


def get_completion(
    prompt: str,
    model: str = "Qwen3-8B",
    timeout: float = 120.0,
    max_retries: int = 11,
    backoff_factor: float = 2.0,
    system = None,
    seed: Optional[int] = None,
    return_meta: bool = False,
    **kwargs
) -> Union[str, Tuple[str, Optional[str]]]:
    """
    Get completion from an OpenAI-compatible API (or from Anthropic for ``claude*`` models)
    with retry logic and timeout.

    Determinism: for non-reasoning models we default ``temperature`` to 0.0 when
    the caller does not specify one, and forward ``seed`` to the API. This makes
    runs near-deterministic (the OpenAI ``seed`` is best-effort: the server-side
    ``system_fingerprint`` can still shift, so this is near- not bit-exact).

    Args:
        prompt: The prompt to send
        model: Model to use (abbreviations resolve via model_abbrev_to_id; explicit
            IDs pass through verbatim)
        max_retries: Maximum number of attempts on transient errors (rate limits, timeouts,
            connection and server errors) and empty completions
        backoff_factor: Factor to multiply backoff time by after each retry
        timeout: Timeout for the request
        seed: Optional integer seed forwarded to the OpenAI API for reproducibility
        return_meta: If True, return (content, system_fingerprint) so callers can
            log the fingerprint to a run manifest. Default False keeps the str contract.
        **kwargs: Additional arguments to pass to the OpenAI API; max_tokens, temperature, etc.
    Returns:
        Generated completion text, or (text, system_fingerprint) when return_meta=True

    Raises:
        Exception: If all retries fail
    """
    model_id = model_abbrev_to_id.get(model, model)

    # Provider dispatch: a `claude*` model id (or LLM_PROVIDER=anthropic) routes to Anthropic
    # (the ensemble judge); everything else uses the OpenAI-compatible path. The OpenAI
    # token-bucket limiter does not gate Anthropic calls; Anthropic enforces its own rate limits.
    if model_id.lower().startswith("claude") or os.environ.get("LLM_PROVIDER") == "anthropic":
        return _anthropic_completion(prompt, model_id, system, timeout, max_retries,
                                     backoff_factor, return_meta, **kwargs)

    # Non-Anthropic providers (OpenAI, Gemini, DeepSeek, self-hosted open-weight) all speak the
    # OpenAI Chat Completions API; pick the right endpoint/key by model prefix.
    client, _provider = _openai_compatible_client(model_id)

    # Reasoning models restrict the chat-completions params (max_tokens -> max_completion_tokens,
    # temperature/top_p rejected). This applies to OpenAI's o-series / gpt-5 only; Gemini,
    # DeepSeek, and open-weight chat models take the standard params (keep temperature).
    is_reasoning = _provider == "openai" and (model_id.startswith('o') or model_id.startswith('gpt-5'))

    # Determinism defaults (non-reasoning only; reasoning models reject temperature).
    if not is_reasoning and 'temperature' not in kwargs:
        kwargs['temperature'] = 0.0
    if seed is not None and 'seed' not in kwargs:
        kwargs['seed'] = seed

    # Messages and final kwargs do not change across retries, so they are built once.
    messages = (
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        if system is not None
        else [{"role": "user", "content": prompt}]
    )
    # Normalise max_tokens -> max_completion_tokens for every model.
    api_kwargs = {('max_completion_tokens' if k == 'max_tokens' else k): v
                  for k, v in kwargs.items()}
    if is_reasoning:
        api_kwargs.pop('temperature', None)              # only default (1) allowed
        api_kwargs.pop('top_p', None)                    # also unsupported
        # Reasoning models (o-series, gpt-5*) spend the completion budget on hidden
        # reasoning before the visible answer; too small a cap returns an empty completion
        # (which the retry loop below eventually raises on). Floor the cap high: billing is
        # for tokens actually used, not the cap, so this is free insurance against starved
        # or empty outputs on tiny-output tasks (yes/no, one-line names, short JSON).
        _floor = int(os.environ.get('OPENAI_REASONING_MAX_TOKENS', '64000'))
        _cur = api_kwargs.get('max_completion_tokens')
        api_kwargs['max_completion_tokens'] = max(int(_cur), _floor) if _cur else _floor
        # Bounded reasoning effort keeps latency/cost sane on these structured tasks.
        api_kwargs.setdefault('reasoning_effort', os.environ.get('OPENAI_REASONING_EFFORT', 'medium'))

    # DeepSeek v4 enables thinking mode by default at the server; set the mode explicitly
    # (default: non-thinking) so bulk binary annotation and quick calls stay fast, cheap and
    # temperature-honoring, while a step can opt into reasoning with a '-think' model suffix
    # (e.g. deepseek-v4-pro-think) or DEEPSEEK_THINKING=enabled. Thinking mode ignores
    # temperature/top_p, so they are dropped there and reasoning_effort is set instead.
    if _provider == "deepseek":
        model_id, _ds_think = _resolve_deepseek_thinking(model_id)
        _eb = dict(api_kwargs.get("extra_body") or {})
        _eb["thinking"] = {"type": "enabled" if _ds_think else "disabled"}
        api_kwargs["extra_body"] = _eb
        if _ds_think:
            api_kwargs.pop("temperature", None)
            api_kwargs.pop("top_p", None)
            api_kwargs.setdefault("reasoning_effort", os.environ.get("DEEPSEEK_REASONING_EFFORT", "medium"))

    # Proactively reserve this call's token budget so parallel workers stay under the
    # org TPM limit instead of colliding and triggering 429 retry storms.
    _RATE_LIMITER.acquire(_estimate_tokens(messages, api_kwargs.get('max_completion_tokens') or 256))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                timeout=timeout,
                **api_kwargs
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                # Empty completion: retry plainly; raise a clear error if it persists.
                # Handled inline (not via a broad except) so unrelated ValueErrors
                # from the SDK are never silently swallowed.
                if attempt == max_retries - 1:
                    raise RuntimeError("Empty completion after retries")
                time.sleep(1.0)
                continue
            if return_meta:
                return content, getattr(response, 'system_fingerprint', None)
            return content

        except _PORTAL_ERRORS as e:
            # See _PORTAL_ERRORS: during a run this usually means a network login page
            # intercepted the request rather than an invalid key (the preflight check catches
            # that before any spend). Wait long enough for a sign-in to complete, a few times,
            # then give up.
            if attempt >= _PORTAL_MAX_RETRIES:
                raise e
            print(f"Auth-shaped error ({type(e).__name__}); this can mean a network login page intercepted the request. "
                  f"Retrying in {_PORTAL_WAIT_S:.0f}s ({attempt+1}/{_PORTAL_MAX_RETRIES})...")
            time.sleep(_PORTAL_WAIT_S)
            continue

        except _TRANSIENT_ERRORS as e:
            if attempt == max_retries - 1:  # Last attempt
                raise e

            # Try to honour the Retry-After header; fall back to short exponential backoff.
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", 0))
            except Exception:
                pass
            if retry_after and retry_after > 0:
                wait_time = retry_after + 0.25
            else:
                # With the defaults (backoff_factor=2, max_retries=11) this sleeps 2, 4, 8,
                # 16, 32 and then 60 s per retry, about six minutes in total, enough to ride
                # out a network interruption of a few minutes. Hard failures cost nothing:
                # 400 and 404 errors are not in _TRANSIENT_ERRORS and raise on the first
                # attempt without sleeping (401 and 403 take the _PORTAL_ERRORS path above).
                wait_time = min(2.0 * (backoff_factor ** attempt), 60.0)
            print(f"API error: {e}; retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
            time.sleep(wait_time)
