"""LLM access layer with swappable providers.

    LLM_PROVIDER=anthropic | openai | local     (default: anthropic if ANTHROPIC_API_KEY, else openai if OPENAI_API_KEY, else none)

Roles: router (routing + claim extraction), generate (answers), judge (evals). Per-provider model env vars:
    anthropic:  ROUTER_MODEL / GENERATE_MODEL / JUDGE_MODEL            (defaults: Haiku 4.5 / Sonnet 5.5 / Sonnet 5.5)
    openai:     OPENAI_ROUTER_MODEL / OPENAI_GENERATE_MODEL / OPENAI_JUDGE_MODEL
    local:      LOCAL_BASE_URL (OpenAI-compatible server: Ollama, LM Studio, vLLM, llama.cpp ...) and LOCAL_MODEL
                (optionally LOCAL_ROUTER_MODEL / LOCAL_GENERATE_MODEL / LOCAL_JUDGE_MODEL)
Tests inject a fake with set_llm(); callers only ever use .complete(model, system, user, max_tokens, temperature, effort).
"""
import contextlib
import json
import os
import re
import threading

from dotenv import load_dotenv

load_dotenv()

OPENAI_DEFAULTS = {"router": "gpt-5.4-mini", "generate": "gpt-5.5", "judge": "gpt-5.5"}
ANTHROPIC_DEFAULTS = {"router": "claude-haiku-4-5-20251001", "generate": "claude-sonnet-5-5", "judge": "claude-sonnet-5-5"}


class LLMUnavailable(RuntimeError):
    pass


class LLMTruncated(RuntimeError):
    """The model ran out of tokens (often spent on hidden reasoning) before finishing: never verify a partial answer."""


_tls_mode = threading.local()


@contextlib.contextmanager
def no_thinking():
    """Callers with short structured outputs (router, claim extractor) ask local reasoning models not to think.
    A context manager (not a new complete() parameter) so every provider and test fake keeps the same signature."""
    prev = getattr(_tls_mode, "off", False)
    _tls_mode.off = True
    try:
        yield
    finally:
        _tls_mode.off = prev


def thinking_off() -> bool:
    """True when hidden reasoning should be switched off for the current call (local models only)."""
    if getattr(_tls_mode, "off", False):
        return True
    if os.environ.get("LOCAL_THINKING_GENERATE") == "on":
        return False
    return os.environ.get("LOCAL_NO_THINK") == "1"


USAGE: dict[str, dict[str, int]] = {}  # model -> {"calls", "input", "output"}; printed by the eval runner (cost visibility)


_usage_lock = threading.Lock()
_tls_usage = threading.local()


def _add(table: dict, model: str, i: int, o: int):
    u = table.setdefault(model, {"calls": 0, "input": 0, "output": 0})
    u["calls"] += 1
    u["input"] += i
    u["output"] += o


def _count(model: str, usage, in_attr: str, out_attr: str):
    i, o = int(getattr(usage, in_attr, 0) or 0), int(getattr(usage, out_attr, 0) or 0)  # o includes hidden reasoning tokens
    with _usage_lock:
        _add(USAGE, model, i, o)
    _add(_tls_usage.__dict__.setdefault("u", {}), model, i, o)


def thread_usage_reset():
    """Start counting usage for the CURRENT thread (evals: one case per call, even when cases run in parallel)."""
    _tls_usage.u = {}


def thread_usage() -> dict:
    return {m: dict(u) for m, u in getattr(_tls_usage, "u", {}).items()}


def provider_name() -> str | None:
    p = (os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if p:
        return p if p in ("anthropic", "openai", "local") else None
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return None


def _models(provider: str | None) -> dict:
    if provider == "openai":
        return {r: os.environ.get(f"OPENAI_{r.upper()}_MODEL", d) for r, d in OPENAI_DEFAULTS.items()}
    if provider == "local":
        base = os.environ.get("LOCAL_MODEL", "local-model")
        return {r: os.environ.get(f"LOCAL_{r.upper()}_MODEL", base) for r in OPENAI_DEFAULTS}
    return {r: os.environ.get(f"{r.upper()}_MODEL", d) for r, d in ANTHROPIC_DEFAULTS.items()}


_M = _models(provider_name())
ROUTER_MODEL, GENERATE_MODEL, JUDGE_MODEL = _M["router"], _M["generate"], _M["judge"]


def describe() -> dict:
    """What the app is running on (shown by /api/health and in the eval report)."""
    p = provider_name()
    return {"provider": p, "router": ROUTER_MODEL, "generate": GENERATE_MODEL, "judge": judge_model(), "judge_provider": judge_provider(),
            "base_url": os.environ.get("LOCAL_BASE_URL") if p == "local" else None}


def _use_system_trust_store():
    import truststore
    truststore.inject_into_ssl()


class AnthropicLLM:
    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set (put it in .env)")
        _use_system_trust_store()
        import anthropic
        self.client = anthropic.Anthropic(api_key=key)

    def complete(self, model: str, system: str, user: str, max_tokens: int = 1500, temperature: float | None = None,
                 effort: str | None = None) -> str:
        """Sonnet 5.5 rejects non-default temperature (400) and runs adaptive thinking by default, so only Haiku gets
        `temperature`; other models get an `effort` level and a max_tokens large enough to cover thinking."""
        kw = {}
        if model.startswith("claude-haiku"):
            if temperature is not None:
                kw["temperature"] = temperature
        elif effort:
            kw["output_config"] = {"effort": effort}
        msg = self.client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                          messages=[{"role": "user", "content": user}], **kw)
        _count(model, getattr(msg, "usage", None), "input_tokens", "output_tokens")
        if msg.stop_reason == "refusal":
            raise RuntimeError("model refused (safety classifier); failing closed")
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatibleLLM:
    """OpenAI Chat Completions API, or any OpenAI-compatible server (local models) when `local=True`."""

    REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

    def __init__(self, local: bool = False):
        self.local = local
        if local:
            base_url = os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1")
            api_key = os.environ.get("LOCAL_API_KEY", "local")
        else:
            base_url = os.environ.get("OPENAI_BASE_URL") or None
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise LLMUnavailable("OPENAI_API_KEY is not set (put it in .env)")
            _use_system_trust_store()
        import openai
        retries = int(os.environ.get("LLM_MAX_RETRIES", 1 if local else 2))  # a hung local call must not block for 3 x timeout
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=float(os.environ.get("LLM_TIMEOUT", 120 if local else 300)), max_retries=retries)

    def _is_reasoning(self, model: str) -> bool:
        return not self.local and model.lower().startswith(self.REASONING_PREFIXES)

    def complete(self, model: str, system: str, user: str, max_tokens: int = 1500, temperature: float | None = None,
                 effort: str | None = None) -> str:
        """Reasoning models (o-series / gpt-5) take max_completion_tokens (which also covers hidden reasoning, so it is
        floored), no temperature, and optional reasoning_effort. Local servers take plain max_tokens/temperature."""
        kw: dict = {}
        if self.local:
            kw["max_tokens"] = max_tokens
            kw["temperature"] = temperature if temperature is not None else 0.2
        elif self._is_reasoning(model):
            kw["max_completion_tokens"] = max(max_tokens, 4000)
            if effort in ("low", "medium", "high"):
                kw["reasoning_effort"] = effort
        else:
            kw["max_completion_tokens"] = max_tokens
            if temperature is not None:
                kw["temperature"] = temperature
        if self.local and thinking_off():
            # Template switch (honoured by Gemma 4 / Qwen3 chat templates) plus the Qwen-style soft switch; saves most output tokens.
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            kw["reasoning_effort"] = "none"  # the switch LM Studio honours for Gemma 4 (probe: 10 tokens instead of 134; a server that rejects it gets the 400-retry below)
            if "qwen" in model.lower():
                user += chr(10) + "/no_think"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            resp = self.client.chat.completions.create(model=model, messages=messages, **kw)
        except Exception as e:  # an optional parameter this model/server rejects: retry once with only the token limit
            msg = str(e)
            if getattr(e, "status_code", None) == 400 and any(k in msg for k in ("temperature", "reasoning_effort", "max_completion_tokens", "max_tokens")):
                token_key = "max_tokens" if (self.local or "max_completion_tokens" in msg) else "max_completion_tokens"
                resp = self.client.chat.completions.create(model=model, messages=messages, **{token_key: max_tokens})
            else:
                raise
        _count(model, getattr(resp, "usage", None), "prompt_tokens", "completion_tokens")
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise RuntimeError("provider content filter triggered; failing closed")
        text = choice.message.content or ""
        if self.local:
            text = strip_thinking(text)
            if choice.finish_reason == "length":
                raise LLMTruncated(f"{model} hit max_tokens={max_tokens} before finishing (visible chars: {len(text)})")
        return text


_THINK = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.S | re.I)


def strip_thinking(text: str) -> str:
    """Local reasoning models (Qwen, Gemma, ...) may put their hidden reasoning in the reply: remove it before anything is verified."""
    text = _THINK.sub("", text)
    if re.search(r"</think(?:ing)?>", text, re.I):  # a template that opens the block itself leaves only the closing tag
        text = re.split(r"</think(?:ing)?>", text, flags=re.I)[-1]
    return text.strip()


_llm = None
_judge_llm = None


def judge_provider() -> str | None:
    p = (os.environ.get("JUDGE_PROVIDER") or "").strip().lower()
    return p if p in ("anthropic", "openai", "local") else provider_name()


def judge_model() -> str:
    jp = judge_provider()
    return JUDGE_MODEL if jp == provider_name() else _models(jp)["judge"]


class LLMBusy(RuntimeError):
    """Waited too long for a free model slot (the live demo has one GPU)."""


_tls_job = threading.local()


@contextlib.contextmanager
def request_seq(seq: int):
    """Jobs started earlier get the model first (a question already in progress is not overtaken by newcomers)."""
    prev = getattr(_tls_job, "seq", 0)
    _tls_job.seq = seq
    try:
        yield
    finally:
        _tls_job.seq = prev


class FairGate:
    """Counting gate that admits waiters in (job sequence, arrival) order, with a timeout instead of waiting forever."""

    def __init__(self, capacity: int):
        import heapq
        self._hq = heapq
        self.capacity, self.active, self._waiting, self._n = max(1, capacity), 0, [], 0
        self._cond = threading.Condition()

    def acquire(self, seq: int = 0, timeout: float | None = None):
        import time
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            self._n += 1
            entry = (seq, self._n)
            self._hq.heappush(self._waiting, entry)
            while not (self.active < self.capacity and self._waiting[0] == entry):
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    self._waiting.remove(entry)
                    self._hq.heapify(self._waiting)
                    self._cond.notify_all()
                    raise LLMBusy("no free model slot, please try again")
                self._cond.wait(left)
            self._hq.heappop(self._waiting)
            self.active += 1

    def release(self):
        with self._cond:
            self.active -= 1
            self._cond.notify_all()

    def stats(self) -> dict:
        with self._cond:
            return {"active": self.active, "waiting": len(self._waiting), "capacity": self.capacity}


GATE: FairGate | None = None


class GatedLLM:
    def __init__(self, inner, gate: FairGate):
        self.inner, self.gate = inner, gate

    def complete(self, model, system, user, max_tokens=1500, temperature=None, effort=None):
        self.gate.acquire(getattr(_tls_job, "seq", 0), float(os.environ.get("QUEUE_TIMEOUT_S", 240)))
        try:
            return self.inner.complete(model, system, user, max_tokens=max_tokens, temperature=temperature, effort=effort)
        finally:
            self.gate.release()


def queue_stats() -> dict:
    return GATE.stats() if GATE else {"active": 0, "waiting": 0, "capacity": 0}


def ping() -> bool:
    """Tiny uncached model call (startup warm-up and keep-alive, so the local server never unloads the model)."""
    llm = get_llm()
    while isinstance(llm, CachedLLM):
        llm = llm.inner
    with no_thinking():
        llm.complete(ROUTER_MODEL, "Reply with: ok", "ping", max_tokens=8)
    return True


class CachedLLM:
    """LLM_CACHE=1 (evals / development only): identical requests are answered from data/cache/llm, so re-running an
    unchanged prompt costs nothing. The key covers provider, model, prompts and every sampling parameter.
    LLM_CACHE=readonly (live demo): lookups only, nothing is ever written, so no question text reaches the disk."""

    def __init__(self, inner, tag: str):
        import hashlib
        from pathlib import Path
        self.inner, self.tag, self._sha = inner, tag, hashlib.sha256
        self.dir = Path(__file__).resolve().parent.parent / "data" / "cache" / "llm"
        if os.environ.get("LLM_CACHE") != "readonly":
            self.dir.mkdir(parents=True, exist_ok=True)

    def complete(self, model, system, user, max_tokens=1500, temperature=None, effort=None):
        mode = [thinking_off(), os.environ.get("LOCAL_THINKING_GENERATE")] if self.tag == "local" else None
        key = self._sha(json.dumps([self.tag, model, system, user, max_tokens, temperature, effort, mode], ensure_ascii=False).encode("utf-8")).hexdigest()
        f = self.dir / f"{key}.txt"
        if f.exists():
            return f.read_text(encoding="utf-8")
        out = self.inner.complete(model, system, user, max_tokens=max_tokens, temperature=temperature, effort=effort)
        if out.strip() and os.environ.get("LLM_CACHE") != "readonly":  # only successful, non-empty completions are cached (errors propagate; an empty reply must be retried)
            f.write_text(out, encoding="utf-8")
        return out


def _build(p):
    if p == "anthropic":
        llm = AnthropicLLM()
    elif p == "openai":
        llm = OpenAICompatibleLLM()
    elif p == "local":
        llm = OpenAICompatibleLLM(local=True)
    else:
        raise LLMUnavailable("no LLM configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY (or LLM_PROVIDER=local) in .env")
    global GATE
    if GATE is None:  # one shared gate: the GPU (or the API quota) serves a few calls at a time, the rest wait in order
        GATE = FairGate(int(os.environ.get("MAX_CONCURRENT_ANSWERS", 2 if p == "local" else 8)))
    llm = GatedLLM(llm, GATE)
    return CachedLLM(llm, p) if os.environ.get("LLM_CACHE") in ("1", "readonly") else llm


def get_judge_llm():
    """The judge may run on a different provider than the generator (JUDGE_PROVIDER=anthropic, say)."""
    global _judge_llm
    if _llm is not None:  # an injected fake serves every role
        return _llm
    jp = judge_provider()
    if jp == provider_name():
        return get_llm()
    if _judge_llm is None:
        _judge_llm = _build(jp)
    return _judge_llm


def get_llm():
    global _llm
    if _llm is None:
        _llm = _build(provider_name())
    return _llm


def set_llm(obj):
    """Inject a fake (anything with .complete(model, system, user, max_tokens, temperature, effort)) or None to reset."""
    global _llm, _judge_llm
    _llm = obj
    _judge_llm = None


def llm_available() -> bool:
    if _llm is not None:
        return True
    p = provider_name()
    if p == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if p == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    return p == "local"


def extract_json(text: str):
    """Parse the first JSON object/array in an LLM reply (tolerates code fences and prose)."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    firsts = sorted((t.find(c), c) for c in "{[" if t.find(c) >= 0)  # the earliest opener decides object vs array
    for i, start in firsts:
        end = t.rfind("}" if start == "{" else "]")
        if end > i:
            try:
                return json.loads(t[i:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in LLM output")
