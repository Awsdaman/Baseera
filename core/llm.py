"""LLM access layer with swappable providers.

    LLM_PROVIDER=anthropic | openai | local     (default: anthropic if ANTHROPIC_API_KEY, else openai if OPENAI_API_KEY, else none)

Roles: router (routing + claim extraction), generate (answers), judge (evals). Per-provider model env vars:
    anthropic:  ROUTER_MODEL / GENERATE_MODEL / JUDGE_MODEL            (defaults: Haiku 4.5 / Sonnet 5.5 / Sonnet 5.5)
    openai:     OPENAI_ROUTER_MODEL / OPENAI_GENERATE_MODEL / OPENAI_JUDGE_MODEL
    local:      LOCAL_BASE_URL (OpenAI-compatible server: Ollama, LM Studio, vLLM, llama.cpp ...) and LOCAL_MODEL
                (optionally LOCAL_ROUTER_MODEL / LOCAL_GENERATE_MODEL / LOCAL_JUDGE_MODEL)
Tests inject a fake with set_llm(); callers only ever use .complete(model, system, user, max_tokens, temperature, effort).
"""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

OPENAI_DEFAULTS = {"router": "gpt-5.4-mini", "generate": "gpt-5.5", "judge": "gpt-5.5"}
ANTHROPIC_DEFAULTS = {"router": "claude-haiku-4-5-20251001", "generate": "claude-sonnet-5-5", "judge": "claude-sonnet-5-5"}


class LLMUnavailable(RuntimeError):
    pass


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
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=float(os.environ.get("LLM_TIMEOUT", 300)))

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
        choice = resp.choices[0]
        if choice.finish_reason == "content_filter":
            raise RuntimeError("provider content filter triggered; failing closed")
        return choice.message.content or ""


_llm = None
_judge_llm = None


def judge_provider() -> str | None:
    p = (os.environ.get("JUDGE_PROVIDER") or "").strip().lower()
    return p if p in ("anthropic", "openai", "local") else provider_name()


def judge_model() -> str:
    jp = judge_provider()
    return JUDGE_MODEL if jp == provider_name() else _models(jp)["judge"]


def _build(p):
    if p == "anthropic":
        return AnthropicLLM()
    if p == "openai":
        return OpenAICompatibleLLM()
    if p == "local":
        return OpenAICompatibleLLM(local=True)
    raise LLMUnavailable("no LLM configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY (or LLM_PROVIDER=local) in .env")


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
        p = provider_name()
        if p == "anthropic":
            _llm = AnthropicLLM()
        elif p == "openai":
            _llm = OpenAICompatibleLLM()
        elif p == "local":
            _llm = OpenAICompatibleLLM(local=True)
        else:
            raise LLMUnavailable("no LLM configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY (or LLM_PROVIDER=local) in .env")
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
