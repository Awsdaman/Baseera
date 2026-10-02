"""Thin LLM wrapper. Tests inject a fake with set_llm(); production uses the Anthropic SDK."""
import json
import os
import re

from dotenv import load_dotenv

load_dotenv()

ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "claude-haiku-4-5-20251001")
GENERATE_MODEL = os.environ.get("GENERATE_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-sonnet-5-5")


class LLMUnavailable(RuntimeError):
    pass


class AnthropicLLM:
    def __init__(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set (put it in .env)")
        import truststore
        truststore.inject_into_ssl()
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


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = AnthropicLLM()
    return _llm


def set_llm(obj):
    """Inject a fake (anything with .complete(model, system, user, max_tokens, temperature)) or None to reset."""
    global _llm
    _llm = obj


def llm_available() -> bool:
    if _llm is not None:
        return True
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


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
