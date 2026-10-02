"""The real AnthropicLLM wrapper against a stubbed SDK client: parameters must be valid for Sonnet 5.5 and Haiku 4.5."""
from types import SimpleNamespace

import pytest

from core import llm as L


class StubMessages:
    def __init__(self, stop="end_turn"):
        self.calls, self.stop = [], stop

    def create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(stop_reason=self.stop, content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="ok")])


def make(stop="end_turn"):
    obj = L.AnthropicLLM.__new__(L.AnthropicLLM)
    obj.client = SimpleNamespace(messages=StubMessages(stop))
    return obj


def test_sonnet_never_gets_temperature_and_gets_effort():
    m = make()
    assert m.complete(L.GENERATE_MODEL, "s", "u", max_tokens=6000, temperature=0.2, effort="medium") == "ok"
    kw = m.client.messages.calls[0]
    assert "temperature" not in kw and kw["output_config"] == {"effort": "medium"} and kw["max_tokens"] == 6000


def test_haiku_gets_temperature_and_no_effort():
    m = make()
    m.complete(L.ROUTER_MODEL, "s", "u", max_tokens=120, temperature=0.0, effort="low")
    kw = m.client.messages.calls[0]
    assert kw["temperature"] == 0.0 and "output_config" not in kw


def test_thinking_blocks_are_ignored_and_refusal_fails_closed():
    assert make().complete(L.GENERATE_MODEL, "s", "u") == "ok"
    with pytest.raises(RuntimeError):
        make("refusal").complete(L.GENERATE_MODEL, "s", "u")
