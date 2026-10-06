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
    assert m.complete("claude-sonnet-5-5", "s", "u", max_tokens=6000, temperature=0.2, effort="medium") == "ok"
    kw = m.client.messages.calls[0]
    assert "temperature" not in kw and kw["output_config"] == {"effort": "medium"} and kw["max_tokens"] == 6000


def test_haiku_gets_temperature_and_no_effort():
    m = make()
    m.complete("claude-haiku-4-5-20251001", "s", "u", max_tokens=120, temperature=0.0, effort="low")
    kw = m.client.messages.calls[0]
    assert kw["temperature"] == 0.0 and "output_config" not in kw


def test_thinking_blocks_are_ignored_and_refusal_fails_closed():
    assert make().complete("claude-sonnet-5-5", "s", "u") == "ok"
    with pytest.raises(RuntimeError):
        make("refusal").complete("claude-sonnet-5-5", "s", "u")


# ---------------------------------------------------------------- OpenAI / local (OpenAI-compatible) backend
class StubChat:
    def __init__(self, reject=None, finish="stop", text="مرحبا"):
        self.calls, self.reject, self.finish, self.text = [], reject, finish, text
        self.completions = self

    def create(self, **kw):
        self.calls.append(kw)
        if self.reject and any(k in kw for k in self.reject):
            err = RuntimeError(f"Unsupported parameter: {sorted(self.reject)}")
            err.status_code = 400
            raise err
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=self.finish, message=SimpleNamespace(content=self.text))])


def oai(local=False, **kw):
    obj = L.OpenAICompatibleLLM.__new__(L.OpenAICompatibleLLM)
    obj.local = local
    obj.client = SimpleNamespace(chat=StubChat(**kw))
    return obj


def test_openai_standard_model_params():
    m = oai()
    assert m.complete("gpt-4o", "sys", "usr", max_tokens=6000, temperature=0.2, effort="medium") == "مرحبا"
    kw = m.client.chat.calls[0]
    assert kw["max_completion_tokens"] == 6000 and kw["temperature"] == 0.2 and "reasoning_effort" not in kw and "max_tokens" not in kw
    assert kw["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]


def test_openai_reasoning_model_floors_tokens_and_has_no_temperature():
    m = oai()
    m.complete("gpt-5-mini", "s", "u", max_tokens=120, temperature=0.0, effort="low")
    kw = m.client.chat.calls[0]
    assert kw["max_completion_tokens"] >= 4000 and "temperature" not in kw and kw["reasoning_effort"] == "low"


def test_openai_retries_without_rejected_optional_params():
    m = oai(reject={"temperature"})
    assert m.complete("gpt-4o", "s", "u", max_tokens=500, temperature=0.5) == "مرحبا"
    calls = m.client.chat.calls
    assert len(calls) == 2 and "temperature" not in calls[1] and calls[1]["max_completion_tokens"] == 500


def test_openai_content_filter_fails_closed():
    import pytest
    with pytest.raises(RuntimeError):
        oai(finish="content_filter").complete("gpt-4o", "s", "u")


def test_local_backend_uses_plain_max_tokens_and_default_temperature():
    m = oai(local=True)
    m.complete("qwen2.5:14b", "s", "u", max_tokens=700, effort="medium")
    kw = m.client.chat.calls[0]
    assert kw["max_tokens"] == 700 and kw["temperature"] == 0.2 and "max_completion_tokens" not in kw and "reasoning_effort" not in kw


def test_provider_selection_and_models(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    assert L.provider_name() is None and not L.llm_available()
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert L.provider_name() == "openai"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    assert L.provider_name() == "anthropic"           # Anthropic wins when both are set unless LLM_PROVIDER says otherwise
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert L.provider_name() == "openai"
    monkeypatch.setenv("LLM_PROVIDER", "local")
    assert L.provider_name() == "local" and L.llm_available()
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    assert L.provider_name() is None
    monkeypatch.setenv("LOCAL_MODEL", "my-model")
    assert L._models("local") == {"router": "my-model", "generate": "my-model", "judge": "my-model"}
    assert L._models("openai")["generate"] == "gpt-5.5"


def test_judge_can_use_a_different_provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    assert L.judge_provider() == "anthropic" and L.provider_name() == "openai"
    assert L.judge_model() == "claude-sonnet-5-5"
    assert L.describe()["judge_provider"] == "anthropic"
    monkeypatch.delenv("JUDGE_PROVIDER")
    assert L.judge_provider() == "openai"


def test_injected_fake_serves_the_judge_role_too():
    fake = object()
    L.set_llm(fake)
    assert L.get_judge_llm() is fake


def test_token_usage_is_counted_per_model():
    L.USAGE.clear()

    class U:
        prompt_tokens, completion_tokens = 120, 450

    m = oai()
    orig = m.client.chat.create
    m.client.chat.create = lambda **kw: SimpleNamespace(**vars(orig(**kw)), usage=U())
    m.complete("gpt-5.5", "s", "u")
    m.complete("gpt-5.5", "s", "u")
    assert L.USAGE["gpt-5.5"] == {"calls": 2, "input": 240, "output": 900}


def test_llm_cache_serves_identical_requests_without_calling_the_provider(tmp_path):
    class Inner:
        calls = 0

        def complete(self, model, system, user, max_tokens=1500, temperature=None, effort=None):
            Inner.calls += 1
            return f"answer-{Inner.calls}"

    c = L.CachedLLM(Inner(), "unit-test-provider")
    c.dir = tmp_path
    a = c.complete("m", "sys", "usr", max_tokens=100, effort="low")
    assert c.complete("m", "sys", "usr", max_tokens=100, effort="low") == a and Inner.calls == 1
    assert c.complete("m", "sys", "usr2", max_tokens=100, effort="low") != a and Inner.calls == 2   # different prompt -> new call
    assert c.complete("m", "sys", "usr", max_tokens=100, effort="high") != a and Inner.calls == 3    # different effort -> new call


def test_llm_cache_does_not_store_failures(tmp_path):
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("429")

    c = L.CachedLLM(Boom(), "x")
    c.dir = tmp_path
    import pytest
    with pytest.raises(RuntimeError):
        c.complete("m", "s", "u")
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- local reasoning models
def test_strip_thinking_removes_reasoning_blocks_and_orphan_closing_tags():
    assert L.strip_thinking("<think>الخطوة الأولى...</think>\nالجواب النهائي [[qa:x:1]]") == "الجواب النهائي [[qa:x:1]]"
    assert L.strip_thinking("<thinking>a</thinking>b<think>c</think>d") == "bd"
    assert L.strip_thinking("hidden reasoning here</think>real answer") == "real answer"
    assert L.strip_thinking("plain answer") == "plain answer"


def test_local_backend_strips_thinking_and_can_switch_it_off(monkeypatch):
    m = oai(local=True, text="<think>reasoning</think>الجواب")
    assert m.complete("qwen3-14b", "s", "u") == "الجواب"
    monkeypatch.setenv("LOCAL_NO_THINK", "1")
    m.complete("qwen3-14b", "s", "question")
    assert m.client.chat.calls[-1]["messages"][1]["content"].endswith("/no_think")
    monkeypatch.delenv("LOCAL_NO_THINK")
    m.complete("qwen3-14b", "s", "question")
    assert not m.client.chat.calls[-1]["messages"][1]["content"].endswith("/no_think")


def test_openai_backend_never_strips_or_alters_replies():
    m = oai(text="<think>kept</think>x")
    assert m.complete("gpt-5.5", "s", "u") == "<think>kept</think>x"


def test_passage_size_is_tunable_for_small_context_models(monkeypatch):
    from core.generate import format_passages
    p = [{"id": "a:1", "type": "qa", "source": "s", "text": "x" * 2000, "text_en": None, "grade": None}]
    assert len(format_passages(p)) > 900
    monkeypatch.setenv("LLM_PASSAGE_CHARS", "200")
    assert len(format_passages(p)) < 400


def test_thinking_switch_and_empty_replies_are_not_cached(tmp_path, monkeypatch):
    from core import llm as L
    assert not L.thinking_off()
    monkeypatch.setenv("LOCAL_NO_THINK", "1")
    assert L.thinking_off()
    monkeypatch.setenv("LOCAL_THINKING_GENERATE", "on")
    assert not L.thinking_off()
    with L.no_thinking():
        assert L.thinking_off()          # router / extractor calls always switch thinking off

    class Inner:
        n = 0
        def complete(self, *a, **k):
            Inner.n += 1
            return ""
    c = L.CachedLLM(Inner(), "local")
    c.dir = tmp_path
    c.complete("m", "s", "u"); c.complete("m", "s", "u")
    assert Inner.n == 2 and not list(tmp_path.iterdir())


def test_default_prompt_is_unchanged_and_local_profile_adds_the_checklist(monkeypatch):
    from core import generate as G
    ps = [{"id": "qa:x:1", "type": "qa", "source": "s", "text": "t"}]
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    default = G.build_prompts("q", "ب", "ar", ps)
    assert "CHECKLIST" not in default[1]
    monkeypatch.setenv("LLM_PROVIDER", "local")
    local = G.build_prompts("q", "ب", "ar", ps)
    assert local[0] == default[0] and "CHECKLIST" in local[1] and local[1].index("PASSAGES") < local[1].index("CHECKLIST")
    retry = G.build_prompts("q", "ب", "ar", ps, error=["Citation [[x]] was NOT retrieved for this question"], previous="p")[1]
    assert "FIX:" in retry and "PASSAGES list" in retry


def test_fair_gate_serves_older_jobs_first_and_times_out():
    import threading
    import time
    g = L.FairGate(1)
    g.acquire(0)
    order = []

    def waiter(seq):
        g.acquire(seq, timeout=5)
        order.append(seq)
        g.release()

    ts = [threading.Thread(target=waiter, args=(s,)) for s in (7, 3, 5)]
    for t in ts:
        t.start()
        time.sleep(0.05)
    assert g.stats() == {"active": 1, "waiting": 3, "capacity": 1}
    g.release()
    for t in ts:
        t.join()
    assert order == [3, 5, 7]                       # lowest job sequence first, not arrival order
    g.acquire(0)
    with pytest.raises(L.LLMBusy):
        g.acquire(1, timeout=0.1)
    assert g.stats()["waiting"] == 0


def test_readonly_cache_reads_but_never_writes(tmp_path, monkeypatch):
    class Inner:
        def complete(self, *a, **k):
            return "fresh"

    monkeypatch.setenv("LLM_CACHE", "readonly")
    c = L.CachedLLM(Inner(), "unit-test-provider")
    c.dir = tmp_path
    assert c.complete("m", "s", "u") == "fresh"
    assert list(tmp_path.iterdir()) == []           # privacy: the live demo never writes question text to disk
    monkeypatch.setenv("LLM_CACHE", "1")
    c.complete("m", "s", "u")
    monkeypatch.setenv("LLM_CACHE", "readonly")
    assert c.complete("m", "s", "u") == "fresh" and len(list(tmp_path.iterdir())) == 1
