import pytest

from core import llm as L


class FakeLLM:
    """Scripted replies per model name; records every call."""

    def __init__(self, router=None, generate=None, extract=None):
        self.replies = {L.ROUTER_MODEL: list(router or []), L.GENERATE_MODEL: list(generate or [])}
        self.extract = extract
        self.calls = []

    def complete(self, model, system, user, max_tokens=1000, temperature=0.0):
        self.calls.append((model, system, user))
        if self.extract is not None and "Extract every checkable" in system:
            return self.extract
        q = self.replies.get(model, [])
        if not q:
            raise AssertionError(f"unexpected LLM call to {model}")
        return q.pop(0) if len(q) > 1 else q[0]


@pytest.fixture
def fake_llm():
    def make(**kw):
        f = FakeLLM(**kw)
        L.set_llm(f)
        return f
    yield make
    L.set_llm(None)


@pytest.fixture(autouse=True)
def no_vectors(monkeypatch, request):
    """Unit tests use keyword retrieval only (no embedding model load); mark a test @pytest.mark.vectors to use the real index."""
    if request.node.get_closest_marker("vectors"):
        return
    from core import retrieve as R
    orig = R.retrieve
    monkeypatch.setattr(R, "retrieve", lambda q, per_type=None, vectors=False, dorar=False: orig(q, per_type=per_type, vectors=False, dorar=dorar))
