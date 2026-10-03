"""Slow sanity check of verify mode on synthetic data (run with: python -m pytest -m slow). Full report: evals/synth_verify.py."""
import random

import pytest

from core import verifier_mode as V
from evals import synth_verify as SV

pytestmark = pytest.mark.slow


def test_synthetic_verse_classes_are_recognised():
    cases = SV.verse_cases(4, random.Random(3))
    ok = {"verified": 0, "misquoted": 0, "fabricated": 0}
    n = {"verified": 0, "misquoted": 0, "fabricated": 0}
    for c in cases:
        n[c["label"]] += 1
        ok[c["label"]] += V.check_verse(c["text"])["verdict"] == c["label"]
    assert ok["verified"] == n["verified"]                       # exact text and fragments are never rejected
    assert ok["misquoted"] / n["misquoted"] >= 0.9               # was 73% before the shortlist fix
    assert ok["fabricated"] / n["fabricated"] >= 0.8
