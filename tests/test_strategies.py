"""Pytest wrapper around the eval harness.

Asserts that each (strategy, fixture) pair achieves WER below a per-class
threshold. Skips cleanly when fixtures or the server are unavailable so the
suite is friendly to CI environments that haven't been provisioned yet.

Default scope is tight (a handful of short fixtures + long-form) so the
gate runs in ~5 min. For the full corpus-WER comparison vs NVIDIA's
published 1.69%, use `python -m tests.eval_harness` directly with no
limits — that's slow.

Run:

    pytest tests/test_strategies.py -v

    PARAKEET_URL=http://192.168.0.172:8777 pytest tests/test_strategies.py -v

    # Tighter scope while iterating
    pytest tests/test_strategies.py -v -k "chunked_v2 and short"
"""
from __future__ import annotations

import asyncio
import os

import pytest

from tests.eval_harness import (
    _fixture_records,
    _load_manifest,
    _run_one,
    REST_STRATEGIES,
    WS_STRATEGIES,
    FIXTURES,
)

SERVER_URL = os.environ.get("PARAKEET_URL", "http://localhost:8777")
REQUEST_TIMEOUT_S = float(os.environ.get("PARAKEET_TEST_TIMEOUT_S", "900"))
INCLUDE_LONGFORM = os.environ.get("PARAKEET_TEST_INCLUDE_LONGFORM", "1") == "1"
SHORT_LIMIT = int(os.environ.get("PARAKEET_TEST_SHORT_LIMIT", "5"))

# Per (strategy, fixture-kind) WER ceilings.
#
# Parakeet on LibriSpeech-clean is ~1.7%. We keep `full` / `_v2` strict (≤10%)
# on short to catch real engine regressions; the streaming `_v2` engine gets
# slightly looser short-clip headroom because its first-chunk right context
# is silence padding, which can flip one phoneme on a 3-5s utterance (=
# 10-15% WER from a single word). Legacy chunked/progressive get a wide
# short-clip gate because their middle-token-merge boundary artifacts are
# documented pre-existing behavior — the whole reason `_v2` exists.
#
# Long-form thresholds reflect what's reasonable on TED-LIUM 3 (harder than
# LibriSpeech — natural speech, varied speakers, no read text).
WER_THRESHOLDS = {
    "full":           {"short": 0.10, "longform": 0.15},
    "chunked_v2":     {"short": 0.10, "longform": 0.15},
    "progressive_v2": {"short": 0.20, "longform": 0.15},
    "chunked":        {"short": 0.75, "longform": 0.20},
    "progressive":    {"short": 0.75, "longform": 0.20},
}


@pytest.fixture(scope="session")
def normalizer():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer
    except ImportError:
        pytest.skip("whisper-normalizer not installed; pip install -r tests/requirements-test.txt")
    return EnglishTextNormalizer()


@pytest.fixture(scope="session")
def server_ready():
    """Skip the whole module if the server isn't reachable."""
    import httpx

    try:
        resp = httpx.get(f"{SERVER_URL.rstrip('/')}/readyz", timeout=5.0)
    except Exception as exc:
        pytest.skip(f"Server at {SERVER_URL} unreachable: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"Server at {SERVER_URL} not ready: {resp.status_code} {resp.text[:200]}")
    return SERVER_URL


def _collect_cases(manifest):
    records = _fixture_records(manifest, short_limit=SHORT_LIMIT, include_longform=INCLUDE_LONGFORM)
    cases = []
    for fixture in records:
        for strategy in REST_STRATEGIES + WS_STRATEGIES:
            # Skip WS strategies on the long-form fixtures by default — each
            # request has a ~30s teardown overhead that doesn't add useful
            # signal on a 10-25 min talk. Override with
            # PARAKEET_TEST_LONGFORM_WS=1 if you want them.
            if (
                fixture["kind"] == "longform"
                and strategy in WS_STRATEGIES
                and os.environ.get("PARAKEET_TEST_LONGFORM_WS", "0") != "1"
            ):
                continue
            cases.append((fixture, strategy))
    return cases


def _case_id(fixture: dict, strategy: str) -> str:
    return f"{fixture['kind']}::{fixture['id'][:24]}::{strategy}"


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        if not (FIXTURES / "manifest.json").exists():
            metafunc.parametrize("case", [], ids=[])
            return
        manifest = _load_manifest()
        cases = _collect_cases(manifest)
        ids = [_case_id(f, s) for f, s in cases]
        metafunc.parametrize("case", cases, ids=ids)


def test_strategy(case, normalizer, server_ready):
    fixture, strategy = case
    result = asyncio.run(_run_one(server_ready, fixture, strategy, normalizer, REQUEST_TIMEOUT_S))
    if result.error:
        pytest.fail(f"{strategy} on {fixture['id']} errored: {result.error}")
    assert result.wer is not None, "WER was None — reference text empty?"
    threshold = WER_THRESHOLDS[strategy][fixture["kind"]]
    assert result.wer <= threshold, (
        f"WER {result.wer * 100:.2f}% exceeds {threshold * 100:.1f}% threshold "
        f"for {fixture['kind']} fixture with strategy={strategy}.\n"
        f"  Reference (truncated): {fixture['text'][:200]!r}\n"
        f"  Hypothesis (truncated): {result.text[:200]!r}"
    )
