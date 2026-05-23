"""ASR regression / quality harness.

Drives the running Parakeet server (PARAKEET_URL env, default
http://localhost:8777) through every strategy against two datasets:

  - **short**: LibriSpeech test-clean per-utterance (up to 2620 clips).
    Aggregated to a single corpus-WER number per strategy, directly
    comparable to NVIDIA's published 1.69% on the same split.

  - **longform**: TED-LIUM 3 long-form (11 real TED talks, 5-25 min each).
    Per-talk WER; lets us see how `full` / `chunked_v2` / `progressive_v2`
    behave on natural long-form audio (no synthetic concat).

WER is computed via `EnglishTextNormalizer` (Whisper) + `jiwer.wer` — the
same combo the Open ASR Leaderboard uses.

Usage:

    python -m tests.download_fixtures
    PARAKEET_URL=http://192.168.0.172:8777 python -m tests.eval_harness

WS strategies (`progressive`, `progressive_v2`) have a ~30s per-request
teardown overhead, so the harness caps them at `--ws-limit` short fixtures
(default 50). REST strategies run on all of them.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

FIXTURES = Path(__file__).parent / "fixtures"
RESULTS = Path(__file__).parent / "results"

REST_STRATEGIES = ("full", "chunked", "chunked_v2")
WS_STRATEGIES = ("progressive", "progressive_v2")
ALL_STRATEGIES = REST_STRATEGIES + WS_STRATEGIES


def _import_deps():
    try:
        import jiwer  # noqa: F401
        import httpx  # noqa: F401
        import websockets  # noqa: F401
        from rich.console import Console  # noqa: F401
        from whisper_normalizer.english import EnglishTextNormalizer  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}. Install via `pip install -r tests/requirements-test.txt`."
        )


@dataclass
class FixtureResult:
    """One (strategy, fixture) cell."""

    fixture_id: str
    fixture_kind: str  # "short" or "longform"
    fixture_duration_s: float
    strategy: str
    wall_s: float
    wer: float | None
    text: str
    reference: str
    error: str | None = None
    first_partial_s: float | None = None  # WS only
    partials_received: int = 0  # WS only


@dataclass
class HarnessReport:
    server_url: str
    results: list[FixtureResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "server_url": self.server_url,
            "started_at": self.started_at,
            "results": [r.__dict__ for r in self.results],
        }


def _load_manifest() -> dict:
    manifest = FIXTURES / "manifest.json"
    if not manifest.exists():
        sys.exit(
            f"No fixtures at {manifest}. Run `python -m tests.download_fixtures` first."
        )
    return json.loads(manifest.read_text())


def _fixture_records(manifest: dict, short_limit: int | None, include_longform: bool) -> list[dict]:
    """Flat list of {id, path (absolute), duration_s, text, source, kind}."""
    records: list[dict] = []
    short = manifest["short"]
    if short_limit is not None:
        short = short[:short_limit]
    for s in short:
        records.append({**s, "path": str(FIXTURES / s["path"]), "kind": "short"})
    if include_longform:
        for l in manifest.get("longform", []):
            records.append({**l, "path": str(FIXTURES / l["path"]), "kind": "longform"})
    return records


def _wer(reference: str, hypothesis: str, normalizer) -> float | None:
    import jiwer

    ref_n = normalizer(reference).strip()
    hyp_n = normalizer(hypothesis).strip()
    if not ref_n:
        return None
    return float(jiwer.wer(ref_n, hyp_n))


def _corpus_wer(results: list[FixtureResult], normalizer) -> tuple[float | None, int, int]:
    """Pool all refs+hyps and compute one corpus-level WER (total_edits / total_ref_words).

    This is what NVIDIA / leaderboards report — NOT mean-of-per-utterance-WER.
    Returns (wer, total_ref_words, total_hyp_words).
    """
    import jiwer

    refs = []
    hyps = []
    for r in results:
        if r.error or r.wer is None:
            continue
        refs.append(normalizer(r.reference).strip())
        hyps.append(normalizer(r.text).strip())
    refs = [r for r in refs if r]
    hyps = hyps[: len(refs)]
    if not refs:
        return None, 0, 0
    total_ref_words = sum(len(r.split()) for r in refs)
    total_hyp_words = sum(len(h.split()) for h in hyps)
    return float(jiwer.wer(refs, hyps)), total_ref_words, total_hyp_words


async def _run_rest(
    url: str, audio_path: str, strategy: str, timeout_s: float
) -> tuple[str, float]:
    """POST one file. Returns (text, wall_seconds)."""
    import httpx

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            resp = await client.post(
                f"{url.rstrip('/')}/v1/audio/transcriptions",
                params={"strategy": strategy},
                files=files,
            )
    wall = time.monotonic() - t0
    if resp.status_code != 200:
        raise RuntimeError(
            f"POST returned {resp.status_code}: {resp.text[:500]}"
        )
    payload = resp.json()
    text = payload.get("text") or _segments_to_text(payload.get("segments") or [])
    return text, wall


def _segments_to_text(segments: list[dict]) -> str:
    return " ".join((s.get("text") or "").strip() for s in segments).strip()


def _wav_meta(path: str) -> int:
    with contextlib.closing(wave.open(path, "rb")) as w:
        return w.getframerate()


async def _run_ws(
    url: str, audio_path: str, strategy: str, timeout_s: float
) -> tuple[str, float, float | None, int]:
    """Stream a WAV file through the unified WS endpoint.

    Feeds as fast as the network allows — measures end-to-end quality, not
    real-time pacing.
    Returns (text, wall_seconds, first_partial_seconds, partials_count).
    """
    import websockets

    ws_url = url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    ws_endpoint = f"{ws_url}/v1/audio/transcriptions"
    sample_rate = _wav_meta(audio_path)
    audio_bytes = Path(audio_path).read_bytes()

    config = {
        "sample_rate": sample_rate,
        "channels": 1,
        "bytes_per_sample": 2,
        "format": "wav",
        "strategy": strategy,
    }

    t0 = time.monotonic()
    first_partial_at: float | None = None
    partials = 0
    final_text: str | None = None
    final_segments: list[dict] = []

    async with websockets.connect(ws_endpoint, max_size=2**26, ping_timeout=60) as ws:
        await ws.send(json.dumps(config))
        sender = asyncio.create_task(_stream_bytes(ws, audio_bytes, chunk=64 * 1024))
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype == "segments_batch":
                    if first_partial_at is None:
                        first_partial_at = time.monotonic() - t0
                    partials += 1
                elif mtype == "refined_transcription":
                    final_text = msg.get("text") or _segments_to_text(msg.get("segments") or [])
                    final_segments = msg.get("segments") or []
                elif mtype == "final_transcription" or mtype == "final":
                    if not final_text:
                        final_text = msg.get("text") or _segments_to_text(msg.get("segments") or [])
                        final_segments = msg.get("segments") or []
                    break
                elif mtype == "error":
                    raise RuntimeError(f"Server error: {msg.get('error')}")
        finally:
            sender.cancel()
            with contextlib.suppress(Exception):
                await sender

    wall = time.monotonic() - t0
    text = final_text if final_text is not None else _segments_to_text(final_segments)
    return text or "", wall, first_partial_at, partials


async def _stream_bytes(ws, data: bytes, chunk: int):
    for off in range(0, len(data), chunk):
        await ws.send(data[off : off + chunk])
    await ws.send(b"")  # EOF signal


async def _run_one(
    server_url: str,
    fixture: dict,
    strategy: str,
    normalizer,
    timeout_s: float,
) -> FixtureResult:
    first_partial_s = None
    partials = 0
    try:
        if strategy in REST_STRATEGIES:
            text, wall = await _run_rest(server_url, fixture["path"], strategy, timeout_s)
        else:
            text, wall, first_partial_s, partials = await _run_ws(
                server_url, fixture["path"], strategy, timeout_s
            )
        wer = _wer(fixture["text"], text, normalizer)
        return FixtureResult(
            fixture_id=fixture["id"],
            fixture_kind=fixture["kind"],
            fixture_duration_s=fixture["duration_s"],
            strategy=strategy,
            wall_s=wall,
            wer=wer,
            text=text,
            reference=fixture["text"],
            first_partial_s=first_partial_s,
            partials_received=partials,
        )
    except Exception as exc:
        return FixtureResult(
            fixture_id=fixture["id"],
            fixture_kind=fixture["kind"],
            fixture_duration_s=fixture["duration_s"],
            strategy=strategy,
            wall_s=0.0,
            wer=None,
            text="",
            reference=fixture["text"],
            error=f"{type(exc).__name__}: {exc}",
        )


def _print_tables(report: HarnessReport, strategies: list[str], normalizer) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # --- Aggregate short corpus WER ---
    agg = Table(title="Short corpus WER (LibriSpeech test-clean, vs NVIDIA's 1.69%)", show_lines=False)
    agg.add_column("Strategy", style="cyan")
    agg.add_column("Utterances", justify="right")
    agg.add_column("Ref words", justify="right")
    agg.add_column("WER %", justify="right", style="bold")
    agg.add_column("Wall time", justify="right")
    agg.add_column("RTFx", justify="right")
    for s in strategies:
        short_results = [r for r in report.results if r.strategy == s and r.fixture_kind == "short"]
        if not short_results:
            continue
        wer, total_ref, _total_hyp = _corpus_wer(short_results, normalizer)
        good = [r for r in short_results if not r.error]
        wall = sum(r.wall_s for r in good)
        audio = sum(r.fixture_duration_s for r in good)
        wer_str = f"{wer * 100:.2f}" if wer is not None else "n/a"
        rtfx = (audio / wall) if wall > 0 else 0
        agg.add_row(s, str(len(good)), str(total_ref), wer_str, f"{wall:.1f}s", f"{rtfx:.1f}x")
    console.print(agg)

    # --- Long-form per-talk WER ---
    longform_results = [r for r in report.results if r.fixture_kind == "longform"]
    if longform_results:
        by_fix: dict[str, dict[str, FixtureResult]] = {}
        for r in longform_results:
            by_fix.setdefault(r.fixture_id, {})[r.strategy] = r

        lf_table = Table(title="Long-form WER % (TED-LIUM 3 long-form, lower = better)", show_lines=True)
        lf_table.add_column("Talk", style="cyan", no_wrap=True)
        lf_table.add_column("Duration", justify="right")
        for s in strategies:
            lf_table.add_column(s, justify="right")
        rtfx_table = Table(title="Long-form RTFx", show_lines=True)
        rtfx_table.add_column("Talk", style="cyan", no_wrap=True)
        rtfx_table.add_column("Duration", justify="right")
        for s in strategies:
            rtfx_table.add_column(s, justify="right")

        for fid in sorted(by_fix):
            row_w = [fid]
            row_r = [fid]
            any_r = next(iter(by_fix[fid].values()))
            row_w.append(f"{any_r.fixture_duration_s:.0f}s")
            row_r.append(f"{any_r.fixture_duration_s:.0f}s")
            for s in strategies:
                r = by_fix[fid].get(s)
                if r is None:
                    row_w.append("-")
                    row_r.append("-")
                elif r.error:
                    row_w.append("[red]ERR[/red]")
                    row_r.append("[red]ERR[/red]")
                else:
                    row_w.append(f"{r.wer * 100:.2f}" if r.wer is not None else "n/a")
                    rtfx = r.fixture_duration_s / r.wall_s if r.wall_s > 0 else 0
                    row_r.append(f"{rtfx:.1f}x")
            lf_table.add_row(*row_w)
            rtfx_table.add_row(*row_r)
        console.print(lf_table)
        console.print(rtfx_table)

        # Long-form corpus WER
        agg_lf = Table(title="Long-form corpus WER (pooled across all talks)", show_lines=False)
        agg_lf.add_column("Strategy", style="cyan")
        agg_lf.add_column("Talks", justify="right")
        agg_lf.add_column("Ref words", justify="right")
        agg_lf.add_column("Audio (min)", justify="right")
        agg_lf.add_column("WER %", justify="right", style="bold")
        agg_lf.add_column("Wall (s)", justify="right")
        agg_lf.add_column("RTFx", justify="right")
        for s in strategies:
            rs = [r for r in longform_results if r.strategy == s]
            if not rs:
                continue
            wer, ref_words, _ = _corpus_wer(rs, normalizer)
            good = [r for r in rs if not r.error]
            wall = sum(r.wall_s for r in good)
            audio = sum(r.fixture_duration_s for r in good)
            wer_str = f"{wer * 100:.2f}" if wer is not None else "n/a"
            rtfx = (audio / wall) if wall > 0 else 0
            agg_lf.add_row(s, str(len(good)), str(ref_words), f"{audio/60:.1f}", wer_str, f"{wall:.1f}", f"{rtfx:.1f}x")
        console.print(agg_lf)

    # --- Errors ---
    errors = [r for r in report.results if r.error]
    if errors:
        err_table = Table(title="Errors", show_lines=False)
        err_table.add_column("Fixture", style="cyan")
        err_table.add_column("Strategy", style="magenta")
        err_table.add_column("Error", style="red")
        for r in errors[:30]:
            err_table.add_row(f"{r.fixture_kind}::{r.fixture_id[:40]}", r.strategy, (r.error or "")[:200])
        console.print(err_table)
        if len(errors) > 30:
            console.print(f"...and {len(errors) - 30} more errors (see results JSON)")


async def run(
    server_url: str,
    strategies: list[str],
    short_limit_rest: int | None,
    short_limit_ws: int | None,
    include_longform: bool,
    timeout_s: float,
) -> HarnessReport:
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()
    manifest = _load_manifest()

    # Strategies can have different short-fixture caps (REST is cheap;
    # WS has a ~30s per-request teardown that makes 2620 prohibitive).
    rest_records = _fixture_records(manifest, short_limit_rest, include_longform)
    ws_records = _fixture_records(manifest, short_limit_ws, include_longform)

    rest_strats = [s for s in strategies if s in REST_STRATEGIES]
    ws_strats = [s for s in strategies if s in WS_STRATEGIES]

    rest_total = len(rest_records) * len(rest_strats)
    ws_total = len(ws_records) * len(ws_strats)
    total = rest_total + ws_total
    report = HarnessReport(server_url=server_url)

    done = 0

    async def _execute(records, strats):
        nonlocal done
        # Iterate strategy-outer so model state warms in sequence per strategy.
        for strategy in strats:
            for fixture in records:
                done += 1
                if done == 1 or done % 50 == 0 or done == total:
                    print(f"[{done}/{total}] {strategy} on {fixture['kind']}::{fixture['id'][:40]}", flush=True)
                result = await _run_one(server_url, fixture, strategy, normalizer, timeout_s)
                report.results.append(result)
                if result.error:
                    print(f"  ! {strategy} on {fixture['id'][:40]}: {result.error[:120]}")

    await _execute(rest_records, rest_strats)
    await _execute(ws_records, ws_strats)

    return report, normalizer


def main() -> None:
    _import_deps()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default=os.environ.get("PARAKEET_URL", "http://localhost:8777"))
    parser.add_argument(
        "--strategies",
        default=",".join(ALL_STRATEGIES),
        help=f"Comma list. Default: all {ALL_STRATEGIES}.",
    )
    parser.add_argument(
        "--short-limit",
        type=int,
        default=None,
        help="Cap short fixtures for REST strategies (default: all in manifest, ~2620).",
    )
    parser.add_argument(
        "--ws-limit",
        type=int,
        default=50,
        help="Cap short fixtures for WS strategies due to ~30s per-request teardown (default 50).",
    )
    parser.add_argument(
        "--no-longform",
        action="store_true",
        help="Skip the long-form TED-LIUM fixtures.",
    )
    parser.add_argument("--timeout", type=float, default=900.0, help="Per-request timeout seconds.")
    parser.add_argument("--out", default=None, help="Optional JSON results dump path.")
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    bad = [s for s in strategies if s not in ALL_STRATEGIES]
    if bad:
        sys.exit(f"Unknown strategies: {bad}. Valid: {ALL_STRATEGIES}")

    report, normalizer = asyncio.run(
        run(
            server_url=args.server,
            strategies=strategies,
            short_limit_rest=args.short_limit,
            short_limit_ws=args.ws_limit,
            include_longform=not args.no_longform,
            timeout_s=args.timeout,
        )
    )

    _print_tables(report, strategies, normalizer)

    if args.out:
        out_path = Path(args.out)
    else:
        RESULTS.mkdir(exist_ok=True)
        out_path = RESULTS / f"run-{int(report.started_at)}.json"
    out_path.write_text(json.dumps(report.as_dict(), indent=2))
    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    main()
