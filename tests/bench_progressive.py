"""Backend-only progressive-stream throughput probe.

Drives the deployed Parakeet WS endpoint with a single audio file as fast
as the network allows, with explicit control over `vad_enabled`. Reports
total wall time, audio duration, RTFx, time-to-first-partial, and the
count of segments / peaks / refined messages received.

Removes every UI variable (no MediaRecorder, no decodeAudioData, no
React state) so we can attribute the perceived slowness to the server
vs the client.

Usage:
    PARAKEET_URL=http://192.168.0.172:8777 \
      venv/bin/python -m tests.bench_progressive \
      --fixture GaryFlake --vad off
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
import wave
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"


def _wav_meta(path: str) -> tuple[int, float]:
    with contextlib.closing(wave.open(path, "rb")) as w:
        rate = w.getframerate()
        frames = w.getnframes()
    return rate, frames / float(rate)


async def run(server_url: str, audio_path: Path, vad_enabled: bool, chunk_kb: int) -> dict:
    import websockets

    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
    endpoint = f"{ws_url}/v1/audio/transcriptions"
    sample_rate, audio_duration_s = _wav_meta(str(audio_path))
    audio_bytes = audio_path.read_bytes()

    config = {
        "sample_rate": sample_rate,
        "channels": 1,
        "bytes_per_sample": 2,
        "format": "wav",
        "strategy": "progressive",
        "progressive_refinement": False,
        "vad_enabled": vad_enabled,
    }

    t0 = time.monotonic()
    first_partial_at: float | None = None
    counts = {"segments_batch": 0, "peaks": 0, "refined_transcription": 0}
    final_text: str | None = None

    async with websockets.connect(endpoint, max_size=2**26, ping_timeout=60) as ws:
        await ws.send(json.dumps(config))

        async def _send_all() -> None:
            chunk = chunk_kb * 1024
            for off in range(0, len(audio_bytes), chunk):
                await ws.send(audio_bytes[off : off + chunk])
            await ws.send(b"")  # documented EOF

        sender = asyncio.create_task(_send_all())
        send_done_at: float | None = None
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=600.0)
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                mtype = msg.get("type")
                if mtype in counts:
                    counts[mtype] += 1
                    if mtype == "segments_batch" and first_partial_at is None:
                        first_partial_at = time.monotonic() - t0
                elif mtype == "final_transcription":
                    final_text = msg.get("text") or ""
                    break
                elif mtype == "error":
                    raise RuntimeError(f"server error: {msg.get('error')}")
                if send_done_at is None and sender.done():
                    send_done_at = time.monotonic() - t0
        finally:
            sender.cancel()
            with contextlib.suppress(Exception):
                await sender

    wall = time.monotonic() - t0
    return {
        "audio_duration_s": audio_duration_s,
        "wall_s": wall,
        "rtfx": audio_duration_s / wall if wall > 0 else 0.0,
        "first_partial_s": first_partial_at,
        "send_done_s": send_done_at,
        "segments_batches": counts["segments_batch"],
        "peaks_messages": counts["peaks"],
        "refined_messages": counts["refined_transcription"],
        "final_chars": len(final_text or ""),
        "vad_enabled": vad_enabled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server",
        default=os.environ.get("PARAKEET_URL", "http://localhost:8777"),
    )
    parser.add_argument(
        "--fixture",
        default="GaryFlake",
        help="Long-form fixture id (without .wav). See tests/fixtures/longform/.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Override fixture lookup with an explicit WAV path.",
    )
    parser.add_argument(
        "--vad",
        choices=("on", "off", "both"),
        default="both",
        help="Which VAD mode(s) to benchmark.",
    )
    parser.add_argument("--chunk-kb", type=int, default=64)
    args = parser.parse_args()

    if args.path:
        audio = Path(args.path)
    else:
        audio = FIXTURES / "longform" / f"{args.fixture}.wav"
    if not audio.exists():
        sys.exit(f"Audio not found: {audio}")

    modes: list[bool]
    if args.vad == "on":
        modes = [True]
    elif args.vad == "off":
        modes = [False]
    else:
        modes = [False, True]

    print(f"server : {args.server}")
    print(f"audio  : {audio} ({audio.stat().st_size} bytes)")
    print()

    rows: list[dict] = []
    for vad in modes:
        label = "vad ON " if vad else "vad OFF"
        print(f"== {label} ==", flush=True)
        result = asyncio.run(run(args.server, audio, vad, args.chunk_kb))
        rows.append(result)
        print(
            f"  audio_duration : {result['audio_duration_s']:.1f}s\n"
            f"  wall           : {result['wall_s']:.2f}s\n"
            f"  RTFx           : {result['rtfx']:.2f}x\n"
            f"  first partial  : {result['first_partial_s']:.2f}s (after stream open)\n"
            f"  send finished  : {result['send_done_s']:.2f}s\n"
            f"  segments_batch : {result['segments_batches']}\n"
            f"  peaks msgs     : {result['peaks_messages']}\n"
            f"  refined msgs   : {result['refined_messages']}\n"
            f"  text chars     : {result['final_chars']}",
            flush=True,
        )
        print()

    if len(rows) == 2:
        off, on = rows
        speedup = (on["wall_s"] / off["wall_s"]) if off["wall_s"] > 0 else 0.0
        print(f"vad-ON / vad-OFF wall ratio: {speedup:.2f}x  (>1 means VAD is slower)")


if __name__ == "__main__":
    main()
