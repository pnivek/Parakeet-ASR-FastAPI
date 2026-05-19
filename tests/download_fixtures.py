"""Download ASR regression / quality fixtures to tests/fixtures/.

Two datasets, both public on HuggingFace Hub (no auth required):

  - **Short**: LibriSpeech test-clean, the canonical short-utterance benchmark.
    Up to 2620 utterances (the full test set). Default: all of them — this is
    what we score against NVIDIA's published 1.69% number per-utterance.

  - **Long-form**: TED-LIUM 3 long-form (`distil-whisper/tedlium-long-form`),
    11 full TED talks (5-25 min each, ~2.5h total) with single reference
    transcript per talk. Real long-form audio — not synthetic concatenation —
    used by the Open ASR Leaderboard for long-form WER. References are
    lowercase / no-punct (TED-LIUM convention); the Whisper normalizer
    handles this at scoring time.

Downloads bypass the `datasets` library (broken on Python 3.14 due to a
dill incompatibility) and read parquet shards directly via huggingface_hub
+ pyarrow.

After first run, `tests/fixtures/manifest.json` is:

    {
      "sample_rate": 16000,
      "short":    [{id, path, duration_s, text, source}, ...],
      "longform": [{id, path, duration_s, text, source}, ...]
    }

Re-running is idempotent; existing fixtures are kept unless --force is passed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_RATE = 16000

LIBRISPEECH_REPO = "openslr/librispeech_asr"
LIBRISPEECH_PARQUET = "clean/test/0000.parquet"

TEDLIUM_REPO = "distil-whisper/tedlium-long-form"
TEDLIUM_PARQUET = "data/test-00000-of-00001-7a1bb92f62e929b8.parquet"


def _import_deps():
    try:
        import huggingface_hub  # noqa: F401
        import numpy as np  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        import soundfile as sf  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"Missing dependency: {exc}. Install via `pip install -r tests/requirements-test.txt`."
        )


def _save_wav(path: Path, samples_f32, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples_f32, sample_rate, subtype="PCM_16")


def _download_short(n: int) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    print(f"Fetching {LIBRISPEECH_REPO}:{LIBRISPEECH_PARQUET} ...")
    parquet_path = hf_hub_download(LIBRISPEECH_REPO, LIBRISPEECH_PARQUET, repo_type="dataset")
    table = pq.read_table(parquet_path)
    print(f"  parquet has {table.num_rows} rows; saving first {n}.")
    rows = table.slice(0, min(n, table.num_rows)).to_pylist()

    items: list[dict] = []
    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate {sr} for {row['id']}")
        rel = f"short/{row['id']}.wav"
        _save_wav(FIXTURES / rel, samples.astype(np.float32, copy=False), SAMPLE_RATE)
        items.append(
            {
                "id": row["id"],
                "path": rel,
                "duration_s": float(len(samples)) / SAMPLE_RATE,
                "text": row["text"].strip(),
                "source": "librispeech-test-clean",
            }
        )
        if (i + 1) % 200 == 0:
            print(f"  saved {i + 1}/{len(rows)} short fixtures")
    print(f"  saved {len(items)} short fixtures")
    return items


def _download_longform(n: int) -> list[dict]:
    from huggingface_hub import hf_hub_download
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf

    print(f"Fetching {TEDLIUM_REPO}:{TEDLIUM_PARQUET} ...")
    parquet_path = hf_hub_download(TEDLIUM_REPO, TEDLIUM_PARQUET, repo_type="dataset")
    table = pq.read_table(parquet_path)
    print(f"  parquet has {table.num_rows} talks; saving first {n}.")
    rows = table.slice(0, min(n, table.num_rows)).to_pylist()

    items: list[dict] = []
    for i, row in enumerate(rows):
        samples, sr = sf.read(io.BytesIO(row["audio"]["bytes"]))
        if sr != SAMPLE_RATE:
            # TED-LIUM is natively 16k, but guard anyway.
            raise RuntimeError(f"Unexpected sample rate {sr} for {row['speaker_id']}")
        speaker = row["speaker_id"]
        rel = f"longform/{speaker}.wav"
        _save_wav(FIXTURES / rel, samples.astype(np.float32, copy=False), SAMPLE_RATE)
        items.append(
            {
                "id": speaker,
                "path": rel,
                "duration_s": float(len(samples)) / SAMPLE_RATE,
                "text": row["text"].strip(),
                "source": "tedlium3-long-form",
            }
        )
        print(f"  saved {speaker}: {items[-1]['duration_s']:.0f}s ({len(items[-1]['text'].split())} ref words)")
    return items


def download(n_short: int, n_longform: int, force: bool) -> dict:
    """Download fixtures, returning the manifest dict."""
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest_path = FIXTURES / "manifest.json"

    if manifest_path.exists() and not force:
        print(f"manifest already exists at {manifest_path}. Use --force to refresh.")
        return json.loads(manifest_path.read_text())

    short_items = _download_short(n_short)
    longform_items = _download_longform(n_longform)

    manifest = {
        "sample_rate": SAMPLE_RATE,
        "short": short_items,
        "longform": longform_items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    total_dur = sum(s["duration_s"] for s in short_items) + sum(l["duration_s"] for l in longform_items)
    print(
        f"\nWrote {len(short_items)} short fixtures + {len(longform_items)} long-form talks "
        f"({total_dur / 60:.1f} min total audio)."
    )
    print(f"Manifest: {manifest_path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-short",
        type=int,
        default=2620,
        help="LibriSpeech test-clean utterances (default 2620 = full test set).",
    )
    parser.add_argument(
        "--n-longform",
        type=int,
        default=11,
        help="TED-LIUM long-form talks (default 11 = full test set).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if manifest exists.")
    args = parser.parse_args()
    _import_deps()
    download(args.n_short, args.n_longform, args.force)


if __name__ == "__main__":
    main()
