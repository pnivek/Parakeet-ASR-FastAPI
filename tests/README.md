# ASR regression / quality harness

Drives the running Parakeet server through every processing strategy against
two public datasets:

| Dataset | What | Why |
|---|---|---|
| **LibriSpeech test-clean** (2620 utterances) | Per-utterance read-speech, ~7.3h | Direct apples-to-apples comparison vs NVIDIA's published **1.69%** WER. Aggregated to one corpus-WER number per strategy. |
| **TED-LIUM 3 long-form** (11 talks, ~2.5h) | Real long-form recordings, 5-25 min/talk | Honest long-form WER. The Open ASR Leaderboard's canonical long-form benchmark. No synthetic concatenation. |

WER uses `EnglishTextNormalizer` (Whisper's text normalizer) + `jiwer` —
the same combo as the [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard).

## Quick start

```bash
python3 -m venv venv
venv/bin/pip install -r tests/requirements-test.txt

# One-time fixture download (~5 min, ~1 GB on disk)
venv/bin/python -m tests.download_fixtures

# Run the full eval against a deployment
PARAKEET_URL=http://192.168.0.172:8777 venv/bin/python -m tests.eval_harness

# Or pytest gate (fast: handful of short fixtures + long-form)
PARAKEET_URL=http://192.168.0.172:8777 venv/bin/pytest tests/test_strategies.py -v
```

## What each output column means

**Short corpus WER table** — pooled across all short fixtures. This is the
number to stack against NVIDIA's 1.69%. Computed as
`total_edits / total_reference_words` (true corpus WER, NOT mean-of-means).

**Long-form WER table** — per-talk WER for each strategy on the 11 TED talks.
Lets you see if `split_full` matches `full` on a 25-min file, or if some
talks degrade more than others.

**Long-form corpus WER** — pooled across all 11 talks. Strategy-level number
for long-form quality.

**RTFx** — `audio_duration / wall_time`. Higher is faster than real-time.

## Strategy coverage and per-strategy limits

| Strategy | Transport | Default short cap | Notes |
|---|---|---|---|
| `full` | REST POST | all 2620 | Single-pass encode + decode of the full waveform |
| `split_full` | REST POST | all 2620 | Sequential FULL passes over slices with seam-stitching |
| `streaming` | WS stream | 50 (`--ws-limit`) | ffmpeg PCM stream into `StreamingPrevBatchedEngine` |

WS `streaming` has ~30s of per-request teardown overhead (ffmpeg flush +
right-context wait), so running it on 2620 utterances is 22h of wall time
for no extra signal. The harness caps it at `--ws-limit` (default 50).

For long-form, WS `streaming` is excluded by default (a 25-min talk through
WS adds 30s overhead but the talk itself takes 4+ min anyway — fine in
theory, but not informative when we already have REST numbers). Override by
running with `--strategies streaming` explicitly.

## Pytest thresholds

`test_strategies.py` uses per-`(strategy, fixture-kind)` ceilings. Defaults:

| Strategy | short (≤30s LibriSpeech) | longform (5-25 min TED talks) |
|---|---|---|
| `full` | 10% | 15% |
| `split_full` | 10% | 15% |
| `streaming` | 20% | 15% |

Parakeet's NVIDIA-reported short-clip number on LibriSpeech-clean is 1.69%,
so any short-clip failure on `full` / `split_full` indicates a real engine
bug. The `streaming` short headroom (20%) is wider because its first chunk's
right context is silence padding, which can flip one phoneme on a 3-5s
utterance (one wrong word = 10-15% WER at this length).

For an exhaustive comparison vs NVIDIA's published number, use the harness
CLI directly (not pytest) with no `--short-limit` — that runs all 2620
utterances and prints the corpus WER you can stack against 1.69%.

## Environment knobs

| Var | Default | Effect |
|---|---|---|
| `PARAKEET_URL` | `http://localhost:8777` | Server to drive |
| `PARAKEET_TEST_TIMEOUT_S` | `900` | Per-request timeout |
| `PARAKEET_TEST_INCLUDE_LONGFORM` | `1` | Include long-form fixtures in pytest |
| `PARAKEET_TEST_SHORT_LIMIT` | `5` | How many short fixtures pytest tests |
| `PARAKEET_TEST_LONGFORM_WS` | `0` | Run WS strategies on long-form too |

CLI equivalents on `eval_harness.py`: `--server`, `--timeout`,
`--no-longform`, `--short-limit`, `--ws-limit`, `--strategies`.

## Disk

After running `download_fixtures.py` with defaults:

```
tests/fixtures/
  manifest.json         ~1 MB
  short/*.wav           2620 files, ~580 MB total
  longform/*.wav        11 files, ~280 MB total
                        ~860 MB total
```

`tests/fixtures/` and `tests/results/` are gitignored.

## Known limitations

1. **`full` errors above MAX_FULL_WAVEFORM_S** (default 1440s server-side).
   The TED `BillGates` talk is 1506s — explicit `?strategy=full` against it
   will error. `split_full` handles it via sequential FULL passes. The harness
   records the error rather than crashing.
2. **The harness measures correctness end-to-end, not real-time pacing.**
   It feeds WS bytes as fast as the network allows; "emission lag" numbers
   reflect server processing, not realistic client throttling.
3. **No legacy long-form fixture.** We removed the synthetic concat-30 /
   concat-120 files because they introduce unnatural transitions the model
   wasn't trained on and aren't comparable to any published number. TED-LIUM
   serves the long-form role properly.
