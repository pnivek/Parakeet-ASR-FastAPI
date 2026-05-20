# Transcription strategies

A Parakeet Playground guide to picking the right strategy, what the model can and can't do, and where our server's defaults come from. This page eventually renders inside the SPA at `/docs/strategies`.

> **Source vs. server.** Every fact about the model itself is sourced from NVIDIA's [`nvidia/parakeet-tdt-0.6b-v2` model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and the upstream [NeMo](https://github.com/NVIDIA/NeMo) library. Everything about *strategies* (`auto` / `full` / `chunked` / `progressive`) and our defaults is our server's wrapper layer.

---

## TL;DR — pick a strategy

| Your situation | Use | Why |
|---|---|---|
| Offline file, ≤ 24 min, you want it done as fast as possible | `full` | Single-pass through the encoder; up to ~170× RTFx on a single stream. |
| Offline file, any length | `chunked` (REST default via `auto`) | Same engine as `full` driven over a sliding window, never OOMs, identical quality within noise. |
| Live mic / WebSocket streaming | `progressive` (WS default via `auto`) | Same engine again but emits partial segments as they commit. ~6 s emission lag. |
| Don't want to think about it | `auto` | REST → `chunked`. WS → `progressive`. Always works. |

If `progressive` is selected on a REST upload the server returns `400`; the client maps stale values to `chunked` defensively. (Why: see [Why we don't allow `progressive` on REST](#why-progressive-only-makes-sense-over-a-websocket).)

---

## What the model can physically do

Parakeet TDT-0.6B-v2 is a **FastConformer-TDT** model trained by NVIDIA. The encoder uses **full self-attention** in training. Decoding is a Token-and-Duration Transducer (greedy by default, no external LM).

Hard facts from NVIDIA's model card:

| Property | Value |
|---|---|
| Architecture | FastConformer encoder + TDT decoder |
| Parameters | 600 M |
| Attention | Full attention (no local/chunked attention in training) |
| Sample rate | 16 kHz mono |
| Supported audio I/O | wav / flac (mono) — our server also handles mp3, m4a, ogg, webm, etc. by piping through ffmpeg |
| **Max audio length, single pass** | **24 minutes** |
| Subsampling factor | 8 (an encoder frame ≈ 80 ms) |
| NVIDIA-published WER on LibriSpeech test-clean | 1.69 % |
| Supported timestamps | char, word, segment |

The 24-minute ceiling is real, not a soft limit. Full attention scales O(N²) in sequence length; above 24 minutes the model wasn't trained to attend that far and the attention matrix is too big regardless. **You cannot do a single FULL pass on longer audio**, no matter how much GPU memory you have.

---

## How the four strategies differ

All four strategies share the **same primitives**: the FastConformer encoder and `decoding_computer` (the TDT label-loop). They differ in how much audio the encoder sees per call, whether decoder state is threaded across calls, and when results go back to the client.

### `full` — single-pass offline

- Encoder runs once over the entire waveform.
- Decoder runs once with `prev_batched_state=None` — fresh start, drains everything.
- **No emission lag.** Tokens come back when the whole pass is done.
- Our `_transcribe_full` bypasses NeMo's `transcribe()` wrapper because of a [CUDA-graph bug on short audio](#avg_logprob-and-cuda-graphs) — but uses the same encoder + decoder.

**Quality**: 1.37 % WER on LibriSpeech test-clean (our measurement; better than NVIDIA's 1.69 % via `transcribe()` because we sidestep the wrapper bug).
**Throughput**: ~170× RTFx single-stream on a DGX Spark.
**Hard limit**: ≤ 24 min, or ≤ `MAX_FULL_WAVEFORM_S` (env, default 1440 s).

### `chunked` — sliding window, offline

- Audio is already in memory but processed through `StreamingPrevBatchedEngine` chunk-by-chunk.
- Each call to the engine sees a 25 s buffer: **10 s left context + 10 s chunk + 5 s right context** (configurable via `STREAMING_LEFT_CONTEXT_S` / `STREAMING_CHUNK_S` / `STREAMING_RIGHT_CONTEXT_S`).
- The middle 10 s is what gets *emitted* per call; the trailing 5 s is lookahead the decoder uses but doesn't commit (it commits on the next call).
- **Critically**, the decoder's `prev_batched_state` is threaded across chunk calls — the TDT label-loop state is carried, so sentence boundaries stay coherent and you don't get the cold-start corruption that naive splitting would produce.

**Quality**: 1.33 % WER on LibriSpeech test-clean. Within noise of `full`.
**Throughput**: ~65× RTFx single-stream.
**Limits**: handles anything from 1 s to multi-hour files. No upper bound.

### `progressive` — same engine, live I/O

Algorithmically identical to `chunked`. The differences are entirely about *when audio arrives and when results go back*:

- **Input** comes over a WebSocket as binary frames (whatever the browser captures — WebM/Opus from `MediaRecorder`, raw WAV, etc.). Each chunk is piped through `ffmpeg -f s16le -ac 1 -ar 16000` to normalize to 16 kHz mono PCM.
- **Output** is pushed back mid-stream as `segments_batch` messages — partials appear in the UI while the user is still talking.
- On EOF (empty binary frame), a `final_transcription` closes the stream.

**Emission lag**: ~7.5 s with the offline-like 10-10-5 preset, or ~6 s with the live 10-2-2 preset (`live_latency=true` swaps to `STREAMING_LIVE_CHUNK_S=2`, `STREAMING_LIVE_RIGHT_CONTEXT_S=2`). That lag is the model's architectural floor — the decoder needs that right-context to commit a token. **You cannot beat it without switching to a cache-aware streaming checkpoint** (e.g. NVIDIA's [`parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)).

### `auto` — server picks

- REST request → `chunked`.
- WebSocket request → `progressive`.

That's it. If you set the env `DEFAULT_STRATEGY` to something other than `auto`, that becomes the picked strategy.

---

## Why `progressive` only makes sense over a WebSocket

Two reasons, both about the HTTP request lifecycle:

1. **Input is one-shot on REST.** `multipart/form-data` delivers the entire body before the handler runs. There's nothing "streaming in" to feed a live producer. `await file.read()` returns the complete buffer at once.
2. **Output is one-shot on REST.** A single HTTP response, one body. There's no way to push `segments_batch` messages as they commit. Even with chunked transfer encoding you'd be streaming the *response*, but the *input* is still already complete.

Asking for `progressive` over REST would silently degrade to "wait until done, then return all segments at once" — which is exactly what `chunked` already does. The server returns `400` instead of doing the silent demotion because if a developer asks for `progressive`, they want live partials and need to know they're not getting them.

The Maison sidebar disables the `progressive` option outside Live mic mode. If a stale localStorage value picks it on REST anyway, the client transparently maps `progressive → chunked` at submit time.

---

## `progressive_refinement` — the EOF FULL pass

When `progressive_refinement=true` (default) and the total accumulated audio is ≤ `MAX_FULL_WAVEFORM_S` (24 min), the server:

1. Streams partials live as usual during recording.
2. On EOF, runs **one single FULL pass** over the entire accumulated PCM.
3. Replaces the streamed segments with the FULL output as a `refined_transcription` message.
4. Sends `final_transcription` to close.

**It is not a continuous refinement.** It's one shot at the end. The streamed partials are the "preview"; the FULL pass is the "final cut."

If the recording exceeds 24 min, refinement is silently skipped (logged) — FULL can't handle it. The streamed partials remain the final output.

---

## Why we can't just split a long file into FULL passes

The naive approach: chop a 1-hour file into three 20-min slices, run FULL on each, concat. **It would be fast — but quality at the joins is bad.** Two reasons:

1. **Where to cut**: splitting at second 1200 might chop mid-word. Silence detection is its own problem.
2. **Decoder cold-start at every split**: the TDT decoder is stateful. Three independent FULL passes mean three cold starts. The first 1–2 words after every join are statistically worse, and sentence boundaries get jagged.

`StreamingPrevBatchedEngine` (chunked / progressive) solves both:
- **Overlapping windows** mean no word ever falls inside a "missed" segment.
- **`prev_batched_state` threading** means the decoder believes it's doing one long decode that just happens to come in pieces.

Measured impact: chunked is 1.33 % WER vs. full's 1.37 % on LibriSpeech test-clean — essentially identical accuracy. Naive 20-min FULL splits would measurably regress at the joins.

The cost of doing it right is throughput: ~65× RTFx for chunked vs. ~170× RTFx for full. Worth it for long files; not worth it when full fits.

---

## Local attention above 8 minutes — off-spec but supported

Between 8 and 24 minutes, before we run FULL, the server calls:

```python
asr_model.change_attention_model("rel_pos_local_attn", [256, 256], True)
```

That switches the encoder from `rel_pos` (full attention) to `rel_pos_local_attn` with 256-frame left and right context (256 × 80 ms ≈ 20.5 s on each side). This is a NeMo runtime feature, not part of the model's training regime. The threshold is configurable via `LONG_AUDIO_THRESHOLD` (env, default 480 s).

**This is unofficial.** NVIDIA's model card does not mention local attention; the model was trained with full attention only. We use it because it lets FULL keep working between 8 and 24 minutes without OOMs on long buffers, and it's what NVIDIA's own HF Space does. Expect a small but real quality regression vs. true full attention on the same audio.

`chunked` and `progressive` also use this switch internally for their per-chunk encoder calls when the *original* file duration exceeds the threshold — same reasoning.

---

## Latency floor

The streaming presets come from NVIDIA's TDT recommendations:

| Preset | left | chunk | right | Emission lag | Note |
|---|---|---|---|---|---|
| 10-10-5 | 10 s | 10 s | 5 s | ~7.5 s | NeMo's "offline-like quality" recommendation. Default. |
| 10-2-2 | 10 s | 2 s | 2 s | ~6 s | NeMo's "live latency" preset. Slightly worse boundaries. |

`live_latency=true` in the WS config switches to 10-2-2. Anything lower means switching to a different model — `parakeet-tdt-0.6b-v2` was not trained for sub-second emission. If you genuinely need real-time captioning at < 1 s, use a cache-aware streaming checkpoint instead.

---

## Throughput — and why your numbers will differ from NVIDIA's

NVIDIA's published RTFx on the HuggingFace Open-ASR Leaderboard is **3,386×**. That's at **batch size 128** on their reference hardware, summed across a long evaluation. Our numbers (170× for full, 65× for chunked) are **single-stream throughput** — one audio file at a time on a DGX Spark.

Both numbers are real, they measure different things. If you batch 32 streams together on an A100, you'd see something between. The model card's note "RTFx may vary depending on dataset audio duration and batch size" is the official caveat.

GPU-dependent factors:
- **Memory** controls how much audio can be encoded in one pass. ≤ 24 min always works under the trained limits; larger memory just gives you headroom for batches.
- **Tensor cores** (Volta+) accelerate the encoder substantially. CPU-only inference works but is multi-x slower.
- **CUDA graphs** are pinned at startup (`USE_CUDA_GRAPHS=true` default). They give ~30 % decoder speedup but require the encoder output buffer to stay valid between calls — see [`avg_logprob` and CUDA graphs](#avg_logprob-and-cuda-graphs) for the side effect.

NVIDIA's [tested hardware list](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2): A10, A100, A30, H100, L4, L40, Turing T4, Volta V100. Anything Ampere or newer is comfortable; Volta still works.

---

## Tunable env vars

All optional. Defaults shown.

| Var | Default | Effect |
|---|---:|---|
| `DEFAULT_STRATEGY` | `auto` | What `?strategy=auto` resolves to. |
| `MAX_FULL_WAVEFORM_S` | `1440` (24 min) | Hard cap for `full`. Above this, `full` errors or auto-falls-back to `chunked`. |
| `LONG_AUDIO_THRESHOLD` | `480` (8 min) | Above this duration, the encoder is switched to `rel_pos_local_attn` for the session. |
| `STREAMING_LEFT_CONTEXT_S` | `10` | Left context for the offline-like preset. |
| `STREAMING_CHUNK_S` | `10` | Chunk length for the offline-like preset. |
| `STREAMING_RIGHT_CONTEXT_S` | `5` | Right context for the offline-like preset. |
| `STREAMING_LIVE_CHUNK_S` | `2` | Chunk length when `live_latency=true`. |
| `STREAMING_LIVE_RIGHT_CONTEXT_S` | `2` | Right context when `live_latency=true`. |
| `EARLY_BUFFER_TARGET_S` | `15` | PCM seconds buffered before the first WS partial emits. |
| `USE_CUDA_GRAPHS` | `true` | NeMo's FULL_GRAPH-mode CUDA-graph decoder. ~30 % speedup, with the `avg_logprob` caveat. |
| `BATCH_SIZE` | `4` | Default `decoding_computer` batch size. |
| `MAX_URL_BYTES` | 512 MB | Hard cap on URL-ingested audio. |

---

## `avg_logprob` and CUDA graphs

NeMo 2.7.3's `GreedyBatchedTDTLabelLoopingComputer` in FULL_GRAPH mode silently drops the per-token confidence side output, even when `confidence_cfg.preserve_token_confidence=True`. The captured graph doesn't track it.

Practical effect: when `USE_CUDA_GRAPHS=true` (default, 2× faster decode), every `avg_logprob` field in the `verbose_json` response is `null`. Set `USE_CUDA_GRAPHS=false` to populate it, at ~50 % decoder throughput.

`no_speech_prob` is always `null` regardless. Whisper computes it from a dedicated `<|nospeech|>` token; Parakeet TDT has no equivalent token, so there's nothing to derive it from. We refuse to substitute a heuristic.

The Maison segments-view metadata grid shows both as `null` with honest tooltips explaining why.

---

## Quality benchmarks

NVIDIA's published numbers (model card, via `transcribe()`):

| Dataset | WER % |
|---|---:|
| LibriSpeech test-clean | 1.69 |
| LibriSpeech test-other | 3.19 |
| TEDLIUM-v3 | 3.38 |
| SPGI Speech | 2.17 |
| GigaSpeech | 9.74 |
| VoxPopuli | 5.95 |
| Earnings-22 | 11.15 |
| AMI | 11.16 |
| Average | 6.05 |

Our internal harness (`tests/eval_harness.py`) on LibriSpeech test-clean:

| Strategy | WER % | RTFx (single-stream) |
|---|---:|---:|
| `full` (our wrapper-bypass path) | 1.37 | ~170× |
| `chunked` (10-10-5) | 1.33 | ~65× |

We measure slightly better than NVIDIA's `1.69 %` because we bypass NeMo's `transcribe()` wrapper, which has a [FULL_GRAPH CUDA-graph short-audio bug](#avg_logprob-and-cuda-graphs) that corrupts ~23 % of the short-audio tail.

Noise robustness (NVIDIA, MUSAN):

| SNR | WER % | Relative |
|---|---:|---:|
| Clean | 6.05 | — |
| 10 dB | 6.95 | −14.75 % |
| 5 dB | 8.23 | −35.97 % |
| 0 dB | 11.88 | −96.28 % |

Telephony (μ-law 8 kHz): 6.32 % vs. 6.05 % standard — only 4 % relative regression, robust to common downsampling.

---

## FAQ

**Why do you cap `full` at 24 min and not 30 or 60?**
24 minutes is what NVIDIA states in their model card. It's the longest single-pass duration the trained model supports. Above that, full attention either OOMs or produces incoherent output regardless of GPU.

**Can I raise `MAX_FULL_WAVEFORM_S` and try anyway?**
You can, but accuracy degrades and the encoder may OOM. The cap exists to prevent silent failures.

**Why is `chunked`'s WER lower than `full`'s in our table?**
Statistical noise. We're at the model's accuracy floor on a small test set; the 0.04 % gap is within run-to-run variance. The point is that they're functionally equivalent.

**Why is `progressive` not the default for REST too?**
`progressive` provides nothing on REST that `chunked` doesn't — the partials can't be delivered. Routing REST to `chunked` is the honest default; the server explicitly errors on `?strategy=progressive` over REST so a misconfiguration doesn't silently degrade.

**My audio is 30 minutes. What happens?**
`auto` → `chunked` handles it. `full` would error (above the 24-min cap). The server logs the dispatch decision with the resolved strategy.

**What if I want sub-second latency?**
You need a different checkpoint. The architectural floor for `parakeet-tdt-0.6b-v2` is ~6 s with the 10-2-2 preset. Look at [`parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) or [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b).

---

## References

- [Parakeet TDT 0.6B V2 model card (NVIDIA)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [Parakeet docs in HuggingFace Transformers](https://huggingface.co/docs/transformers/en/model_doc/parakeet)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — the upstream library
- [FastConformer paper](https://huggingface.co/papers/2305.05084)
- [HuggingFace Open-ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — RTFx benchmarks
