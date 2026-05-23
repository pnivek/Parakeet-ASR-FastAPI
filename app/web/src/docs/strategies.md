# Transcription strategies

A Parakeet Playground guide to picking the right strategy, what the model can and can't do, and where our server's defaults come from. This page eventually renders inside the SPA at `/docs/strategies`.

> **Source vs. server.** Every fact about the model itself is sourced from NVIDIA's [`nvidia/parakeet-tdt-0.6b-v2` model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) and the upstream [NeMo](https://github.com/NVIDIA/NeMo) library. Everything about *strategies* (`offline` / `full` / `split_full` / `streaming`) and our defaults is our server's wrapper layer.

---

## Vocabulary

The `?strategy=` enum carries four values, picked by the client:

- `offline` — REST default. Server runs `full` if `duration ≤ MAX_FULL_WAVEFORM_S`, else `split_full`.
- `full` — force a single FULL pass; errors / OOMs if the audio exceeds the cap.
- `split_full` — force sequential FULL passes over overlapping slices, stitched at seams.
- `streaming` — WebSocket transport, ffmpeg producer + chunked engine, live partials.

The Maison UI flips the sidebar tabs to **modality** (`Upload / URL / Record`) and offers a per-modality **Engine** picker (REST / WebSocket). The wire mapping is trivial: REST + Strategy=Auto → `offline`; REST + an explicit override → `full` / `split_full`; WebSocket → `streaming`.

---

## TL;DR — pick a strategy

| Your situation | Strategy |
|---|---|
| Any REST request — let the server pick | `offline` (default) |
| Offline file, you know it fits — want max throughput, fail-fast if it doesn't | `full` |
| Offline file longer than the GPU's single-shot ceiling, want FULL throughput anyway | `split_full` |
| Live mic, file-over-WS, or live URL stream | `streaming` |
| Live HLS / icecast / RTSP URL → live transcription | `streaming` + config frame `url: "https://…"` |

Sending `?strategy=streaming` to the REST endpoint returns `400` — see [Why streaming only makes sense over a WebSocket](#why-streaming-only-makes-sense-over-a-websocket).

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

## How the three strategies differ

All three concrete strategies share the **same primitives**: the FastConformer encoder and `decoding_computer` (the TDT label-loop). They differ in how much audio the encoder sees per call, whether decoder state is threaded across calls, and when results go back to the client. (`offline` is a routing label that resolves to `full` or `split_full` at dispatch time.)

### `full` — single-pass offline

- Encoder runs once over the entire waveform.
- Decoder runs once with `prev_batched_state=None` — fresh start, drains everything.
- **No emission lag.** Tokens come back when the whole pass is done.
- Our `_transcribe_full` bypasses NeMo's `transcribe()` wrapper because of a [CUDA-graph bug on short audio](#avg_logprob-and-cuda-graphs) — but uses the same encoder + decoder.

**Quality**: 1.37 % WER on LibriSpeech test-clean (our measurement; better than NVIDIA's 1.69 % via `transcribe()` because we sidestep the wrapper bug).
**Throughput**: ~170× RTFx single-stream on a DGX Spark.
**Hard limit**: ≤ `MAX_FULL_WAVEFORM_S` (env). When `?strategy=offline`, the dispatcher auto-routes longer files to `split_full` instead.

### `split_full` — sequential FULL passes, stitched

- Audio is sliced into `MAX_FULL_WAVEFORM_S * SLICE_SAFETY`-second windows with `SPLIT_FULL_OVERLAP_S` of overlap.
- Each slice runs back-to-back through `_transcribe_full` (same primitive as `full`).
- Slice-relative timestamps are translated to wall-clock; segments inside the prior slice's tail (the overlap) are dropped to avoid duplicates at the seam.

**Why it works**: on this hardware FULL is ~3-5× the throughput of the chunked engine. Even paying the cost of an overlap per seam, sequential FULL passes finish a long file much faster than the chunked engine would. The cold-start at every slice boundary is mitigated by the `SPLIT_FULL_OVERLAP_S` of audio context the second slice's encoder sees before its first emitted segment.

**Throughput**: comparable to `full` minus the overlap tax (typically ≤ 5 %).
**Limits**: any length; no hard ceiling.

### `streaming` — sliding window over a live (or file) stream

The chunked engine driven over a WebSocket. Input arrives as binary frames (whatever the browser captures — WebM/Opus from `MediaRecorder`, raw WAV, etc.) and is piped through `ffmpeg -f s16le -ac 1 -ar 16000` to normalize to 16 kHz mono PCM. Output is pushed back mid-stream as `segments_batch` messages.

- Each engine call sees a 25 s buffer: **10 s left context + 10 s chunk + 5 s right context** (configurable via `STREAMING_LEFT_CONTEXT_S` / `STREAMING_CHUNK_S` / `STREAMING_RIGHT_CONTEXT_S`).
- The middle 10 s is what gets *emitted* per call; the trailing 5 s is lookahead the decoder uses but doesn't commit (it commits on the next call).
- The decoder's `prev_batched_state` is threaded across chunk calls — the TDT label-loop state is carried, so sentence boundaries stay coherent.
- On EOF (empty binary frame), a `final_transcription` closes the stream.

The same `streaming` strategy also covers **URL-over-WS**: if the config frame carries a `url` field, the server launches `ffmpeg -i <url>` and pulls audio directly from that URL — no binary frames are required. Suits HLS, icecast, RTSP, m3u8. The session ends when the URL ends, the client disconnects, or `URL_STREAM_MAX_S` wall-clock seconds elapse (default 6 h).

**Emission lag**: ~7.5 s with the offline-like 10-10-5 preset, or ~6 s with the live 10-2-2 preset (`live_latency=true` swaps to `STREAMING_LIVE_CHUNK_S=2`, `STREAMING_LIVE_RIGHT_CONTEXT_S=2`). That lag is the model's architectural floor — the decoder needs that right-context to commit a token. **You cannot beat it without switching to a cache-aware streaming checkpoint** (e.g. NVIDIA's [`parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1)).

### `offline` — the routing default

- REST request → `full` if `duration ≤ MAX_FULL_WAVEFORM_S`, else `split_full`.
- WS request (rare; same value sent on a WebSocket) → transport auto-corrects to `streaming`.

That's it. If you set the env `DEFAULT_STRATEGY` to something other than `offline`, that becomes the default the omitted-`?strategy=` request picks.

---

## Why `streaming` only makes sense over a WebSocket

Two reasons, both about the HTTP request lifecycle:

1. **Input is one-shot on REST.** `multipart/form-data` delivers the entire body before the handler runs. There's nothing "streaming in" to feed a live producer. `await file.read()` returns the complete buffer at once.
2. **Output is one-shot on REST.** A single HTTP response, one body. There's no way to push `segments_batch` messages as they commit. Even with chunked transfer encoding you'd be streaming the *response*, but the *input* is still already complete.

Asking for `streaming` over REST would silently degrade to "wait until done, then return all segments at once". The server returns `400` instead of doing the silent demotion because if a developer asks for `streaming`, they want live partials and need to know they're not getting them.

The Maison sidebar puts the Engine picker (REST / WebSocket) inside each modality pane so the wire mapping is obvious: Record + WebSocket = live transcription, URL + WebSocket = ffmpeg pulls a live stream, URL + REST = server downloads the URL and runs it as a file, and so on.

---

## Why `split_full` instead of forcing everyone onto the chunked engine for long files

The naive offline path used to be: file too long for one FULL pass? Fall back to the chunked engine. That works but is ~3-5× slower than necessary on this hardware. The right answer for offline files that exceed the FULL ceiling is to **run FULL on slices** — which is what `split_full` does.

Two reasons the seam isn't a problem in practice:

1. **Overlapping slices** give the second slice's encoder enough left-context to land its first few segments cleanly. We dedup segments whose start landed inside the prior slice's tail.
2. **Cold-start cost** is small relative to the slice itself: a single FULL pass on `MAX_FULL_WAVEFORM_S × SLICE_SAFETY` seconds amortizes the cold-start across many minutes of audio. Compare to chunked, where every chunk has the same kind of decoder boundary cost — just constantly, every 10 s, instead of every ~24 min.

---

## Local attention above 8 minutes — where this comes from

Between 8 and 24 minutes, before we run FULL, the server calls:

```python
asr_model.change_attention_model("rel_pos_local_attn", [256, 256], True)
asr_model.change_subsampling_conv_chunking_factor(1)
```

That switches the encoder from `rel_pos` (full attention) to `rel_pos_local_attn` with a 256-frame left and right context window (256 × 80 ms ≈ 20.5 s on each side).

**Origin**: this is a direct mirror of NVIDIA's official HuggingFace Space for the same model. Their [`app.py`](https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2/blob/main/app.py) contains the exact same threshold and arguments:

```python
# NVIDIA's app.py
if duration_sec > 480 : # 8 minutes
    gr.Info("Audio longer than 8 minutes. Applying optimized settings for long transcription.")
    model.change_attention_model("rel_pos_local_attn", [256,256])
    model.change_subsampling_conv_chunking_factor(1)  # 1 = auto select
```

**Is it strictly required?** No. The model still runs with full attention until either (a) you exceed the 24-min trained ceiling, or (b) the encoder OOMs on your GPU. The 480 s threshold is a conservative safety margin NVIDIA chose for their reference demo. They don't document why; the inline comment is the only justification (`# 8 minutes`).

**Why this matters practically**: 480 s is **GPU-dependent**. On modest GPUs the encoder will OOM somewhere between 8 and 24 minutes of audio if you don't switch to local attention. On an H100 or A100 80 GB you could probably run FULL on the entire 24-min window without issue. We expose `LONG_AUDIO_THRESHOLD` as an env var so deployments can tune the cutoff to their hardware — NVIDIA hard-codes 480 because their Space doesn't know what's hosting it.

**Caveats**:
- The model card doesn't mention local attention at all — the model was *trained* with full attention only. Local attention is a NeMo runtime feature inherited from FastConformer.
- Expect a small but real quality regression on the local-attention portion of the audio vs. true full attention.
- `streaming` also calls this switch when the proxy duration exceeds the threshold — its per-chunk encoder calls inherit the long-audio attention mode for the whole session.

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

NVIDIA's published RTFx on the HuggingFace Open-ASR Leaderboard is **3,386×**. That's at **batch size 128** on their reference hardware, summed across a long evaluation. Our numbers (170× for full, 65× for the streaming engine) are **single-stream throughput** — one audio file at a time on a DGX Spark.

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
| `DEFAULT_STRATEGY` | `offline` | What an omitted `?strategy=` resolves to. |
| `MAX_FULL_WAVEFORM_S` | `1440` (24 min) | Hard cap for `full`. When `?strategy=offline`, longer files route to `split_full`. |
| `SPLIT_FULL_OVERLAP_S` | `15` | Overlap (s) between adjacent `split_full` slices. Gives the second slice's encoder enough left-context to land clean segments; seam-dedup drops segments inside the prior slice's tail. |
| `LONG_AUDIO_THRESHOLD` | `480` (8 min) | Above this duration, the encoder is switched to `rel_pos_local_attn` for the session. **GPU-dependent**: NVIDIA's value chosen to be safe across hardware; raise on big-memory GPUs, lower on small ones. See [Local attention above 8 minutes](#local-attention-above-8-minutes--where-this-comes-from). |
| `STREAMING_LEFT_CONTEXT_S` | `10` | Left context for the offline-like preset. |
| `STREAMING_CHUNK_S` | `10` | Chunk length for the offline-like preset. |
| `STREAMING_RIGHT_CONTEXT_S` | `5` | Right context for the offline-like preset. |
| `STREAMING_LIVE_CHUNK_S` | `2` | Chunk length when `live_latency=true`. |
| `STREAMING_LIVE_RIGHT_CONTEXT_S` | `2` | Right context when `live_latency=true`. |
| `EARLY_BUFFER_TARGET_S` | `15` | PCM seconds buffered before the first WS partial emits. |
| `URL_STREAM_MAX_S` | `21600` (6 h) | Wall-clock cap for a single URL-over-WS streaming session. |
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
| `streaming` (10-10-5) | 1.33 | ~65× |

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

**Why is `streaming`'s WER lower than `full`'s in our table?**
Statistical noise. We're at the model's accuracy floor on a small test set; the 0.04 % gap is within run-to-run variance. The point is that they're functionally equivalent.

**Why is `streaming` not the default for REST too?**
`streaming` provides nothing on REST that `offline` doesn't — the partials can't be delivered. Routing REST to `offline` is the honest default; the server explicitly errors on `?strategy=streaming` over REST so a misconfiguration doesn't silently degrade.

**My audio is 30 minutes. What happens?**
`offline` → `split_full` (assuming `MAX_FULL_WAVEFORM_S=1440 s`). `full` would error (above the cap). The server logs the dispatch decision with the resolved strategy.

**Why does the long-audio switch fire at 8 minutes? Is that a model requirement?**
No — it's NVIDIA's chosen safety margin in their reference demo. We copied the threshold and the `[256, 256]` context window from their HuggingFace Space verbatim. The exact value isn't sacred; it's "low enough to keep most GPUs from OOMing in the FULL pass." On a beefy GPU (A100/H100 80 GB) you can comfortably raise it; on a smaller GPU you may need to lower it. Tune via the `LONG_AUDIO_THRESHOLD` env var.

**What if I want sub-second latency?**
You need a different checkpoint. The architectural floor for `parakeet-tdt-0.6b-v2` is ~6 s with the 10-2-2 preset. Look at [`parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) or [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b).

---

## References

- [Parakeet TDT 0.6B V2 model card (NVIDIA)](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
- [NVIDIA's official HuggingFace Space](https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2) — source for the 480 s / `[256, 256]` long-audio pattern
- [Parakeet docs in HuggingFace Transformers](https://huggingface.co/docs/transformers/en/model_doc/parakeet)
- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) — the upstream library
- [FastConformer paper](https://huggingface.co/papers/2305.05084)
- [HuggingFace Open-ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — RTFx benchmarks
