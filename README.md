# Parakeet ASR

A FastAPI wrapper around NVIDIA's `parakeet-tdt-0.6b-v2` covering three workloads — short-batch, long-batch, and buffered-streaming WebSocket — behind one strategy dispatcher.

> ### A note on "streaming"
> `parakeet-tdt-0.6b-v2` is an **offline** model. NVIDIA trained it with full attention for a 24-minute single-pass ceiling — it is not a cache-aware streaming model. Riva does not offer this model as a streaming endpoint ([NVIDIA forum](https://forums.developer.nvidia.com/t/support-parakeet-tdt-0-6b-v2-en/337223)). What this project calls `progressive` is **buffered streaming with a ~4–15 s emission lag** (the model needs right context before committing tokens). If you need true low-latency live ASR, switch checkpoints to a cache-aware streaming variant such as [`parakeet_realtime_eou_120m-v1`](https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1) or [`nemotron-speech-streaming-en-0.6b`](https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b). For everything else — batch transcription, near-real-time captioning, multi-hour archives — this server covers it.

## Features

- **State-of-the-art offline ASR** — `nvidia/parakeet-tdt-0.6b-v2` (Token-and-Duration Transducer).
- **Three processing strategies**, one decoding pipeline (encoder + `decoding_computer` with optional `prev_batched_state` threading):
  - `chunked` *(REST default via `auto`)* — offline waveform fed chunk-by-chunk through NVIDIA's `StreamingBatchedAudioBuffer + decoding_computer + prev_batched_state` engine. Sentence-bounded segments, works across the full duration range.
  - `progressive` *(WS default via `auto`)* — same engine driven over an ffmpeg PCM stream, emitting sentence-bounded partials as new tokens commit.
  - `full` — single-pass encode + decode of the whole waveform. Fastest on long offline audio (~170× RTFx vs `chunked`'s ~65×), matches NVIDIA's published 1.69 % WER on LibriSpeech test-clean. Bypasses NeMo's `transcribe()` wrapper to dodge a FULL_GRAPH-mode CUDA-graph short-audio bug.
- **Whisper-compatible response shape** — segments follow OpenAI Whisper's `verbose_json` (`id`, `seek`, `start`, `end`, `text`, `tokens`, `temperature`, `avg_logprob`, `compression_ratio`, `no_speech_prob`). Top-level fields include `task`, `language`, `duration` alongside our extension fields (strategy, csv_content, srt_content, etc.).
- **Emission lag for streaming**: `(total_buffer − chunk) / 2` seconds — ~7.5 s with the offline-like 10-10-5 preset, ~6 s with the live 10-2-2 preset.
- **Per-token timestamps** recovered from the stateful TDT merge — segment starts/ends align with the `full` strategy within ~40 ms.
- **Optional end-of-stream refinement** — `progressive_refinement` runs a single FULL pass over the accumulated PCM at EOF and replaces the streamed segments with offline-quality output (when audio ≤ `MAX_FULL_WAVEFORM_S`).
- **Verified at scale** — 3 h files complete cleanly through `progressive` with 100% audio coverage (~205 s ASR time on a DGX Spark, 1121 segments).
- **Versatile output** — plain text, segment list (with start/end), CSV, SRT.
- **Interactive Web UI** — file upload, strategy selector, segment playback, downloads.
- **Health probes** — `/health` for liveness + introspection, `/readyz` for readiness gating.
- **Docker-ready** — single image, configurable via `.env` or environment variables.

## Requirements

- Python 3.10+
- NeMo Toolkit 2.7.3 (the stateful TDT streaming utilities require ≥ 2.4.0)
- NVIDIA GPU + CUDA 12.x recommended; the server runs on CPU but transcription will be much slower
- Docker (optional but recommended)
- Dependencies in `app/requirements.txt`

## Installation

### Docker (recommended)

```bash
git clone https://github.com/pnivek/Parakeet-ASR-FastAPI.git
cd Parakeet-ASR-FastAPI
docker build -t parakeet-asr .
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8777:8777 parakeet-asr
```

### Manual

```bash
git clone https://github.com/pnivek/Parakeet-ASR-FastAPI.git
cd Parakeet-ASR-FastAPI
python -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
cd app && python main.py
```

## Configuration

Configure via environment variables or `app/.env`.

### Core

| Variable | Description | Default |
|----------|-------------|---------|
| `BATCH_SIZE` | ASR batch size during chunked inference | 4 |
| `NUM_WORKERS` | DataLoader workers for `transcribe()` | 0 |
| `TRANSCRIBE_CHUNK_LEN` | Legacy chunk length (s) | 30 |
| `TRANSCRIBE_OVERLAP` | Legacy chunk overlap (s) | 5 |
| `LONG_AUDIO_THRESHOLD` | Switch to `rel_pos_local_attn` for audio longer than this (s) | 480 |
| `PORT` | HTTP/WS port | 8777 |
| `LOG_LEVEL` | Python logging level | INFO |

### Strategy dispatch

| Variable | Description | Default |
|----------|-------------|---------|
| `DEFAULT_STRATEGY` | `auto` \| `full` \| `chunked` \| `progressive`. `auto` routes to `chunked` for REST and `progressive` for WS. Aliases `chunked_v2` / `progressive_v2` still accepted for backward compatibility. | auto |
| `MAX_FULL_WAVEFORM_S` | Hard cap for `full`; longer audio routes to `chunked`. Default matches NeMo's full-attention ceiling. | 1440 |
| `EARLY_BUFFER_TARGET_S` | Progressive mode — PCM seconds buffered before the first partial. | 15 |

### Stateful streaming engine

`BatchedFrameASRTDT` powers both `chunked` and `progressive`. Buffer geometry follows NVIDIA's `<left>-<chunk>-<right>` presets in seconds.

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_STATEFUL_CHUNKED` | Enable the stateful engine for `chunked` and `progressive`. When false, both fall back to the legacy independent-chunk consumer. | true |
| `STATEFUL_MAX_DURATION_S` | Above this duration, `chunked` falls back to the legacy path (~4× faster on long audio, slightly lower boundary quality). `0` disables the cap. | 1800 |
| `STREAMING_LEFT_CONTEXT_S` | Left context for the offline-like 10-10-5 preset | 10 |
| `STREAMING_CHUNK_S` | Chunk length for the offline-like preset | 10 |
| `STREAMING_RIGHT_CONTEXT_S` | Right context for the offline-like preset | 5 |
| `STREAMING_LIVE_CHUNK_S` | Chunk length when client opts into `live_latency` (10-2-2 preset, ~6 s emission lag) | 2 |
| `STREAMING_LIVE_RIGHT_CONTEXT_S` | Right context for the live preset | 2 |

Measured wall-clock on a DGX Spark (single-stream):

| Path | RTF | 30 min file | 3 h file |
|------|-----|-------------|----------|
| Stateful (10-10-5) | ~0.022 | ~40 s | ~4 min |
| Legacy fallback | ~0.005 | ~9 s | ~1 min |

### CUDA graph decoder

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_CUDA_GRAPHS` | NeMo's CUDA-graph RNNT/TDT decoder (FULL_GRAPH mode). Captures the decoder label loop into a single graph; encoder runs eagerly. ~30% speedup on the FULL transcribe path. Implementation details in `.claude/plans/cuda-graph-investigation.md`. | true |

`empty_cache()` calls between transcribes are skipped when graphs are on — releasing caching-allocator blocks invalidates the static-buffer addresses captured in the graph (NeMo issue #14727). PyTorch's allocator handles unreferenced blocks fine on its own.

## API

### REST

```
POST /v1/audio/transcriptions
```

Form upload (`file=@…`), optional query parameters:

| Query param | Description |
|-------------|-------------|
| `strategy` | `auto` \| `full` \| `chunked` \| `progressive`. `auto` honors `DEFAULT_STRATEGY`. `chunked_v2` / `progressive_v2` accepted as aliases. |
| `chunk_length`, `chunk_overlap`, `batch_size`, `long_audio_threshold` | Override server defaults. |

Example:

```bash
curl -X POST -F "file=@audio.mp3" \
  "http://localhost:8777/v1/audio/transcriptions?strategy=chunked"
```

Response:

```json
{
  "text": "...",
  "segments": [{ "id": 0, "start": 0.0, "end": 11.32, "text": "...sentence..." }, ...],
  "language": "en",
  "strategy": "chunked",
  "transcription_time_seconds": 1.13,
  "total_request_time_server_seconds": 1.45,
  "audio_duration_seconds": 21.55,
  "csv_content": "...",
  "srt_content": "..."
}
```

### WebSocket — unified

```
WS /v1/audio/transcriptions
```

The first frame is a JSON config; subsequent frames are audio bytes; an empty binary frame or the text `"END"` signals end-of-stream.

Config frame:

```json
{
  "sample_rate": 16000,
  "channels": 1,
  "bytes_per_sample": 2,
  "format": "wav",
  "chunk_length": 30.0,
  "chunk_overlap": 5.0,
  "batch_size": 1,
  "long_audio_threshold": 480.0,
  "strategy": "progressive",
  "progressive_refinement": true,
  "live_latency": false
}
```

Messages from the server (one or more):

| `type` | When | Notable fields |
|--------|------|----------------|
| `segments_batch` | After each engine commit (sentence-bounded) | `segments[]` |
| `refined_transcription` | After EOF if `progressive_refinement` triggered | `segments`, `text`, `transcription_time`, `audio_duration_seconds` |
| `final_transcription` | Last message before close | `text`, `segments`, `transcription_time`, `total_segments`, `csv_content`, `srt_content`, `refinement_applied` |
| `error` | On server error | `error` |

### WebSocket — legacy aliases

These remain as thin compatibility wrappers and forward to the unified handler with `strategy` forced.

```
WS /v1/audio/transcriptions/ws_upload   # forces strategy=chunked
WS /v1/audio/transcriptions/ws_stream   # forces strategy=progressive
```

### Health

```
GET /health
```

Always 200 once the process is up. Returns model + decoder + config snapshot. Suitable for both liveness and observability.

```
GET /readyz
```

200 with `{"status":"ready","model_loaded":true}` once the ASR model is loaded; 503 with `{"status":"not_ready"}` otherwise. Use this for orchestrator readiness gates.

## Strategy guidance

| If… | Use |
|-----|-----|
| Anything REST, any duration — pick the default | `auto` (→ `chunked`) |
| Anything WS, any duration — pick the default | `auto` (→ `progressive`) |
| Offline file, want fastest single-pass throughput on long audio | `full` (~170× RTFx vs `chunked`'s ~65× on multi-min clips) |
| Streaming audio, ~6 s emission lag, prefer fewer/cleaner emissions | `progressive` + `live_latency: true` (10-2-2 preset) |
| Streaming audio, ~7.5 s emission lag, best chunk-boundary quality | `progressive` + `live_latency: false` (10-10-5 preset, default) |
| Streaming audio, want offline-quality replacement at EOF | `progressive` + `progressive_refinement: true` (default) |

### About the latency numbers

The 10-10-5 and 10-2-2 presets come from NVIDIA's TDT streaming benchmarks. Both deliver *buffered* streaming: the engine emits tokens for an audio window only after it has heard `right_context_s` seconds beyond that window. With `total_buffer = left + chunk + right`, the emission lag is roughly `(total_buffer − chunk) / 2`:

| Preset | Lag | Note |
|--------|-----|------|
| 10-10-5 (offline-like) | ~7.5 s | NVIDIA's "results similar to offline" recommendation |
| 10-2-2 (live) | ~6 s | NVIDIA's "live" preset for this model |

If you need sub-second emission, you need a cache-aware streaming checkpoint (see the [streaming note](#a-note-on-streaming) at the top). This model can't deliver lower latency than its right-context requirement.

## Web Interface

Open `http://localhost:8777/` in a browser. The bundled SPA (React + Vite + TypeScript, sources in `app/web/`) covers:

- **File upload** — drag-drop or click-pick, multipart `POST /v1/audio/transcriptions` with live progress bar.
- **URL ingest** — paste any public http(s) audio URL; the server fetches it (512 MB cap) and runs it through the same pipeline.
- **Live mic capture** — `MediaRecorder(audio/webm;codecs=opus)` over `WS /v1/audio/transcriptions`. Captured audio stays in the player so segments are seekable after stop. **Browsers require a secure context for mic access** — see [Live mic & secure-context requirement](#live-mic--secure-context-requirement) below if you've deployed to a LAN IP.
- **Format-driven output** — `response_format` selector (`json` / `verbose_json` / `text` / `srt` / `vtt`) swaps the result view. `verbose_json` shows the full Whisper segment table, per-segment metadata expansion (tokens, temperature, compression_ratio, avg_logprob, no_speech_prob — honest tooltips on the fields that are unavailable or non-applicable for Parakeet TDT), and a clickable word timeline when `timestamp_granularities[]=word`.
- **Audio playback + segment seek** — one shared `<audio>` element. Clicking a segment row or a word seeks the audio. The playing segment and word get a live highlight.
- **Persisted settings** — strategy, response format, granularities, `live_latency`, `progressive_refinement`, batch_size, etc., all kept in `localStorage` (Zustand `persist` middleware). The `progressive` strategy is auto-disabled outside Live mic and translated to `chunked` on REST submission so a stale persisted value can't return a 400.

### Live mic & secure-context requirement

`navigator.mediaDevices.getUserMedia` (mic capture) is gated to **secure contexts** by every modern browser:

- `https://…` — works.
- `http://localhost`, `http://127.0.0.1`, `http://[::1]` — work (loopback is treated as secure).
- `http://<any-other-host>` (e.g. a LAN IP like `http://192.168.0.172:8777`) — **blocked**. `navigator.mediaDevices` will be `undefined` and the UI shows an inline message explaining how to reach a secure origin.

Two easy paths to get the mic working without setting up TLS:

```bash
# 1) SSH local forward — open http://localhost:8777 in your browser
ssh -L 8777:localhost:8777 your-gpu-host
```

```bash
# 2) Vite dev server — works against any backend, proxies REST + WS
cd app/web
PARAKEET_URL=http://your-gpu-host:8777 npm run dev
# then visit http://localhost:5173
```

For shared / multi-user access, put HTTPS in front of the container (Caddy, Traefik, nginx). Once the origin is `https://…`, mic capture works for everyone.

## Development

### Backend

```bash
python -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
cd app && LOG_LEVEL=DEBUG python main.py
```

### Frontend (live-reload against the backend)

```bash
cd app/web
npm install
npm run dev        # Vite at http://localhost:5173, proxies /v1, /health, /readyz to :8777
```

Override the backend target with `PARAKEET_URL=http://host:port npm run dev` (e.g., when developing the SPA against a remote GPU box). For a production build, `npm run build` emits to `app/static/`, which the FastAPI server already mounts.

See `app/web/README.md` for the client architecture overview.

## Troubleshooting

- **Model load OOM** — Parakeet-TDT-0.6B-v2 needs roughly 4 GB GPU memory in bf16. Reduce to fp16/fp32 with care; this server pins bf16 at load on CUDA-bf16-capable hardware.
- **`cudaErrorIllegalAddress` on repeated transcribes** — should not happen with the current defaults (graphs on, no `empty_cache` between calls). If you re-introduce `torch.cuda.empty_cache()` while `USE_CUDA_GRAPHS=true`, this will return. Flip `USE_CUDA_GRAPHS=false` to confirm graphs are the cause.
- **Stateful chunked feels slow on multi-hour files** — that's by design; the FIFO engine is batch-1 and trades throughput for boundary quality. `STATEFUL_MAX_DURATION_S` controls the auto-fallback to the faster independent-chunk path.

## License

[MIT](LICENSE).

## Acknowledgements

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Parakeet-TDT model and the `BatchedFrameASRTDT` streaming utilities.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio) for audio processing.
