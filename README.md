# Parakeet ASR

High-performance Automatic Speech Recognition (ASR) server built on NVIDIA's NeMo Parakeet-TDT-0.6B-v2. REST and WebSocket APIs, an interactive web UI, and a stateful streaming pipeline that carries decoder state across chunk boundaries.

## Features

- **State-of-the-art ASR** — `nvidia/parakeet-tdt-0.6b-v2` (Token-and-Duration Transducer).
- **Three processing strategies** behind one dispatcher:
  - `full` — single transcribe pass, best quality, up to `MAX_FULL_WAVEFORM_S` (default 24 min).
  - `chunked` — stateful sliding-window via `BatchedFrameASRTDT`. Decoder state carries across chunks, eliminating boundary duplication/drop artifacts. Falls back to a legacy independent-chunk path above `STATEFUL_MAX_DURATION_S` (~4× faster for very long files).
  - `progressive` — live WebSocket streaming. Drives the same `BatchedFrameASRTDT` engine chunk-by-chunk over an ffmpeg PCM stream, emitting sentence-bounded partials as the middle-token merge commits them.
- **Per-token timestamps** recovered from the stateful TDT merge — segment starts/ends align with the `full` strategy within ~40ms.
- **Optional end-of-stream refinement** — `progressive_refinement` runs a single FULL pass over the accumulated PCM at EOF and replaces the streamed segments with offline-quality output.
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
| `DEFAULT_STRATEGY` | `auto` \| `full` \| `chunked` \| `progressive`. `auto` picks based on duration and stream state. | auto |
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
| `STREAMING_LIVE_CHUNK_S` | Chunk length when client opts into `live_latency` (10-2-2 preset, ~4s latency) | 2 |
| `STREAMING_LIVE_RIGHT_CONTEXT_S` | Right context for the live preset | 2 |

Measured wall-clock on a DGX Spark (single-stream):

| Path | RTF | 30 min file | 3 h file |
|------|-----|-------------|----------|
| Stateful (10-10-5) | ~0.022 | ~40 s | ~4 min |
| Legacy fallback | ~0.005 | ~9 s | ~1 min |

### Experimental — CUDA graph decoder

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_CUDA_GRAPHS` | Enable NeMo's CUDA-graph RNNT/TDT decoder. Currently default off — H1 (single-thread executor) was tested and doesn't unblock; investigation parked at `.claude/plans/cuda-graph-investigation.md`. | false |

## API

### REST

```
POST /v1/audio/transcriptions
```

Form upload (`file=@…`), optional query parameters:

| Query param | Description |
|-------------|-------------|
| `strategy` | `auto` \| `full` \| `chunked`. `auto` honors `DEFAULT_STRATEGY`. |
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
| File ≤ 24 min, want best quality | `full` |
| File 24 min – 30 min, want best chunked quality | `chunked` (stateful) |
| File > 30 min | `chunked` (auto-falls-back to legacy above `STATEFUL_MAX_DURATION_S`) |
| Live audio, low latency more important than chunk-boundary quality | `progressive` + `live_latency: true` |
| Live audio, want offline-quality output after EOF | `progressive` + `progressive_refinement: true` (default) |

The 10-10-5 and 10-2-2 presets come from NVIDIA's TDT streaming benchmarks; 10-10-5 yields "results similar to offline," 10-2-2 yields ~4 s live latency at slightly lower quality.

## Web Interface

Open `http://localhost:8777/` in a browser. Features:

- Upload an audio file.
- Pick REST, WebSocket Full Upload, or WebSocket Live Stream.
- Choose a strategy (`auto` / `full` / `chunked` / `progressive`).
- Toggle `progressive_refinement` and `live_latency`.
- Inspect the resolved strategy, transcription time, and segment table (click a row to scrub).
- Download CSV / SRT.
- Optional debug log panel.

## Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt
cd app && LOG_LEVEL=DEBUG python main.py
```

## Troubleshooting

- **Model load OOM** — Parakeet-TDT-0.6B-v2 needs roughly 4 GB GPU memory in bf16. Reduce to fp16/fp32 with care; this server pins bf16 at load on CUDA-bf16-capable hardware.
- **`cudaErrorIllegalAddress` on repeated transcribes** — keep `USE_CUDA_GRAPHS=false` (the default). The investigation notes are in `.claude/plans/cuda-graph-investigation.md`.
- **Stateful chunked feels slow on multi-hour files** — that's by design; the FIFO engine is batch-1 and trades throughput for boundary quality. `STATEFUL_MAX_DURATION_S` controls the auto-fallback to the faster independent-chunk path.

## License

[MIT](LICENSE).

## Acknowledgements

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Parakeet-TDT model and the `BatchedFrameASRTDT` streaming utilities.
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework.
- [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio) for audio processing.
