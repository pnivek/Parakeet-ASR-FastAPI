# Parakeet Hybrid Streaming Refactor

> **Branch:** `refactor-hybrid-streaming`
> **Based on:** `streaming-asr-updates` (the latest code with model lifecycle management)
> **Implementation method:** task-by-task via subagent-driven-development

**Goal:** Evolve the Parakeet ASR server into a configurable hybrid transcription service with a unified pipeline that adapts its processing strategy based on audio duration, latency requirements, and VRAM constraints.

**Architecture:** Single consolidated WebSocket endpoint + REST endpoint with a strategy dispatcher. The dispatcher selects between three modes — `full` (feed full waveform to NeMo's `transcribe()`), `chunked` (manual sliding-window with batched NeMo calls), and `progressive` (start transcribing once enough audio has been buffered, optionally refine at EOF). The ffmpeg pipeline stays as the audio decoder. The independent-PCM-chunk approach from `ws_stream` is replaced with NeMo's native streaming API for real-time use.

**Tech Stack:** Python 3.10+, FastAPI, NeMo Toolkit 2.2+, torchaudio, ffmpeg, NVIDIA Parakeet-TDT-0.6B-v2 (with planned v3 upgrade path)

---

## References & Documentation

### NVIDIA Model Card
- **HF Model Card:** https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2
  - "Transcribes audio segments **up to 24 minutes** in a single pass"
  - Architecture: FastConformer-TDT, 600M params, full attention
- **NGC Page:** https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-0.6b-tdt
  - "Long audio transcription, supporting audio up to 24 minutes long with full attention (on A100 80GB) or **up to 3 hours with local attention**"

### NVIDIA Reference Implementation
- **HF Space (NVIDIA's demo):** https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2/blob/main/app.py
  - The canonical reference: loads the model, applies `change_attention_model("rel_pos_local_attn", [256,256])` for audio > 480s, transcribes, reverts settings
  - Our `_apply_model_settings_for_session` already mirrors this exactly
- **NeMo Streaming/Buffered Inference Script:** https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py
  - The reference for stateful streaming: uses `left_context_secs`, `chunk_secs`, `right_context_secs`
  - Key config: `10-10-5` (left=10s, chunk=10s, right=5s) "gives results similar to offline"
  - Stateful decoder: carries decoder state between chunks (no independent re-transcription)
- **NeMo Buffered Inference README:** https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/README.md
  - "The major difference between streaming ASR and buffered ASR is the chunk size and the total context buffer size"

### NVIDIA Developer Guidance
- **Developer Forums:** https://forums.developer.nvidia.com/t/chopping-the-long-audios-for-transcription/326023
  - "With a fastconformer model, transcription may work **out of the box** without audio splitting"
  - To reduce memory further: `model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[64, 64])`
- **TDT Merge Discussion:** https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/discussions/13
  - "Conventional merging methods don't work well for TDT"
  - Use the built-in merge algorithm (auto-detected in NeMo's streaming script)
- **Benchmarking (E2E Networks):** https://www.e2enetworks.com/blog/benchmarking-asr-models-nvidia-l4-parakeet-whisper-nemotron
  - "Parakeet bf16 batch=8 hits 238× real-time throughput"
  - "Beam search on Parakeet is strictly counterproductive: 2× slower with no accuracy gain"

### What the Branch Already Gets Right
- `_apply_model_settings_for_session()` — per-session attention/dtype config (matches NVIDIA's HF Space exactly)
- `_revert_model_to_global_original_state()` — clean teardown with CUDA cache clear
- `model_access_lock` — exclusive access guard for the single global model
- `_deduplicate_segments()` — correct overlap dedup (only used in ws_upload, not ws_stream)
- `create_audio_chunks()` — correct sliding-window chunking
- `_perform_asr_transcription()` — clean abstraction over NeMo's `transcribe()`
- `load_and_preprocess_audio()` — async_audio()` — BytesIO support (used in ws_upload, not REST)

---

## Proposed Approach

### Core insight from NVIDIA docs

The model handles full audio internally up to 24 minutes (full attention) or 3 hours (local attention). **The right default is to just feed the full waveform to `transcribe()`.** Manual chunking is only needed when the audio exceeds VRAM or the response must be streaming. The ffmpeg independent-chunk approach discards decoder state and degrades quality for no benefit.

### Strategy resolution

```
Strategy decision (per request):

Input: audio_duration, client_strategy, is_streaming, available_vram

  strategy = client_strategy  # explicit override
  if strategy == "auto" or not set:
    if is_streaming and client wants early results:
      strategy = "progressive"
    elif audio_duration * vram_bytes_per_second > available_vram:
      strategy = "chunked"     # too big for VRAM
    else:
      strategy = "full"        # let NeMo handle it
```

### Processing modes

| Mode | How it works | Quality | Latency | Memory | Use case |
|------|-------------|---------|---------|--------|----------|
| `full` | `transcribe([full_waveform], timestamps=True)` | Best | Highest (wait for full) | Highest | Offline files < 24min |
| `chunked` | Manual sliding window + batch transcribe | Good (boundary artifacts) | Medium | Low | Files > VRAM limit |
| `progressive` | Start chunked early → opt. re-transcribe full at EOF | Good→Best on refine | Low → better over time | Medium→High | Live streaming, long meetings |

### Endpoint consolidation

| New Endpoint | Replaces | Strategy | Notes |
|-------------|----------|----------|-------|
| `POST /v1/audio/transcriptions` | Same | `full` default, `chunked` available via query params | REST stays as-is (strategy-aware) |
| `WS /v1/audio/transcriptions` | `ws_upload` + `ws_stream` | Configurable per client config | Single WS endpoint, `strategy` in client config |
| *(deprecated)* | `ws_upload` | — | Behind compat alias |
| *(deprecated)* | `ws_stream` | — | Behind compat alias |

---

## Step-by-Step Plan

### Phase 0: Repository Setup

#### Task 0.1: Create working branch
```bash
git checkout -b refactor-hybrid-streaming
```

#### Task 0.2: Save this plan
```bash
mkdir -p .hermes/plans/
# write_file .hermes/plans/2026-05-17_171500-parakeet-hybrid-streaming-refactor.md
git add .hermes/plans/
git commit -m "docs: add hybrid streaming refactor plan"
```

---

### Phase 1: Code Quality — Fix REST Temp File I/O

**Files:**
- Modify: `app/main.py` (transcribe_endpoint_rest, ~lines 1370-1480)

**What:** The REST endpoint still writes uploaded audio to a temp file on disk, then reads it back with `torchaudio.load`. The `ws_upload` endpoint already does it right with `io.BytesIO`. Align REST with the ws_upload pattern.

**Changes:**
1. Replace `tempfile.NamedTemporaryFile` + `shutil.copyfileobj` + `torchaudio.load(path)` with:
   ```python
   audio_bytes = await file.read()
   buffer = io.BytesIO(audio_bytes)
   waveform, total_duration = await load_and_preprocess_audio(buffer, MODEL_SAMPLE_RATE, request_id)
   ```
2. Remove the `temp_audio_file_path` variable and the cleanup block in `finally`
3. Keep the `file.file.close()` in the finally block

**Risk:** None. ws_upload already does this. Pure performance improvement.

---

### Phase 2: Strategy System — Config, Resolution, Constants

**Files:**
- Modify: `app/main.py` (env vars, config section)
- Modify: `app/.env` (new defaults)
- Modify: `app/utils.py` (new function)

**What:** Add the strategy constants, configuration enum, and resolution function. This is plumbing — no processing logic yet.

#### Task 2.1: Add strategy types and defaults

In `main.py`, add:
```python
from enum import Enum

class ProcessingStrategy(strategy_type)

class ProcessingStrategy(str, Enum):
    AUTO = "auto"
    FULL = "full"
    CHUNKED = "chunked = "chunked"
    PROGRESSIVE = "progressive"

# New env vars with defaults
MAX_FULL_WAVEFORM_S = float(os.getenv("MAX_FULL_WAVEFORM_S", 1440.0))  # 24 min — full attention limit per NVIDIA
MAX_FULL_WAVEFORM_S_STR = "MAX_FULL_WAVEFORM_S"  # for os.getenv
# Actually: 24 * 60 = 1440s is the full attention limit per model card.
# But practically VRAM-gated. Make configurable.
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "full")
EARLY_BUFFER_TARGET_S = float(os.getenv("EARLY_BUFFER_TARGET_S", 15.0))
```

In `.env`, add:
```
# --- Strategy Configuration ---
# Processing strategy: "auto", "full", "chunked", "progressive"
# "auto" resolves based on audio duration, VRAM estimate, and streaming state
DEFAULT_STRATEGY=auto
# Max audio duration for full-waveform mode (in seconds).
# Full attention limit: 24 min (1440s) per NVIDIA model card.
# Local attention extends to ~3 hours. VRAM-gated in practice.
MAX_FULL_WAVEFORM_S=1440
# Early buffer target for progressive streaming (seconds before first transcript)
EARLY_BUFFER_TARGET_S=15
```

Reference: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 — "up to 24 minutes in a single pass"

#### Task 2.2: Add strategy resolution function

```python
async def resolve_strategy(
    audio_duration_s: Optional[float],
    client_config: dict,
    is_streaming: bool = False,
) -> ProcessingStrategy:
    """
    Resolve the processing strategy based on audio duration, client config,
    VRAM constraints, and streaming state.
    """
    requested = client_config.get("strategy", DEFAULT_STRATEGY)
    if requested != ProcessingStrategy.AUTO:
        return ProcessingStrategy(requested)

    # Auto-resolution
    if is_streaming and client_config.get("early_buffer_target_s", 0) > 0:
        return ProcessingStrategy.PROGRESSIVE

    if audio_duration_s is not None and audio_duration_s > MAX_FULL_WAVEFORM_S:
        return ProcessingStrategy.CHUNKED

    return ProcessingStrategy.FULL
```

#### Task 2.3: Update parse_request_config and centralize config parsing

Extend `parse_request_config` to accept strategy params. Extend `parse_websocket_config` to accept `strategy` and `early_buffer_target_s`.

```python
def parse_request_config(
    ...existing params...,
    strategy: Optional[...],
    strategy: Optional[str] = None,
    early_buffer_target_s: Optional[float] = None,
) -> dict:
    config = {...existing...}
    config["strategy"] = strategy if strategy else DEFAULT_STRATEGY
    config["early_buffer_target_s"] = early_buffer_target_s if early_buffer_target_s else EARLY_BUFFER_TARGET_S
    return config
```

---

### Phase 3: REST Strategy Deployment

**Files:**
- Modify: `app/main.py` (transcribe_endpoint_rest)

**What:** Make the REST endpoint strategy-aware. Currently it always does `full` (full waveform to `transcribe()`). Add a `strategy` query param. When `strategy=chunked`, split the waveform into chunks via `create_audio_chunks` and batch-transcribe them (like ws_upload does), then deduplicate.

The REST endpoint already does the right thing for `strategy=full` (the current code). We just need to add the `chunked` path and the query parameter.

#### Task 3.1: Add strategy query parameter to REST endpoint

```python
@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(
    file: UploadFile = File(...),
    ...existing params...,
    strategy: Optional[str] = Query(None, description="Processing strategy: auto, full, chunked"),
    early_buffer_target_s: Optional[float] = Query(None, description="..."),
):
    client_config = parse_request_config(..., strategy=strategy, ...)
    resolved = await resolve_strategy(audio_duration_s=total_duration, client_config=..., is_streaming=False)
```

#### Task 3.2: Add chunked processing path for REST

When strategy resolves to `chunked`:
```python
if resolved == ProcessingStrategy.CHUNKED:
    chunks, offsets = create_audio_chunks(
        waveform_tensor, MODEL_SAMPLE_RATE,
        client_config["chunk_length"], client_config["chunk_overlap"]
    )
    all_raw_segments = []
    for batch_start in range(0, len(chunks), client_config["batch_size"]):
        batch = chunks[batch_start:batch_start+batch_size]
        batch_offsets = offsets[batch_start:batch_start+batch_size]
        hyps, _ = await _perform_asr_transcription(..., batch, batch_size=len(batch))
        segs = _process_hypotheses_to_segments(hyps, batch_offsets)
        all_raw_segments.extend(segs)
    segments = _deduplicate_segments(all_raw_segments, client_config["chunk_overlap"] / 2.0)
```

---

### Phase 4: WebSocket Consolidation — Single Endpoint

**Files:**
- Modify: `app/main.py`
- Create: `app/routers/ws_router.py` (optional, might be clean enough inline)

**What:** Merge `ws_upload` and `ws_stream` into a single `WS /v1/audio/transcriptions` endpoint. The client config's `strategy` field selects which pipeline to run. The ffmpeg pipeline stays as the decoder for streaming scenarios.

#### Task 4.1: Create the consolidated WS endpoint

```python
@app.websocket("/v1/audio/transcriptions")
async def websocket_transcribe(websocket: WebSocket):
    """
    Single WebSocket endpoint for audio transcription.
    
    Client sends JSON config first, then audio data, then "END".
    Server selects processing strategy based on config.
    """
    # 1. Accept + receive config
    # 2. Parse config (strategy, chunk_length, etc.)
    # 3. Resolve strategy
    # 4. Route to handler:
    #    - full: accumulate all bytes → load_and_preprocess_audio → transcribe → send
    #    - chunked: accumulate → chunk → batch transcribe → dedup → send
    #    - progressive: ffmpeg pipeline → start chunking at buffer target → stream results
    # 5. Return final_transcription
```

#### Task 4.2: Add backward-compat aliases

```python
@app.websocket("/v1/audio/transcriptions/ws_upload")
async def websocket_upload_alias(websocket: WebSocket):
    # Forward to consolidated with implied config
    ...
```

Actually simpler: keep `ws_upload` and `ws_stream` as thin wrappers that inject implied config and call the handler.

---

### Phase 5: Progressive Streaming Implementation

**Files:**
- Modify: `app/main.py` (handle_streaming_pcm → refactor)
- Modify: `app/main.py` (new: progressive handler for progressive mode)

**What:** This is the core new feature. The progressive mode:
1. Starts the ffmpeg pipeline (same as current `ws_stream`)
2. Once `EARLY_BUFFER_TARGET_S` seconds of PCM have been accumulated, creates a chunk from the available data and transcribes it
3. Sends partial results immediately
4. As more PCM arrives, creates contiguous chunks (sliding window)
5. When the stream ends (EOF signal):
   - **If `progressive_refinement=True`**: feed the *entire accumulated waveform* to `transcribe()` (full attention with local attention), then replace/merge partial results
   - **If `progressive_refinement=False`**: just send the final deduplicated chunked output

Reference: This is the "buffered" mode from NVIDIA's streaming script — the buffer accumulates audio, then the model processes it in chunks with overlap.

#### Task 5.1: Refactor ffmpeg pipeline to support progressive output

The current `handle_streaming_pcm` processes everything through a producer-consumer queue. We need to modify the consumer to:
1. Accumulate PCM continuously (already happens in the buffer)
2. After `early_buffer_target_s` PCM has been read, start creating chunks as fixed-size windows advancing by stride
3. The chunks should include earlier audio as "left context" — meaning the first chunk covers `[0, early_buffer_target_s)` but subsequent chunks overlap by the configured amount

The key change: instead of starting chunk creation from PCM offset 0, delay the first chunk until we have enough buffer to cover the first `chunk_length + overlap` window.

Actually, re-reading the current code more carefully — the `handle_streaming_pcm` already does sliding window chunk creation in the PCM domain. The change is just:
1. Don't start the consumer until `early_buffer_target_s` worth of PCM is in the buffer
2. Keep everything else the same

#### Task 5.2: Add progressive refinement at EOF

When the producer signals EOF and all audio has been received:
1. Collect all PCM bytes that were fed to ffmpeg
2. Create a full waveform from them
3. If `progressive_refinement=True` and duration < `MAX_FULL_WAVEFORM_S`:
   - Apply full model settings (or local attention for long audio)
   - Call `transcribe([full_waveform], timestamps=True)`
   - Process segments from the full transcribe
   - **Merge with already-sent segments**: use start-time alignment to replace overlapping segments with the higher-quality full-transcribe versions
   - Send updated final_transcription

This is the "best of both worlds" — low latency for initial results, perfect quality for the final transcript.

---

### Phase 6: Model Enhancements

**Files:**
- Modify: `app/main.py` (model loading, lifecycle)
- Modify: `app/.env` (new vars)

#### Task 6.1: Add VRAM-based constraint estimation

```python
# Estimate per-second VRAM cost for full-waveform mode
# Based on: 600M params, bf16 = ~1.2GB model weight
# FastConformer has ~86x downsampling; attention is O(n²) in sequence length
# At 16kHz, 24 min = 23M samples → ~267K encoder frames
# With full attention: O(267K²) activations at encoder depth
# Empirical: ~22GB on A100 for 24min, local attention ~6GB for same
def estimate_vram_for_full_waveform(duration_s: float, use_local_attn: bool) -> float:
    """
    Rough VRAM estimate for full-waveform mode.
    
    Reference: NGC page — 24 min full attention on A100 80GB,
    3 hours with local attention.
    """
    if use_local_attn:
        # Rule of thumb: ~2GB base + ~1GB per 30 min of audio
        return 2.0 + (duration_s / 1800.0) * 4.0
    else:
        # Full attention: quadratic in encoder frames
        # Base ~2GB model, grows with sequence length
        scale = duration_s / 1440.0
        return 2.0 + scale * 20.0  # ~22GB at 24 min
```

Reference: https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-0.6b-tdt

#### Task 6.2: Add preserve_context support

TDT models support a `preserve_context` mechanism where the decoder's hidden state from one chunk is carried to the next. This can improve chunk boundary accuracy.

In `_process_hypotheses_to_segments` and the chunked processing loop, check if `hypothesis.timestamp` includes alignment data that can be used to merge across chunks.

Reference: NeMo's streaming script auto-detects TDT and uses `merge_algo="tdt"` with its specialized merge — https://github.com/NVIDIA/NeMo/blob/main/examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py

---

### Phase 7: Health, Observability, Client Experience

**Files:**
- Modify: `app/main.py` (new endpoints)
- Modify: `app/index.html` (update UI for new features)

#### Task 7.1: Health endpoint

```python
@app.get("/health")
async/health")
async def health_check():
    """Health check for load balancers."""
    return {
        "status": "healthy" if asr_model else "unhealthy",
        "model": ASR_MODEL_NAME,
        "device": str(next(asr_model.parameters()).device) if asr_model else "none",
        "model_loaded": asr_model is not None,
        "ready": asr_model is not None,
    }

@app.get("/async/readyz")
async def readiness_check():
    """Readiness probe — 503 if model not loaded."""
    if not asr_model:
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready", "model": ASR_MODEL_NAME}
```

#### Task 7.2: Update Web UI (index.html)

- Add strategy selector dropdown
- Add "progressive_refinement" checkbox
- Update segment display to show intermediate streaming results properly
- Show processing mode in the transcription results

---

### Phase 8: Cleanup & Documentation

**Files:**
- Modify: `README.md`
- Modify: `app/index.html`
- Maybe: `app/requirements.txt` (NeMo version bump for v3 support)

#### Task 8.1: Update README

Document:
- Strategy modes and when to use each
- All query parameters and env vars
- New unified WebSocket protocol
- Backward-compat aliases
- Example curl commands for each strategy

#### Task 8.2: Deprecation notices

Add deprecation warnings to old endpoint aliases (`ws_upload`, `ws_stream`).

---

## Files Likely to Change

| File | What changes |
|------|-------------|
| `app/main.py` | Major: strategy system, endpoint consolidation, progressive mode, health endpoints |
| `app/utils.py` | Minor: maybe add VRAM estimator or strategy helpers |
| `app/.env` | Add new env vars (DEFAULT_STRATEGY, MAX_FULL_WAVEFORM_S, EARLY_BUFFER_TARGET_S) |
| `app/index.html` | Add strategy/refinement controls, show processing mode |
| `app/requirements.txt` | Maybe bump nemo_toolkit version |
| `README.md` | New docs for strategy system |
| `.hermes/plans/2026-05-17_171500-parakeet-hybrid-streaming-refactor.md` | This file |

## Files That Should NOT Change

| File | Rationale |
|------|-----------|
| `app/.env` existing vars (BATCH_SIZE, etc.) | Keep existing, add new ones |
| Dockerfile | Works fine, no changes needed |
| LICENSE | MIT, no change |

## Testing & Validation

### Unit-level
- `_deduplicate_segments`: test with synthetic overlapping segments
- `resolve_strategy`: test each combination of inputs → expected output
- `estimate_vram_for_full_waveform`: check thresholds align with NGC specs

### Integration
- REST endpoint with each query parameter combination
- WS endpoint with each strategy
- Progressive mode: send audio, verify first result arrives within `early_buffer_target_s + processing_time`
- Refinement: send a file, check that final output matches full/mode transcription

### Validation against NVIDIA reference
- Transcribe the same audio through:
  1. NVIDIA's HF Space demo
  2. Our REST endpoint (strategy=full)
  3. Our WS endpoints (each strategy)
- Compare WER between the three (should be equivalent)

## Risks & Tradeoffs

### Risks

1. **TDT merge complexity** — The progressive refinement merge (partial chunked results → full-waveform replacement) requires careful segment alignment. If segments shift between chunked and full modes, the merge could create duplicates or gaps. Mitigation: use `_deduplicate_segments` with strict thresholds on the merge.

2. **VRAM pressure from dual approach** — In progressive mode, both the chunked pipeline and the full-waveform refinement are active near EOF. Could spike VRAM. Mitigation: clear chunked pipeline state before calling full transcribe; run `torch.cuda.empty_cache()` between phases.

3. **ffmpeg as a long-running subprocess** — The current `ws_stream` starts ffmpeg per connection, which is fine. But if the consumer backs up, the ffmpeg process buffers output. With hours-long audio, this could be significant. Mitigation: the `asyncio.Queue(maxsize=batch_size*2)` provides backpressure.

4. **NeMo API stability** — The `change_attention_model` and `change_subsampling_conv_chunking_factor` APIs are not officially documented as stable. They could change between NeMo versions. Mitigation: wrap in try/except with fallback; pin NeMo version in requirements.txt.

5. **Progressive mode quality gap** — If `progressive_refinement=False`, the output quality depends entirely on the chunked approach. For very long audio (>3 hours), the chunks are independent and boundary artifacts accumulate. Mitigation: document that refinement=True is recommended for production use.

### Tradeoffs

| Decision | Chosen | Alternative | Rationale |
|----------|--------|-------------|-----------|
| WS endpoints | Single consolidated | Three separate | Single endpoint = simpler client protoco, one code path to maintain |
| Progressive refinement | Optional (default True) | Always on | Refinement doubles processing time for long files; some use cases (e.g., live captions) don't need it |
| ffmpeg for audio decode | Keep | torchaudio direct torchaudio | ffmpeg handles all codecs and resamples on-the-fly; torchaudio can't decode some formats without ffmpeg |
| Batch size interpretation | Real batching for chunked; no-op for full | — | Matches NeMo's actual behavior; batch solubility documented by NVIDIA |
| VRAM estimation | Heuristic | Fixed threshold | VRAM depends on precision, attention mode, and batch orchestration; a slider in a config is more practical than perfect estimation |

## Open Questions

1. **Should progressive refinement run on the full accumulated PCM or re-feed through ffmpeg?** The ffmpeg output is already 16kHz mono PCM, so the accumulated PCM bytes are the same as what ffmpeg would produce. We can feed them directly to `load_and_preprocess_audio(audio_source=io.BytesIO(pcm_bytes))`.

2. **Is `progressive_refinement` with the full `transcribe()` call compatible with the streaming model lock?** Yes — `_apply_model_settings_for_session` and the model lock already serialize access. The refinement just needs to acquire the lock again (or check if it's still held).

3. **Should we support Parakeet-TDT v3 (multilingual)?** The branch currently uses v2. v3 supports 25 European languages. Worth adding as a config option since the API is identical and it's just a model name swap.

4. **What's the actual per-second VRAM cost of `transcribe()` with local attention on a DGX Spark (Blackwell 128GB)?** We should test empirically. The NGC spec was measured on A100 80GB. Blackwell's SM121 might have different memory characteristics.
