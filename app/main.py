import bisect
import gc
import os
import io
import math
import re
import time
import json
import base64
import asyncio
import logging
import threading
import functools
import concurrent.futures
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional, Tuple, List
import subprocess
import uvicorn

import numpy as np

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.asr_model import ASRModel as NeMoASRModelType

# Streaming engine — NVIDIA's blessed pattern (NeMo PR #9106).
# Lives in its own module to keep main.py from growing further.
from streaming_v2 import StreamingPrevBatchedEngine, tokens_to_sentence_segments, tokens_to_words

from dotenv import load_dotenv

from utils import (
    generate_srt_content,
    generate_csv_content,
)

load_dotenv()

# --- Logging Configuration ---
# Configure logging level and format for the application.
# Level can be set via LOG_LEVEL environment variable (e.g., "DEBUG", "INFO", "WARNING").
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("parakeet-asr.main")
logger.info(f"Logging configured with level: {log_level_str}")

# --- Application Configuration ---
# Core application settings, mostly loaded from environment variables with defaults.

# Number of workers for NeMo's internal DataLoader during transcription.
# 0 typically means operations run in the main data loading thread.
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))

# Target sample rate for the ASR model. Audio will be resampled to this rate.
MODEL_SAMPLE_RATE = 16000

# For uvicorn server
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8777))

# Name of the NeMo ASR model to load from HuggingFace or local cache.
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"

# Threshold (in seconds) to determine if "long audio" specific model settings
# (like local attention) should be applied. This applies to the decision duration,
# which can be total audio length (REST) or ASR chunk length (WebSockets).
LONG_AUDIO_THRESHOLD_S = float(os.getenv("LONG_AUDIO_THRESHOLD", 480.0))

# Server-side ASR chunking configuration. These can be overridden by client in requests.
TRANSCRIBE_CHUNK_LEN = float(os.getenv("TRANSCRIBE_CHUNK_LEN", 30.0)) # Duration of each ASR processing chunk.
TRANSCRIBE_OVERLAP = float(os.getenv("TRANSCRIBE_OVERLAP", 5.0))   # Overlap between ASR processing chunks.
CHUNKING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1)) # Max number of ASR chunks processed together by the model.

# Size of PCM data chunks read from ffmpeg's stdout in the streaming producer.
FFMPEG_PCM_CHUNK_SIZE_BYTES = int(os.getenv("FFMPEG_PCM_CHUNK_SIZE_BYTES", 16384))


class ProcessingStrategy(str, Enum):
    """Transcription dispatch modes.

    Four wire values, picked by the client via `?strategy=`:

      - OFFLINE     REST/accumulate transport. Server picks FULL (≤ cap) or
                    SPLIT_FULL (above) by audio duration. The lowest-surprise
                    default for REST.
      - FULL        Force single-pass encode → decode of the whole waveform.
                    Errors / OOMs if the audio exceeds MAX_FULL_WAVEFORM_S.
      - SPLIT_FULL  Force sequential FULL passes over overlapping slices,
                    stitched at seams.
      - STREAMING   WebSocket transport, ffmpeg producer + chunked engine,
                    live partials. Required for live mic / URL live streams.
    """
    OFFLINE = "offline"
    FULL = "full"
    SPLIT_FULL = "split_full"
    STREAMING = "streaming"


# Strategy configuration.
# DEFAULT_STRATEGY=offline is the lowest-surprise default for REST (full ≤ cap,
# split_full above). WS connections always resolve to STREAMING regardless via
# transport auto-correct inside resolve_strategy().
# MAX_FULL_WAVEFORM_S caps when `full` mode is selected by offline resolution
# (NVIDIA's 24-minute full-attention ceiling for parakeet-tdt-0.6b-v2).
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", ProcessingStrategy.OFFLINE.value).lower()
MAX_FULL_WAVEFORM_S = float(os.getenv("MAX_FULL_WAVEFORM_S", 1440.0))
# Wall-clock cap (seconds) for a single URL-over-WS streaming session. Default
# 6 h bounds runaway live-stream transcriptions (HLS/icecast/RTSP) so a
# forgotten browser tab can't tie up the model lock indefinitely.
URL_STREAM_MAX_S = float(os.getenv("URL_STREAM_MAX_S", 21600.0))

# Streaming context windows. NeMo's reference recommends 10-10-5 for offline-like
# quality, 10-2-2 for live latency. These drive `StreamingPrevBatchedEngine`
# for the STREAMING dispatch path.
STREAMING_LEFT_CONTEXT_S = float(os.getenv("STREAMING_LEFT_CONTEXT_S", 10.0))
STREAMING_CHUNK_S = float(os.getenv("STREAMING_CHUNK_S", 10.0))
STREAMING_RIGHT_CONTEXT_S = float(os.getenv("STREAMING_RIGHT_CONTEXT_S", 5.0))
STREAMING_LIVE_CHUNK_S = float(os.getenv("STREAMING_LIVE_CHUNK_S", 2.0))
STREAMING_LIVE_RIGHT_CONTEXT_S = float(os.getenv("STREAMING_LIVE_RIGHT_CONTEXT_S", 2.0))

# Streaming mode: PCM buffered before emitting the first partial.
EARLY_BUFFER_TARGET_S = float(os.getenv("EARLY_BUFFER_TARGET_S", 15.0))

# CUDA graph decoder for RNNT/TDT. NeMo 2.7.x defaults to ON (FULL_GRAPH mode).
# Earlier attempts crashed with cudaErrorIllegalAddress on the 2nd transcribe;
# root cause was `torch.cuda.empty_cache()` between requests invalidating the
# static buffer addresses baked into the captured graph (NeMo issue #14727).
# That cache-flush is now gated on `not USE_CUDA_GRAPHS`, so flipping graphs
# on is safe. Stress: 30/30 FULL transcribes pass at p99=0.16s (~30% faster
# than eager). Investigation notes at .claude/plans/cuda-graph-investigation.md.
USE_CUDA_GRAPHS = os.getenv("USE_CUDA_GRAPHS", "true").lower() == "true"

logger.info(
    f"Configuration loaded:\n"
    f"  App: workers={NUM_WORKERS}, sample_rate={MODEL_SAMPLE_RATE}, port={PORT}\n"
    f"  Model: {ASR_MODEL_NAME}, long_audio_threshold_for_model_settings={LONG_AUDIO_THRESHOLD_S}s\n"
    f"  Chunking Defaults: length={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s, batch_cap={CHUNKING_BATCH_SIZE}\n"
    f"  Streaming (ffmpeg): pcm_read_chunk_size={FFMPEG_PCM_CHUNK_SIZE_BYTES}B\n"
    f"  Strategy: default={DEFAULT_STRATEGY}, max_full_waveform_s={MAX_FULL_WAVEFORM_S}, url_stream_max_s={URL_STREAM_MAX_S}, early_buffer_target_s={EARLY_BUFFER_TARGET_S}\n"
    f"  Streaming windows: offline {STREAMING_LEFT_CONTEXT_S}-{STREAMING_CHUNK_S}-{STREAMING_RIGHT_CONTEXT_S}, "
    f"live {STREAMING_LEFT_CONTEXT_S}-{STREAMING_LIVE_CHUNK_S}-{STREAMING_LIVE_RIGHT_CONTEXT_S}\n"
    f"  CUDA graphs: {USE_CUDA_GRAPHS}"
)

# --- FastAPI App Setup ---
app = FastAPI(title="Parakeet ASR Service", version="1.0.0")

# Registry of in-flight streaming cancel events, keyed by session_id.
# Populated by the WebSocket entry handler before invoking
# handle_streaming_pcm / handle_streaming_url, removed in its `finally`.
# The /v1/audio/streaming/cancel/{session_id} HTTP endpoint sets the
# event for that session — out-of-band signal that the WS-buffered
# binary frames can't beat. Without this, the client's `socket.close()`
# waits for `bufferedAmount` to drain before the close frame even
# reaches the server; the engine keeps processing all the queued audio
# in the meantime. With this, the cancel arrives in one HTTP round
# trip independent of WS buffering, and engine bail latency is bounded
# to one chunk in flight.
_streaming_cancel_events: Dict[str, asyncio.Event] = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files Setup for serving index.html
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
if not os.path.exists(static_dir): # Ensure static directory exists
    os.makedirs(static_dir, exist_ok=True)
    logger.info(f"Created static directory at {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

def _collect_health_info() -> dict:
    """Snapshot of model load + decoding/runtime config for /health."""
    info: dict = {
        "model_loaded": asr_model is not None,
        "model_name": ASR_MODEL_NAME,
        "config": {
            "default_strategy": DEFAULT_STRATEGY,
            "max_full_waveform_s": MAX_FULL_WAVEFORM_S,
            "url_stream_max_s": URL_STREAM_MAX_S,
            "long_audio_threshold_s": LONG_AUDIO_THRESHOLD_S,
            "use_cuda_graphs": USE_CUDA_GRAPHS,
            "early_buffer_target_s": EARLY_BUFFER_TARGET_S,
            "streaming_context_s": {
                "offline_left": STREAMING_LEFT_CONTEXT_S,
                "offline_chunk": STREAMING_CHUNK_S,
                "offline_right": STREAMING_RIGHT_CONTEXT_S,
                "live_chunk": STREAMING_LIVE_CHUNK_S,
                "live_right": STREAMING_LIVE_RIGHT_CONTEXT_S,
            },
        },
    }
    try:
        info["decoding_strategy"] = str(asr_model.cfg.decoding.strategy) if asr_model else None
        info["decoding_preserve_alignments"] = bool(asr_model.cfg.decoding.get("preserve_alignments", False)) if asr_model else None
    except Exception as e:
        info["decoding_introspect_error"] = str(e)
    if asr_model is None:
        return info
    try:
        info["device"] = str(next(asr_model.parameters()).device)
        info["dtype"] = str(next(asr_model.parameters()).dtype)
    except Exception as e:
        info["param_iter_error"] = str(e)
    try:
        inferer = asr_model.decoding.decoding
        info["inferer_class"] = type(inferer).__name__
        info["use_cuda_graph_decoder"] = getattr(inferer, "use_cuda_graph_decoder", None)
        computer = getattr(inferer, "decoding_computer", None)
        if computer is not None:
            info["computer_class"] = type(computer).__name__
            info["cuda_graphs_mode"] = repr(getattr(computer, "cuda_graphs_mode", "<unset>"))
            info["allow_cuda_graphs"] = getattr(computer, "allow_cuda_graphs", None)
    except Exception as e:
        info["decoder_introspect_error"] = str(e)
    return info


@app.get("/health")
async def health():
    """Liveness + introspection. Always 200 once the process is up. For readiness gating, use /readyz."""
    info = _collect_health_info()
    info["status"] = "ok" if info["model_loaded"] else "loading"
    return info


@app.post("/v1/audio/streaming/cancel/{session_id}")
async def cancel_streaming_session(session_id: str):
    """Out-of-band cancel for a live streaming WebSocket session.

    Sets the session's cancel_event, which the streaming consumer +
    engine poll between chunks. Independent of the WS itself — used
    by the browser client to stop the engine immediately on Stop
    without waiting for the WS's bufferedAmount + close-handshake
    round trip. See `_streaming_cancel_events` at top of file.

    Returns 404 if the session isn't currently in-flight (either
    never existed, already finished, or not a streaming session).
    The 404 isn't an error from the client's perspective — it just
    means there's nothing to cancel — but we surface it so misuse
    surfaces in test runs.
    """
    ev = _streaming_cancel_events.get(session_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Streaming session {session_id!r} not found")
    ev.set()
    logger.info(f"({session_id}) HTTP cancel received; cancel_event set.")
    return {"cancelled": session_id}


@app.get("/readyz")
async def readyz():
    """Readiness probe. 200 when the ASR model is loaded; 503 otherwise."""
    if asr_model is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})
    return {"status": "ready", "model_loaded": True}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def get_index_page():
    """Serves the main HTML page for the UI."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        logger.warning(f"index.html not found at {index_path}")
        return HTMLResponse("<h1>ASR Service UI not found.</h1><p>Ensure index.html is in the 'static' directory.</p>", status_code=404)

# --- Global Model Variables & Initialization ---
asr_model: Optional[NeMoASRModelType] = None
global_original_model_device_str: str = "cpu"  # Default, will be updated after model load
global_original_model_dtype_torch: torch.dtype = torch.float32 # Default

# Dual-decoder pinning — both RNNTDecoding instances built once at startup.
try:
    logger.info(f"Loading ASR model: {ASR_MODEL_NAME}...")
    # Load the pre-trained NeMo ASR model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    if asr_model is not None:
        # Disable dithering in the preprocessor for consistent feature extraction in eval mode
        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.eval() # Set model to evaluation mode

        # Pin the model to its resting state ONCE at load — CUDA+bf16 if available,
        # else CPU+fp32. We do NOT change device/dtype per request: NeMo 2.7.3 leaves
        # the encoder in a broken state if you flip device/dtype after a transcribe(),
        # producing `cudaErrorIllegalAddress` on the next forward. NVIDIA's HF Space
        # also loads-and-stays; we now mirror that pattern.
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
        target_dtype = (
            torch.bfloat16
            if (target_device == "cuda" and torch.cuda.is_bf16_supported())
            else torch.float32
        )
        asr_model = asr_model.to(device=target_device, dtype=target_dtype)
        # change_subsampling_conv_chunking_factor(1) once — works around the
        # NeMo 2.7.3 bug in subsampling.py:442 where -1 routes to a forward
        # path calling MaskedConvSequential(x) without the required `lengths` arg.
        try:
            asr_model.change_subsampling_conv_chunking_factor(1)
        except Exception as e_chunk:
            logger.warning(f"change_subsampling_conv_chunking_factor(1) at load failed: {e_chunk}")

        # Disable CUDA graphs at the CONFIG level. Runtime patches on the
        # decoding_computer don't survive — NeMo's transcribe(timestamps=True)
        # internally toggles cfg.decoding.compute_timestamps and re-runs
        # change_decoding_strategy(), which REBUILDS the decoder from config.
        # The rebuilt computer gets use_cuda_graph_decoder=True (default) and
        # the FULL_GRAPH mode comes back on the first transcribe.
        #
        # Fix at the durable layer: edit the config + apply now so any future
        # rebuild keeps cuda graphs off. Also pre-set compute_timestamps=True
        # so transcribe(timestamps=True) doesn't trigger the internal rebuild
        # on the first call (would otherwise hit a brief inconsistent state).
        # Pin the decoding strategy at the CONFIG level. Doing this once at
        # load and never re-running change_decoding_strategy() preserves the
        # captured CUDA graph in decoding_computer across requests (re-running
        # rebuilds the computer and re-captures). compute_timestamps=True is
        # set up-front so nothing internal flips it later.
        try:
            from omegaconf import open_dict
            cfg = asr_model.cfg.decoding
            with open_dict(cfg):
                cfg.compute_timestamps = True
                # greedy_batch keeps the fast GreedyBatchedTDTInfer with its
                # captured graph; parakeet-tdt checkpoints sometimes ship
                # with strategy unset or 'greedy' (non-batched, ~20× slower).
                cfg.strategy = "greedy_batch"
                cfg.preserve_alignments = False
                if "greedy" in cfg:
                    with open_dict(cfg.greedy):
                        cfg.greedy.use_cuda_graph_decoder = USE_CUDA_GRAPHS
                # Token-level confidence — would populate `avg_logprob` on
                # Whisper-shaped segments. Empirical finding (NeMo 2.7.3):
                # the captured FULL_GRAPH CUDA graph silently drops the side
                # output even when this flag is set, so `avg_logprob` stays
                # None until USE_CUDA_GRAPHS=false (~2× decode slowdown).
                # Kept enabled so non-graph deployments populate it for free.
                if "confidence_cfg" in cfg:
                    with open_dict(cfg.confidence_cfg):
                        cfg.confidence_cfg.preserve_token_confidence = True
                        cfg.confidence_cfg.exclude_blank = True
                else:
                    cfg.confidence_cfg = {
                        "preserve_token_confidence": True,
                        "preserve_frame_confidence": False,
                        "preserve_word_confidence": False,
                        "exclude_blank": True,
                        "aggregation": "mean",
                    }
            asr_model.change_decoding_strategy(cfg, verbose=False)
            computer = getattr(asr_model.decoding.decoding, "decoding_computer", None)
            logger.info(
                f"Decoding strategy pinned: greedy_batch, "
                f"use_cuda_graph_decoder={USE_CUDA_GRAPHS}, compute_timestamps=True, "
                f"preserve_token_confidence=True, "
                f"cuda_graphs_mode={getattr(computer, 'cuda_graphs_mode', '<no computer>')!r}"
            )
        except Exception as e_cg:
            logger.warning(f"Decoding strategy pin failed: {e_cg}", exc_info=True)

        # Globals now reflect the pinned resting state, not where NeMo first put it.
        global_original_model_device_str = str(next(asr_model.parameters()).device)
        global_original_model_dtype_torch = next(asr_model.parameters()).dtype

        logger.info(
            f"ASR model '{ASR_MODEL_NAME}' loaded and pinned. "
            f"Resting device: {global_original_model_device_str}, "
            f"dtype: {global_original_model_dtype_torch}."
        )
except Exception as e:
    asr_model = None # Ensure asr_model is None if loading fails
    logger.critical(
        f"FATAL: Could not load ASR model '{ASR_MODEL_NAME}'. Application might not function correctly. Error: {e}",
        exc_info=True
    )

# Silero VAD — loaded once at module init. Used by handle_streaming_pcm to
# drop silent windows before they hit the engine queue. We load the ONNX
# variant (~3x faster per 32ms frame than the PyTorch jit on CPU) to keep
# the producer's hot read loop from throttling chunk throughput on long
# files. Failure to load is non-fatal — VAD stays off and the pipeline
# behaves like pre-VAD.
silero_vad_model: Optional[object] = None
try:
    from silero_vad import load_silero_vad  # type: ignore[import-untyped]
    silero_vad_model = load_silero_vad(onnx=True)
    logger.info("Silero VAD model loaded successfully (ONNX, runs CPU).")
except Exception as e_vad_load:
    silero_vad_model = None
    logger.warning(
        f"Silero VAD unavailable ({e_vad_load!r}); streaming pipeline will not "
        f"gate silence. Set vad_enabled=false on the client to suppress the warning per request."
    )

# Asynchronous lock to ensure exclusive access to the ASR model during transcription calls.
# This prevents concurrent modifications to model state (e.g., device, dtype, attention settings).
model_access_lock = asyncio.Lock()

# Dedicated single-thread executor for ASR calls. CUDA graphs are stream-bound:
# a graph captured on stream A cannot be safely replayed on stream B. Python's
# default ThreadPoolExecutor (used by asyncio.to_thread) hands work to any free
# worker, so successive transcribes can land on different threads → different
# current CUDA streams → graph replay UB → cudaErrorIllegalAddress.
# Pinning all ASR work to one thread keeps the stream identity stable. Eager
# (graphs-off) path also goes through it for consistency; the lock above
# already serializes transcribes, so single-worker doesn't reduce concurrency
# we were actually using.
_asr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")


async def _run_on_asr_executor(fn, *args, **kwargs):
    """asyncio.to_thread() analogue that pins to the dedicated ASR executor."""
    loop = asyncio.get_running_loop()
    if kwargs:
        return await loop.run_in_executor(_asr_executor, functools.partial(fn, *args, **kwargs))
    return await loop.run_in_executor(_asr_executor, fn, *args)


def _log_torch_stream(request_id: str, label: str) -> None:
    """Diagnostic helper: log current thread id and CUDA stream pointer.
    Direct evidence for H1 (thread/stream drift) when graphs are enabled."""
    try:
        tid = threading.get_ident()
        if torch.cuda.is_available():
            stream = torch.cuda.current_stream().cuda_stream
            logger.info(f"({request_id}) {label}: thread={tid} cuda_stream={stream:x}")
        else:
            logger.info(f"({request_id}) {label}: thread={tid} (no cuda)")
    except Exception as e:
        logger.warning(f"({request_id}) {label}: stream-introspect failed: {e}")


async def load_and_preprocess_audio(
    audio_source: str | io.BytesIO,
    target_sample_rate: int,
    request_id: str = "req"
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Asynchronously loads audio from a file path or BytesIO object,
    resamples it to the target sample rate, converts to mono,
    and returns it as a 1D PyTorch tensor along with its duration.

    Args:
        audio_source: Path to the audio file (str) or a BytesIO object.
        target_sample_rate: The desired sample rate for the output waveform.
        request_id: Identifier for logging purposes.

    Returns:
        A tuple containing the waveform tensor (1D float32) and its duration in seconds.
        Returns (None, 0.0) if loading or processing fails.
    """
    waveform_tensor: Optional[torch.Tensor] = None
    audio_duration_s: float = 0.0
    source_description = audio_source if isinstance(audio_source, str) else "BytesIO object"

    try:
        # Offload synchronous torchaudio.load to a separate thread
        wf, sr = await asyncio.to_thread(torchaudio.load, audio_source)

        # Resample if necessary
        if sr != target_sample_rate:
            wf = await asyncio.to_thread(
                torchaudio.functional.resample, wf, sr, target_sample_rate
            )
        
        # Convert to mono by averaging channels if stereo or multi-channel
        if wf.shape[0] > 1:
            wf = torch.mean(wf, dim=0, keepdim=True)

        waveform_tensor = wf.squeeze(0) # Remove channel dimension if mono, ensure 1D
        audio_duration_s = waveform_tensor.shape[0] / target_sample_rate
        
        if waveform_tensor.numel() == 0: # Check for empty tensor after processing
            logger.warning(f"({request_id}) Audio processing resulted in an empty waveform from {source_description}.")
            return None, 0.0
            
        return waveform_tensor, audio_duration_s
        
    except Exception as e_load:
        logger.error(
            f"({request_id}) Failed to load/preprocess audio from {source_description}. Error: {e_load}",
            exc_info=True
        )
        return None, 0.0


async def _apply_model_settings_for_session(
    decision_duration_s: float,
    target_processing_device: str,
    target_operational_dtype: torch.dtype,
    long_audio_threshold_config: float,
    request_id: str = "req"
) -> bool:
    """
    Switch to long-audio attention mode if needed for this session.

    The model is pinned to its resting state (CUDA+bf16 or CPU+fp32, with
    subsampling_conv_chunking_factor=1) at load time and stays there. We do
    NOT change device or dtype per-request — NeMo 2.7.3 leaves the encoder in
    a broken state when device/dtype is flipped between transcribes, producing
    cudaErrorIllegalAddress on the next forward. Mirrors NVIDIA's HF Space.

    target_processing_device and target_operational_dtype are accepted for
    caller compatibility but ignored; the session's actual device/dtype is
    whatever the model is pinned to.

    Returns:
        True iff long-audio attention (rel_pos_local_attn) was activated and
        will need to be reverted at end of session.
    """
    global asr_model
    if asr_model is None:
        logger.error(f"({request_id}) ASR model is None in _apply_model_settings_for_session.")
        return False

    if decision_duration_s <= long_audio_threshold_config:
        logger.debug(
            f"({request_id}) Session: short audio ({decision_duration_s:.2f}s <= {long_audio_threshold_config:.2f}s); "
            f"keeping resting attention model. No state change."
        )
        return False

    logger.info(
        f"({request_id}) Session: long audio ({decision_duration_s:.2f}s > {long_audio_threshold_config:.2f}s); "
        f"switching to rel_pos_local_attn[256,256]."
    )
    try:
        # ASRModuleMixin.change_attention_model takes only (self_attention_model,
        # att_context_size, update_config) — it does NOT accept a device kwarg.
        # Internally it forwards self.device to the encoder, so new modules
        # land on the model's current device automatically.
        await _run_on_asr_executor(
            asr_model.change_attention_model,
            "rel_pos_local_attn", [256, 256], True,
        )
        # New RelPositionMultiHeadAttentionLongformer modules are constructed
        # with default float32 weights — even though the rest of the model is
        # at its pinned dtype (e.g. bf16). Forward then dies with
        # "mat1 and mat2 must have the same dtype". Cast the whole model to
        # its current dtype so new modules join the bf16 majority.
        pinned_dtype = global_original_model_dtype_torch
        await _run_on_asr_executor(asr_model.to, dtype=pinned_dtype)
        return True
    except Exception as e_long:
        logger.warning(f"({request_id}) Session: Failed to apply long-audio attention: {e_long}")
        return False


async def _revert_model_to_global_original_state(
    long_audio_settings_were_active_for_session: bool,
    session_processing_device: str, # Retained for caller compat; only used for CUDA cache clear
    request_id: str = "req"
):
    """
    Revert the long-audio attention switch (if it ran) and clear CUDA cache.

    The model's device/dtype is NOT changed — it lives on the resting state
    set at load time (CUDA+bf16 or CPU+fp32). See _apply_model_settings_for_session
    for the rationale (NeMo 2.7.3 cudaErrorIllegalAddress on device/dtype churn).
    """
    global asr_model
    if asr_model is None:
        logger.error(f"({request_id}) ASR model is None in _revert_model_to_global_original_state.")
        return

    try:
        if long_audio_settings_were_active_for_session:
            logger.info(f"({request_id}) End of Session: Reverting to rel_pos attention.")
            try:
                # See _apply_model_settings_for_session for signature explanation.
                await _run_on_asr_executor(
                    asr_model.change_attention_model,
                    "rel_pos", None, True,
                )
                # Same dtype-mismatch fix as the apply path: new modules from
                # change_attention_model are float32 by default; cast back to
                # the model's pinned dtype.
                pinned_dtype = global_original_model_dtype_torch
                await _run_on_asr_executor(asr_model.to, dtype=pinned_dtype)
            except Exception as e_rev_long_specific:
                logger.warning(f"({request_id}) End of Session: Failed to revert long-audio attention: {e_rev_long_specific}")

        # Historical cargo-cult call — torch.cuda.empty_cache() between
        # transcribes is fatal when NeMo's CUDA-graph decoder is on. The
        # captured graph holds raw pointers into LabelLoopingState's static
        # buffers; empty_cache releases the underlying caching-allocator
        # blocks, so the next graph replay reads freed/reused memory and
        # the kernel hits cudaErrorIllegalAddress (NeMo issue #14727).
        # PyTorch's caching allocator already releases unreferenced
        # blocks under pressure, so we only gc/empty_cache when graphs
        # are off AND we're on CUDA. Even then it's optional — kept here
        # so the legacy path stays bit-identical to its previous behavior.
        if (
            not USE_CUDA_GRAPHS
            and session_processing_device == "cuda"
            and torch.cuda.is_available()
        ):
            await _run_on_asr_executor(gc.collect)
            await _run_on_asr_executor(torch.cuda.empty_cache)
    except Exception as e_restore_globally:
        logger.error(f"({request_id}) Error during final model state reversion: {e_restore_globally}", exc_info=True)


def parse_request_config(
    c_len: Optional[float] = None,
    c_ov: Optional[float] = None,
    b_size: Optional[int] = None,
    l_thresh: Optional[float] = None,
    strategy: Optional[str] = None,
    early_buffer_target_s: Optional[float] = None,
    live_latency: Optional[bool] = None,
) -> dict:
    """
    Parses and validates common ASR request configuration parameters.

    Uses global default values (TRANSCRIBE_CHUNK_LEN, etc.) if specific
    parameters are not provided. Raises ValueError for invalid parameter values.

    Args:
        c_len: Desired chunk length in seconds for ASR processing.
        c_ov: Desired chunk overlap in seconds for ASR processing.
        b_size: Batch size for ASR model inference.
        l_thresh: Long audio threshold in seconds to determine model settings
                  (e.g., attention mechanism).
        strategy: Processing strategy. One of ProcessingStrategy values.
                  None falls back to DEFAULT_STRATEGY. Unknown values raise.
        early_buffer_target_s: Streaming mode — PCM seconds buffered before
                  the first partial transcript is emitted.
        live_latency: When True, streaming mode uses the 10-2-2 preset for ~4s
                  latency; when False/None, uses 10-10-5 for offline-like quality.

    Returns:
        A dictionary containing the validated configuration parameters.

    Raises:
        ValueError: If any parameter value is outside its allowed range.
    """
    requested_strategy = (strategy or DEFAULT_STRATEGY).lower()
    try:
        strategy_enum = ProcessingStrategy(requested_strategy)
    except ValueError:
        valid = [s.value for s in ProcessingStrategy]
        raise ValueError(f"Invalid strategy '{requested_strategy}'. Must be one of: {valid}.")

    config = {
        "chunk_length": c_len if c_len is not None else TRANSCRIBE_CHUNK_LEN,
        "chunk_overlap": c_ov if c_ov is not None else TRANSCRIBE_OVERLAP,
        "batch_size": b_size if b_size is not None else CHUNKING_BATCH_SIZE,
        "long_audio_threshold": l_thresh if l_thresh is not None else LONG_AUDIO_THRESHOLD_S,
        "strategy": strategy_enum,
        "early_buffer_target_s": (
            early_buffer_target_s if early_buffer_target_s is not None else EARLY_BUFFER_TARGET_S
        ),
        "live_latency": bool(live_latency) if live_latency is not None else False,
    }

    if not (0 < config["chunk_length"] <= 300): # Max 5 minutes chunk
        raise ValueError("chunk_length must be > 0 and <= 300 seconds.")
    if not (0 <= config["chunk_overlap"] < config["chunk_length"]):
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_length.")
    if not (1 <= config["batch_size"] <= 32): # Practical limit for batch size
        raise ValueError("batch_size must be between 1 and 32.")
    if not (0 <= config["long_audio_threshold"] <= 3600): # Max 1 hour threshold
        raise ValueError("long_audio_threshold must be >= 0 and <= 3600 seconds.")
    if not (0 < config["early_buffer_target_s"] <= 300):
        raise ValueError("early_buffer_target_s must be > 0 and <= 300 seconds.")

    return config


# Slice-safety margin for SPLIT_FULL. The per-slice duration is
# MAX_FULL_WAVEFORM_S * SLICE_SAFETY so we stay under the FULL-mode ceiling
# even after accounting for the overlap tail.
SLICE_SAFETY = float(os.getenv("SPLIT_FULL_SLICE_SAFETY", 0.95))
SPLIT_FULL_OVERLAP_S = float(os.getenv("SPLIT_FULL_OVERLAP_S", 15.0))


def resolve_strategy(
    audio_duration_s: Optional[float],
    client_config: dict,
    is_streaming: bool,
) -> ProcessingStrategy:
    """
    Resolve `?strategy=` to a concrete dispatch mode.

    Cascade:
      - OFFLINE      → STREAMING if the connection is actually a WS
                       (transport auto-correct); else if duration is unknown
                       (WS-accumulate path before load) return OFFLINE as a
                       placeholder so the caller re-resolves later; else
                       FULL if duration ≤ MAX_FULL_WAVEFORM_S, else SPLIT_FULL.
      - FULL / SPLIT_FULL / STREAMING → returned as-is.

    `MAX_FULL_WAVEFORM_S` (env, default 1440s = 24min) is the deployment's
    verified FULL ceiling. SPLIT_FULL routes long offline files to back-to-back
    FULL passes at FULL throughput (~3-5× faster than the chunked engine on
    this hardware).
    """
    requested = client_config.get("strategy", ProcessingStrategy.OFFLINE)
    if isinstance(requested, str):
        requested = ProcessingStrategy(requested)

    if requested == ProcessingStrategy.OFFLINE:
        if is_streaming:
            # Client asked for offline on a WS — fall through to streaming
            # rather than error; the engine on this socket is the streaming
            # one regardless.
            return ProcessingStrategy.STREAMING
        if audio_duration_s is None:
            # WS-accumulate path before we know the duration: defer to the
            # caller, which calls back into resolve_strategy() with the now-
            # known duration after load_and_preprocess_audio.
            return ProcessingStrategy.OFFLINE
        if audio_duration_s <= MAX_FULL_WAVEFORM_S:
            return ProcessingStrategy.FULL
        return ProcessingStrategy.SPLIT_FULL

    # Explicit FULL / SPLIT_FULL / STREAMING → return as-is.
    return requested


def parse_websocket_config(client_cfg: dict) -> dict:
    """
    Parses and validates WebSocket-specific client configuration.

    This extends `parse_request_config` with parameters relevant to
    audio stream characteristics like sample rate, channels, and format.

    Args:
        client_cfg: A dictionary containing configuration sent by the WebSocket client.
                    Expected keys include "sample_rate", "channels", "bytes_per_sample",
                    "format", and optionally "chunk_length", "chunk_overlap", etc.

    Returns:
        A dictionary containing the validated and combined configuration.

    Raises:
        ValueError: If required fields are missing or parameter values are invalid.
                    Specifically, client must send a real file format (e.g. "wav", "mp3")
                    and not "pcm" for the 'format' field in streaming mode, as ffmpeg
                    needs to know the input container/codec.
    """
    required_fields = ["sample_rate", "channels", "bytes_per_sample", "format"]
    missing_fields = [field for field in required_fields if field not in client_cfg]
    if missing_fields:
        raise ValueError(f"Missing required WebSocket configuration fields: {', '.join(missing_fields)}")

    # Client must specify the actual audio format for ffmpeg to decode.
    # 'pcm' is too generic for ffmpeg's input format detection.
    audio_format = str(client_cfg["format"]).lower()

    # Get common ASR config (chunk_length, overlap, batch_size, long_audio_threshold, strategy)
    asr_config = parse_request_config(
        client_cfg.get("chunk_length"),
        client_cfg.get("chunk_overlap"),
        client_cfg.get("batch_size"),
        client_cfg.get("long_audio_threshold"),
        strategy=client_cfg.get("strategy"),
        early_buffer_target_s=client_cfg.get("early_buffer_target_s"),
        live_latency=client_cfg.get("live_latency"),
    )

    # Combine with WebSocket-specific audio stream parameters
    asr_config.update({
        "sample_rate": int(client_cfg["sample_rate"]),
        "channels": int(client_cfg["channels"]),
        "bytes_per_sample": int(client_cfg["bytes_per_sample"]), # Bytes per sample of original audio
        "format": audio_format,
    })
    
    # Add any other client-provided parameters that aren't already handled
    for key, value in client_cfg.items():
        if key not in asr_config and key not in required_fields:
            asr_config[key] = value
            
    if not (1 <= asr_config["channels"] <= 8): # Practical limit for channels
        raise ValueError("Number of audio channels must be between 1 and 8.")
    # sample_rate and bytes_per_sample are noted but primarily used by ffmpeg or initial processing;
    # the ASR model itself expects MODEL_SAMPLE_RATE mono.

    # VAD + HPF knobs (range validation; types already coerced by the
    # generic forwarder above). Use sentinel `_UNSET` so callers omitting
    # a field fall through to defaults inside _build_vad_state.
    if 'vad_threshold' in asr_config:
        v = float(asr_config['vad_threshold'])
        if not (0.0 <= v <= 1.0):
            raise ValueError('vad_threshold must be in [0, 1].')
        asr_config['vad_threshold'] = v
    if 'vad_consecutive' in asr_config:
        v_int = int(asr_config['vad_consecutive'])
        if not (1 <= v_int <= 32):
            raise ValueError('vad_consecutive must be in [1, 32].')
        asr_config['vad_consecutive'] = v_int
    for k in ('vad_hangover_ms', 'vad_pad_min_gap_ms', 'vad_pad_duration_ms'):
        if k in asr_config:
            v = float(asr_config[k])
            if not (0 <= v <= 10_000):
                raise ValueError(f'{k} must be in [0, 10000] ms.')
            asr_config[k] = v
    if 'hpf_hz' in asr_config:
        v = float(asr_config['hpf_hz'])
        if not (0 <= v <= 1000):
            raise ValueError('hpf_hz must be in [0, 1000].')
        asr_config['hpf_hz'] = v

    return asr_config


def _segments_to_words(segments: List[dict]) -> List[dict]:
    """Compute word-level entries from a batch of segments for the WS stream.

    Each segment may carry per-token timestamps in `token_times` (streaming
    engine emits real values from the decoder; offline path likewise). When
    present we use them directly so word boundaries land on the decoder's
    actual emission times. When absent (older payloads or alternate code
    paths) we fall back to uniform interpolation across the segment span.

    Returns the flattened list across all input segments in order.
    """
    if asr_model is None:
        return []
    out: List[dict] = []
    for seg in segments:
        seg_tokens = seg.get("tokens") or []
        if not seg_tokens:
            continue
        seg_token_times = seg.get("token_times")
        if seg_token_times and len(seg_token_times) == len(seg_tokens):
            times = [float(t) for t in seg_token_times]
        else:
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start))
            n = len(seg_tokens)
            if n == 1:
                times = [seg_start]
            else:
                span = max(seg_end - seg_start, 0.0)
                times = [seg_start + (i / (n - 1)) * span for i in range(n)]
        try:
            out.extend(tokens_to_words(seg_tokens, times, asr_model.tokenizer))
        except Exception:
            # Defensive — if the tokenizer chokes on a particular batch we'd
            # rather emit segments without words than drop the whole frame.
            continue
    return out


# Silero VAD operates on 32 ms (512-sample) frames at 16 kHz mono s16le.
_VAD_FRAME_SAMPLES = 512
_VAD_FRAME_BYTES = _VAD_FRAME_SAMPLES * 2  # int16 LE
_VAD_FRAME_MS = 32.0
_VAD_NOISE_AMPLITUDE = 1e-3  # ~-60 dBFS gaussian; protects TDT decoder timing


def _build_vad_state(client_config: dict) -> dict:
    """Build the per-session VAD state + config.

    Two orthogonal axes:
      - `enabled`  : whether the silence-clip state machine drops bytes
                     before they reach the engine. User-controlled via
                     the "Voice detection" sidebar toggle.
      - `meter`    : whether Silero inference runs *at all* to populate
                     the speech_ms counter that drives the live RTFx
                     metric. Always on when the model is loaded —
                     inference is cheap (~1ms per 32ms frame) and the
                     metric needs it even when clip is off.

    If Silero is unavailable, both flip to False and the filter fast-
    paths through with `speech_ms` stuck at 0 (client falls back to —).
    """
    meter = silero_vad_model is not None
    enabled = bool(client_config.get('vad_enabled', True)) and meter
    if meter:
        try:
            silero_vad_model.reset_states()  # type: ignore[attr-defined]
        except Exception:
            pass
    return {
        'enabled': enabled,
        'meter': meter,
        'model': silero_vad_model,
        'threshold': float(client_config.get('vad_threshold', 0.5)),
        'consecutive': max(1, int(client_config.get('vad_consecutive', 3))),
        'hangover_ms': float(client_config.get('vad_hangover_ms', 500)),
        'pad_min_gap_ms': float(client_config.get('vad_pad_min_gap_ms', 400)),
        # Default 0 — we don't synthesize audio for the model by default.
        # The model sees the speech-only stream as cut by VAD. Power
        # users can re-enable via the Advanced toggle if they ever want
        # to test the old behaviour.
        'pad_duration_ms': float(client_config.get('vad_pad_duration_ms', 0)),
        # Mutable state
        'mode': 'silent',           # 'silent' | 'speech'
        'consec': 0,                # consecutive loud frames toward N-of-consecutive
        'pending': bytearray(),     # candidate frames pre-confirmation
        'silent_run_ms': 0.0,       # ms of pure silent since last speech
        'last_speech_ms': 0.0,      # cumulative engine ms at last loud frame (hangover)
        'engine_ms': 0.0,           # engine-clock ms (only counts forwarded audio)
        'in_buf': bytearray(),      # incoming bytes pending 32 ms frame alignment
        # Speech-clock → wall-clock translation. The engine sees a
        # contiguous "speech-only" stream when VAD drops silence; token
        # timestamps it emits are on the engine clock. To map back to
        # the wall-clock of the original recording (the audio player
        # plays the untouched track) we track `wall_ms` independently:
        # wall_ms increments by _VAD_FRAME_MS for every VAD frame the
        # filter actually consumes, regardless of what's done with it.
        # engine_ms already tracks what the engine has seen. The offset
        # at any moment is `wall_ms - engine_ms`.
        #
        # The offset is piecewise-constant from the engine's POV — it
        # only changes during silent/pending stretches (where engine_ms
        # doesn't advance) and at silent→speech confirmation (where
        # padding + pending flush advance engine_ms without advancing
        # wall_ms). We append one breakpoint per confirmation. Lookup
        # is bisect_right against time_map_keys; empty time_map ↔
        # identity ↔ VAD off or no silence ever dropped.
        'wall_ms': 0.0,
        'time_map': [],             # list[(engine_t_s, wall_offset_s)]
        'time_map_keys': [],        # parallel list of engine_t_s for bisect
        # Stats for end-of-session logging
        'dropped_frames': 0,
        'speech_frames': 0,
        'padded_ms': 0.0,
        # Cumulative wall-clock ms classified as speech by Silero VAD,
        # *regardless* of the clip-state-machine outcome. Ticks per frame
        # where prob >= threshold, including frames the gate later
        # drops as part of pending/hangover bookkeeping. Drives the
        # live RTFx denominator (speech_committed_s / speech_received_s)
        # and is the single source of truth for "how much speech have
        # we heard". 0 if the Silero model isn't loaded.
        'speech_ms': 0.0,
        # Cumulative wall-clock ms covered by segments the engine has
        # committed (sum of post-translation seg.end - seg.start). Lives
        # on vad_state so both PCM and URL consumer loops can update it
        # in place and the final-payload assembly can read it without
        # extra nonlocal plumbing.
        'committed_speech_ms': 0.0,
    }


def _vad_filter(pcm_bytes: bytes, st: dict) -> bytes:
    """Run Silero on incoming PCM, return the engine-bound subset.

    Two responsibilities, kept distinct:

    1. **Metric (always-on when Silero is loaded).** Every 32 ms frame
       runs VAD inference; `speech_ms` ticks for any frame with
       `prob >= threshold`. This feeds the live RTFx counter regardless
       of whether the user has clipping enabled — silence still has to
       fall out of both numerator and denominator for the metric to be
       honest.

    2. **Clip state machine (gated on `st['enabled']`).** When clip is
       on, drops silence, hangs over speech, optionally pads onsets —
       same behavior as before. When clip is off, every frame is
       forwarded to the engine, but the metric counters still tick.

    Bytes returned should be appended to the engine's PCM accumulator.

    Timestamps emitted by the engine are on the speech-only "engine"
    timeline when clip is on. Callers should translate them back to
    wall-clock with `_translate_segment_times(seg, vad_state)` before
    exposing them to the client — the audio player plays the
    unfiltered wall-clock track and word highlights need to match it.
    When clip is off, engine clock == wall clock (no time_map entries).
    """
    # No Silero at all — pass through, advance engine_ms for byte
    # accounting, speech_ms stays at 0 (client will fall back to —).
    if not st['meter']:
        st['engine_ms'] += (len(pcm_bytes) / 2) / 16.0  # 16 samples/ms @ 16 kHz
        return pcm_bytes

    st['in_buf'].extend(pcm_bytes)
    out = bytearray()
    model = st['model']
    threshold = st['threshold']
    clip_enabled = st['enabled']
    consec_target = st['consecutive']
    hangover_ms = st['hangover_ms']
    pad_min = st['pad_min_gap_ms']
    pad_dur = st['pad_duration_ms']

    while len(st['in_buf']) >= _VAD_FRAME_BYTES:
        frame_bytes = bytes(st['in_buf'][:_VAD_FRAME_BYTES])
        del st['in_buf'][:_VAD_FRAME_BYTES]

        # Every consumed VAD frame is 32ms of wall-clock — we always
        # received it from the audio source. engine_ms ticks only when
        # we forward bytes to the engine (below). The running difference
        # wall_ms - engine_ms is the wall-clock offset we snapshot at
        # each silent→speech confirmation.
        st['wall_ms'] += _VAD_FRAME_MS

        samples_np = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        samples_t = torch.from_numpy(samples_np)
        try:
            with torch.no_grad():
                prob = float(model(samples_t, 16000).item())  # type: ignore[misc]
        except Exception:
            # If VAD inference fails on any frame, fall back to forwarding it.
            out.extend(frame_bytes)
            st['engine_ms'] += _VAD_FRAME_MS
            continue
        is_loud = prob >= threshold
        # Metric counter — independent of clipping. Counts every frame
        # the model classified as speech, including ones the gate later
        # drops as part of pending/hangover bookkeeping.
        if is_loud:
            st['speech_ms'] += _VAD_FRAME_MS

        # Clip off → forward every frame, skip the state-machine entirely.
        # Counters still tick (above) so the metric stays honest.
        if not clip_enabled:
            out.extend(frame_bytes)
            st['engine_ms'] += _VAD_FRAME_MS
            continue

        if st['mode'] == 'silent':
            if is_loud:
                st['consec'] += 1
                st['pending'].extend(frame_bytes)
                if st['consec'] >= consec_target:
                    # Confirmed speech. Optionally pre-pad if the prior gap
                    # was long enough; protects the TDT decoder's blank-
                    # token timing assumptions.
                    if st['silent_run_ms'] >= pad_min and pad_dur > 0:
                        pad_n = int(pad_dur * 16)  # 16 samples/ms @ 16 kHz
                        pad_pcm = (
                            np.random.randn(pad_n) * _VAD_NOISE_AMPLITUDE * 32768.0
                        ).astype(np.int16).tobytes()
                        out.extend(pad_pcm)
                        st['engine_ms'] += pad_dur
                        st['padded_ms'] += pad_dur
                    # Flush the pending speech frames.
                    pending_ms = (len(st['pending']) / 2) / 16.0
                    out.extend(st['pending'])
                    st['engine_ms'] += pending_ms
                    st['pending'].clear()
                    st['mode'] = 'speech'
                    st['consec'] = 0
                    st['silent_run_ms'] = 0.0
                    st['last_speech_ms'] = st['engine_ms']
                    st['speech_frames'] += 1
                    # Record the new (engine_t, offset) breakpoint. Offset
                    # is computed from the independently tracked wall/engine
                    # clocks: any tokens emitted at engine_t >= this point
                    # map back to wall_t = engine_t + offset.
                    eng_s = st['engine_ms'] / 1000.0
                    off_s = (st['wall_ms'] - st['engine_ms']) / 1000.0
                    st['time_map'].append((eng_s, off_s))
                    st['time_map_keys'].append(eng_s)
                else:
                    # Still pending. Frame stays buffered (wall already ticked
                    # at top of loop; engine_ms unchanged).
                    st['silent_run_ms'] += _VAD_FRAME_MS
                    st['dropped_frames'] += 1
            else:
                # Real silent frame. wall already ticked; engine doesn't.
                st['consec'] = 0
                st['pending'].clear()
                st['silent_run_ms'] += _VAD_FRAME_MS
                st['dropped_frames'] += 1
        else:  # mode == 'speech'
            # Always forward in speech mode. wall and engine both tick by
            # 32ms here, so the offset stays flat — no new breakpoint
            # needed (the most recent breakpoint still describes the
            # current segment of the timeline).
            out.extend(frame_bytes)
            st['engine_ms'] += _VAD_FRAME_MS
            st['speech_frames'] += 1
            if is_loud:
                st['last_speech_ms'] = st['engine_ms']
            elif st['engine_ms'] - st['last_speech_ms'] > hangover_ms:
                # Drop back to silent — done with this utterance.
                st['mode'] = 'silent'
                st['consec'] = 0
                st['silent_run_ms'] = 0.0

    return bytes(out)


def _wall_time_s(engine_t_s: float, vad_state: dict) -> float:
    """Translate an engine-clock timestamp back to wall-clock seconds.

    Identity when the time map is empty (VAD off, or VAD on but no silence
    was ever dropped). Otherwise picks the offset from the most recent
    breakpoint whose engine-time key is <= engine_t_s. Values before the
    very first breakpoint use that first breakpoint's offset — they
    correspond to anything emitted before the first silent→speech
    confirmation, which in practice is rare but bounded.
    """
    time_map = vad_state.get('time_map') or []
    if not time_map:
        return float(engine_t_s)
    keys = vad_state.get('time_map_keys') or [m[0] for m in time_map]
    idx = bisect.bisect_right(keys, engine_t_s) - 1
    if idx < 0:
        return float(engine_t_s) + float(time_map[0][1])
    return float(engine_t_s) + float(time_map[idx][1])


def _translate_segment_times(segments: List[dict], vad_state: dict) -> None:
    """In-place: rewrite each segment's start/end/token_times from engine
    clock to wall clock using `vad_state`'s time map. No-op when time map
    is empty (VAD off — engine and wall clocks coincide)."""
    if not vad_state.get('time_map'):
        return
    for seg in segments:
        if 'start' in seg:
            seg['start'] = round(_wall_time_s(float(seg['start']), vad_state), 3)
        if 'end' in seg:
            seg['end'] = round(_wall_time_s(float(seg['end']), vad_state), 3)
        token_times = seg.get('token_times')
        if token_times:
            seg['token_times'] = [
                round(_wall_time_s(float(t), vad_state), 3) for t in token_times
            ]


def _accumulate_committed_speech(segments: List[dict], vad_state: dict) -> None:
    """Add the wall-clock spans of newly committed segments to the
    cumulative `committed_speech_ms` counter. Call AFTER
    `_translate_segment_times` so the spans are in wall-clock terms."""
    vad_state['committed_speech_ms'] += sum(
        max(0.0, float(s.get('end', 0.0)) - float(s.get('start', 0.0))) * 1000.0
        for s in segments
    )


def _streaming_counters(vad_state: dict) -> dict:
    """Snapshot of the streaming counters the client uses to compute the
    live RTFx (speech_committed_s / speech_received_s) and to surface
    audio receipt in the tooltip. Splatted into segments_batch /
    partial_segment / final_transcription payloads via `**`."""
    return {
        'audio_received_s': round(vad_state.get('wall_ms', 0.0) / 1000.0, 3),
        'speech_received_s': round(vad_state.get('speech_ms', 0.0) / 1000.0, 3),
        'speech_committed_s': round(vad_state.get('committed_speech_ms', 0.0) / 1000.0, 3),
    }


async def handle_streaming_pcm(
    websocket: WebSocket,
    session_id: str,
    processing_device: str,
    client_config: dict
):
    """
    Handles the live STREAMING pipeline for a WebSocket connection.

    Producer/consumer pattern:
    - Producer: receives audio from the WS, pipes through ffmpeg → 16 kHz mono
      PCM, segments into engine-sized chunks, puts (tensor, offset_s) onto a
      queue.
    - Consumer: drains the queue and feeds each chunk into the
      StreamingPrevBatchedEngine. Newly committed sentence-bounded segments
      are emitted as `segments_batch` messages.

    On EOF: emits `final_transcription` aggregated from the streamed segments.

    Args:
        websocket: The active WebSocket connection.
        session_id: A unique identifier for this streaming session.
        processing_device: The device ("cuda" or "cpu") for ASR model inference.
        client_config: Parsed configuration dictionary from the client, including
                       chunk_length, overlap, batch_size, format, etc.
    """
    sent_segments_pcm: List[dict] = []  # all segments sent to client, for final aggregation

    # Per-session VAD state. Disabled gracefully if Silero isn't loaded or
    # the client opted out. See _vad_filter for the actual gating logic.
    vad_state = _build_vad_state(client_config)
    if vad_state['enabled']:
        logger.info(
            f"({session_id}) Stream: VAD enabled "
            f"(threshold={vad_state['threshold']:.2f}, "
            f"consec={vad_state['consecutive']}, "
            f"hangover={vad_state['hangover_ms']:g}ms, "
            f"pad>{vad_state['pad_min_gap_ms']:g}ms→{vad_state['pad_duration_ms']:g}ms)."
        )
    else:
        logger.info(f"({session_id}) Stream: VAD disabled.")

    live_latency = bool(client_config.get("live_latency", False))

    # The producer pushes engine-sized chunks with zero overlap; the engine
    # owns its own context window via its FIFO buffer.
    engine_chunk_len_s = STREAMING_LIVE_CHUNK_S if live_latency else STREAMING_CHUNK_S
    right_context_secs = STREAMING_LIVE_RIGHT_CONTEXT_S if live_latency else STREAMING_RIGHT_CONTEXT_S
    engine_total_buffer_s = STREAMING_LEFT_CONTEXT_S + engine_chunk_len_s + right_context_secs
    preset_label = (
        f"{STREAMING_LEFT_CONTEXT_S:g}-{engine_chunk_len_s:g}-{right_context_secs:g} "
        f"{'live' if live_latency else 'offline-like'}"
    )
    asr_chunk_len_s = engine_chunk_len_s
    asr_chunk_overlap_s = 0.0
    logger.info(
        f"({session_id}) Stream: chunk={engine_chunk_len_s:g}s, "
        f"buffer={engine_total_buffer_s:g}s ({preset_label})."
    )

    # Target PCM characteristics (output from ffmpeg, input to ASR chunker)
    target_pcm_sample_rate = MODEL_SAMPLE_RATE # 16000 Hz
    target_pcm_bytes_per_sample = 2 # For s16le (16-bit signed little-endian PCM)
    
    samples_per_asr_chunk = int(asr_chunk_len_s * target_pcm_sample_rate)
    samples_per_asr_stride = int((asr_chunk_len_s - asr_chunk_overlap_s) * target_pcm_sample_rate)
    
    bytes_per_asr_chunk_target_pcm = samples_per_asr_chunk * target_pcm_bytes_per_sample
    bytes_per_asr_stride_target_pcm = samples_per_asr_stride * target_pcm_bytes_per_sample

    # Queue for (audio_tensor, offset_s) tuples from producer to consumer
    # Maxsize helps manage backpressure if consumer is slower.
    chunk_queue: asyncio.Queue[Optional[Tuple[torch.Tensor, float]]] = asyncio.Queue(
        maxsize=client_config["batch_size"] * 2 # Allow some buffering
    )
    producer_done_event = asyncio.Event() # Signals producer has finished all its tasks
    # Flips True when the client disconnects (or we otherwise want to bail).
    # Set by the producer's disconnect handlers (feed_ffmpeg_stdin catches
    # WebSocketDisconnect / RuntimeError on close); consumed by the consumer
    # to break out of its drain loop and skip the engine.flush() trailing
    # work, and by the engine itself via the cancel_check callback to bail
    # between forward passes. Without this, when the client closed mid-
    # upload the consumer kept processing all already-queued chunks (a
    # multi-second tail of GPU work the user never sees).
    cancel_event = asyncio.Event()

    accumulated_asr_processing_time_s: float = 0.0
    total_duration_processed_seconds_for_asr: float = 0.0 # Based on PCM from ffmpeg

    def _create_asr_tensor_from_bytes(pcm_bytes: bytes) -> torch.Tensor:
        """Synchronous helper to convert raw s16le PCM bytes to a float32 tensor."""
        # Convert s16le bytes to int16 tensor, then to float32 in range [-1.0, 1.0]
        return torch.frombuffer(pcm_bytes, dtype=torch.int16).float() / 32768.0

    async def producer():
        """
        Producer coroutine:
        1. Receives audio from WebSocket.
        2. Feeds it to ffmpeg.
        3. Reads standardized PCM from ffmpeg.
        4. Creates ASR chunks (tensors) and puts them on `chunk_queue`.
        """
        nonlocal total_duration_processed_seconds_for_asr
        # Buffer for PCM data read from ffmpeg, used to form ASR chunks
        pcm_buffer_for_asr_chunks = bytearray()
        # Keeps track of total samples *advanced* in the ffmpeg output stream
        # to calculate correct time offsets for ASR chunks.
        total_samples_in_pcm_buffer_for_offset_calc = 0
        total_bytes_fed_to_ffmpeg = 0 # For logging/debugging

        # ffmpeg command:
        # -i pipe:0 : Read input from stdin
        # -f s16le : Output format: signed 16-bit little-endian PCM
        # -ac 1 : Output audio channels: 1 (mono)
        # -ar str(MODEL_SAMPLE_RATE) : Output audio sample rate: 16000 Hz
        # -acodec pcm_s16le : Output codec: PCM s16le
        # pipe:1 : Write output to stdout
        # -hide_banner -loglevel error : Reduce ffmpeg's console noise
        # Optional high-pass filter to cut rumble/hum/DC before VAD + ASR see
        # the signal. Pulled from client config; 0 (or unset) disables.
        hpf_hz = float(client_config.get('hpf_hz', 0) or 0)
        af_args: List[str] = []
        if hpf_hz > 0:
            af_args = ['-af', f'highpass=f={hpf_hz:g}']
        ffmpeg_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0',  # Input from stdin
            *af_args,        # Optional audio filter chain (e.g. highpass)
            '-f', 's16le',   # Output format: signed 16-bit PCM
            '-ac', '1',      # Output channels: mono
            '-ar', str(MODEL_SAMPLE_RATE), # Output sample rate
            '-acodec', 'pcm_s16le', # Output codec
            'pipe:1'         # Output to stdout
        ]
        logger.info(f"({session_id}) Stream Producer: Starting ffmpeg with command: {' '.join(ffmpeg_command)}")
        
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE  # Capture stderr for error reporting
        )
        logger.info(f"({session_id}) Stream Producer: ffmpeg process started (PID: {process.pid}).")

        async def feed_ffmpeg_stdin():
            """Reads from WebSocket and writes to ffmpeg's stdin."""
            nonlocal total_bytes_fed_to_ffmpeg
            try:
                while True:
                    if websocket.application_state != WebSocketState.CONNECTED:
                        logger.info(f"({session_id}) Feed ffmpeg: WebSocket no longer connected. Cancelling pipeline.")
                        cancel_event.set()
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        break

                    try:
                        # Timeout for receive to prevent indefinite blocking if client goes silent
                        message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"({session_id}) Feed ffmpeg: Timeout waiting for message from client. Assuming stream ended.")
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close() # Close stdin to signal ffmpeg to finish
                        break
                    except RuntimeError as e_ws_recv: # FastAPI can raise this on disconnect
                        if "disconnect" in str(e_ws_recv).lower():
                             logger.info(f"({session_id}) Feed ffmpeg: WebSocket disconnected during receive.")
                        else:
                            logger.error(f"({session_id}) Feed ffmpeg: WebSocket receive error: {e_ws_recv}", exc_info=True)
                        # Client gone — short-circuit the rest of the pipeline.
                        # See `cancel_event` declaration in handle_streaming_pcm
                        # for the propagation chain.
                        cancel_event.set()
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        break # Exit loop on disconnect or critical error

                    if 'text' in message and message['text'].upper() == "END":
                        logger.info(f"({session_id}) Feed ffmpeg: 'END' signal received from client. Closing ffmpeg stdin.")
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        break # End of stream signaled by client
                    
                    if 'bytes' in message:
                        chunk_data = message['bytes']
                        # An empty binary frame is the documented EOF signal
                        # (alongside the text "END"). Honor it immediately so
                        # the trailing segment lands within ~1s of client Stop
                        # instead of after the 30s receive timeout above.
                        if not chunk_data:
                            logger.info(f"({session_id}) Feed ffmpeg: empty binary frame (EOF). Closing ffmpeg stdin.")
                            if process.stdin and not process.stdin.is_closing():
                                process.stdin.close()
                            break
                        if process.stdin and not process.stdin.is_closing():
                            try:
                                process.stdin.write(chunk_data)
                                await process.stdin.drain()
                                total_bytes_fed_to_ffmpeg += len(chunk_data)
                            except (BrokenPipeError, ConnectionResetError) as e_pipe:
                                logger.warning(f"({session_id}) Feed ffmpeg: Pipe to ffmpeg broken: {e_pipe}. Ffmpeg might have exited.")
                                break # ffmpeg likely terminated
                        else:
                            logger.warning(f"({session_id}) Feed ffmpeg: ffmpeg stdin closed or unavailable, cannot send data.")
                            break
            except WebSocketDisconnect:
                logger.info(f"({session_id}) Feed ffmpeg: WebSocket disconnected by client. Cancelling pipeline.")
                cancel_event.set()
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close()
            except Exception as e_feed:
                logger.error(f"({session_id}) Feed ffmpeg: Unexpected error: {e_feed}", exc_info=True)
                cancel_event.set()
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close() # Attempt to clean up
            finally:
                logger.info(f"({session_id}) Feed ffmpeg: Exiting task. Total bytes sent to ffmpeg stdin: {total_bytes_fed_to_ffmpeg}.")
                # Ensure stdin is closed if not already
                if process.stdin and not process.stdin.is_closing():
                    try:
                        process.stdin.close()
                        # await process.stdin.wait_closed() # Optional: ensure it's fully closed
                    except Exception as e_close_stdin:
                        logger.warning(f"({session_id}) Feed ffmpeg: Error closing ffmpeg stdin in finally: {e_close_stdin}")
                # On cancel, terminate ffmpeg immediately so it doesn't
                # spend the next several seconds draining its buffered
                # input — that drain is exactly what kept the server
                # "going" after the client clicked Stop on a long file
                # upload. read_ffmpeg_stdout sees stdout EOF and exits.
                if cancel_event.is_set() and process.returncode is None:
                    try:
                        process.terminate()
                    except ProcessLookupError:
                        pass


        async def read_ffmpeg_stdout_and_queue_chunks():
            """Reads PCM from ffmpeg's stdout, creates ASR chunks, and queues them."""
            nonlocal pcm_buffer_for_asr_chunks, total_samples_in_pcm_buffer_for_offset_calc
            nonlocal total_duration_processed_seconds_for_asr
            
            temp_total_pcm_bytes_read_from_ffmpeg = 0 # For calculating total_duration_processed_seconds_for_asr
            try:
                while True:
                    if process.stdout is None:
                        logger.warning(f"({session_id}) Read ffmpeg: stdout stream is None. Cannot read.")
                        break
                    
                    # Read a block of PCM data from ffmpeg's stdout
                    pcm_data_from_ffmpeg = await process.stdout.read(FFMPEG_PCM_CHUNK_SIZE_BYTES)
                    
                    if not pcm_data_from_ffmpeg:
                        logger.info(f"({session_id}) Read ffmpeg: EOF received from ffmpeg stdout. Stream finished.")
                        break # ffmpeg closed its stdout, indicating end of conversion
                    
                    # Run VAD layer (no-op when disabled). Returns only
                    # speech-confirmed bytes; silent windows are dropped so
                    # the engine queue stays drained.
                    speech_only_pcm = _vad_filter(pcm_data_from_ffmpeg, vad_state)
                    pcm_buffer_for_asr_chunks.extend(speech_only_pcm)
                    temp_total_pcm_bytes_read_from_ffmpeg += len(speech_only_pcm)
                    
                    chunks_created_this_read_cycle = 0
                    # Create as many full ASR chunks as possible from the current buffer
                    while len(pcm_buffer_for_asr_chunks) >= bytes_per_asr_chunk_target_pcm:
                        asr_chunk_bytes = bytes(pcm_buffer_for_asr_chunks[:bytes_per_asr_chunk_target_pcm])
                        
                        # Offload tensor creation as it can be a minor CPU blip
                        tensor = await asyncio.to_thread(_create_asr_tensor_from_bytes, asr_chunk_bytes)
                        
                        # Calculate time offset for this chunk
                        current_chunk_offset_s = total_samples_in_pcm_buffer_for_offset_calc / target_pcm_sample_rate
                        
                        await chunk_queue.put((tensor, current_chunk_offset_s))
                        chunks_created_this_read_cycle += 1
                        
                        # Advance the buffer by the stride (not the full chunk length due to overlap)
                        pcm_buffer_for_asr_chunks = pcm_buffer_for_asr_chunks[bytes_per_asr_stride_target_pcm:]
                        total_samples_in_pcm_buffer_for_offset_calc += samples_per_asr_stride
                    
                    if chunks_created_this_read_cycle > 0:
                        logger.debug(f"({session_id}) Read ffmpeg: Created {chunks_created_this_read_cycle} ASR chunks. "
                                     f"PCM buffer size: {len(pcm_buffer_for_asr_chunks)}B. Queue size: {chunk_queue.qsize()}")

            except Exception as e_read_stdout:
                logger.error(f"({session_id}) Read ffmpeg: Error reading ffmpeg stdout: {e_read_stdout}", exc_info=True)
            finally:
                logger.info(f"({session_id}) Read ffmpeg: Exiting stdout reading task.")
                # After EOF, process any remaining data in the buffer as a final partial chunk
                min_final_asr_chunk_samples = int(target_pcm_sample_rate * 0.1) # e.g., 0.1 seconds minimum
                min_final_asr_chunk_bytes = target_pcm_bytes_per_sample * min_final_asr_chunk_samples
                
                if len(pcm_buffer_for_asr_chunks) >= min_final_asr_chunk_bytes:
                    final_asr_chunk_bytes = bytes(pcm_buffer_for_asr_chunks)
                    final_tensor = await asyncio.to_thread(_create_asr_tensor_from_bytes, final_asr_chunk_bytes)
                    
                    if final_tensor.numel() > 0: # Ensure tensor is not empty
                        final_chunk_offset_s = total_samples_in_pcm_buffer_for_offset_calc / target_pcm_sample_rate
                        await chunk_queue.put((final_tensor, final_chunk_offset_s))
                        logger.info(f"({session_id}) Read ffmpeg: Queued final partial ASR chunk of {final_tensor.shape[0]} samples.")
                    else:
                        logger.debug(f"({session_id}) Read ffmpeg: Final partial ASR chunk resulted in an empty tensor. Discarding.")
                elif len(pcm_buffer_for_asr_chunks) > 0:
                    logger.debug(f"({session_id}) Read ffmpeg: Discarding final remaining {len(pcm_buffer_for_asr_chunks)} bytes from PCM buffer (too small).")

                # Calculate total duration based on all PCM bytes successfully read from ffmpeg
                total_duration_processed_seconds_for_asr = temp_total_pcm_bytes_read_from_ffmpeg / \
                                                           (target_pcm_sample_rate * target_pcm_bytes_per_sample)
                logger.info(f"({session_id}) Read ffmpeg: Total raw PCM bytes read from ffmpeg: {temp_total_pcm_bytes_read_from_ffmpeg}B "
                            f"({total_duration_processed_seconds_for_asr:.2f}s of audio).")
                if vad_state['enabled']:
                    total_frames = vad_state['speech_frames'] + vad_state['dropped_frames']
                    if total_frames > 0:
                        speech_pct = 100.0 * vad_state['speech_frames'] / total_frames
                        logger.info(
                            f"({session_id}) Stream: VAD summary — "
                            f"{vad_state['speech_frames']}/{total_frames} frames forwarded "
                            f"({speech_pct:.1f}% speech, "
                            f"{vad_state['dropped_frames'] * _VAD_FRAME_MS / 1000:.2f}s of silence dropped, "
                            f"{vad_state['padded_ms'] / 1000:.2f}s of low-noise padding injected)."
                        )
        try:
            # Run stdin feeder and stdout reader concurrently
            feed_task = asyncio.create_task(feed_ffmpeg_stdin())
            read_task = asyncio.create_task(read_ffmpeg_stdout_and_queue_chunks())
            
            await asyncio.gather(feed_task, read_task) # Wait for both to complete
            
            # Ensure ffmpeg process is cleaned up
            if process.stdin and not process.stdin.is_closing():
                process.stdin.close() # Should be closed by feed_ffmpeg_stdin already

            # Wait for ffmpeg to exit and capture any remaining stderr
            # communicate() should be called after stdin is closed and stdout/stderr pipes are drained (by read_task)
            stdout_rem, stderr_rem = await process.communicate()
            
            if stderr_rem:
                logger.warning(f"({session_id}) ffmpeg stderr output: {stderr_rem.decode(errors='ignore').strip()}")
            
            if process.returncode != 0 and process.returncode is not None:
                logger.error(f"({session_id}) ffmpeg process exited with error code {process.returncode}.")
            else:
                logger.info(f"({session_id}) ffmpeg process finished successfully (return code: {process.returncode}).")

        except Exception as e_prod_ffmpeg_main:
            logger.error(f"({session_id}) Stream Producer (ffmpeg main loop) Error: {e_prod_ffmpeg_main}", exc_info=True)
            if process and process.returncode is None: # If ffmpeg is still running
                try:
                    logger.warning(f"({session_id}) Terminating ffmpeg process due to error.")
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0) # Wait for termination
                except ProcessLookupError:
                    logger.debug(f"({session_id}) ffmpeg process already exited.")
                except asyncio.TimeoutError:
                    logger.warning(f"({session_id}) Timeout waiting for ffmpeg to terminate. Killing.")
                    process.kill()
                    await process.wait()
                except Exception as e_term:
                    logger.error(f"({session_id}) Error during ffmpeg termination: {e_term}")
        finally:
            logger.info(f"({session_id}) Stream Producer (ffmpeg): Signaling completion to consumer.")
            producer_done_event.set() # Signal that producer is done
            await chunk_queue.put(None) # Sentinel to signal consumer to stop

    async def consumer():
        """
        Consumer coroutine. Drains chunk_queue and feeds each tensor into
        `StreamingPrevBatchedEngine` (the v2 engine — NVIDIA's blessed
        prev_batched_state pattern). After each step, newly committed
        sentence-bounded segments are sent as `segments_batch` messages.
        At EOF (sentinel), `engine.flush()` closes out the trailing context.
        """
        nonlocal accumulated_asr_processing_time_s, sent_segments_pcm

        if asr_model is None:
            return

        engine = StreamingPrevBatchedEngine(
            asr_model_instance=asr_model,
            chunk_secs=engine_chunk_len_s,
            left_context_secs=STREAMING_LEFT_CONTEXT_S,
            right_context_secs=right_context_secs,
            request_id=f"WS-Stream-{session_id}-eng",
            # Engine bails between its internal forward passes when this
            # returns True (set by the producer's disconnect handlers).
            cancel_check=lambda: cancel_event.is_set(),
        )
        logger.info(
            f"({session_id}) Stream: emission_lag={engine.emission_lag_secs:.2f}s "
            f"(chunk={engine.chunk_secs}s + right={right_context_secs}s)"
        )

        total_engine_chunks = 0
        # Stream the in-progress (uncommitted) sentence buffer to the
        # client between real commits so the UI doesn't sit blank for
        # the 4–6 s emission lag while the user is speaking. Decoupled
        # from `live_latency` — partials are cheap and useful for any
        # streaming source (including file uploads, where they preview
        # progress on a slow upload). `live_latency` now controls only
        # the chunk preset (10-2-2 vs 10-10-5).
        partial_emit_enabled = True
        try:
            while True:
                item = await chunk_queue.get()
                if item is None:
                    chunk_queue.task_done()
                    break
                # Bail before doing any GPU work if the client disconnected.
                # The producer's disconnect handlers set cancel_event; we
                # check it after every queue read so the bail latency is at
                # most one engine chunk (typically 2–10s of audio compute).
                if cancel_event.is_set():
                    chunk_queue.task_done()
                    break
                tensor, _offset_s = item
                chunk_queue.task_done()
                samples_np = (
                    tensor.detach().cpu().numpy()
                    if hasattr(tensor, "detach")
                    else np.asarray(tensor, dtype=np.float32)
                )
                await _run_on_asr_executor(engine.feed_float32, samples_np)
                total_engine_chunks += 1
                new_segs = engine.pop_committed_segments()
                if new_segs and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        # When VAD is on, engine timestamps are speech-clock —
                        # translate them back to the wall clock of the original
                        # recording before the client sees them. No-op when VAD
                        # is off (time_map is empty).
                        _translate_segment_times(new_segs, vad_state)
                        _accumulate_committed_speech(new_segs, vad_state)
                        new_words = _segments_to_words(new_segs)
                        await websocket.send_json({
                            "type": "segments_batch",
                            "segments": new_segs,
                            "words": new_words,
                            **_streaming_counters(vad_state),
                        })
                        sent_segments_pcm.extend(new_segs)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) Stream: segment send failed: {e_send}")
                # In-flight peek of the next sentence (uncommitted tokens).
                # Read-only; the engine still commits naturally on .!?.
                if partial_emit_enabled and websocket.application_state == WebSocketState.CONNECTED:
                    partial = engine.peek_partial_segment()
                    if partial is not None:
                        try:
                            _translate_segment_times([partial], vad_state)
                            partial_words = _segments_to_words([partial])
                            await websocket.send_json({
                                "type": "partial_segment",
                                "segment": partial,
                                "words": partial_words,
                                **_streaming_counters(vad_state),
                            })
                        except Exception as e_send:
                            logger.warning(f"({session_id}) Stream: partial send failed: {e_send}")

            # EOF — pad + flush to commit trailing tokens. Skipped on
            # cancel: the client has gone, and flush() is one more
            # forward pass we'd rather not run.
            if not cancel_event.is_set():
                await _run_on_asr_executor(engine.flush)
                final_partials = engine.pop_final_segments()
                if final_partials and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        _translate_segment_times(final_partials, vad_state)
                        _accumulate_committed_speech(final_partials, vad_state)
                        final_words = _segments_to_words(final_partials)
                        await websocket.send_json({
                            "type": "segments_batch",
                            "segments": final_partials,
                            "words": final_words,
                            **_streaming_counters(vad_state),
                        })
                        sent_segments_pcm.extend(final_partials)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) Stream: final segment send failed: {e_send}")

            logger.info(
                f"({session_id}) Stream: processed {total_engine_chunks} engine chunks "
                f"in {engine.asr_time_s:.2f}s ASR time, emitted {len(sent_segments_pcm)} segments."
                f"{' (cancelled)' if cancel_event.is_set() else ''}"
            )
        finally:
            accumulated_asr_processing_time_s += engine.asr_time_s
            engine.reset()
            # On cancel the producer may still be mid-put on a full
            # queue. Drain so it can unblock, finish its finally, and
            # the gather() in the caller can complete.
            while not chunk_queue.empty():
                try:
                    chunk_queue.get_nowait()
                    chunk_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    break

    # Register the cancel event so /v1/audio/streaming/cancel/{session_id}
    # can flip it out-of-band. Unregistered in the outermost finally below.
    _streaming_cancel_events[session_id] = cancel_event

    # Main execution block for handle_streaming_pcm
    try:
        logger.info(f"({session_id}) Streaming Pipeline (ffmpeg-based): Starting producer and consumer tasks.")
        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())
        logger.info(f"({session_id}) Streaming Pipeline (ffmpeg-based): Producer and consumer tasks have completed.")

        # Consolidate all transcribed text from segments
        # Note: sent_segments_pcm might not be perfectly ordered if deduplication is added later for streaming.
        # For now, assume they are appended in rough chronological order.
        final_transcribed_text_pcm = " ".join(s["text"] for s in sent_segments_pcm).strip()

        if websocket.application_state == WebSocketState.CONNECTED:
            logger.info(f"({session_id}) Streaming: Sending final_transcription message to client. "
                        f"Total ASR input duration (from ffmpeg PCM): {total_duration_processed_seconds_for_asr:.2f}s")

            final_message_payload = {
                "type": "final_transcription",
                # Whisper-compatible fields
                "task": "transcribe",
                "language": "english",
                "duration": round(total_duration_processed_seconds_for_asr, 3),
                "text": final_transcribed_text_pcm,
                "segments": sent_segments_pcm,
                "words": _segments_to_words(sent_segments_pcm),
                # Extensions:
                "transcription_time": round(accumulated_asr_processing_time_s, 3),
                "total_segments": len(sent_segments_pcm),
                "final_duration_processed_seconds": round(total_duration_processed_seconds_for_asr, 3),
                "csv_content": generate_csv_content(sent_segments_pcm),
                "srt_content": generate_srt_content(sent_segments_pcm),
                "vtt_content": _segments_to_vtt(sent_segments_pcm),
                "streaming_mode": client_config.get("format", "unknown"),
                **_streaming_counters(vad_state),
            }
            await websocket.send_json(final_message_payload)
            logger.info(f"({session_id}) Streaming: Final transcription message sent.")
        else:
            logger.info(f"({session_id}) Streaming: WebSocket disconnected before final_transcription could be sent.")

    except Exception as e_pipeline_main:
        logger.error(f"({session_id}) Main Streaming Pipeline Error: {e_pipeline_main}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "error": f"Critical server error in streaming pipeline: {str(e_pipeline_main)}"})
            except Exception as e_send_err_critical:
                logger.warning(f"({session_id}) Could not send critical error message to client after pipeline failure: {e_send_err_critical}")
    finally:
        logger.info(f"({session_id}) Streaming Pipeline (ffmpeg-based): Final cleanup.")
        _streaming_cancel_events.pop(session_id, None)
        # Ensure producer_done_event is set, and a sentinel is in the queue if not already guaranteed by producer's finally.
        if not producer_done_event.is_set():
            producer_done_event.set()
        try:
            # Try to put a sentinel if producer might have exited prematurely without doing so.
            # This helps ensure consumer exits cleanly.
            if chunk_queue.empty(): # Only if empty, to avoid multiple sentinels if one is already there
                chunk_queue.put_nowait(None)
        except (asyncio.QueueFull, Exception):
            # Queue might be full if consumer also exited prematurely, or other rare conditions.
            logger.warning(f"({session_id}) Streaming: Could not put sentinel in queue during final pipeline cleanup (queue full or other error).")


async def handle_streaming_url(
    websocket: WebSocket,
    session_id: str,
    processing_device: str,
    client_config: dict,
    source_url: str,
):
    """
    Live-stream URL pipeline. Same consumer + queue + VAD pipeline as
    `handle_streaming_pcm`, but the producer launches `ffmpeg -i <url>` so
    audio is pulled from the URL directly (no client binary frames). Suits
    HLS playlists, icecast streams, RTSP feeds, m3u8 — anything ffmpeg can
    open.

    Termination: closes when the URL ends (ffmpeg EOF), the client
    disconnects, or `URL_STREAM_MAX_S` wall-clock seconds elapse.
    """
    sent_segments_pcm: List[dict] = []

    vad_state = _build_vad_state(client_config)
    if vad_state['enabled']:
        logger.info(
            f"({session_id}) URL Stream: VAD enabled "
            f"(threshold={vad_state['threshold']:.2f}, "
            f"consec={vad_state['consecutive']}, "
            f"hangover={vad_state['hangover_ms']:g}ms)."
        )
    else:
        logger.info(f"({session_id}) URL Stream: VAD disabled.")

    live_latency = bool(client_config.get("live_latency", False))
    engine_chunk_len_s = STREAMING_LIVE_CHUNK_S if live_latency else STREAMING_CHUNK_S
    right_context_secs = STREAMING_LIVE_RIGHT_CONTEXT_S if live_latency else STREAMING_RIGHT_CONTEXT_S
    asr_chunk_len_s = engine_chunk_len_s
    asr_chunk_overlap_s = 0.0

    target_pcm_sample_rate = MODEL_SAMPLE_RATE
    target_pcm_bytes_per_sample = 2
    samples_per_asr_chunk = int(asr_chunk_len_s * target_pcm_sample_rate)
    samples_per_asr_stride = int((asr_chunk_len_s - asr_chunk_overlap_s) * target_pcm_sample_rate)
    bytes_per_asr_chunk_target_pcm = samples_per_asr_chunk * target_pcm_bytes_per_sample
    bytes_per_asr_stride_target_pcm = samples_per_asr_stride * target_pcm_bytes_per_sample

    chunk_queue: asyncio.Queue[Optional[Tuple[torch.Tensor, float]]] = asyncio.Queue(
        maxsize=client_config["batch_size"] * 2
    )
    producer_done_event = asyncio.Event()
    cancel_event = asyncio.Event()

    accumulated_asr_processing_time_s: float = 0.0
    total_duration_processed_seconds_for_asr: float = 0.0

    def _create_asr_tensor_from_bytes(pcm_bytes: bytes) -> torch.Tensor:
        return torch.frombuffer(pcm_bytes, dtype=torch.int16).float() / 32768.0

    async def watch_ws_disconnect():
        """Set cancel_event when the client goes away or the wall-clock cap
        fires. The producer's ffmpeg poll picks the event up and tears
        ffmpeg down."""
        deadline = asyncio.get_event_loop().time() + URL_STREAM_MAX_S
        try:
            while not cancel_event.is_set():
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.warning(
                        f"({session_id}) URL Stream: hit URL_STREAM_MAX_S={URL_STREAM_MAX_S:g}s cap; closing."
                    )
                    cancel_event.set()
                    return
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=min(remaining, 5.0))
                except asyncio.TimeoutError:
                    continue
                if msg.get("type") == "websocket.disconnect":
                    logger.info(f"({session_id}) URL Stream: client disconnected.")
                    cancel_event.set()
                    return
        except asyncio.CancelledError:
            pass
        except Exception:
            cancel_event.set()

    async def producer():
        nonlocal total_duration_processed_seconds_for_asr
        pcm_buffer_for_asr_chunks = bytearray()
        total_samples_in_pcm_buffer_for_offset_calc = 0

        hpf_hz = float(client_config.get('hpf_hz', 0) or 0)
        af_args: List[str] = []
        if hpf_hz > 0:
            af_args = ['-af', f'highpass=f={hpf_hz:g}']
        ffmpeg_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', source_url,  # Input from URL (HLS / icecast / RTSP / m3u8 / etc.)
            *af_args,
            '-f', 's16le',
            '-ac', '1',
            '-ar', str(MODEL_SAMPLE_RATE),
            '-acodec', 'pcm_s16le',
            'pipe:1',
        ]
        logger.info(f"({session_id}) URL Stream Producer: starting ffmpeg -i {source_url!r}.")

        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logger.info(f"({session_id}) URL Stream Producer: ffmpeg PID={process.pid}.")

        temp_total_pcm_bytes_read_from_ffmpeg = 0
        try:
            while True:
                if cancel_event.is_set():
                    logger.info(f"({session_id}) URL Stream Producer: cancel_event set; terminating ffmpeg.")
                    break
                if process.stdout is None:
                    break
                try:
                    pcm_data_from_ffmpeg = await asyncio.wait_for(
                        process.stdout.read(FFMPEG_PCM_CHUNK_SIZE_BYTES), timeout=2.0,
                    )
                except asyncio.TimeoutError:
                    continue  # re-check cancel_event
                if not pcm_data_from_ffmpeg:
                    logger.info(f"({session_id}) URL Stream Producer: ffmpeg EOF.")
                    break

                speech_only_pcm = _vad_filter(pcm_data_from_ffmpeg, vad_state)
                pcm_buffer_for_asr_chunks.extend(speech_only_pcm)
                temp_total_pcm_bytes_read_from_ffmpeg += len(speech_only_pcm)

                while len(pcm_buffer_for_asr_chunks) >= bytes_per_asr_chunk_target_pcm:
                    asr_chunk_bytes = bytes(pcm_buffer_for_asr_chunks[:bytes_per_asr_chunk_target_pcm])
                    tensor = await asyncio.to_thread(_create_asr_tensor_from_bytes, asr_chunk_bytes)
                    current_chunk_offset_s = total_samples_in_pcm_buffer_for_offset_calc / target_pcm_sample_rate
                    await chunk_queue.put((tensor, current_chunk_offset_s))
                    pcm_buffer_for_asr_chunks = pcm_buffer_for_asr_chunks[bytes_per_asr_stride_target_pcm:]
                    total_samples_in_pcm_buffer_for_offset_calc += samples_per_asr_stride
        finally:
            # Trailing partial chunk
            min_final_asr_chunk_bytes = target_pcm_bytes_per_sample * int(target_pcm_sample_rate * 0.1)
            if len(pcm_buffer_for_asr_chunks) >= min_final_asr_chunk_bytes:
                try:
                    final_tensor = await asyncio.to_thread(_create_asr_tensor_from_bytes, bytes(pcm_buffer_for_asr_chunks))
                    if final_tensor.numel() > 0:
                        final_chunk_offset_s = total_samples_in_pcm_buffer_for_offset_calc / target_pcm_sample_rate
                        await chunk_queue.put((final_tensor, final_chunk_offset_s))
                except Exception as e_final_chunk:
                    logger.warning(f"({session_id}) URL Stream Producer: final-chunk queue error: {e_final_chunk}")

            total_duration_processed_seconds_for_asr = temp_total_pcm_bytes_read_from_ffmpeg / (
                target_pcm_sample_rate * target_pcm_bytes_per_sample
            )

            # Tear ffmpeg down if still running.
            if process.returncode is None:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except ProcessLookupError:
                    pass
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            # Drain stderr for log visibility.
            try:
                _stdout_rem, stderr_rem = await process.communicate()
                if stderr_rem:
                    logger.warning(f"({session_id}) URL Stream Producer: ffmpeg stderr: {stderr_rem.decode(errors='ignore').strip()}")
            except Exception:
                pass
            producer_done_event.set()
            await chunk_queue.put(None)

    async def consumer():
        nonlocal accumulated_asr_processing_time_s, sent_segments_pcm
        if asr_model is None:
            return
        engine = StreamingPrevBatchedEngine(
            asr_model_instance=asr_model,
            chunk_secs=engine_chunk_len_s,
            left_context_secs=STREAMING_LEFT_CONTEXT_S,
            right_context_secs=right_context_secs,
            request_id=f"WS-URL-{session_id}-eng",
            # Same cancel propagation as the PCM path — the watcher task
            # above (watch_ws_disconnect) sets cancel_event on client
            # disconnect / URL_STREAM_MAX_S hit; engine bails between
            # forward passes.
            cancel_check=lambda: cancel_event.is_set(),
        )
        # Partials decoupled from `live_latency` — URL is always live; the
        # preset only controls chunk size. Partials give the user preview
        # text even when the user opted into the higher-throughput
        # 10-10-5 preset.
        partial_emit_enabled = True
        try:
            while True:
                item = await chunk_queue.get()
                if item is None:
                    chunk_queue.task_done()
                    break
                if cancel_event.is_set():
                    chunk_queue.task_done()
                    break
                tensor, _offset_s = item
                chunk_queue.task_done()
                samples_np = (
                    tensor.detach().cpu().numpy()
                    if hasattr(tensor, "detach")
                    else np.asarray(tensor, dtype=np.float32)
                )
                await _run_on_asr_executor(engine.feed_float32, samples_np)
                new_segs = engine.pop_committed_segments()
                if new_segs and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        _translate_segment_times(new_segs, vad_state)
                        _accumulate_committed_speech(new_segs, vad_state)
                        new_words = _segments_to_words(new_segs)
                        await websocket.send_json({
                            "type": "segments_batch",
                            "segments": new_segs,
                            "words": new_words,
                            **_streaming_counters(vad_state),
                        })
                        sent_segments_pcm.extend(new_segs)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) URL Stream: segment send failed: {e_send}")
                if partial_emit_enabled and websocket.application_state == WebSocketState.CONNECTED:
                    partial = engine.peek_partial_segment()
                    if partial is not None:
                        try:
                            _translate_segment_times([partial], vad_state)
                            partial_words = _segments_to_words([partial])
                            await websocket.send_json({
                                "type": "partial_segment",
                                "segment": partial,
                                "words": partial_words,
                                **_streaming_counters(vad_state),
                            })
                        except Exception as e_send:
                            logger.warning(f"({session_id}) URL Stream: partial send failed: {e_send}")

            # Skip the trailing flush + final-segment send on cancel —
            # see PCM consumer for the same pattern.
            if not cancel_event.is_set():
                await _run_on_asr_executor(engine.flush)
                final_partials = engine.pop_final_segments()
                if final_partials and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        _translate_segment_times(final_partials, vad_state)
                        _accumulate_committed_speech(final_partials, vad_state)
                        final_words = _segments_to_words(final_partials)
                        await websocket.send_json({
                            "type": "segments_batch",
                            "segments": final_partials,
                            "words": final_words,
                            **_streaming_counters(vad_state),
                        })
                        sent_segments_pcm.extend(final_partials)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) URL Stream: final segment send failed: {e_send}")
        finally:
            accumulated_asr_processing_time_s += engine.asr_time_s
            engine.reset()
            # Drain so producer can unblock from a potentially-full put.
            while not chunk_queue.empty():
                try:
                    chunk_queue.get_nowait()
                    chunk_queue.task_done()
                except (asyncio.QueueEmpty, ValueError):
                    break

    # Register the cancel event so the HTTP cancel endpoint can flip it
    # out-of-band (see _streaming_cancel_events at top of file).
    _streaming_cancel_events[session_id] = cancel_event

    try:
        watcher_task = asyncio.create_task(watch_ws_disconnect())
        try:
            await asyncio.gather(producer(), consumer())
        finally:
            watcher_task.cancel()
            try:
                await watcher_task
            except (asyncio.CancelledError, Exception):
                pass

        final_transcribed_text_pcm = " ".join(s["text"] for s in sent_segments_pcm).strip()
        if websocket.application_state == WebSocketState.CONNECTED:
            final_message_payload = {
                "type": "final_transcription",
                "task": "transcribe",
                "language": "english",
                "duration": round(total_duration_processed_seconds_for_asr, 3),
                "text": final_transcribed_text_pcm,
                "segments": sent_segments_pcm,
                "words": _segments_to_words(sent_segments_pcm),
                "strategy": ProcessingStrategy.STREAMING.value,
                "transcription_time": round(accumulated_asr_processing_time_s, 3),
                "total_segments": len(sent_segments_pcm),
                "final_duration_processed_seconds": round(total_duration_processed_seconds_for_asr, 3),
                "csv_content": generate_csv_content(sent_segments_pcm),
                "srt_content": generate_srt_content(sent_segments_pcm),
                "vtt_content": _segments_to_vtt(sent_segments_pcm),
                "streaming_mode": "url",
                **_streaming_counters(vad_state),
            }
            await websocket.send_json(final_message_payload)
    except Exception as e_pipeline:
        logger.error(f"({session_id}) URL Stream Pipeline Error: {e_pipeline}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"type": "error", "error": f"Critical server error in URL stream: {e_pipeline}"})
            except Exception:
                pass
    finally:
        _streaming_cancel_events.pop(session_id, None)
        if not producer_done_event.is_set():
            producer_done_event.set()
        try:
            if chunk_queue.empty():
                chunk_queue.put_nowait(None)
        except (asyncio.QueueFull, Exception):
            pass


class _RestCancelled(Exception):
    """Internal sentinel — the REST client disconnected mid-transcription;
    unwind to the 499 response without logging it as a server error."""


async def _watch_request_disconnect(
    request: Request,
    cancel_event: threading.Event,
    request_id: str,
    poll_s: float = 0.4,
) -> None:
    """Poll for an HTTP client disconnect and set `cancel_event`.

    uvicorn keeps an endpoint coroutine running after the client goes
    away (the request body was already received), so a long offline
    transcription would otherwise run to completion holding the model
    lock — blocking every subsequent request. Setting the event lets
    `_transcribe_split_full` bail between slices and free the lock
    promptly.
    """
    try:
        while not cancel_event.is_set():
            if await request.is_disconnected():
                cancel_event.set()
                logger.info(f"({request_id}) REST: client disconnected — cancelling in-flight transcription.")
                return
            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.warning(f"({request_id}) REST: disconnect watcher error: {e}")


async def _watch_ws_disconnect(
    websocket: WebSocket,
    cancel_event: threading.Event,
    session_id: str,
) -> None:
    """Set `cancel_event` when the WS client goes away mid-inference.

    Same idea as the REST watcher: a disconnected client shouldn't leave a
    long offline transcription running to completion holding the model
    lock. Detected by awaiting receive() — a closed socket yields a
    `websocket.disconnect` message (or raises once already closed).
    Only run AFTER the accumulate loop has stopped receiving, so there's
    no competing reader.
    """
    try:
        while not cancel_event.is_set():
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                cancel_event.set()
                logger.info(f"({session_id}) WS: client disconnected — cancelling in-flight transcription.")
                return
            # Ignore any stray frames sent after EOF.
    except asyncio.CancelledError:
        pass
    except Exception:
        # receive() raises once the socket is gone — treat as disconnect.
        cancel_event.set()
        logger.info(f"({session_id}) WS: receive ended — cancelling in-flight transcription.")


async def _transcribe_full(
    waveform: torch.Tensor,
    audio_duration_s: float,
    request_id: str = "full",
) -> Tuple[List[dict], float]:
    """Single-pass offline transcription via encoder + `decoding_computer`.

    Bypasses NeMo 2.7.3's `asr_model.transcribe()` wrapper entirely. That
    wrapper interacts badly with FULL_GRAPH-mode CUDA graphs on short audio
    (the captured decoder graph reads stale data from the encoder-output
    static buffer when audio is shorter than the captured shape, producing
    23 % empty / 33 % catastrophic outputs on the LibriSpeech test-clean
    short tail — see the regression harness in `tests/`).

    Same encoder + `decoding_computer` pipeline as `StreamingPrevBatchedEngine`,
    but single-shot: encoder runs once on the full waveform, decoder runs once
    with `prev_batched_state=None`, tokens are mapped to sentence-bounded
    segments via the shared helper.

    Caller MUST hold `model_access_lock`. `_apply_model_settings_for_session`
    must have run first so local-attention mode is engaged for long audio.
    """
    if asr_model is None:
        logger.error(f"({request_id}) full: asr_model is None.")
        return [], 0.0

    device = next(asr_model.parameters()).device
    dtype = next(asr_model.parameters()).dtype

    waveform_1d = waveform.squeeze().to(device=device, dtype=torch.float32).contiguous()
    audio_batch = waveform_1d.unsqueeze(0)
    audio_lengths = torch.tensor([waveform_1d.shape[0]], dtype=torch.long, device=device)

    decoding_computer = asr_model.decoding.decoding.decoding_computer
    tokenizer = asr_model.tokenizer

    model_cfg = asr_model._cfg
    feature_stride_sec = float(model_cfg.preprocessor["window_stride"])
    encoder_subsampling_factor = int(asr_model.encoder.subsampling_factor)
    encoder_stride_s = feature_stride_sec * encoder_subsampling_factor

    def _run() -> Tuple[List[dict], float]:
        t0 = time.time()
        with torch.inference_mode():
            with torch.amp.autocast(device.type, dtype=dtype):
                encoder_output, encoder_output_len = asr_model(
                    input_signal=audio_batch,
                    input_signal_length=audio_lengths,
                )
            # The captured joint graph was warmed in bf16 — force matching dtype
            # on the inputs we hand back to it (same pattern the streaming engine uses).
            encoder_output = encoder_output.transpose(1, 2).to(dtype=dtype)  # [B, T, C]

            with torch.amp.autocast(device.type, dtype=dtype):
                chunk_hyps, _, _ = decoding_computer(
                    x=encoder_output,
                    out_len=encoder_output_len,
                    prev_batched_state=None,
                )

            n_tokens = int(chunk_hyps.current_lengths[0].item())
            if n_tokens == 0:
                return [], time.time() - t0
            token_ids = chunk_hyps.transcript[0, :n_tokens].detach().cpu().tolist()
            frame_idx = chunk_hyps.timestamps[0, :n_tokens].detach().cpu().tolist()
            token_times = [max(0.0, float(f) * encoder_stride_s) for f in frame_idx]
            # If the decoder was configured with preserve_token_confidence
            # (and the captured graph honored it), read per-token confidence
            # and convert to log probabilities for avg_logprob.
            conf_tensor = getattr(chunk_hyps, "token_confidence", None) or getattr(chunk_hyps, "confidence", None)
            token_logprobs: Optional[List[Optional[float]]] = None
            if conf_tensor is not None:
                try:
                    conf_vals = conf_tensor[0, :n_tokens].detach().cpu().tolist()
                    token_logprobs = [
                        (math.log(max(float(c), 1e-10)) if c is not None else None)
                        for c in conf_vals
                    ]
                except Exception:
                    token_logprobs = None

        segments = tokens_to_sentence_segments(
            token_ids, token_times, tokenizer, token_logprobs=token_logprobs,
        )
        return segments, time.time() - t0

    segments, asr_time = await _run_on_asr_executor(_run)
    logger.info(
        f"({request_id}) full: dur={audio_duration_s:.2f}s "
        f"asr_t={asr_time:.2f}s segs={len(segments)}"
    )
    return segments, asr_time


async def _transcribe_split_full(
    waveform: torch.Tensor,
    audio_duration_s: float,
    client_config: dict,
    request_id: str = "split_full",
    cancel_event: Optional["threading.Event"] = None,
) -> Tuple[List[dict], float]:
    """Sequential FULL passes over overlapping slices, stitched at seams.

    For audio longer than `MAX_FULL_WAVEFORM_S` we can't fit one encoder
    pass on the available GPU memory, but we can run back-to-back FULL
    passes at FULL throughput — ~3-5× faster than the CHUNKED engine on
    this hardware. Each slice is `MAX_FULL_WAVEFORM_S * SLICE_SAFETY`
    seconds long with `SPLIT_FULL_OVERLAP_S` of overlap; the overlap
    gives the second slice's encoder enough left-context to land its
    first few segments cleanly, and we drop segments at the seam whose
    start landed inside the prior slice's tail to avoid duplicates.

    Caller MUST hold `model_access_lock`.
    """
    if asr_model is None:
        logger.error(f"({request_id}) split_full: asr_model is None.")
        return [], 0.0

    slice_s = MAX_FULL_WAVEFORM_S * SLICE_SAFETY
    overlap_s = SPLIT_FULL_OVERLAP_S
    if overlap_s >= slice_s:
        raise ValueError(
            f"SPLIT_FULL_OVERLAP_S ({overlap_s}) must be less than slice "
            f"length ({slice_s:.1f}s = MAX_FULL_WAVEFORM_S * SLICE_SAFETY)."
        )
    step_s = slice_s - overlap_s
    sr = MODEL_SAMPLE_RATE

    waveform_1d = waveform.squeeze()
    if waveform_1d.dim() != 1:
        waveform_1d = waveform_1d.reshape(-1)

    all_segs: List[dict] = []
    asr_total = 0.0
    seg_id = 0
    start_s = 0.0
    slice_idx = 0

    while start_s < audio_duration_s:
        if cancel_event is not None and cancel_event.is_set():
            logger.info(
                f"({request_id}) split_full: cancelled before slice {slice_idx + 1} "
                f"(start={start_s:.1f}s of {audio_duration_s:.1f}s)."
            )
            break

        end_s = min(audio_duration_s, start_s + slice_s)
        sub = waveform_1d[int(start_s * sr) : int(end_s * sr)]
        sub_duration_s = end_s - start_s
        slice_idx += 1
        logger.info(
            f"({request_id}) split_full: slice {slice_idx} "
            f"[{start_s:.1f}s, {end_s:.1f}s] ({sub_duration_s:.1f}s)"
        )

        segs, asr_t = await _transcribe_full(
            waveform=sub,
            audio_duration_s=sub_duration_s,
            request_id=f"{request_id}-split{slice_idx}",
        )
        asr_total += asr_t

        # Translate slice-relative timestamps back to wall-clock.
        for s in segs:
            s["start"] = float(s.get("start", 0.0)) + start_s
            s["end"] = float(s.get("end", 0.0)) + start_s

        # Drop segments in the overlap with the prior slice's tail to keep
        # the seam clean. 0.30s of slop tolerates minor decoder jitter.
        if all_segs and segs:
            prior_end = float(all_segs[-1]["end"])
            segs = [s for s in segs if s["start"] > prior_end - 0.30]

        for s in segs:
            s["id"] = seg_id
            seg_id += 1
        all_segs.extend(segs)

        if end_s >= audio_duration_s:
            break
        start_s += step_s

    logger.info(
        f"({request_id}) split_full: dur={audio_duration_s:.2f}s "
        f"asr_t={asr_total:.2f}s segs={len(all_segs)} slices={slice_idx}"
    )
    return all_segs, asr_total


_VALID_RESPONSE_FORMATS = {"json", "text", "srt", "verbose_json", "vtt"}


def _build_openai_response(
    *,
    response_format: str,
    segments: List[dict],
    words: Optional[List[dict]],
    full_text: str,
    duration_s: float,
    asr_time_s: float,
    total_server_time_s: float,
    resolved_strategy: ProcessingStrategy,
) -> Response:
    """Render a transcription result into the OpenAI-compatible shape the client
    asked for via `response_format`. Falls back to JSON shape on unknown values.

    Formats:
      - `json` (default): {"text": ...} only — matches OpenAI exactly.
      - `text`           : raw transcript text, content-type text/plain
      - `srt`            : SubRip subtitle file, content-type text/plain
      - `vtt`            : WebVTT subtitle file, content-type text/plain
      - `verbose_json`   : full Whisper verbose_json shape (segments, tokens,
                          timestamps, etc.) plus our server-side extensions
                          (strategy, *_seconds, csv_content, srt_content).
    """
    if response_format == "text":
        return PlainTextResponse(full_text)

    if response_format == "srt":
        return PlainTextResponse(generate_srt_content(segments), media_type="text/plain")

    if response_format == "vtt":
        return PlainTextResponse(_segments_to_vtt(segments), media_type="text/vtt")

    if response_format == "verbose_json":
        body = {
            # OpenAI verbose_json shape
            "task": "transcribe",
            "language": "english",
            "duration": round(duration_s, 3),
            "text": full_text,
            "segments": segments,
            # Server extensions (not in OpenAI spec):
            "strategy": resolved_strategy.value,
            "transcription_time_seconds": round(asr_time_s, 3),
            "total_request_time_server_seconds": total_server_time_s,
            "audio_duration_seconds": round(duration_s, 3),
            "csv_content": generate_csv_content(segments),
            "srt_content": generate_srt_content(segments),
            "vtt_content": _segments_to_vtt(segments),
        }
        if words is not None:
            body["words"] = words
        return JSONResponse(content=body)

    # Default / "json": minimal, just text — matches OpenAI's compact response.
    return JSONResponse(content={"text": full_text})


def _segments_to_vtt(segments: List[dict]) -> str:
    """WebVTT serialization of segments. Simple and stdlib-only."""
    def _fmt(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    lines = ["WEBVTT", ""]
    for s in segments:
        lines.append(f"{_fmt(s.get('start', 0.0))} --> {_fmt(s.get('end', 0.0))}")
        lines.append((s.get("text") or "").strip())
        lines.append("")
    return "\n".join(lines)


_MAX_URL_BYTES = 512 * 1024 * 1024  # 512 MB hard cap on URL-ingested audio
_URL_FETCH_TIMEOUT_S = 60


async def _fetch_audio_url(url: str, request_id: str) -> Tuple[bytes, str]:
    """Download audio bytes from a URL. Returns (bytes, inferred_filename).

    Runs urllib in a worker thread so it doesn't block the event loop. Enforces
    a 512 MB cap and a 60 s connect+read timeout. Rejects non-http(s) schemes.
    """
    from urllib.parse import urlparse
    from urllib.request import urlopen, Request as UrlRequest
    from urllib.error import URLError, HTTPError

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme '{parsed.scheme}'; must be http or https.")

    def _fetch_sync() -> Tuple[bytes, str]:
        req = UrlRequest(url, headers={"User-Agent": "parakeet-asr/1.0"})
        with urlopen(req, timeout=_URL_FETCH_TIMEOUT_S) as resp:
            ctype = resp.headers.get("Content-Type", "")
            clen = resp.headers.get("Content-Length")
            if clen and int(clen) > _MAX_URL_BYTES:
                raise ValueError(f"URL audio is {int(clen)} bytes, exceeds 512 MB cap.")
            data = resp.read(_MAX_URL_BYTES + 1)
            if len(data) > _MAX_URL_BYTES:
                raise ValueError(f"URL audio exceeds 512 MB cap during read.")
            name = os.path.basename(parsed.path) or "audio-from-url"
            logger.info(f"({request_id}) URL fetch: {len(data)} bytes from {url} (content-type={ctype!r}).")
            return data, name

    try:
        return await asyncio.to_thread(_fetch_sync)
    except HTTPError as e:
        raise ValueError(f"URL fetch failed: HTTP {e.code} {e.reason}") from e
    except URLError as e:
        raise ValueError(f"URL fetch failed: {e.reason}") from e


@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(
    request: Request,
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    # OpenAI Whisper API-compatible form fields (multipart). All optional —
    # only `file` is required. We accept-and-ignore the parameters Parakeet
    # can't honor (model is fixed; greedy decoding means temperature is 0;
    # we don't bias on `prompt`); they're listed here so OpenAI-SDK clients
    # don't 422 when sending the standard payload.
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: Optional[float] = Form(0.0),
    # Server-specific extensions (query string — keep them out of the
    # multipart form so OpenAI clients don't have to know about them).
    chunk_length: Optional[float] = Query(None),
    chunk_overlap: Optional[float] = Query(None),
    batch_size: Optional[int] = Query(None),
    long_audio_threshold: Optional[float] = Query(None),
    strategy: Optional[str] = Query(None, description="offline (default; full ≤ cap, split_full above) | full | split_full | streaming."),
):
    """
    POST /v1/audio/transcriptions — OpenAI Whisper API drop-in.

    Parameters follow the OpenAI spec (multipart form):
      - file (required) — audio file (wav, mp3, ogg, flac, m4a, etc.)
      - response_format ∈ {json, text, srt, vtt, verbose_json}, default `json`
      - model, language, prompt, temperature — accepted, with caveats:
          model     : ignored (server is fixed to parakeet-tdt-0.6b-v2)
          language  : ignored (Parakeet 0.6b is English-only)
          prompt    : ignored (no prompt biasing in this model)
          temperature: ignored (we use greedy decoding, effectively 0.0)
      - timestamp_granularities[] ∈ {segment, word}, default ["segment"]
          adding "word" populates a top-level `words` array (only on
          response_format=verbose_json).

    Server extensions (query string):
      - chunk_length, chunk_overlap, batch_size, long_audio_threshold
      - strategy ∈ {offline, full, split_full, streaming}
    """
    if not asr_model:
        logger.error("REST Request: ASR model is not available.")
        return JSONResponse(status_code=503, content={"error": "ASR model not available. Service is initializing or encountered an error."})

    request_id = base64.urlsafe_b64encode(os.urandom(6)).decode() # Short unique ID for logging

    # Exactly one of `file` (multipart) or `url` (form field) must be set. The
    # URL path fetches the bytes server-side and then re-enters the same
    # pipeline as a file upload.
    if (file is None) == (not url):
        return JSONResponse(
            status_code=400,
            content={"error": "Provide exactly one of `file` (multipart) or `url` (form field)."},
        )
    if file is not None:
        logger.info(f"({request_id}) REST request received for file: '{file.filename}'. Content-type: {file.content_type}")
    else:
        logger.info(f"({request_id}) REST request received for url: {url!r}.")

    # Validate response_format up front so we fail fast on bad client input.
    if response_format not in _VALID_RESPONSE_FORMATS:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid response_format '{response_format}'. Must be one of: {sorted(_VALID_RESPONSE_FORMATS)}"},
        )

    # OpenAI's timestamp_granularities[] uses bracket-suffixed multipart keys,
    # which FastAPI's `Form()` doesn't bind cleanly. Read the raw form to pick
    # them up from either spelling, then normalize. Default to ["segment"]
    # per OpenAI's spec.
    requested_granularities: List[str] = ["segment"]
    try:
        raw_form = await request.form()
        granularities = raw_form.getlist("timestamp_granularities[]") or raw_form.getlist("timestamp_granularities")
        if granularities:
            requested_granularities = [g for g in granularities if g in {"segment", "word"}]
            if not requested_granularities:
                requested_granularities = ["segment"]
    except Exception:
        pass  # If form re-read fails, fall back to default.
    want_word_timestamps = "word" in requested_granularities

    # Log (but ignore) OpenAI params we can't honor on Parakeet so callers can
    # see them being accepted rather than 422'd.
    ignored = []
    if model is not None:
        ignored.append(f"model={model!r} (server is fixed to {ASR_MODEL_NAME})")
    if language is not None and language.lower() not in {"en", "english"}:
        ignored.append(f"language={language!r} (model is English-only)")
    if prompt:
        ignored.append("prompt=<provided> (Parakeet has no prompt biasing)")
    if temperature and temperature != 0.0:
        ignored.append(f"temperature={temperature} (greedy decoding only)")
    if ignored:
        logger.info(f"({request_id}) REST: accepted-but-ignored OpenAI params: {'; '.join(ignored)}")

    try:
        # Parse and validate common ASR configuration from query parameters
        client_config_rest = parse_request_config(
            chunk_length, chunk_overlap, batch_size, long_audio_threshold,
            strategy=strategy,
        )
        logger.info(f"({request_id}) REST: Parsed request config: {client_config_rest}")
    except ValueError as e_config:
        logger.warning(f"({request_id}) REST: Invalid request parameters: {e_config}")
        return JSONResponse(status_code=400, content={"error": f"Invalid request parameter: {str(e_config)}"})

    long_audio_settings_applied_this_session = False
    
    # Determine processing device and data type for this session
    session_processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    session_target_operational_dtype = torch.float32
    if session_processing_device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        session_target_operational_dtype = torch.bfloat16
        logger.info(f"({request_id}) REST Session: Will use bfloat16 on CUDA for ASR computation.")
    else:
        logger.info(f"({request_id}) REST Session: Will use float32 on {session_processing_device} for ASR computation.")

    start_time_total_request_processing = time.time()
    final_response_content: Optional[dict] = None  # error-path JSON body
    success_response: Optional[Response] = None    # set on the happy path
    response_status_code: int = 200

    try:
        # Acquire lock for exclusive ASR model access
        async with model_access_lock:
            logger.debug(f"({request_id}) REST: Acquired ASR model access lock.")
            try:
                if file is not None:
                    audio_bytes = await file.read()
                    logger.info(f"({request_id}) REST: Read {len(audio_bytes)} bytes from upload '{file.filename}'.")
                else:
                    try:
                        audio_bytes, _fetched_name = await _fetch_audio_url(url, request_id)
                    except ValueError as e_url:
                        logger.warning(f"({request_id}) REST: URL fetch failed: {e_url}")
                        return JSONResponse(status_code=400, content={"error": str(e_url)})

                waveform_tensor, total_audio_duration_s = await load_and_preprocess_audio(
                    audio_source=io.BytesIO(audio_bytes),
                    target_sample_rate=MODEL_SAMPLE_RATE,
                    request_id=request_id
                )

                if waveform_tensor is None or total_audio_duration_s == 0:
                    logger.warning(f"({request_id}) REST: Audio processing failed or resulted in empty audio.")
                    response_status_code = 400 # Bad Request
                    final_response_content = {"error": "Audio processing failed or the audio file is empty/corrupted."}
                elif asr_model is None: # Re-check, though checked at start
                    logger.critical(f"({request_id}) REST: ASR model became unavailable during locked operation.")
                    response_status_code = 503 # Service Unavailable
                    final_response_content = {"error": "ASR model became unavailable during processing."}
                else:
                    resolved_strategy = resolve_strategy(
                        audio_duration_s=total_audio_duration_s,
                        client_config=client_config_rest,
                        is_streaming=False,
                    )
                    logger.info(f"({request_id}) REST: Resolved strategy = '{resolved_strategy.value}' for {total_audio_duration_s:.2f}s of audio.")

                    if resolved_strategy == ProcessingStrategy.STREAMING:
                        response_status_code = 400
                        final_response_content = {
                            "error": "Strategy 'streaming' requires a WebSocket connection. "
                                     "For REST, pick 'full' or 'split_full' "
                                     "(or omit `strategy` for offline auto-routing)."
                        }
                    else:
                        # Apply model settings (long/short audio attention) for this session.
                        # For REST, the decision for long/short audio settings is based on the total file duration.
                        long_audio_settings_applied_this_session = await _apply_model_settings_for_session(
                            decision_duration_s=total_audio_duration_s,
                            target_processing_device=session_processing_device,
                            target_operational_dtype=session_target_operational_dtype,
                            long_audio_threshold_config=client_config_rest["long_audio_threshold"],
                            request_id=request_id
                        )
                        logger.info(f"({request_id}) REST: ASR model settings applied for session. Long audio specific settings active: {long_audio_settings_applied_this_session}.")

                        # Watch for a client disconnect during the (possibly
                        # long) inference. uvicorn keeps this coroutine running
                        # after the client cancels, so without this a cancelled
                        # big-file job would hold the model lock to completion
                        # and block every later request. SPLIT_FULL polls
                        # cancel_event between slices and bails; FULL is one
                        # shot, so the watcher only buys us the chance to skip
                        # the final-result send if the client is already gone.
                        rest_cancel_event = threading.Event()
                        rest_watcher = asyncio.create_task(
                            _watch_request_disconnect(request, rest_cancel_event, request_id)
                        )
                        try:
                            if resolved_strategy == ProcessingStrategy.SPLIT_FULL:
                                # Sequential FULL passes over overlapping
                                # slices — keeps FULL-mode throughput on files
                                # longer than the GPU's single-shot ceiling.
                                segments, asr_processing_time_s = await _transcribe_split_full(
                                    waveform=waveform_tensor,
                                    audio_duration_s=total_audio_duration_s,
                                    client_config=client_config_rest,
                                    request_id=f"REST-{request_id}",
                                    cancel_event=rest_cancel_event,
                                )
                            else:
                                # FULL: single-pass through encoder + decoding_computer.
                                # Bypasses NeMo's transcribe() wrapper to dodge the
                                # short-audio CUDA-graph bug (see _transcribe_full).
                                # (Single pass — not cooperatively cancellable.)
                                segments, asr_processing_time_s = await _transcribe_full(
                                    waveform=waveform_tensor,
                                    audio_duration_s=total_audio_duration_s,
                                    request_id=f"REST-{request_id}",
                                )
                        finally:
                            rest_watcher.cancel()

                        if rest_cancel_event.is_set():
                            # Client went away mid-transcription — discard the
                            # aborted result; the lock is already releasing.
                            logger.info(f"({request_id}) REST: client disconnected; discarding cancelled result.")
                            response_status_code = 499
                            final_response_content = {"error": "Client disconnected; transcription cancelled."}
                            raise _RestCancelled()

                        full_transcribed_text = " ".join(s['text'] for s in segments).strip()
                        total_server_processing_time_s = round(time.time() - start_time_total_request_processing, 3)

                        # Build word-level timestamps if the client requested them
                        # (timestamp_granularities=["word"]). Aggregates each segment's
                        # token IDs + the per-segment start/end to give real per-word
                        # timestamps via the SentencePiece word-boundary marker.
                        words_for_response: Optional[List[dict]] = None
                        if want_word_timestamps and asr_model is not None:
                            words_for_response = []
                            for seg in segments:
                                seg_tokens = seg.get("tokens") or []
                                seg_start = seg.get("start", 0.0)
                                seg_end = seg.get("end", seg_start)
                                if not seg_tokens:
                                    continue
                                # Interpolate per-token times uniformly within the segment —
                                # decoder per-token timestamps already landed in the segment
                                # start/end, but we don't carry the per-token vector through
                                # the helper. Even spacing across the segment span is a tight
                                # approximation; the boundary tokens hit start and end exactly.
                                n = len(seg_tokens)
                                if n == 1:
                                    times = [seg_start]
                                else:
                                    span = max(seg_end - seg_start, 0.0)
                                    times = [seg_start + (i / (n - 1)) * span for i in range(n)]
                                seg_words = tokens_to_words(seg_tokens, times, asr_model.tokenizer)
                                words_for_response.extend(seg_words)

                        success_response = _build_openai_response(
                            response_format=response_format,
                            segments=segments,
                            words=words_for_response,
                            full_text=full_transcribed_text,
                            duration_s=total_audio_duration_s,
                            asr_time_s=asr_processing_time_s,
                            total_server_time_s=total_server_processing_time_s,
                            resolved_strategy=resolved_strategy,
                        )
                        response_status_code = 200
                        logger.info(
                            f"({request_id}) REST: Transcription successful (strategy={resolved_strategy.value}, format={response_format}). "
                            f"Duration: {total_audio_duration_s:.2f}s, ASR time: {asr_processing_time_s:.2f}s, segments: {len(segments)}."
                        )

            except _RestCancelled:
                # Normal cancellation (client disconnected) — the 499 body is
                # already set; just unwind to the finally (model revert).
                pass
            except Exception as e_locked_rest_processing:
                import traceback
                tb_str = traceback.format_exc()
                logger.error(f"({request_id}) REST: Error occurred during locked ASR processing: {e_locked_rest_processing!r}\n{tb_str}", exc_info=False)
                # Ensure a response is set if not already
                if final_response_content is None:
                    response_status_code = 500 # Internal Server Error
                    final_response_content = {
                        "error": "An unexpected error occurred during transcription processing.",
                        "detail": f"{type(e_locked_rest_processing).__name__}: {e_locked_rest_processing}",
                        "traceback": tb_str.splitlines()[-15:],  # last 15 lines, for debugging
                    }
            finally:
                logger.debug(f"({request_id}) REST: Releasing ASR model lock and reverting model state.")
                # Always revert model state at the end of the locked block
                await _revert_model_to_global_original_state(
                    long_audio_settings_were_active_for_session=long_audio_settings_applied_this_session,
                    session_processing_device=session_processing_device, # The device used in this session
                    request_id=f"{request_id}-rest_model_revert"
                )
                logger.info(f"({request_id}) REST: ASR Model state reverted after session.")
        
        # Success path: return the rendered OpenAI-shaped response.
        if success_response is not None:
            return success_response

        # Error path: build a JSON error response from final_response_content.
        if final_response_content is None:
            logger.error(f"({request_id}) REST: No response prepared after model lock release. Setting generic error.")
            response_status_code = 500
            final_response_content = {"error": "An unknown error occurred while processing the REST request."}
        return JSONResponse(status_code=response_status_code, content=final_response_content)

    except Exception as e_outer_rest_handler:
        # Catch-all for errors outside the model lock, e.g., initial parameter parsing issues
        # that were not caught by `parse_request_config`'s ValueError.
        logger.error(f"({request_id}) REST: Unhandled outer error in endpoint: {e_outer_rest_handler}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "An unexpected server error occurred in the REST endpoint.", "detail": str(e_outer_rest_handler)})
    finally:
        # Ensure file object from UploadFile is closed if FastAPI hasn't handled it.
        if file is not None and hasattr(file, 'file') and file.file and not file.file.closed:
            await asyncio.to_thread(file.file.close)

        source_label = (
            f"file '{file.filename}'" if file is not None else f"url {url!r}"
        )
        logger.info(
            f"({request_id}) REST request for {source_label} completed with status code "
            f"{response_status_code if 'response_status_code' in locals() else 'unknown'}."
        )


async def _ws_accumulate_then_process(
    websocket: WebSocket,
    session_id: str,
    client_config: dict,
    session_processing_device: str,
    session_target_operational_dtype: torch.dtype,
    resolved_strategy: ProcessingStrategy,
    log_prefix: str,
) -> Optional[dict]:
    """
    Accumulate all WS binary frames until "END", then transcribe via FULL or
    SPLIT_FULL. Returns the final_transcription payload dict to be sent by
    the caller after the model lock is released.

    `resolved_strategy` may be the OFFLINE placeholder when the caller didn't
    know the duration upfront — we re-resolve here after load_and_preprocess
    so it concretizes to FULL or SPLIT_FULL based on the now-known duration.

    Returns None on disconnect or empty input; otherwise the payload dict.
    """
    accumulated = bytearray()
    logger.info(f"({session_id}) {log_prefix}: Waiting to receive audio data...")
    while True:
        if websocket.application_state != WebSocketState.CONNECTED:
            raise WebSocketDisconnect(code=1001, reason="Client disconnected during file data transfer.")
        try:
            message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.warning(f"({session_id}) {log_prefix}: Timeout waiting for audio data chunk from client.")
            raise WebSocketDisconnect(code=1008, reason="Timeout waiting for file data from client.")
        if message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect(code=message.get('code', 1000))
        if 'text' in message and message['text']:
            if message['text'].upper() == "END":
                logger.info(f"({session_id}) {log_prefix}: 'END' signal received. Bytes received: {len(accumulated)}.")
                break
            logger.warning(f"({session_id}) {log_prefix}: Unexpected text during data transfer: {message['text'][:100]}")
        elif 'bytes' in message and message['bytes']:
            accumulated.extend(message['bytes'])
            if len(accumulated) % (1024 * 1024 * 5) < len(message['bytes']):
                logger.debug(f"({session_id}) {log_prefix}: Received {len(accumulated)} bytes so far...")
        else:
            logger.warning(f"({session_id}) {log_prefix}: Unexpected message type: {message.get('type')}")

    if not accumulated:
        raise ValueError("No audio data received before 'END' signal.")

    logger.info(f"({session_id}) {log_prefix}: Loading {len(accumulated)} accumulated bytes.")
    waveform, audio_duration_s = await load_and_preprocess_audio(
        audio_source=io.BytesIO(bytes(accumulated)),
        target_sample_rate=MODEL_SAMPLE_RATE,
        request_id=session_id,
    )
    accumulated.clear()
    if waveform is None or audio_duration_s == 0:
        raise ValueError(f"{log_prefix}: Audio loading or preprocessing resulted in empty audio.")
    if asr_model is None:
        raise RuntimeError(f"{log_prefix}: ASR model became None during processing.")

    # Two-step strategy resolution: if the first pass returned the OFFLINE
    # placeholder (because we didn't know duration yet), re-resolve now that
    # load_and_preprocess_audio has given us the real duration.
    if resolved_strategy == ProcessingStrategy.OFFLINE:
        resolved_strategy = resolve_strategy(
            audio_duration_s=audio_duration_s,
            client_config=client_config,
            is_streaming=False,
        )
        logger.info(
            f"({session_id}) {log_prefix}: re-resolved strategy with duration → "
            f"{resolved_strategy.value}"
        )

    # Apply model settings against the actual duration now that we know it.
    long_audio_active = await _apply_model_settings_for_session(
        decision_duration_s=audio_duration_s,
        target_processing_device=session_processing_device,
        target_operational_dtype=session_target_operational_dtype,
        long_audio_threshold_config=client_config["long_audio_threshold"],
        request_id=session_id,
    )
    logger.info(
        f"({session_id}) {log_prefix}: Model settings applied. "
        f"Duration={audio_duration_s:.2f}s, long-audio={long_audio_active}, strategy={resolved_strategy.value}"
    )

    # Watch for a mid-inference disconnect so a cancelled job doesn't run
    # to completion holding the model lock (mirrors the REST path).
    # SPLIT_FULL polls cancel_event between slices and bails; FULL is
    # one shot.
    cancel_event = threading.Event()
    ws_watcher = asyncio.create_task(_watch_ws_disconnect(websocket, cancel_event, session_id))
    try:
        if resolved_strategy == ProcessingStrategy.SPLIT_FULL:
            segments, asr_t = await _transcribe_split_full(
                waveform=waveform,
                audio_duration_s=audio_duration_s,
                client_config=client_config,
                request_id=f"WS-{session_id}",
                cancel_event=cancel_event,
            )
        else:
            # FULL — single pass; not cooperatively cancellable, but the
            # watcher still lets us skip the final send below.
            segments, asr_t = await _transcribe_full(
                waveform=waveform,
                audio_duration_s=audio_duration_s,
                request_id=f"WS-{session_id}",
            )
    finally:
        ws_watcher.cancel()
        # Revert under the lock owner's lifecycle (we are still inside the lock).
        await _revert_model_to_global_original_state(
            long_audio_settings_were_active_for_session=long_audio_active,
            session_processing_device=session_processing_device,
            request_id=f"{session_id}-{log_prefix.lower().replace(' ', '_')}_revert",
        )

    if cancel_event.is_set() or websocket.application_state != WebSocketState.CONNECTED:
        logger.info(f"({session_id}) {log_prefix}: client disconnected before final transcription.")
        return None

    text = " ".join(s.get('text', '') for s in segments).strip()
    return {
        "type": "final_transcription",
        # Whisper-compatible fields
        "task": "transcribe",
        "language": "english",
        "duration": round(audio_duration_s, 3),
        "text": text,
        "segments": segments,
        # Extensions (non-Whisper):
        "strategy": resolved_strategy.value,
        "transcription_time_seconds": round(asr_t, 3),
        "total_segments": len(segments),
        "final_duration_processed_seconds": round(audio_duration_s, 3),
        "csv_content": generate_csv_content(segments),
        "srt_content": generate_srt_content(segments),
        "vtt_content": _segments_to_vtt(segments),
    }


async def _ws_handle_unified(
    websocket: WebSocket,
    log_prefix: str = "WS",
) -> None:
    """
    Unified WebSocket entry. Reads the JSON config first frame, then either:

      - if `config.url` is set → `handle_streaming_url` (ffmpeg `-i <url>`
        live-stream transcription; no binary frames are read);
      - else if `config.strategy=streaming` → `handle_streaming_pcm` (ffmpeg
        stdin pipe, audio bytes come on binary frames);
      - else → `_ws_accumulate_then_process` (accumulate-then-process for
        FULL / SPLIT_FULL).
    """
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    await websocket.accept()
    logger.info(f"({session_id}) {log_prefix}: WebSocket connection accepted.")

    # Send session_id to the client so it can hit the cancel HTTP
    # endpoint without waiting for WS buffer drain. Safe to send
    # before any client config — WS is full-duplex once accepted.
    try:
        await websocket.send_json({"type": "session", "session_id": session_id})
    except Exception as e_session_send:
        logger.warning(f"({session_id}) {log_prefix}: failed to send session id: {e_session_send}")

    if not asr_model:
        logger.error(f"({session_id}) {log_prefix}: ASR model not available.")
        await websocket.send_json({"type": "error", "error": "ASR model not available."})
        await websocket.close(code=1011)
        return

    session_processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    session_target_operational_dtype = torch.float32
    if session_processing_device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        session_target_operational_dtype = torch.bfloat16
    logger.info(
        f"({session_id}) {log_prefix}: device={session_processing_device}, dtype={session_target_operational_dtype}."
    )

    final_payload: Optional[dict] = None
    try:
        config_text = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
        config_dict = json.loads(config_text)

        # URL-over-WS branch: config contains a `url` string, no binary
        # frames follow. The streaming pipeline pulls audio from ffmpeg
        # `-i <url>` instead of stdin.
        url_in_config = (config_dict.get("url") or "").strip() if isinstance(config_dict.get("url"), str) else ""

        client_config = parse_websocket_config(config_dict)
        logger.info(f"({session_id}) {log_prefix}: parsed client config: {client_config}")

        resolved = resolve_strategy(
            audio_duration_s=None,
            client_config=client_config,
            is_streaming=True,
        )
        logger.info(f"({session_id}) {log_prefix}: resolved strategy = {resolved.value}.")

        if url_in_config:
            # Validate the URL up front (same http/https-only check the REST
            # URL ingest uses). Anything else → reject.
            from urllib.parse import urlparse
            parsed = urlparse(url_in_config)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError(
                    f"Unsupported URL scheme '{parsed.scheme}'; must be http or https."
                )
            long_audio_active = False
            async with model_access_lock:
                try:
                    long_audio_active = await _apply_model_settings_for_session(
                        decision_duration_s=client_config["chunk_length"],
                        target_processing_device=session_processing_device,
                        target_operational_dtype=session_target_operational_dtype,
                        long_audio_threshold_config=client_config["long_audio_threshold"],
                        request_id=session_id,
                    )
                    await handle_streaming_url(
                        websocket, session_id, session_processing_device,
                        client_config, url_in_config,
                    )
                except Exception as e_stream:
                    logger.error(f"({session_id}) {log_prefix}: URL stream error: {e_stream}", exc_info=True)
                    if websocket.application_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_json({"type": "error", "error": f"Server error during URL stream: {e_stream}"})
                        except Exception:
                            pass
                finally:
                    await _revert_model_to_global_original_state(
                        long_audio_settings_were_active_for_session=long_audio_active,
                        session_processing_device=session_processing_device,
                        request_id=f"{session_id}-url_stream_revert",
                    )
        elif resolved == ProcessingStrategy.STREAMING:
            # Streaming pipeline: ffmpeg producer/consumer feeding the v2 engine.
            # We don't know audio duration upfront in streaming; use chunk_length
            # as the proxy for the long-audio attention decision.
            long_audio_active = False
            async with model_access_lock:
                try:
                    long_audio_active = await _apply_model_settings_for_session(
                        decision_duration_s=client_config["chunk_length"],
                        target_processing_device=session_processing_device,
                        target_operational_dtype=session_target_operational_dtype,
                        long_audio_threshold_config=client_config["long_audio_threshold"],
                        request_id=session_id,
                    )
                    await handle_streaming_pcm(
                        websocket, session_id, session_processing_device, client_config,
                    )
                except Exception as e_stream:
                    logger.error(f"({session_id}) {log_prefix}: streaming error: {e_stream}", exc_info=True)
                    if websocket.application_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_json({"type": "error", "error": f"Server error during streaming: {e_stream}"})
                        except Exception:
                            pass
                finally:
                    await _revert_model_to_global_original_state(
                        long_audio_settings_were_active_for_session=long_audio_active,
                        session_processing_device=session_processing_device,
                        request_id=f"{session_id}-stream_revert",
                    )
        else:
            # FULL / SPLIT_FULL / OFFLINE-placeholder — accumulate first
            # (no lock), then load, acquire lock, transcribe, revert,
            # release lock, send final.
            async with model_access_lock:
                try:
                    final_payload = await _ws_accumulate_then_process(
                        websocket=websocket,
                        session_id=session_id,
                        client_config=client_config,
                        session_processing_device=session_processing_device,
                        session_target_operational_dtype=session_target_operational_dtype,
                        resolved_strategy=resolved,
                        log_prefix=log_prefix,
                    )
                except Exception as e_proc:
                    logger.error(f"({session_id}) {log_prefix}: processing error: {e_proc}", exc_info=True)
                    if websocket.application_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_json({"type": "error", "error": f"Server error: {e_proc}"})
                        except Exception:
                            pass

        if final_payload and websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json(final_payload)
            logger.info(f"({session_id}) {log_prefix}: final_transcription sent.")

    except asyncio.TimeoutError:
        logger.warning(f"({session_id}) {log_prefix}: Timeout waiting for initial client config.")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": "Timeout: No configuration received."})
            except Exception: pass
    except json.JSONDecodeError as e_json:
        logger.warning(f"({session_id}) {log_prefix}: Bad JSON config: {e_json}")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": f"Invalid JSON configuration: {e_json}"})
            except Exception: pass
    except ValueError as e_val:
        logger.warning(f"({session_id}) {log_prefix}: Value error: {e_val}")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": str(e_val)})
            except Exception: pass
    except WebSocketDisconnect as e_disc:
        logger.info(f"({session_id}) {log_prefix}: WebSocket disconnected. code={e_disc.code} reason={e_disc.reason}")
    except Exception as e_outer:
        logger.error(f"({session_id}) {log_prefix}: Unhandled exception: {e_outer}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": "An unexpected server error occurred."})
            except Exception: pass
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            try:
                await websocket.close(code=1000)
            except Exception as e_close:
                logger.warning(f"({session_id}) {log_prefix}: close error: {e_close}")
        logger.info(f"({session_id}) {log_prefix}: WebSocket session ended.")


@app.websocket("/v1/audio/transcriptions")
async def websocket_transcribe_unified(websocket: WebSocket):
    """
    Unified WebSocket endpoint. Client sends a JSON config first frame; server
    resolves the processing strategy and dispatches:

      - config carries `url`     → ffmpeg `-i <url>` live-stream transcription
                                   (HLS, icecast, RTSP, m3u8); no binary frames.
      - strategy=streaming       → ffmpeg producer + chunked engine over the
                                   binary frames the client sends.
      - strategy=full            → accumulate-then-process, single-pass via
                                   encoder + decoding_computer.
      - strategy=split_full      → accumulate-then-process, sequential FULL
                                   passes over slices.
      - strategy=offline (default for WS-accumulate) → routed to FULL or
                                   SPLIT_FULL after load_and_preprocess gives
                                   us the duration.
    """
    await _ws_handle_unified(websocket, log_prefix="WS")


if __name__ == "__main__":
    
    if not asr_model:
        logger.critical(
            f"ASR Model ('{ASR_MODEL_NAME}') failed to load. "
            "The Parakeet ASR FastAPI server cannot start. "
            "Please check model availability, network connection (if downloading), "
            "and logs for details on the loading error."
        )
    else:
        logger.info(f"Attempting to start Uvicorn server on host 'localhost', port {PORT}.")
        logger.info("The ASR service will be available once Uvicorn starts successfully.")
        logger.info(f"Loaded ASR model: {ASR_MODEL_NAME}")
        logger.info(f"Default ASR chunk length: {TRANSCRIBE_CHUNK_LEN}s, overlap: {TRANSCRIBE_OVERLAP}s")
        
        # Note: Uvicorn's `workers` parameter here refers to Uvicorn workers (processes).
        # NeMo's `NUM_WORKERS` (for DataLoader) is a separate concept.
        # For simplicity in development, often Uvicorn is run with 1 worker.
        # For production, multiple Uvicorn workers might be used, but this requires
        # careful consideration of how the global ASR model is shared or replicated.
        # The current setup with a global model and asyncio.Lock is best suited for
        # a single Uvicorn worker process managing multiple concurrent asyncio tasks.
        uvicorn.run(
            "main:app",             # FastAPI app instance string
            host=HOST,              # Host to bind to
            port=PORT,              # Port to listen on
            workers=1,              # Single worker — global model + asyncio.Lock
            log_level=log_level_str.lower(),
            # WebSocket keepalive — defaults of 20s ping interval + 20s timeout
            # are way too aggressive for the streaming endpoints: a long file
            # being streamed over WS or a long-running mic session can saturate
            # the socket with binary chunks + outbound segments_batch messages,
            # which delays ping/pong handshakes and trips a 1011 close. Push
            # the timeout to 5 minutes so the connection survives realistic
            # workloads while still catching genuinely dead clients.
            ws_ping_interval=30,
            ws_ping_timeout=300,
        )