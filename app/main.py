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
from enum import Enum
from typing import Awaitable, Callable, Optional, Tuple, List
import subprocess
import uvicorn

import numpy as np

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import torchaudio
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.asr_model import ASRModel as NeMoASRModelType
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.streaming_utils import (
    BatchedFrameASRTDT,
    AudioFeatureIterator,
)

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
    """Transcription dispatch modes. See README and resolve_strategy() for selection rules."""
    AUTO = "auto"
    FULL = "full"
    CHUNKED = "chunked"
    PROGRESSIVE = "progressive"


# Strategy configuration.
# DEFAULT_STRATEGY=auto lets resolve_strategy() pick based on duration/streaming.
# MAX_FULL_WAVEFORM_S caps when `full` mode is selected by auto-resolution
# (NVIDIA's 24-minute full-attention ceiling for parakeet-tdt-0.6b-v2).
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", ProcessingStrategy.AUTO.value).lower()
MAX_FULL_WAVEFORM_S = float(os.getenv("MAX_FULL_WAVEFORM_S", 1440.0))

# Streaming context windows for the stateful (BatchedFrameASRTDT) engine — Phase 6.
# NeMo's reference recommends 10-10-5 for offline-like quality, 10-2-2 for live latency.
STREAMING_LEFT_CONTEXT_S = float(os.getenv("STREAMING_LEFT_CONTEXT_S", 10.0))
STREAMING_CHUNK_S = float(os.getenv("STREAMING_CHUNK_S", 10.0))
STREAMING_RIGHT_CONTEXT_S = float(os.getenv("STREAMING_RIGHT_CONTEXT_S", 5.0))
STREAMING_LIVE_CHUNK_S = float(os.getenv("STREAMING_LIVE_CHUNK_S", 2.0))
STREAMING_LIVE_RIGHT_CONTEXT_S = float(os.getenv("STREAMING_LIVE_RIGHT_CONTEXT_S", 2.0))

# Progressive mode: PCM buffered before emitting the first partial.
EARLY_BUFFER_TARGET_S = float(os.getenv("EARLY_BUFFER_TARGET_S", 15.0))

# Phase 6 — Stateful chunked engine (BatchedFrameASRTDT). When true, the
# chunked strategy carries decoder state across chunks instead of running
# independent overlapping transcribe() calls. Better quality (no boundary
# duplicates / drops), but currently produces a single concatenated text
# segment rather than per-utterance timestamps. Toggle false to fall back to
# the legacy independent-chunk path (which keeps timestamps).
USE_STATEFUL_CHUNKED = os.getenv("USE_STATEFUL_CHUNKED", "false").lower() == "true"

# Stateful engine ties up the model lock proportional to audio duration
# (chunks process sequentially with batch_size=1). Measured RTF ≈ 0.022 on
# DGX Spark — a 30 min file takes ~40s, a 3h file takes ~4min. Above the
# threshold, fall back to the legacy independent-chunk path (which batches
# 4 chunks per call and is ~20× faster, at the cost of some boundary
# quality). Set to 0 to disable the cap (always use stateful when on).
STATEFUL_MAX_DURATION_S = float(os.getenv("STATEFUL_MAX_DURATION_S", 1800.0))

logger.info(
    f"Configuration loaded:\n"
    f"  App: workers={NUM_WORKERS}, sample_rate={MODEL_SAMPLE_RATE}, port={PORT}\n"
    f"  Model: {ASR_MODEL_NAME}, long_audio_threshold_for_model_settings={LONG_AUDIO_THRESHOLD_S}s\n"
    f"  Chunking Defaults: length={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s, batch_cap={CHUNKING_BATCH_SIZE}\n"
    f"  Streaming (ffmpeg): pcm_read_chunk_size={FFMPEG_PCM_CHUNK_SIZE_BYTES}B\n"
    f"  Strategy: default={DEFAULT_STRATEGY}, max_full_waveform_s={MAX_FULL_WAVEFORM_S}, early_buffer_target_s={EARLY_BUFFER_TARGET_S}\n"
    f"  Stateful streaming: offline {STREAMING_LEFT_CONTEXT_S}-{STREAMING_CHUNK_S}-{STREAMING_RIGHT_CONTEXT_S}, "
    f"live {STREAMING_LEFT_CONTEXT_S}-{STREAMING_LIVE_CHUNK_S}-{STREAMING_LIVE_RIGHT_CONTEXT_S}\n"
    f"  Stateful chunked engine: USE_STATEFUL_CHUNKED={USE_STATEFUL_CHUNKED}, "
    f"STATEFUL_MAX_DURATION_S={STATEFUL_MAX_DURATION_S}"
)

# --- FastAPI App Setup ---
app = FastAPI(title="Parakeet ASR Service", version="1.0.0")
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
    """Snapshot of model load + decoding/runtime config for /health and /v1/debug/state."""
    info: dict = {
        "model_loaded": asr_model is not None,
        "model_name": ASR_MODEL_NAME,
        "config": {
            "default_strategy": DEFAULT_STRATEGY,
            "max_full_waveform_s": MAX_FULL_WAVEFORM_S,
            "long_audio_threshold_s": LONG_AUDIO_THRESHOLD_S,
            "use_stateful_chunked": USE_STATEFUL_CHUNKED,
            "stateful_max_duration_s": STATEFUL_MAX_DURATION_S,
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


@app.get("/readyz")
async def readyz():
    """Readiness probe. 200 when the ASR model is loaded; 503 otherwise."""
    if asr_model is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "model_loaded": False})
    return {"status": "ready", "model_loaded": True}


@app.get("/v1/debug/state", include_in_schema=False)
async def debug_state():
    """Back-compat alias — same payload as /health for the test/debug scripts that already query it."""
    return _collect_health_info()


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
        try:
            from omegaconf import open_dict
            cfg = asr_model.cfg.decoding
            with open_dict(cfg):
                cfg.compute_timestamps = True
                # Explicit fast default — parakeet-tdt checkpoints can ship with
                # strategy unset or 'greedy', which would make the FULL path and
                # the legacy chunked fallback fall through to the non-batched
                # GreedyTDTInfer (~20× slower than GreedyBatchedTDTInfer).
                cfg.strategy = "greedy_batch"
                cfg.preserve_alignments = False
                if "greedy" in cfg:
                    with open_dict(cfg.greedy):
                        cfg.greedy.use_cuda_graph_decoder = False
            # NOTE: we do NOT switch to strategy=greedy + preserve_alignments here
            # even when USE_STATEFUL_CHUNKED=true. Doing so globally would force
            # the FAST greedy_batch decoder off the model and make the legacy
            # chunked fallback (and the FULL path) ~20× slower. Instead we swap
            # to greedy transiently inside _transcribe_chunked_stateful only.
            asr_model.change_decoding_strategy(cfg, verbose=False)
            inferer = asr_model.decoding.decoding
            computer = getattr(inferer, "decoding_computer", None)
            final_mode = getattr(computer, "cuda_graphs_mode", "<no computer>")
            final_allow = getattr(computer, "allow_cuda_graphs", "<no computer>")
            logger.info(
                f"Decoding strategy reapplied with use_cuda_graph_decoder=False, "
                f"compute_timestamps=True. cuda_graphs_mode={final_mode!r}, "
                f"allow_cuda_graphs={final_allow!r}"
            )
        except Exception as e_cg:
            logger.warning(f"config-level cuda-graph disable failed: {e_cg}", exc_info=True)

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

# Asynchronous lock to ensure exclusive access to the ASR model during transcription calls.
# This prevents concurrent modifications to model state (e.g., device, dtype, attention settings).
model_access_lock = asyncio.Lock()


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
        await asyncio.to_thread(
            asr_model.change_attention_model,
            "rel_pos_local_attn", [256, 256], True,
        )
        # New RelPositionMultiHeadAttentionLongformer modules are constructed
        # with default float32 weights — even though the rest of the model is
        # at its pinned dtype (e.g. bf16). Forward then dies with
        # "mat1 and mat2 must have the same dtype". Cast the whole model to
        # its current dtype so new modules join the bf16 majority.
        pinned_dtype = global_original_model_dtype_torch
        await asyncio.to_thread(asr_model.to, dtype=pinned_dtype)
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
                await asyncio.to_thread(
                    asr_model.change_attention_model,
                    "rel_pos", None, True,
                )
                # Same dtype-mismatch fix as the apply path: new modules from
                # change_attention_model are float32 by default; cast back to
                # the model's pinned dtype.
                pinned_dtype = global_original_model_dtype_torch
                await asyncio.to_thread(asr_model.to, dtype=pinned_dtype)
            except Exception as e_rev_long_specific:
                logger.warning(f"({request_id}) End of Session: Failed to revert long-audio attention: {e_rev_long_specific}")

        if session_processing_device == "cuda" and torch.cuda.is_available():
            await asyncio.to_thread(gc.collect)
            await asyncio.to_thread(torch.cuda.empty_cache)
    except Exception as e_restore_globally:
        logger.error(f"({request_id}) Error during final model state reversion: {e_restore_globally}", exc_info=True)


async def _perform_asr_transcription(
    asr_model_instance: NeMoASRModelType,
    audio_input_list: List[torch.Tensor], # Expected to be float32 from load_and_preprocess_audio
    batch_size_for_transcribe_call: int,
    num_asr_workers: int,
    request_id: str = "asr"
) -> Tuple[Optional[List[Hypothesis]], float]:
    """
    Performs ASR transcription using the NeMo model.

    Args:
        asr_model_instance: The loaded NeMo ASR model.
        audio_input_list: A list of 1D float32 audio tensors for transcription.
        batch_size_for_transcribe_call: Batch size for the model's transcribe method.
        num_asr_workers: Number of workers for the model's internal DataLoader.
        request_id: Identifier for logging.

    Returns:
        A tuple containing the list of Hypothesis objects and the ASR processing time in seconds.
        Returns (None, 0.0) on failure.
    """
    if not audio_input_list or not asr_model_instance:
        logger.warning(f"({request_id}) _perform_asr_transcription called with no audio or no model.")
        return None, 0.0
        
    hypotheses: Optional[List[Hypothesis]] = None
    asr_processing_duration_s: float = 0.0
    
    try:
        model_device = next(asr_model_instance.parameters()).device
        # The model's compute layers (encoder, decoder) should be in target_operational_dtype (e.g. bfloat16)
        # as set by _apply_model_settings_for_session.
        # The NeMo AudioPreprocessor internally expects/handles float32 input signals.
    except Exception as e_dev:
        logger.error(f"({request_id}) Error getting model device: {e_dev}", exc_info=True)
        raise
    
    audio_for_preprocessor_on_device: List[torch.Tensor] = []
    try:
        # Ensure audio tensors are on the model's device AND are float32 for the preprocessor
        # (as per NeMo preprocessor warning/behavior).
        for audio_tensor in audio_input_list:
            if audio_tensor.dtype != torch.float32:
                logger.warning(f"({request_id}) Input audio tensor was not float32 ({audio_tensor.dtype}), casting.")
            audio_for_preprocessor_on_device.append(
                audio_tensor.to(device=model_device, dtype=torch.float32)
            )
    except Exception as e_mov:
        logger.error(f"({request_id}) Error moving audio to device '{model_device}' and ensuring float32: {e_mov}", exc_info=True)
        raise
    
    start_time = time.time()
    try:
        raw_hypotheses = await asyncio.to_thread(
            asr_model_instance.transcribe,
            audio=audio_for_preprocessor_on_device, # Pass float32 audio list
            batch_size=batch_size_for_transcribe_call,
            num_workers=num_asr_workers,
            return_hypotheses=True,  # Required for accessing detailed timestamp info
            timestamps=True,         # Request word/segment timestamps
            verbose=False            # Reduce NeMo's internal verbosity
        )
        asr_processing_duration_s = round(time.time() - start_time, 3)

        # Process raw_hypotheses which can be List[Hypothesis] or List[List[Hypothesis]]
        if raw_hypotheses:
            if all(isinstance(h, Hypothesis) for h in raw_hypotheses):
                hypotheses = raw_hypotheses
            elif all(isinstance(h_list, list) for h_list in raw_hypotheses): # Handle list of lists (common for batch>1)
                hypotheses = [h_item for h_sublist in raw_hypotheses for h_item in h_sublist if isinstance(h_item, Hypothesis)]
            elif isinstance(raw_hypotheses, Hypothesis): # Single hypothesis for single audio
                hypotheses = [raw_hypotheses]
            else:
                logger.warning(f"({request_id}) Unexpected hypothesis format from model.transcribe: {type(raw_hypotheses)}")
        
        if hypotheses:
             logger.debug(f"({request_id}) Transcription successful, {len(hypotheses)} hypotheses obtained in {asr_processing_duration_s}s.")
        else:
            logger.warning(f"({request_id}) Transcription yielded no valid hypotheses in {asr_processing_duration_s}s.")
            
        return hypotheses, asr_processing_duration_s
        
    except Exception as e_trans:
        logger.error(f"({request_id}) Error during asr_model.transcribe call: {e_trans}", exc_info=True)
        raise


def _process_hypotheses_to_segments(
    batch_hypotheses: Optional[List[Hypothesis]], # Made Optional
    batch_offsets_s: List[float],
    request_id: str = "seg_proc"
) -> List[dict]:
    """
    Converts a list of NeMo Hypothesis objects into a list of segment dictionaries.
    Each segment contains start time, end time, and text.

    Args:
        batch_hypotheses: A list of Hypothesis objects from NeMo ASR.
        batch_offsets_s: A list of time offsets (in seconds) corresponding to the
                         start of each audio chunk that generated a hypothesis.
        request_id: Identifier for logging.

    Returns:
        A list of segment dictionaries.
    """
    all_segments: List[dict] = []
    if not batch_hypotheses: # Check if None or empty
        logger.debug(f"({request_id}) No hypotheses provided to process into segments.")
        return all_segments
        
    if len(batch_hypotheses) != len(batch_offsets_s):
        logger.warning(f"({request_id}) Mismatch between number of hypotheses ({len(batch_hypotheses)}) and offsets ({len(batch_offsets_s)}). Cannot process segments accurately.")
        return all_segments 
    
    for hyp_idx, hyp_obj in enumerate(batch_hypotheses):
        if hyp_obj is None:
            logger.debug(f"({request_id}) Hypothesis at index {hyp_idx} is None, skipping.")
            continue
            
        chunk_offset_s = batch_offsets_s[hyp_idx]
        
        # Check for NeMo's detailed timestamp structure
        if hasattr(hyp_obj, "timestamp") and hyp_obj.timestamp and isinstance(hyp_obj.timestamp, dict):
            # 'segment' level timestamps are usually word groups or phrases
            segments_in_hyp = hyp_obj.timestamp.get("segment", []) 
            if not segments_in_hyp and hasattr(hyp_obj, "text") and hyp_obj.text:
                 # Fallback if 'segment' is empty but 'word' timestamps might exist or just plain text
                word_timestamps = hyp_obj.timestamp.get("word", [])
                if word_timestamps:
                    logger.debug(f"({request_id}) No 'segment' timestamps, but found {len(word_timestamps)} 'word' timestamps for hypothesis {hyp_idx}. Combining them.")
                    current_segment_text = []
                    current_segment_start = -1
                    for i, word_info in enumerate(word_timestamps):
                        word_text = word_info.get("word", "").strip()
                        word_start = word_info.get("start_offset", -1.0) # NeMo uses start_offset/end_offset for words
                        word_end = word_info.get("end_offset", -1.0)
                        if not word_text or word_start < 0 or word_end < word_start : continue

                        if current_segment_start == -1:
                            current_segment_start = word_start
                        current_segment_text.append(word_text)
                        
                        # Heuristic: end segment on punctuation or if next word is significantly later
                        is_last_word = (i == len(word_timestamps) - 1)
                        next_word_start = word_timestamps[i+1].get("start_offset", -1.0) if not is_last_word else -1.0
                        
                        if word_text.endswith(('.', '?', '!')) or is_last_word or \
                           (not is_last_word and next_word_start > word_end + 0.5): # End segment if >0.5s gap
                            segment_text_final = " ".join(current_segment_text)
                            start_time = round(current_segment_start + chunk_offset_s, 3)
                            end_time = round(word_end + chunk_offset_s, 3)
                            all_segments.append({"start": start_time, "end": end_time, "text": segment_text_final})
                            current_segment_text = []
                            current_segment_start = -1
                    if current_segment_text : # remaining text
                        segment_text_final = " ".join(current_segment_text)
                        start_time = round(current_segment_start + chunk_offset_s, 3)
                        # Estimate end time if only one word and no proper end
                        end_time = round((word_timestamps[-1].get("end_offset", current_segment_start + 1.0)) + chunk_offset_s, 3)
                        all_segments.append({"start": start_time, "end": end_time, "text": segment_text_final})

                elif hyp_obj.text: # No segment or word timestamps, use full text as one segment
                     logger.debug(f"({request_id}) No 'segment' or 'word' timestamps, using full hypothesis text for hypothesis {hyp_idx}.")
                     all_segments.append({
                        "start": chunk_offset_s, 
                        "end": round(chunk_offset_s + (len(hyp_obj.text.split()) * 0.5), 3), # Rough estimate for end
                        "text": hyp_obj.text.strip()
                    })

            for seg_idx, seg_meta in enumerate(segments_in_hyp): # Original loop for 'segment' level
                seg_text = seg_meta.get("segment", "").strip()
                if not seg_text: continue
                
                start_time = round(seg_meta.get("start", 0.0) + chunk_offset_s, 3)
                end_time = round(seg_meta.get("end", 0.0) + chunk_offset_s, 3)
                
                if start_time < 0 or end_time < start_time:
                    logger.warning(f"({request_id}) Invalid segment timing: start={start_time}, end={end_time} for text '{seg_text}'. Skipping.")
                    continue
                all_segments.append({"start": start_time, "end": end_time, "text": seg_text})
        
        elif hasattr(hyp_obj, "text") and hyp_obj.text: # Fallback if no timestamp attribute at all
            logger.debug(f"({request_id}) Hypothesis {hyp_idx} has no 'timestamp' attribute, using full text.")
            all_segments.append({
                "start": chunk_offset_s, 
                "end": round(chunk_offset_s + (len(hyp_obj.text.split()) * 0.5), 3), # Rough estimate
                "text": hyp_obj.text.strip()
            })
            
    # Add default keys if missing from any segment (simplifies downstream processing)
    final_output_segments = []
    for i, seg in enumerate(all_segments):
        seg_template = {"id": i, "seek":0, "tokens":[], "temperature":0.0, "avg_logprob":None, "compression_ratio":None, "no_speech_prob":None}
        seg_template.update(seg) # Override defaults with actual segment data
        final_output_segments.append(seg_template)

    logger.debug(f"({request_id}) Processed {len(batch_hypotheses)} hypotheses into {len(final_output_segments)} segments.")
    return final_output_segments


def _deduplicate_segments(
    raw_segments: List[dict],
    overlap_threshold_seconds: float = 0.3
) -> List[dict]:
    """
    Deduplicates a list of transcribed segments by merging or removing overlapping ones.

    The function sorts segments by start time. It iterates through them, deciding whether
    to keep, merge, or discard segments based on their temporal relationship with the
    previously accepted segment and the `overlap_threshold_seconds`.

    Args:
        raw_segments: A list of segment dictionaries. Each dictionary is expected
                      to have at least 'start', 'end', and 'text' keys.
        overlap_threshold_seconds: The maximum allowed overlap (in seconds) between
                                   the end of one segment and the start of the next
                                   before they are considered significantly overlapping.
                                   This also influences merging logic.

    Returns:
        A list of deduplicated and cleaned segment dictionaries, with updated 'id' fields.
        Returns an empty list if input is empty or segments are malformed.
    """
    if not raw_segments:
        return []

    # Attempt to sort segments; return empty if essential keys are missing causing TypeError
    try:
        # Sort by start time, then by end time as a secondary criterion.
        sorted_segments = sorted(raw_segments, key=lambda s: (s.get('start', float('inf')), s.get('end', float('inf'))))
    except TypeError:
        logger.warning("(_deduplicate_segments) Segments list contained items missing 'start' or 'end' keys, or they were not comparable. Returning empty list.")
        return [] # Segments are malformed for sorting

    final_segments: List[dict] = []
    prev_seg: Optional[dict] = None

    for current_segment in sorted_segments:
        # Ensure basic structure of the current segment
        if not all(key in current_segment for key in ["start", "end", "text"]):
            logger.debug(f"(_deduplicate_segments) Skipping segment due to missing keys: {current_segment.get('text', 'N/A')[:30]}")
            continue

        if prev_seg is None:
            # This is the first valid segment
            current_segment["id"] = len(final_segments)
            final_segments.append(current_segment)
            prev_seg = current_segment
            continue

        # prev_seg is guaranteed to be not None here
        current_start = current_segment["start"]
        prev_end = prev_seg["end"]

        # Condition 1: Current segment starts after (or very slightly before) previous segment ends.
        # This means they are distinct or have a minor, acceptable overlap.
        if current_start >= prev_end - overlap_threshold_seconds:
            # If there's a slight overlap, adjust the end of the previous segment
            # to ensure no temporal overlap in the final list.
            if current_start < prev_end:
                # Ensure prev_seg end doesn't go before its start
                prev_seg["end"] = max(prev_seg["start"], current_start - 0.001)
            
            current_segment["id"] = len(final_segments)
            final_segments.append(current_segment)
            prev_seg = current_segment
        else:
            # Condition 2: Current segment overlaps significantly with the previous segment.
            # This is the more complex case requiring a decision to replace or discard.
            
            # Heuristic: If the current segment is much shorter and ends not much later
            # than the previous one, it's likely a less complete version of the same utterance.
            # The 0.7 factor means if current is less than 70% of prev's duration.
            # The overlap_threshold_seconds / 2 provides a small buffer for the end time.
            current_duration = current_segment["end"] - current_segment["start"]
            prev_duration = prev_seg["end"] - prev_seg["start"]

            if current_segment["end"] < prev_end + (overlap_threshold_seconds / 2.0) and \
               current_duration < prev_duration * 0.7:
                # Discard current segment as it seems to be a less complete, overlapping part
                logger.debug(f"(_deduplicate_segments) Discarding shorter overlapping segment: '{current_segment['text'][:30]}...'")
                continue
            else:
                # Replace previous segment with current segment if current segment is preferred
                # (e.g., longer, or starts earlier but considered more complete by this logic path).
                logger.debug(f"(_deduplicate_segments) Replacing segment '{prev_seg['text'][:30]}...' with '{current_segment['text'][:30]}...'")
                current_segment["id"] = prev_seg["id"] # Retain ID of the segment being replaced
                final_segments[-1] = current_segment
                prev_seg = current_segment

    # Final cleanup: ensure segments have valid durations (end > start)
    cleaned_segments: List[dict] = []
    for i, seg in enumerate(final_segments):
        if seg["end"] <= seg["start"]:
            # If duration is zero or negative, but there's text, give it a minimal duration.
            if seg["text"].strip():
                seg["end"] = seg["start"] + 0.001 # Minimal positive duration
                seg["id"] = len(cleaned_segments)
                cleaned_segments.append(seg)
            # If no text and invalid duration, it's likely an artifact; discard.
        else:
            seg["id"] = len(cleaned_segments) # Re-assign ID based on final position
            cleaned_segments.append(seg)
            
    logger.info(f"(_deduplicate_segments) Raw segments: {len(raw_segments)}, Deduplicated segments: {len(cleaned_segments)}")
    return cleaned_segments


def parse_request_config(
    c_len: Optional[float] = None,
    c_ov: Optional[float] = None,
    b_size: Optional[int] = None,
    l_thresh: Optional[float] = None,
    strategy: Optional[str] = None,
    early_buffer_target_s: Optional[float] = None,
    live_latency: Optional[bool] = None,
    progressive_refinement: Optional[bool] = None,
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
        strategy: Processing strategy override. One of ProcessingStrategy values.
                  None falls back to DEFAULT_STRATEGY.
        early_buffer_target_s: Progressive mode — PCM seconds buffered before
                  the first partial transcript is emitted.
        live_latency: When True, progressive mode uses the 10-2-2 streaming preset
                  for ~4s latency; when False/None, uses 10-10-5 for offline-like quality.
        progressive_refinement: When True (default), progressive mode runs an
                  extra full-pass at EOF if duration <= MAX_FULL_WAVEFORM_S.

    Returns:
        A dictionary containing the validated configuration parameters.

    Raises:
        ValueError: If any parameter value is outside its allowed range.
    """
    requested_strategy = (strategy or DEFAULT_STRATEGY).lower()
    try:
        strategy_enum = ProcessingStrategy(requested_strategy)
    except ValueError:
        raise ValueError(
            f"Invalid strategy '{requested_strategy}'. "
            f"Must be one of: {[s.value for s in ProcessingStrategy]}."
        )

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
        "progressive_refinement": (
            bool(progressive_refinement) if progressive_refinement is not None else True
        ),
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


def resolve_strategy(
    audio_duration_s: Optional[float],
    client_config: dict,
    is_streaming: bool,
) -> ProcessingStrategy:
    """
    Resolve the processing strategy for a request.

    Explicit non-AUTO strategies from client_config win. Otherwise:
    - Streaming with unknown or live duration -> PROGRESSIVE.
    - Known duration <= MAX_FULL_WAVEFORM_S -> FULL (_apply_model_settings_for_session
      will engage local attention past LONG_AUDIO_THRESHOLD_S as needed).
    - Otherwise -> CHUNKED.
    """
    requested = client_config.get("strategy", ProcessingStrategy.AUTO)
    if isinstance(requested, str):
        requested = ProcessingStrategy(requested)

    if requested != ProcessingStrategy.AUTO:
        return requested

    if is_streaming:
        return ProcessingStrategy.PROGRESSIVE

    if audio_duration_s is None:
        return ProcessingStrategy.FULL

    if audio_duration_s <= MAX_FULL_WAVEFORM_S:
        return ProcessingStrategy.FULL

    return ProcessingStrategy.CHUNKED


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
        progressive_refinement=client_cfg.get("progressive_refinement"),
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

    return asr_config


async def handle_streaming_pcm(
    websocket: WebSocket,
    session_id: str,
    processing_device: str,
    client_config: dict
):
    """
    Handles the audio streaming pipeline for a WebSocket connection using ffmpeg.

    This function sets up a producer-consumer pattern:
    - Producer:
        - Receives audio byte chunks from the WebSocket client.
        - Pipes these bytes to an `ffmpeg` subprocess.
        - `ffmpeg` decodes/resamples the input audio to 16kHz mono PCM.
        - Reads the standardized PCM output from `ffmpeg`.
        - Buffers and segments this PCM data into ASR-ready chunks (fixed duration).
        - Puts these (audio_tensor, offset_s) tuples onto an asyncio Queue.
    - Consumer:
        - Retrieves ASR chunks from the queue.
        - Batches them according to `client_config["batch_size"]`.
        - Performs ASR transcription using `_perform_asr_transcription`.
        - Processes hypotheses into segments.
        - Sends `segments_batch` messages back to the client via WebSocket.

    Finally, it sends a `final_transcription` message with aggregated results.

    Args:
        websocket: The active WebSocket connection.
        session_id: A unique identifier for this streaming session.
        processing_device: The device ("cuda" or "cpu") for ASR model inference.
        client_config: Parsed configuration dictionary from the client, including
                       chunk_length, overlap, batch_size, format, etc.
    """
    sent_segments_pcm: List[dict] = [] # Stores all segments sent to client for final aggregation

    live_latency = bool(client_config.get("live_latency", False))
    # USE_STATEFUL_CHUNKED also gates progressive: when enabled the live partial
    # pipeline runs `StreamingTdtEngine` (BatchedFrameASRTDT chunk-by-chunk)
    # instead of the legacy independent-chunk consumer. Same flag means callers
    # get coherent decoder-state continuity in BOTH offline and live mode.
    use_stateful_progressive = USE_STATEFUL_CHUNKED

    if use_stateful_progressive:
        # Engine chunk + buffer follow the live or offline-like preset.
        engine_chunk_len_s = STREAMING_LIVE_CHUNK_S if live_latency else STREAMING_CHUNK_S
        if live_latency:
            engine_total_buffer_s = STREAMING_LEFT_CONTEXT_S + STREAMING_LIVE_CHUNK_S + STREAMING_LIVE_RIGHT_CONTEXT_S
            preset_label = f"10-{STREAMING_LIVE_CHUNK_S:g}-{STREAMING_LIVE_RIGHT_CONTEXT_S:g} live"
        else:
            engine_total_buffer_s = STREAMING_LEFT_CONTEXT_S + STREAMING_CHUNK_S + STREAMING_RIGHT_CONTEXT_S
            preset_label = f"{STREAMING_LEFT_CONTEXT_S:g}-{STREAMING_CHUNK_S:g}-{STREAMING_RIGHT_CONTEXT_S:g} offline-like"
        # The producer pushes engine-sized chunks with zero overlap; the engine
        # holds its own context window via the FIFO buffer.
        asr_chunk_len_s = engine_chunk_len_s
        asr_chunk_overlap_s = 0.0
        logger.info(
            f"({session_id}) Stream: stateful TDT engine — chunk={engine_chunk_len_s:g}s, "
            f"buffer={engine_total_buffer_s:g}s ({preset_label})."
        )
    elif live_latency:
        # Legacy independent-chunk path: smaller chunks ≈ faster partials, no
        # state carry. Quality is lower than the stateful path but no rebuild.
        asr_chunk_len_s = STREAMING_LIVE_CHUNK_S
        asr_chunk_overlap_s = min(client_config.get("chunk_overlap", 0.0), STREAMING_LIVE_CHUNK_S - 0.1)
        if asr_chunk_overlap_s < 0:
            asr_chunk_overlap_s = 0.0
        logger.info(
            f"({session_id}) Stream: legacy + live_latency → {asr_chunk_len_s}s chunks "
            f"(overlap {asr_chunk_overlap_s:.2f}s) instead of client's {client_config['chunk_length']}s."
        )
    else:
        asr_chunk_len_s = client_config["chunk_length"]
        asr_chunk_overlap_s = client_config["chunk_overlap"]

    # `progressive_refinement` (defaults True): after EOF, if the full audio
    # fits the FULL-mode ceiling, run a single transcribe pass over the entire
    # buffered PCM and emit a `refined_transcription` message — better quality
    # than the streamed partials, since FULL gets the whole context at once.
    progressive_refinement = bool(client_config.get("progressive_refinement", True))
    refinement_pcm_accumulator: Optional[bytearray] = bytearray() if progressive_refinement else None
    
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
        ffmpeg_command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', 'pipe:0',  # Input from stdin
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
                        logger.info(f"({session_id}) Feed ffmpeg: WebSocket no longer connected. Closing ffmpeg stdin.")
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
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        break # Exit loop on disconnect or critical error

                    if 'text' in message and message['text'].upper() == "END":
                        logger.info(f"({session_id}) Feed ffmpeg: 'END' signal received from client. Closing ffmpeg stdin.")
                        if process.stdin and not process.stdin.is_closing():
                            process.stdin.close()
                        break # End of stream signaled by client
                    
                    if 'bytes' in message and message['bytes']:
                        chunk_data = message['bytes']
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
                logger.info(f"({session_id}) Feed ffmpeg: WebSocket disconnected by client. Closing ffmpeg stdin.")
                if process.stdin and not process.stdin.is_closing():
                    process.stdin.close()
            except Exception as e_feed:
                logger.error(f"({session_id}) Feed ffmpeg: Unexpected error: {e_feed}", exc_info=True)
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
                    
                    pcm_buffer_for_asr_chunks.extend(pcm_data_from_ffmpeg)
                    temp_total_pcm_bytes_read_from_ffmpeg += len(pcm_data_from_ffmpeg)
                    if refinement_pcm_accumulator is not None:
                        refinement_pcm_accumulator.extend(pcm_data_from_ffmpeg)
                    
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
        Consumer coroutine.

        When `use_stateful_progressive` is True: drains the chunk_queue and
        feeds each tensor to `StreamingTdtEngine`, which advances the
        BatchedFrameASRTDT FIFO buffer one chunk at a time. After each step,
        the middle-token merge is re-run over `frame_asr.all_alignments`;
        newly completed sentences are emitted as `segments_batch` messages.
        At sentinel, `engine.flush()` pushes trailing silence so the merge
        can commit the final tokens.

        Otherwise: legacy independent-chunk consumer (one transcribe() call
        per consumer_batch_size_cap chunks, no cross-chunk state).
        """
        nonlocal accumulated_asr_processing_time_s, sent_segments_pcm

        if use_stateful_progressive and asr_model is not None:
            feature_stride = asr_model._cfg.preprocessor["window_stride"]
            model_stride_s = feature_stride * asr_model.encoder.subsampling_factor
            engine = StreamingTdtEngine(
                asr_model_instance=asr_model,
                chunk_len_s=engine_chunk_len_s,
                total_buffer_s=engine_total_buffer_s,
                model_stride_s=model_stride_s,
                sample_rate=target_pcm_sample_rate,
                request_id=f"WS-Stream-{session_id}-eng",
            )
            total_engine_chunks = 0
            try:
                while True:
                    item = await chunk_queue.get()
                    if item is None:
                        chunk_queue.task_done()
                        break
                    tensor, _offset_s = item
                    chunk_queue.task_done()
                    samples_np = (
                        tensor.detach().cpu().numpy()
                        if hasattr(tensor, "detach")
                        else np.asarray(tensor, dtype=np.float32)
                    )
                    await asyncio.to_thread(engine.feed_float32, samples_np)
                    total_engine_chunks += 1
                    new_segs = engine.pop_committed_segments()
                    if new_segs and websocket.application_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_json({"type": "segments_batch", "segments": new_segs})
                            sent_segments_pcm.extend(new_segs)
                        except Exception as e_send:
                            logger.warning(f"({session_id}) Stream(stateful): segment send failed: {e_send}")

                # EOF — pad + flush to commit trailing tokens
                await asyncio.to_thread(engine.flush)
                final_partials = engine.pop_final_segments()
                if final_partials and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "segments_batch", "segments": final_partials})
                        sent_segments_pcm.extend(final_partials)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) Stream(stateful): final segment send failed: {e_send}")

                logger.info(
                    f"({session_id}) Stream(stateful): processed {total_engine_chunks} engine chunks "
                    f"in {engine.asr_time_s:.2f}s ASR time, emitted {len(sent_segments_pcm)} segments."
                )
            finally:
                accumulated_asr_processing_time_s += engine.asr_time_s
                engine.reset()
            return

        consumer_batch_size_cap = client_config["batch_size"]
        total_asr_chunks_processed_by_consumer = 0
        
        while True:
            batch_audio_tensors: List[torch.Tensor] = []
            batch_offsets_s: List[float] = []
            first_item_in_batch_is_sentinel = False

            # Try to fill a batch up to consumer_batch_size_cap
            for _ in range(consumer_batch_size_cap):
                item: Optional[Tuple[torch.Tensor, float]] = None
                try:
                    # Determine timeout for queue.get()
                    # If batch is partially full, use short timeout to quickly process it.
                    # If producer is done and queue is empty, short timeout to exit soon.
                    # Otherwise, wait longer if batch is empty and producer is active.
                    timeout_val = None # Default: wait indefinitely if producer is active and batch empty
                    if batch_audio_tensors: # Batch is partially filled
                        timeout_val = 0.02 # Short timeout to process what we have
                    elif producer_done_event.is_set(): # Producer is done
                        timeout_val = 0.1 # Short timeout to quickly check for remaining items or sentinel

                    if producer_done_event.is_set() and chunk_queue.empty() and not batch_audio_tensors:
                        # Optimization: if producer is done, queue is empty, and batch is empty,
                        # try non-blocking get to fetch potential sentinel quickly.
                        try:
                            item = chunk_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            # This means queue is truly empty and sentinel might have been processed or is next.
                            # The outer loop's break condition will handle exit.
                            break 
                    else:
                        item = await asyncio.wait_for(chunk_queue.get(), timeout=timeout_val)
                
                except (asyncio.TimeoutError, asyncio.QueueEmpty):
                    # Timeout or queue empty means no more items for this batching iteration.
                    # Break inner loop to process current batch (if any).
                    break 
                
                if item is None: # Sentinel received
                    chunk_queue.task_done() # Acknowledge sentinel
                    # If this sentinel is the first thing we got for this batch,
                    # it means no more audio data will come.
                    if not batch_audio_tensors:
                        first_item_in_batch_is_sentinel = True
                    # Else, if batch_audio_tensors is not empty, the sentinel was picked up after some data.
                    # We'll process the current batch, then the outer loop will break.
                    break # Exit inner loop to process current batch (if any) before stopping.
                
                # Valid audio item received
                audio_tensor, offset_s = item
                batch_audio_tensors.append(audio_tensor)
                batch_offsets_s.append(offset_s)
                chunk_queue.task_done() # Acknowledge item

                # If producer is done and queue is now empty, break to process current batch.
                if producer_done_event.is_set() and chunk_queue.empty():
                    break
            
            if not batch_audio_tensors:
                # No data collected for this batch.
                # If it's because we got a sentinel as the first item, or if producer is done and queue is drained,
                # then it's time to exit the consumer.
                if first_item_in_batch_is_sentinel or (producer_done_event.is_set() and chunk_queue.empty()):
                    logger.info(f"({session_id}) Stream Consumer: No more audio chunks to process. Exiting.")
                    break 
                else:
                    # No data, but producer might still be working, or sentinel not yet received. Continue waiting.
                    continue 

            # We have a batch of audio tensors to process
            total_asr_chunks_processed_by_consumer += len(batch_audio_tensors)
            logger.info(f"({session_id}) Stream Consumer: Processing ASR batch of {len(batch_audio_tensors)} audio chunks. "
                        f"(Total ASR chunks processed by consumer so far: {total_asr_chunks_processed_by_consumer})")
            
            hypotheses_list, asr_call_duration_s = await _perform_asr_transcription(
                asr_model_instance=asr_model, # type: ignore # Checked at endpoint start
                audio_input_list=batch_audio_tensors,
                batch_size_for_transcribe_call=len(batch_audio_tensors), # Process the whole collected batch
                num_asr_workers=NUM_WORKERS,
                request_id=f"WS-Stream-{session_id}-b{total_asr_chunks_processed_by_consumer}"
            )
            accumulated_asr_processing_time_s += asr_call_duration_s
            
            # Clear CUDA cache periodically if using CUDA to manage memory
            if processing_device == "cuda" and torch.cuda.is_available():
                await asyncio.to_thread(torch.cuda.empty_cache)

            if hypotheses_list:
                segments_from_batch = _process_hypotheses_to_segments(
                    hypotheses_list, batch_offsets_s, f"{session_id}-segproc"
                )
                logger.info(f"({session_id}) Stream Consumer: Generated {len(segments_from_batch)} segments from "
                            f"{len(hypotheses_list)} hypotheses for the current batch.")
                
                if segments_from_batch and websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "segments_batch", "segments": segments_from_batch})
                        sent_segments_pcm.extend(segments_from_batch) # Track all sent segments
                        logger.info(f"({session_id}) Stream Consumer: Successfully sent {len(segments_from_batch)} segments to client. "
                                    f"(Total segments sent this session: {len(sent_segments_pcm)})")
                        await asyncio.sleep(0.001) # Tiny sleep to allow other tasks (e.g., network I/O)
                    except Exception as e_send:
                        logger.warning(f"({session_id}) Stream Consumer: Failed to send segments batch to client: {e_send}", exc_info=True)
                        # If send fails, we still add to sent_segments_pcm for the final transcription if connection is restored
                        # or if we want to log what *would* have been sent.
                        sent_segments_pcm.extend(segments_from_batch)
                elif segments_from_batch: # Segments generated but WebSocket no longer connected
                    sent_segments_pcm.extend(segments_from_batch)
                    logger.info(f"({session_id}) Stream Consumer: Generated {len(segments_from_batch)} segments, "
                                f"but WebSocket is disconnected. Segments stored for potential final summary.")
            else:
                logger.info(f"({session_id}) Stream Consumer: No hypotheses generated from ASR for the current batch, thus no segments to send.")

            # If the sentinel was received and processed along with the last batch of data,
            # now is the time to exit the consumer loop.
            if first_item_in_batch_is_sentinel and not batch_audio_tensors : # Should have been handled by the top break
                 pass # This case should be caught by the break at the start of the loop
            elif item is None and not batch_audio_tensors : # Also should be caught
                 pass


        logger.info(f"({session_id}) Stream Consumer: Finished processing. "
                    f"Total ASR chunks processed: {total_asr_chunks_processed_by_consumer}. "
                    f"Total segments generated and queued/sent: {len(sent_segments_pcm)}.")

    # Main execution block for handle_streaming_pcm
    # The stateful engine needs (greedy, preserve_alignments=True) — swap the
    # decoder before launching producer/consumer, revert in finally. This is
    # a no-op for the legacy path.
    saved_decoder_state: Optional[dict] = None
    try:
        if use_stateful_progressive and asr_model is not None:
            saved_decoder_state = await _swap_decoder_to_stateful_tdt()
            logger.info(f"({session_id}) Stream: decoder swapped to (greedy, preserve_alignments=True) for stateful engine.")
        logger.info(f"({session_id}) Streaming Pipeline (ffmpeg-based): Starting producer and consumer tasks.")
        # Run producer and consumer concurrently
        await asyncio.gather(producer(), consumer())
        logger.info(f"({session_id}) Streaming Pipeline (ffmpeg-based): Producer and consumer tasks have completed.")

        # Consolidate all transcribed text from segments
        # Note: sent_segments_pcm might not be perfectly ordered if deduplication is added later for streaming.
        # For now, assume they are appended in rough chronological order.
        final_transcribed_text_pcm = " ".join(s["text"] for s in sent_segments_pcm).strip()

        # progressive_refinement: re-transcribe the accumulated PCM as a single
        # FULL pass before emitting final_transcription. Cheaper than re-running
        # the whole pipeline and yields offline-quality segments instead of the
        # independent-chunk approximations that were streamed live.
        refined_segments: Optional[List[dict]] = None
        refined_text: Optional[str] = None
        refinement_asr_t: float = 0.0
        if refinement_pcm_accumulator is not None and len(refinement_pcm_accumulator) > 0:
            full_audio_duration_s = (len(refinement_pcm_accumulator) // (target_pcm_bytes_per_sample)) / target_pcm_sample_rate
            if full_audio_duration_s > MAX_FULL_WAVEFORM_S:
                logger.info(
                    f"({session_id}) Stream: progressive_refinement skipped — accumulated audio "
                    f"{full_audio_duration_s:.1f}s > MAX_FULL_WAVEFORM_S ({MAX_FULL_WAVEFORM_S:.0f}s)."
                )
            elif asr_model is not None:
                try:
                    full_pcm_bytes = bytes(refinement_pcm_accumulator)
                    full_tensor = await asyncio.to_thread(_create_asr_tensor_from_bytes, full_pcm_bytes)
                    hypotheses_refined, refinement_asr_t = await _perform_asr_transcription(
                        asr_model_instance=asr_model,
                        audio_input_list=[full_tensor],
                        batch_size_for_transcribe_call=1,
                        num_asr_workers=NUM_WORKERS,
                        request_id=f"WS-Stream-{session_id}-refine",
                    )
                    refined_segments = _process_hypotheses_to_segments(
                        hypotheses_refined,
                        [0.0] * (len(hypotheses_refined) if hypotheses_refined else 0),
                        f"{session_id}-refine",
                    )
                    refined_text = " ".join(s["text"] for s in refined_segments).strip()
                    logger.info(
                        f"({session_id}) Stream: progressive_refinement complete — "
                        f"{len(refined_segments)} refined segs in {refinement_asr_t:.2f}s "
                        f"over {full_audio_duration_s:.1f}s audio."
                    )
                    if websocket.application_state == WebSocketState.CONNECTED:
                        await websocket.send_json({
                            "type": "refined_transcription",
                            "segments": refined_segments,
                            "text": refined_text,
                            "transcription_time": round(refinement_asr_t, 3),
                            "audio_duration_seconds": round(full_audio_duration_s, 3),
                        })
                except Exception as e_refine:
                    logger.warning(
                        f"({session_id}) Stream: progressive_refinement failed ({e_refine!r}); "
                        f"final_transcription will use the live-streamed segments.",
                        exc_info=True,
                    )
                    refined_segments = None
                    refined_text = None

        if websocket.application_state == WebSocketState.CONNECTED:
            logger.info(f"({session_id}) Streaming: Sending final_transcription message to client. "
                        f"Total ASR input duration (from ffmpeg PCM): {total_duration_processed_seconds_for_asr:.2f}s")

            use_refined = refined_segments is not None and refined_text is not None
            final_segments_for_payload = refined_segments if use_refined else sent_segments_pcm
            final_text_for_payload = refined_text if use_refined else final_transcribed_text_pcm
            final_total_asr_t = accumulated_asr_processing_time_s + refinement_asr_t

            final_message_payload = {
                "type": "final_transcription",
                "text": final_text_for_payload,
                "language": "en", # Assuming English, could be made configurable
                "transcription_time": round(final_total_asr_t, 3),
                "total_segments": len(final_segments_for_payload),
                "final_duration_processed_seconds": round(total_duration_processed_seconds_for_asr, 3),
                "csv_content": generate_csv_content(final_segments_for_payload),
                "srt_content": generate_srt_content(final_segments_for_payload),
                "streaming_mode": client_config.get("format", "unknown"), # Reflect client-declared format
                "refinement_applied": use_refined,
            }
            await websocket.send_json(final_message_payload)
            logger.info(
                f"({session_id}) Streaming: Final transcription message sent "
                f"(refinement_applied={use_refined})."
            )
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
        if saved_decoder_state is not None:
            try:
                await _restore_decoder(saved_decoder_state)
                logger.info(f"({session_id}) Stream: decoder restored after stateful session.")
            except Exception as e_restore:
                logger.warning(f"({session_id}) Stream: decoder restore failed: {e_restore}", exc_info=True)


def create_audio_chunks(
    waveform: torch.Tensor,
    sample_rate: int = MODEL_SAMPLE_RATE,
    chunk_len_s: float = TRANSCRIBE_CHUNK_LEN,
    overlap_s: float = TRANSCRIBE_OVERLAP
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Splits a long audio waveform tensor into smaller, overlapping chunks.

    This is typically used for processing very long audio files that cannot be
    fed to the ASR model in one go.

    Args:
        waveform: A 1D PyTorch tensor containing the audio data.
        sample_rate: The sample rate of the input waveform (in Hz).
        chunk_len_s: The desired length of each chunk in seconds.
        overlap_s: The desired overlap between consecutive chunks in seconds.

    Returns:
        A tuple containing:
            - chunks (List[torch.Tensor]): A list of 1D audio tensors, each representing a chunk.
            - offsets (List[float]): A list of floats, where each float is the starting
                                     time (in seconds) of the corresponding chunk in the
                                     original waveform.
    Raises:
        AssertionError: If the input waveform is not 1D.
        ValueError: If `overlap_s` is not less than `chunk_len_s` (i.e., stride <= 0).
    """
    assert waveform.ndim == 1, "Input waveform must be a 1D tensor."
    
    total_duration_s = waveform.shape[0] / sample_rate
    
    stride_s = chunk_len_s - overlap_s
    if stride_s <= 0:
        raise ValueError("Overlap duration must be less than chunk length duration for a positive stride.")

    chunks: List[torch.Tensor] = []
    offsets_s: List[float] = []
    
    current_offset_s = 0.0
    while current_offset_s < total_duration_s:
        start_sample_idx = int(current_offset_s * sample_rate)
        # Ensure end_sample_idx does not exceed waveform length
        end_sample_idx = int(min(total_duration_s, current_offset_s + chunk_len_s) * sample_rate)

        # If the calculated chunk is empty or too small (e.g., due to rounding at the very end)
        if end_sample_idx <= start_sample_idx:
            # If advancing by stride would still be within the audio, continue to next possible chunk
            if current_offset_s + stride_s < total_duration_s:
                current_offset_s += stride_s
                continue
            else: # No more meaningful chunks can be formed
                break
        
        chunk_tensor = waveform[start_sample_idx:end_sample_idx].clone() # Clone to avoid views if tensor is modified
        
        if chunk_tensor.numel() > 0: # Ensure the chunk is not empty
            chunks.append(chunk_tensor)
            offsets_s.append(round(current_offset_s, 3)) # Store offset with precision

        # If this chunk reaches or exceeds the end of the waveform, stop
        if end_sample_idx >= waveform.shape[0]:
            break 
            
        current_offset_s += stride_s

        # Safety break for very tiny residual audio that might cause near-infinite loops if stride is small
        # and remaining audio is smaller than a full chunk but offsets don't align perfectly.
        # If we have chunks, and the last chunk's intended end was before total_duration,
        # but current_offset_s is now >= total_duration, it implies we might be stuck.
        if current_offset_s >= total_duration_s:
            if chunks and (offsets_s[-1] + chunk_len_s < total_duration_s - 0.01): # Last chunk didn't cover end
                 # This condition tries to catch if the last segment was small and we are past the audio
                 # but there was still a tiny bit left. This usually means the last chunk should be the end.
                 pass # Allow one more pass if a tiny sliver is left. The outer while will catch it.
            # Avoid infinite loop if offsets get stuck due to floating point issues on tiny final segments
            if len(chunks) > 1 and offsets_s[-1] == offsets_s[-2]:
                logger.warning(f"(create_audio_chunks) Detected potential stuck loop with duplicate offsets. Breaking.")
                break
    
    logger.debug(f"(create_audio_chunks) Created {len(chunks)} chunks from audio of {total_duration_s:.2f}s.")
    return chunks, offsets_s


async def _transcribe_chunked_waveform(
    waveform: torch.Tensor,
    client_config: dict,
    request_id: str = "chunked",
    on_batch_segments: Optional[Callable[[List[dict], int], Awaitable[bool]]] = None,
) -> Tuple[List[dict], float]:
    """
    Sliding-window chunked transcription of an in-memory waveform.

    Mirrors the pattern used inside `websocket_transcribe_endpoint_full_file_upload`:
    chunk via `create_audio_chunks`, batch through `_perform_asr_transcription`,
    convert to segments, dedup. Used by REST when `?strategy=chunked` (or auto
    resolves there for long audio). Phase 6 will swap this for stateful streaming
    via `BatchedFrameASRTDT`.

    Args:
        on_batch_segments: optional async callback invoked with
            (batch_segments, batch_num) after each ASR batch produces segments.
            Return True to continue, False to abort the loop early (used by WS
            handlers to emit `segments_batch` messages and bail on disconnect).

    Returns:
        Tuple of (deduplicated_segments, total_asr_processing_time_seconds).
        Empty list and 0.0 if the waveform produces no chunks.
    """
    chunk_len_s = client_config["chunk_length"]
    chunk_overlap_s = client_config["chunk_overlap"]
    batch_size = client_config["batch_size"]

    chunks, offsets = create_audio_chunks(
        waveform=waveform,
        sample_rate=MODEL_SAMPLE_RATE,
        chunk_len_s=chunk_len_s,
        overlap_s=chunk_overlap_s,
    )
    if not chunks:
        logger.warning(f"({request_id}) Chunked: create_audio_chunks returned no chunks.")
        return [], 0.0

    logger.info(
        f"({request_id}) Chunked: {len(chunks)} chunks "
        f"(len={chunk_len_s}s overlap={chunk_overlap_s}s), batching by {batch_size}."
    )

    all_raw_segments: List[dict] = []
    total_asr_time = 0.0

    for batch_num, batch_start in enumerate(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[batch_start:batch_start + batch_size]
        batch_offsets = offsets[batch_start:batch_start + batch_size]
        if not batch_chunks:
            continue

        hyps, asr_dur = await _perform_asr_transcription(
            asr_model_instance=asr_model,
            audio_input_list=batch_chunks,
            batch_size_for_transcribe_call=len(batch_chunks),
            num_asr_workers=NUM_WORKERS,
            request_id=f"{request_id}-b{batch_num}",
        )
        total_asr_time += asr_dur

        if hyps:
            batch_segs = _process_hypotheses_to_segments(
                hyps, batch_offsets, f"{request_id}-segproc_b{batch_num}",
            )
            if batch_segs:
                batch_segs.sort(key=lambda s: s.get("start", float("inf")))
                all_raw_segments.extend(batch_segs)
                if on_batch_segments is not None:
                    keep_going = await on_batch_segments(batch_segs, batch_num)
                    if not keep_going:
                        logger.info(
                            f"({request_id}) Chunked: callback signaled abort at batch {batch_num + 1}/"
                            f"{(len(chunks) + batch_size - 1) // batch_size}."
                        )
                        break

    deduped = _deduplicate_segments(
        all_raw_segments,
        chunk_overlap_s / 2.0,
    )
    return deduped, total_asr_time


def _extract_tdt_token_times(
    frame_asr,
    delay: int,
    tokens_per_chunk: int,
    chunk_len_s: float,
    total_buffer_s: float,
    model_stride_s: float,
) -> List[Tuple[int, float]]:
    """
    Mirror BatchedFrameASRTDT.transcribe()'s middle-token merge to recover
    (token_id, audio_time_s) pairs for batch slot 0.

    The class's own transcribe() discards alignment frame indices via
    _alignment_decoder. We walk the same per-chunk slicing rules but keep
    `(frame_idx_in_alignment, token_id)` tuples and convert the alignment
    frame to global audio time via the buffer geometry:

        After chunk `a_idx` is fed, the FIFO buffer ends at audio time
        `(a_idx + 1) * chunk_len_s` and is `total_buffer_s` wide. Encoder
        frame `f` of that chunk's alignment therefore corresponds to
        audio time `(a_idx + 1) * chunk_len_s - total_buffer_s + f * stride`.

    The "middle token" slice picks frames in the centre of the buffer, so
    using the absolute frame index naturally produces the right time even
    though the slice does not start at the chunk's nominal beginning.
    """
    all_alignments = frame_asr.all_alignments[0]
    signal_end_idx = frame_asr.frame_bufferer.signal_end_index[0]
    blank_id = frame_asr.blank_id
    tdt_search_boundary = getattr(frame_asr, "tdt_search_boundary", 4)

    out: List[Tuple[int, float]] = []
    unmerged_ids: List[int] = []

    def _walk_with_times(slice_align, slice_start_idx):
        toks_with_t: List[Tuple[int, int]] = []
        for fi, frame in enumerate(slice_align):
            global_fi = slice_start_idx + fi
            for u in range(len(frame)):
                _, tid = frame[u]
                tid = int(tid)
                if tid != blank_id:
                    toks_with_t.append((global_fi, tid))
        return toks_with_t

    for a_idx, alignment in enumerate(all_alignments):
        if delay == len(alignment):
            offset = 0
        else:
            offset = 1
        base_start = len(alignment) - offset - delay
        base_end = base_start + tokens_per_chunk
        long_start = base_start - tdt_search_boundary
        long_end = base_end

        # Decode "longer" slice (for boundary-search) and the base slice with frame indices.
        longer_with_t = _walk_with_times(alignment[long_start:long_end], long_start)
        base_with_t = _walk_with_times(alignment[base_start:base_end], base_start)

        if not longer_with_t or (signal_end_idx is not None and a_idx >= signal_end_idx):
            continue

        if a_idx == 0 or len(unmerged_ids) == 0:
            use_with_t = base_with_t
        elif len(unmerged_ids) > 0 and len(longer_with_t) > 1:
            id_to_match = unmerged_ids[-1]
            longer_ids_only = [t[1] for t in longer_with_t]
            start_idx = min(len(longer_ids_only) - len(base_with_t), len(longer_ids_only) - 1)
            use_with_t = base_with_t  # fallback when no match
            for i in range(start_idx, -1, -1):
                if longer_ids_only[i] == id_to_match:
                    use_with_t = longer_with_t[i + 1:]
                    break
        else:
            use_with_t = base_with_t

        buffer_end_audio_s = (a_idx + 1) * chunk_len_s
        for frame_idx_in_align, tid in use_with_t:
            time_s = buffer_end_audio_s - total_buffer_s + frame_idx_in_align * model_stride_s
            if time_s < 0.0:
                time_s = 0.0  # clamp pre-roll pad
            out.append((tid, float(time_s)))
            unmerged_ids.append(tid)

    return out


def _segments_from_token_times(
    tokens_with_times: List[Tuple[int, float]],
    tokenizer,
    duration_s: float,
) -> List[dict]:
    """
    Group (token_id, time_s) pairs into sentence-bounded segments.

    Splits on subword tokens whose decoded text ends in '.', '!' or '?'.
    Uses actual per-token timestamps for start/end, not proportional
    distribution.
    """
    if not tokens_with_times:
        return []

    segments: List[dict] = []
    cur_ids: List[int] = []
    cur_start: Optional[float] = None
    cur_last_t: float = 0.0
    seg_id = 0

    def flush(end_t: float):
        nonlocal cur_ids, cur_start, seg_id
        if not cur_ids:
            return
        text = tokenizer.ids_to_text(cur_ids).strip()
        if text:
            start_clamped = max(0.0, cur_start if cur_start is not None else 0.0)
            end_clamped = min(duration_s, max(start_clamped, end_t))
            segments.append({
                "start": round(start_clamped, 3),
                "end": round(end_clamped, 3),
                "text": text,
                "id": seg_id,
            })
            seg_id += 1
        cur_ids = []
        cur_start = None

    for tid, t in tokens_with_times:
        if cur_start is None:
            cur_start = t
        cur_ids.append(tid)
        cur_last_t = t
        try:
            tok = tokenizer.ids_to_tokens([tid])[0]
        except Exception:
            tok = ""
        if tok and tok[-1] in ".!?":
            flush(t)

    if cur_ids:
        flush(cur_last_t)

    return segments


class StreamingTdtEngine:
    """
    Chunk-by-chunk driver for BatchedFrameASRTDT — the streaming counterpart
    of the offline _stateful_chunked_sync helper.

    Owns a BatchedFrameASRTDT instance plus an internal PCM accumulator. As
    callers `feed()` raw 16-bit PCM bytes, the engine extracts features for
    each complete `chunk_len_s` slice, advances the FIFO buffer one slot, and
    runs encoder+decoder over the current buffer state via `_get_batch_preds`.
    Each step appends one entry to `frame_asr.all_alignments[0]`.

    `pop_committed_segments()` re-walks the accumulated alignments via the
    same middle-token merge used in the offline path (`_extract_tdt_token_times`),
    diffs against the previously-emitted token tail, and emits any newly
    completed sentence-bounded segments. In-progress (no terminal '.!?') token
    runs are held in `_sentence_buffer_*` until the sentence closes.

    The engine assumes the model's decoding strategy has already been switched
    to (`greedy`, `preserve_alignments=True`, `fused_batch_size=-1`) by the
    caller — same contract as `_transcribe_chunked_stateful`.

    Not async-safe — call from a single coroutine under `model_access_lock`.
    """

    def __init__(
        self,
        asr_model_instance,
        chunk_len_s: float,
        total_buffer_s: float,
        model_stride_s: float,
        sample_rate: int,
        request_id: str,
    ):
        self.asr_model = asr_model_instance
        self.chunk_len_s = chunk_len_s
        self.total_buffer_s = total_buffer_s
        self.model_stride_s = model_stride_s
        self.sample_rate = sample_rate
        self.request_id = request_id

        self.frame_asr = BatchedFrameASRTDT(
            asr_model=asr_model_instance,
            frame_len=chunk_len_s,
            total_buffer=total_buffer_s,
            batch_size=1,
            stateful_decoding=True,
        )
        self.tokens_per_chunk = math.ceil(chunk_len_s / model_stride_s)
        self.mid_delay = math.ceil((chunk_len_s + (total_buffer_s - chunk_len_s) / 2) / model_stride_s)
        self.samples_per_chunk = int(round(chunk_len_s * sample_rate))

        self._pcm_buffer_f32 = np.zeros(0, dtype=np.float32)
        self._chunks_processed = 0
        self._eof_flushed = False
        self._asr_time_s = 0.0

        # Sentence-emission state — carries across pop_committed_segments() calls so
        # tokens emitted in chunk N can be flushed when the sentence completes in chunk N+k.
        self._sentence_buffer_ids: List[int] = []
        self._sentence_buffer_start: Optional[float] = None
        self._sentence_buffer_last_t: float = 0.0
        self._next_seg_id: int = 0
        self._tokens_emitted_through: int = 0

        # Cached for autocast around inference calls
        self._model_dtype = next(asr_model_instance.parameters()).dtype
        self._device_type = asr_model_instance.device.type

    def feed(self, pcm_bytes: bytes) -> None:
        """Append s16le PCM and process any complete chunks immediately."""
        if pcm_bytes:
            arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self._pcm_buffer_f32 = np.concatenate([self._pcm_buffer_f32, arr])
        self._drain_complete_chunks()

    def feed_float32(self, samples_f32: np.ndarray) -> None:
        """Append already-converted float32 [-1, 1] samples and process any complete chunks."""
        if samples_f32 is not None and samples_f32.size > 0:
            self._pcm_buffer_f32 = np.concatenate(
                [self._pcm_buffer_f32, samples_f32.astype(np.float32, copy=False)]
            )
        self._drain_complete_chunks()

    def _drain_complete_chunks(self) -> None:
        while len(self._pcm_buffer_f32) >= self.samples_per_chunk:
            chunk = self._pcm_buffer_f32[:self.samples_per_chunk]
            self._pcm_buffer_f32 = self._pcm_buffer_f32[self.samples_per_chunk:]
            self._step_one_chunk(chunk)

    def flush(self) -> None:
        """
        Signal EOF: pad any partial PCM, then push `mid_delay * stride` of
        trailing silence so the middle-token merge can commit the tail.
        """
        if self._eof_flushed:
            return
        if len(self._pcm_buffer_f32) > 0:
            pad_n = self.samples_per_chunk - len(self._pcm_buffer_f32)
            self._pcm_buffer_f32 = np.pad(self._pcm_buffer_f32, (0, pad_n))
            self._step_one_chunk(self._pcm_buffer_f32[:self.samples_per_chunk])
            self._pcm_buffer_f32 = np.zeros(0, dtype=np.float32)

        tail_samples = int(self.mid_delay * self.model_stride_s * self.sample_rate)
        if tail_samples > 0:
            self._pcm_buffer_f32 = np.zeros(tail_samples, dtype=np.float32)
            while len(self._pcm_buffer_f32) >= self.samples_per_chunk:
                chunk = self._pcm_buffer_f32[:self.samples_per_chunk]
                self._pcm_buffer_f32 = self._pcm_buffer_f32[self.samples_per_chunk:]
                self._step_one_chunk(chunk)
            if len(self._pcm_buffer_f32) > 0:
                pad_n = self.samples_per_chunk - len(self._pcm_buffer_f32)
                self._pcm_buffer_f32 = np.pad(self._pcm_buffer_f32, (0, pad_n))
                self._step_one_chunk(self._pcm_buffer_f32[:self.samples_per_chunk])
                self._pcm_buffer_f32 = np.zeros(0, dtype=np.float32)

        self._eof_flushed = True

    def _step_one_chunk(self, chunk_f32: np.ndarray) -> None:
        """Extract features for one chunk, slide the FIFO buffer, run encoder+decoder."""
        t0 = time.time()
        device = self.asr_model.device
        audio = torch.from_numpy(chunk_f32.copy()).unsqueeze(0).to(device)
        length = torch.tensor([chunk_f32.shape[0]], device=device)
        with torch.inference_mode(), torch.amp.autocast(self._device_type, dtype=self._model_dtype):
            features, _ = self.frame_asr.raw_preprocessor(input_signal=audio, length=length)
        features = features.squeeze(0).detach().cpu().numpy()

        # Per-chunk preprocessing produces one extra time frame compared to
        # `_feature_frame_len` because of the windowing edge — AudioFeatureIterator
        # avoids this by preprocessing the full waveform once and slicing.
        # Trim/pad to the bufferer's expected `n_frame_len` so the FIFO insert
        # broadcasts cleanly into the [batch, n_feat, total_buffer_len] buffer.
        n_frame_len = self.frame_asr.frame_bufferer.n_frame_len
        if features.shape[1] > n_frame_len:
            features = features[:, :n_frame_len]
        elif features.shape[1] < n_frame_len:
            pad = np.zeros((features.shape[0], n_frame_len - features.shape[1]), dtype=features.dtype)
            features = np.concatenate([features, pad], axis=1)

        # FIFO slide + insert at the right end; returns a list-of-list-of-buffers
        # because BatchedFrameASR expects the data_layer interface to consume it.
        frame_buffers = self.frame_asr.frame_bufferer.get_frame_buffers([features])
        self.frame_asr.data_layer[0].set_signal(frame_buffers[0][:])
        self.frame_asr.frame_bufferer.signal_end[0] = False
        with torch.inference_mode(), torch.amp.autocast(self._device_type, dtype=self._model_dtype):
            self.frame_asr._get_batch_preds()
        self._chunks_processed += 1
        self._asr_time_s += (time.time() - t0)

    def pop_committed_segments(self) -> List[dict]:
        """
        Walk `frame_asr.all_alignments[0]` through the middle-token merge,
        emit any newly committed sentences (those whose terminal '.!?' just
        landed). In-progress sentences stay in the buffer for the next call.
        """
        if self._chunks_processed == 0:
            return []

        try:
            all_tokens = _extract_tdt_token_times(
                frame_asr=self.frame_asr,
                delay=self.mid_delay,
                tokens_per_chunk=self.tokens_per_chunk,
                chunk_len_s=self.chunk_len_s,
                total_buffer_s=self.total_buffer_s,
                model_stride_s=self.model_stride_s,
            )
        except Exception as e:
            logger.warning(
                f"({self.request_id}) StreamingTdtEngine: token extraction failed ({e!r}).",
                exc_info=True,
            )
            return []

        new_tokens = all_tokens[self._tokens_emitted_through:]
        self._tokens_emitted_through = len(all_tokens)

        return self._consume_tokens_into_segments(new_tokens, flush_partial=False)

    def pop_final_segments(self) -> List[dict]:
        """Flush trailing in-progress sentence after EOF."""
        committed = self.pop_committed_segments()
        if self._sentence_buffer_ids:
            text = self.asr_model.tokenizer.ids_to_text(self._sentence_buffer_ids).strip()
            if text:
                committed.append({
                    "start": round(max(0.0, self._sentence_buffer_start or 0.0), 3),
                    "end": round(self._sentence_buffer_last_t, 3),
                    "text": text,
                    "id": self._next_seg_id,
                })
                self._next_seg_id += 1
            self._sentence_buffer_ids = []
            self._sentence_buffer_start = None
        return committed

    def _consume_tokens_into_segments(
        self, new_tokens: List[Tuple[int, float]], flush_partial: bool
    ) -> List[dict]:
        segments: List[dict] = []
        tokenizer = self.asr_model.tokenizer
        for tid, t in new_tokens:
            if self._sentence_buffer_start is None:
                self._sentence_buffer_start = t
            self._sentence_buffer_ids.append(tid)
            self._sentence_buffer_last_t = t
            try:
                tok = tokenizer.ids_to_tokens([tid])[0]
            except Exception:
                tok = ""
            if tok and tok[-1] in ".!?":
                text = tokenizer.ids_to_text(self._sentence_buffer_ids).strip()
                if text:
                    segments.append({
                        "start": round(max(0.0, self._sentence_buffer_start), 3),
                        "end": round(t, 3),
                        "text": text,
                        "id": self._next_seg_id,
                    })
                    self._next_seg_id += 1
                self._sentence_buffer_ids = []
                self._sentence_buffer_start = None
        if flush_partial and self._sentence_buffer_ids:
            text = tokenizer.ids_to_text(self._sentence_buffer_ids).strip()
            if text:
                segments.append({
                    "start": round(max(0.0, self._sentence_buffer_start or 0.0), 3),
                    "end": round(self._sentence_buffer_last_t, 3),
                    "text": text,
                    "id": self._next_seg_id,
                })
                self._next_seg_id += 1
            self._sentence_buffer_ids = []
            self._sentence_buffer_start = None
        return segments

    @property
    def asr_time_s(self) -> float:
        return self._asr_time_s

    def reset(self) -> None:
        try:
            self.frame_asr.reset()
        except Exception:
            pass


def _stateful_chunked_sync(
    waveform_np: np.ndarray,
    chunk_len_s: float,
    total_buffer_s: float,
    model_stride_s: float,
    request_id: str,
) -> Tuple[str, float, List[Tuple[int, float]]]:
    """
    Synchronous core of the BatchedFrameASRTDT stateful chunked engine.

    Designed to be called via asyncio.to_thread from `_transcribe_chunked_stateful`.
    Returns (joined_text, asr_time_s, token_times) where token_times is the
    `(token_id, audio_time_s)` stream produced by mirroring the middle-token
    merge over `frame_asr.all_alignments`. Runs at batch_size=1.
    """
    tokens_per_chunk = math.ceil(chunk_len_s / model_stride_s)
    mid_delay = math.ceil((chunk_len_s + (total_buffer_s - chunk_len_s) / 2) / model_stride_s)

    frame_asr = BatchedFrameASRTDT(
        asr_model=asr_model,
        frame_len=chunk_len_s,
        total_buffer=total_buffer_s,
        batch_size=1,
        stateful_decoding=True,
    )
    try:
        # Pad with `mid_delay * stride * sample_rate` trailing zeros (matches NeMo's
        # `read_audio_file` preprocessing — the "middle token" algorithm needs the
        # tail context to finalize the last chunk).
        sr = asr_model._cfg.sample_rate
        pad_samples = int(mid_delay * model_stride_s * sr)
        if pad_samples > 0:
            samples = np.pad(waveform_np, (0, pad_samples))
        else:
            samples = waveform_np

        # Build the in-memory frame reader and register it as the only batch slot.
        frame_reader = AudioFeatureIterator(
            samples=samples,
            frame_len=chunk_len_s,
            preprocessor=frame_asr.raw_preprocessor,
            device=asr_model.device,
        )
        frame_asr.set_frame_reader(frame_reader, 0)

        t0 = time.time()
        # transcribe() runs infer_logits() (encoder + decoder forward over all
        # buffered chunks) and emits batch_size string hypotheses with the
        # middle-token TDT merge applied across chunks.
        # autocast is required: the model is pinned to bf16 but the audio
        # samples are float32 (NeMo's preprocessor requirement). Without
        # autocast the first matmul fails: "Input type (float) and bias
        # type (c10::BFloat16) should be the same".
        model_dtype = next(asr_model.parameters()).dtype
        device_type = asr_model.device.type
        with torch.inference_mode(), torch.amp.autocast(device_type, dtype=model_dtype):
            outputs = frame_asr.transcribe(tokens_per_chunk=tokens_per_chunk, delay=mid_delay)
        asr_time = time.time() - t0
        text = outputs[0] if outputs else ""

        token_times: List[Tuple[int, float]] = []
        try:
            token_times = _extract_tdt_token_times(
                frame_asr=frame_asr,
                delay=mid_delay,
                tokens_per_chunk=tokens_per_chunk,
                chunk_len_s=chunk_len_s,
                total_buffer_s=total_buffer_s,
                model_stride_s=model_stride_s,
            )
        except Exception as e_tt:
            logger.warning(
                f"({request_id}) Stateful: per-token timestamp extraction failed "
                f"({e_tt!r}); falling back to text-based approximation.",
                exc_info=True,
            )
            token_times = []

        logger.info(
            f"({request_id}) Stateful: chunk={chunk_len_s}s buf={total_buffer_s}s "
            f"stride={model_stride_s:.4f}s tpc={tokens_per_chunk} delay={mid_delay} "
            f"→ {len(text)} chars in {asr_time:.2f}s (token_times={len(token_times)})"
        )
        return text, asr_time, token_times
    finally:
        # Free the per-session decoder state; the FrameBatchASR allocates
        # per-batch buffers that we don't want lingering between requests.
        try:
            frame_asr.reset()
        except Exception:
            pass


async def _swap_decoder_to_stateful_tdt() -> dict:
    """
    Transiently switch the global decoder to (greedy, preserve_alignments=True,
    fused_batch_size=-1) — the configuration BatchedFrameASRTDT needs. Returns
    a `saved` dict that `_restore_decoder` consumes to put things back.

    Caller MUST be holding `model_access_lock`.
    """
    from omegaconf import open_dict
    decoding_cfg = asr_model.cfg.decoding
    saved = {
        "strategy": decoding_cfg.strategy,
        "preserve_alignments": decoding_cfg.get("preserve_alignments", False),
        "fused_batch_size": decoding_cfg.get("fused_batch_size", -1),
    }
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = "greedy"
        decoding_cfg.preserve_alignments = True
        decoding_cfg.fused_batch_size = -1
    await asyncio.to_thread(asr_model.change_decoding_strategy, decoding_cfg, verbose=False)
    return saved


async def _restore_decoder(saved: dict) -> None:
    """Put the decoder strategy back to what `_swap_decoder_to_stateful_tdt` captured."""
    from omegaconf import open_dict
    decoding_cfg = asr_model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.strategy = saved["strategy"]
        decoding_cfg.preserve_alignments = saved["preserve_alignments"]
        decoding_cfg.fused_batch_size = saved["fused_batch_size"]
    await asyncio.to_thread(asr_model.change_decoding_strategy, decoding_cfg, verbose=False)


async def _transcribe_chunked_stateful(
    waveform: torch.Tensor,
    audio_duration_s: float,
    client_config: dict,
    request_id: str = "stateful-chunked",
) -> Tuple[List[dict], float]:
    """
    Stateful chunked transcription via NeMo's BatchedFrameASRTDT.

    Carries decoder state across chunks; the middle-token TDT merge stitches
    chunk outputs into one coherent transcript without boundary duplicates or
    drops. Emits sentence-bounded segments with real per-token timestamps
    recovered by walking `frame_asr.all_alignments` (see
    `_extract_tdt_token_times`). Falls back to proportional approximation only
    if extraction fails.

    The 10-10-5 / 10-2-2 NeMo recommendations map to:
        chunk_len_in_secs  = STREAMING_CHUNK_S
        total_buffer_in_secs = STREAMING_LEFT_CONTEXT_S + STREAMING_CHUNK_S + STREAMING_RIGHT_CONTEXT_S
    """
    if asr_model is None:
        logger.error(f"({request_id}) Stateful: asr_model is None.")
        return [], 0.0

    chunk_len = STREAMING_CHUNK_S
    total_buffer = STREAMING_LEFT_CONTEXT_S + STREAMING_CHUNK_S + STREAMING_RIGHT_CONTEXT_S

    feature_stride = asr_model._cfg.preprocessor["window_stride"]
    model_stride_in_secs = feature_stride * asr_model.encoder.subsampling_factor

    # Convert torch tensor (likely on CPU after load_and_preprocess_audio) to
    # a contiguous float32 numpy array — that's what AudioFeatureIterator expects.
    if waveform.dim() > 1:
        waveform_np = waveform.squeeze().to(dtype=torch.float32).contiguous().cpu().numpy()
    else:
        waveform_np = waveform.to(dtype=torch.float32).contiguous().cpu().numpy()

    # BatchedFrameASRTDT requires strategy=greedy + preserve_alignments=True;
    # globally pinning those would slow FULL and the legacy chunked fallback
    # ~20× (they rely on greedy_batch). Swap transiently here.
    saved = await _swap_decoder_to_stateful_tdt()
    try:
        text, asr_time, token_times = await asyncio.to_thread(
            _stateful_chunked_sync,
            waveform_np,
            chunk_len,
            total_buffer,
            model_stride_in_secs,
            request_id,
        )
    finally:
        await _restore_decoder(saved)

    if not text:
        return [], asr_time

    segments: List[dict] = []
    if token_times:
        try:
            segments = _segments_from_token_times(token_times, asr_model.tokenizer, audio_duration_s)
        except Exception as e_seg:
            logger.warning(
                f"({request_id}) Stateful: building segments from token_times "
                f"failed ({e_seg!r}); falling back to approximation.",
                exc_info=True,
            )
            segments = []

    if not segments:
        segments = _approximate_segments_from_text(text.strip(), audio_duration_s)

    return segments, asr_time


def _should_use_stateful_engine(audio_duration_s: Optional[float], request_id: str) -> bool:
    """
    Resolve which chunked engine to use for a request.

    Returns True iff USE_STATEFUL_CHUNKED is on AND the audio (if known)
    fits within STATEFUL_MAX_DURATION_S. Above the threshold (or when 0
    means disabled), we fall back to the legacy independent-chunk path —
    it batches 4 chunks per call and is ~20× faster on long audio at the
    cost of some boundary-merge quality. Logs the bypass for visibility.
    """
    if not USE_STATEFUL_CHUNKED:
        return False
    if STATEFUL_MAX_DURATION_S <= 0:
        return True  # explicit "no cap" mode
    if audio_duration_s is not None and audio_duration_s > STATEFUL_MAX_DURATION_S:
        logger.info(
            f"({request_id}) Stateful chunked engine bypassed: duration "
            f"{audio_duration_s:.1f}s > STATEFUL_MAX_DURATION_S ({STATEFUL_MAX_DURATION_S:.0f}s). "
            f"Falling back to legacy independent-chunk path."
        )
        return False
    return True


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def _approximate_segments_from_text(text: str, duration_s: float) -> List[dict]:
    """
    Split a concatenated transcript on sentence boundaries and distribute
    timestamps proportionally over the audio duration.

    Used by the stateful chunked engine (BatchedFrameASRTDT.transcribe() only
    returns a string; its internal all_timestamps tracks per-chunk token
    timing BEFORE the middle-token merge, so reconstructing per-utterance
    timestamps after the merge would require duplicating NeMo's merge logic).

    The output timestamps assume uniform speech rate — they're useful for
    SRT/CSV consumption but not frame-accurate. For exact timestamps, use
    the legacy independent-chunk path (USE_STATEFUL_CHUNKED=false).
    """
    if not text or duration_s <= 0:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    if not parts:
        return [{"start": 0.0, "end": round(duration_s, 3), "text": text.strip(), "id": 0}]

    total_chars = sum(len(p) for p in parts)
    if total_chars == 0:
        return [{"start": 0.0, "end": round(duration_s, 3), "text": text.strip(), "id": 0}]

    segments: List[dict] = []
    cursor_s = 0.0
    for i, part in enumerate(parts):
        seg_dur = duration_s * (len(part) / total_chars)
        start_s = cursor_s
        end_s = min(cursor_s + seg_dur, duration_s) if i < len(parts) - 1 else duration_s
        segments.append({
            "start": round(start_s, 3),
            "end": round(end_s, 3),
            "text": part,
            "id": i,
        })
        cursor_s = end_s
    return segments


@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(
    file: UploadFile = File(...),
    chunk_length: Optional[float] = Query(None, description="Duration of audio chunks for ASR in seconds. Uses server default if not set."),
    chunk_overlap: Optional[float] = Query(None, description="Overlap between audio chunks in seconds. Uses server default if not set."),
    batch_size: Optional[int] = Query(None, description="Batch size for ASR model processing. Uses server default if not set."),
    long_audio_threshold: Optional[float] = Query(None, description="Threshold in seconds to apply long audio model settings. Uses server default if not set."),
    strategy: Optional[str] = Query(None, description="Processing strategy: auto (default), full, chunked. Auto picks based on audio duration vs MAX_FULL_WAVEFORM_S."),
):
    """
    Handles REST API requests for audio transcription of a single uploaded file.

    The endpoint expects a file upload. It processes the entire audio file,
    performs transcription, and returns the results as JSON.

    Query Parameters (Optional):
        chunk_length: Overrides server default for ASR internal chunking (if model uses it).
                      Note: For Parakeet TDT, the model might process the whole audio,
                      but this can influence settings if `_apply_model_settings_for_session`
                      uses it for decision duration. For this REST endpoint, total audio
                      duration is used for `_apply_model_settings_for_session`.
        chunk_overlap: Overrides server default for ASR internal chunking overlap.
        batch_size: Overrides server default for batch size during NeMo's `transcribe` call.
        long_audio_threshold: Overrides server default for the threshold that determines
                              if long-audio specific model settings are applied.

    Returns:
        JSONResponse: Contains transcription text, segments, language, timing information,
                      and SRT/CSV content. Returns an error response on failure.
    """
    if not asr_model:
        logger.error("REST Request: ASR model is not available.")
        return JSONResponse(status_code=503, content={"error": "ASR model not available. Service is initializing or encountered an error."})

    request_id = base64.urlsafe_b64encode(os.urandom(6)).decode() # Short unique ID for logging
    logger.info(f"({request_id}) REST request received for file: '{file.filename}'. Content-type: {file.content_type}")

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
    final_response_content: Optional[dict] = None
    response_status_code: int = 200 # Default to 200 OK

    try:
        # Acquire lock for exclusive ASR model access
        async with model_access_lock:
            logger.debug(f"({request_id}) REST: Acquired ASR model access lock.")
            try:
                audio_bytes = await file.read()
                logger.info(f"({request_id}) REST: Read {len(audio_bytes)} bytes from upload '{file.filename}'.")

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

                    if resolved_strategy == ProcessingStrategy.PROGRESSIVE:
                        response_status_code = 400
                        final_response_content = {
                            "error": "Strategy 'progressive' requires a streaming connection. "
                                     "Use a WebSocket endpoint, or pick 'full' or 'chunked' for REST."
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

                        if resolved_strategy == ProcessingStrategy.CHUNKED:
                            if _should_use_stateful_engine(total_audio_duration_s, request_id):
                                segments, asr_processing_time_s = await _transcribe_chunked_stateful(
                                    waveform=waveform_tensor,
                                    audio_duration_s=total_audio_duration_s,
                                    client_config=client_config_rest,
                                    request_id=f"REST-{request_id}",
                                )
                            else:
                                segments, asr_processing_time_s = await _transcribe_chunked_waveform(
                                    waveform=waveform_tensor,
                                    client_config=client_config_rest,
                                    request_id=f"REST-{request_id}",
                                )
                        else:
                            # FULL: feed the entire waveform to NeMo's transcribe in one call.
                            hypotheses_list, asr_processing_time_s = await _perform_asr_transcription(
                                asr_model_instance=asr_model,
                                audio_input_list=[waveform_tensor],
                                batch_size_for_transcribe_call=client_config_rest["batch_size"],
                                num_asr_workers=NUM_WORKERS,
                                request_id=f"REST-{request_id}"
                            )
                            # Single, non-chunked audio file → offset is 0.0 for all hypotheses.
                            segments = _process_hypotheses_to_segments(
                                hypotheses_list,
                                [0.0] * (len(hypotheses_list) if hypotheses_list else 0),
                                request_id
                            )

                        full_transcribed_text = " ".join(s['text'] for s in segments).strip()
                        total_server_processing_time_s = round(time.time() - start_time_total_request_processing, 3)

                        final_response_content = {
                            "text": full_transcribed_text,
                            "segments": segments,
                            "language": "en",
                            "strategy": resolved_strategy.value,
                            "transcription_time_seconds": round(asr_processing_time_s, 3),
                            "total_request_time_server_seconds": total_server_processing_time_s,
                            "csv_content": generate_csv_content(segments),
                            "srt_content": generate_srt_content(segments),
                            "audio_duration_seconds": round(total_audio_duration_s, 3)
                        }
                        response_status_code = 200
                        logger.info(
                            f"({request_id}) REST: Transcription successful (strategy={resolved_strategy.value}). "
                            f"Duration: {total_audio_duration_s:.2f}s, ASR time: {asr_processing_time_s:.2f}s, segments: {len(segments)}."
                        )

            except Exception as e_locked_rest_processing:
                logger.error(f"({request_id}) REST: Error occurred during locked ASR processing: {e_locked_rest_processing}", exc_info=True)
                # Ensure a response is set if not already
                if final_response_content is None:
                    response_status_code = 500 # Internal Server Error
                    final_response_content = {"error": "An unexpected error occurred during transcription processing.", "detail": str(e_locked_rest_processing)}
            finally:
                logger.debug(f"({request_id}) REST: Releasing ASR model lock and reverting model state.")
                # Always revert model state at the end of the locked block
                await _revert_model_to_global_original_state(
                    long_audio_settings_were_active_for_session=long_audio_settings_applied_this_session,
                    session_processing_device=session_processing_device, # The device used in this session
                    request_id=f"{request_id}-rest_model_revert"
                )
                logger.info(f"({request_id}) REST: ASR Model state reverted after session.")
        
        # If, after releasing the lock, no specific response was prepared (should be rare)
        if final_response_content is None:
            logger.error(f"({request_id}) REST: final_response_content is None after model lock release. Setting generic error.")
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
        if hasattr(file, 'file') and file.file and not file.file.closed:
            await asyncio.to_thread(file.file.close)

        logger.info(f"({request_id}) REST request for file '{file.filename}' completed with status code {response_status_code if 'response_status_code' in locals() else 'unknown'}.")


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
    Accumulate all WS binary frames until "END", then transcribe via the FULL
    or CHUNKED engine. Sends `segments_batch` per chunked batch (chunked only)
    and returns the final_transcription payload dict to be sent by the caller
    after the model lock is released.

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

    try:
        if resolved_strategy == ProcessingStrategy.FULL:
            hyps, asr_t = await _perform_asr_transcription(
                asr_model_instance=asr_model,
                audio_input_list=[waveform],
                batch_size_for_transcribe_call=client_config["batch_size"],
                num_asr_workers=NUM_WORKERS,
                request_id=f"WS-{session_id}",
            )
            segments = _process_hypotheses_to_segments(
                hyps, [0.0] * (len(hyps) if hyps else 0), session_id,
            )
        else:  # CHUNKED
            if _should_use_stateful_engine(audio_duration_s, session_id):
                # Stateful engine has no per-batch hook: it computes the whole
                # thing in one transcribe() call. No intermediate segments_batch.
                segments, asr_t = await _transcribe_chunked_stateful(
                    waveform=waveform,
                    audio_duration_s=audio_duration_s,
                    client_config=client_config,
                    request_id=f"WS-{session_id}",
                )
            else:
                # Legacy: emit segments_batch as each independent-chunk batch completes.
                async def _on_batch(batch_segs: List[dict], batch_num: int) -> bool:
                    if websocket.application_state != WebSocketState.CONNECTED:
                        return False
                    try:
                        await websocket.send_json({"type": "segments_batch", "segments": batch_segs})
                    except Exception as e_send:
                        logger.warning(f"({session_id}) {log_prefix}: send segments_batch failed: {e_send}")
                        return False
                    return True
                segments, asr_t = await _transcribe_chunked_waveform(
                    waveform=waveform,
                    client_config=client_config,
                    request_id=f"WS-{session_id}",
                    on_batch_segments=_on_batch,
                )
    finally:
        # Revert under the lock owner's lifecycle (we are still inside the lock).
        await _revert_model_to_global_original_state(
            long_audio_settings_were_active_for_session=long_audio_active,
            session_processing_device=session_processing_device,
            request_id=f"{session_id}-{log_prefix.lower().replace(' ', '_')}_revert",
        )

    if websocket.application_state != WebSocketState.CONNECTED:
        logger.info(f"({session_id}) {log_prefix}: client disconnected before final transcription.")
        return None

    text = " ".join(s.get('text', '') for s in segments).strip()
    return {
        "type": "final_transcription",
        "text": text,
        "language": "en",
        "strategy": resolved_strategy.value,
        "transcription_time_seconds": round(asr_t, 3),
        "total_segments": len(segments),
        "final_duration_processed_seconds": round(audio_duration_s, 3),
        "csv_content": generate_csv_content(segments),
        "srt_content": generate_srt_content(segments),
    }


async def _ws_handle_unified(
    websocket: WebSocket,
    strategy_override: Optional[str] = None,
    log_prefix: str = "WS",
) -> None:
    """
    Shared body for the unified WS endpoint and the two legacy compat aliases.

    strategy_override: when set (used by `/ws_upload` and `/ws_stream` compat
        wrappers), forces the resolved strategy regardless of what the client
        sends. Logs a warning if the client requested something different.
    """
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    await websocket.accept()
    logger.info(f"({session_id}) {log_prefix}: WebSocket connection accepted.")

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

        if strategy_override is not None:
            client_requested = config_dict.get("strategy")
            if client_requested is not None and str(client_requested).lower() not in (strategy_override.lower(), "auto"):
                logger.warning(
                    f"({session_id}) {log_prefix}: client requested strategy={client_requested!r}, "
                    f"but this endpoint forces {strategy_override!r}; overriding."
                )
            config_dict["strategy"] = strategy_override

        client_config = parse_websocket_config(config_dict)
        logger.info(f"({session_id}) {log_prefix}: parsed client config: {client_config}")

        resolved = resolve_strategy(
            audio_duration_s=None,
            client_config=client_config,
            is_streaming=True,
        )
        logger.info(f"({session_id}) {log_prefix}: resolved strategy = {resolved.value}.")

        if resolved == ProcessingStrategy.PROGRESSIVE:
            # Streaming pipeline: ffmpeg producer/consumer. handle_streaming_pcm
            # does its own _apply / _revert via the legacy pattern here, since
            # for streaming we don't know duration upfront and use chunk_length
            # as the proxy.
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
            # FULL or CHUNKED — accumulate first (no lock), then load,
            # acquire lock, transcribe, revert, release lock, send final.
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

      - strategy=full      → accumulate-then-process, single transcribe call.
      - strategy=chunked   → accumulate-then-process, sliding-window batched.
      - strategy=progressive → ffmpeg streaming pipeline with intermediate sends.
      - strategy=auto (default for WS) → progressive.

    Legacy aliases /ws_upload and /ws_stream forward here with strategy
    forced to chunked and progressive respectively.
    """
    await _ws_handle_unified(websocket, strategy_override=None, log_prefix="WS")


@app.websocket("/v1/audio/transcriptions/ws_stream")
async def websocket_transcribe_endpoint_streaming(websocket: WebSocket):
    """
    Legacy compatibility alias for /v1/audio/transcriptions with strategy=progressive.

    Existing clients connecting here get the ffmpeg streaming pipeline regardless
    of the strategy they send. Logs that the legacy path was taken.
    """
    logger.info("WS legacy alias: /ws_stream → forwarding with strategy=progressive.")
    await _ws_handle_unified(websocket, strategy_override="progressive", log_prefix="WS Stream")


@app.websocket("/v1/audio/transcriptions/ws_upload")
async def websocket_transcribe_endpoint_full_file_upload(websocket: WebSocket):
    """
    Legacy compatibility alias for /v1/audio/transcriptions with strategy=chunked.

    Existing clients connecting here get the accumulate-then-chunked pipeline
    regardless of the strategy they send. Logs that the legacy path was taken.
    """
    logger.info("WS legacy alias: /ws_upload → forwarding with strategy=chunked.")
    await _ws_handle_unified(websocket, strategy_override="chunked", log_prefix="WS Upload")


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
            workers=1,              # Number of Uvicorn worker processes (recommend 1 with current global model)
            log_level=log_level_str.lower() # Sync Uvicorn log level with app's
        )