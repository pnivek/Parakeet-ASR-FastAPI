import gc
import os
import io
import time
import json
import shutil
import tempfile
import base64
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple, List
import subprocess
import uvicorn

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

logger.info(
    f"Configuration loaded:\n"
    f"  App: workers={NUM_WORKERS}, sample_rate={MODEL_SAMPLE_RATE}, port={PORT}\n"
    f"  Model: {ASR_MODEL_NAME}, long_audio_threshold_for_model_settings={LONG_AUDIO_THRESHOLD_S}s\n"
    f"  Chunking Defaults: length={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s, batch_cap={CHUNKING_BATCH_SIZE}\n"
    f"  Streaming (ffmpeg): pcm_read_chunk_size={FFMPEG_PCM_CHUNK_SIZE_BYTES}B"
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

        # Store the model's initial device and dtype to revert to after processing
        global_original_model_device_str = str(next(asr_model.parameters()).device)
        global_original_model_dtype_torch = next(asr_model.parameters()).dtype

        logger.info(
            f"ASR model '{ASR_MODEL_NAME}' loaded successfully. "
            f"Original device: {global_original_model_device_str}, "
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
    Configures the ASR model ONCE per session for device, dtype, and long/short audio settings.
    
    The order of operations is important:
    1. Move to target device.
    2. Ensure model is float32 (for safety before structural changes).
    3. Apply attention/subsampling changes (long or short audio mode).
    4. Convert to final target_operational_dtype (e.g., bfloat16) for ASR computation.

    Args:
        decision_duration_s: The audio duration (can be total file or ASR chunk length)
                             used to decide if long audio settings are needed.
        target_processing_device: "cuda" or "cpu".
        target_operational_dtype: The desired dtype for ASR computation (e.g., torch.bfloat16 or torch.float32).
        long_audio_threshold_config: Duration threshold to activate long audio settings.
        request_id: Identifier for logging.

    Returns:
        True if long-audio specific settings (attention/subsampling) were activated for the session.
    """
    global asr_model
    if asr_model is None:
        logger.error(f"({request_id}) ASR model is None in _apply_model_settings_for_session.")
        return False
    
    long_audio_settings_activated_for_session = False

    # 1. Set model to target_processing_device for the session
    if str(next(asr_model.parameters()).device) != target_processing_device:
        await asyncio.to_thread(asr_model.to, device=target_processing_device)
        logger.debug(f"({request_id}) Session: Model moved to device: {target_processing_device}")

    # 2. Ensure model is float32 BEFORE any structural changes (e.g., attention/subsampling)
    # This ensures that model parameter modifications happen from a known, stable dtype.
    if next(asr_model.parameters()).dtype != torch.float32:
        await asyncio.to_thread(asr_model.to, dtype=torch.float32)
        logger.debug(f"({request_id}) Session: Model temporarily set to float32 before structural changes.")

    # 3. Apply/Ensure attention and subsampling settings based on decision_duration_s (model is currently float32)
    if decision_duration_s > long_audio_threshold_config:
        logger.info(f"({request_id}) Session: Decision duration {decision_duration_s:.2f}s > threshold {long_audio_threshold_config:.2f}s. Applying long audio settings.")
        try:
            await asyncio.to_thread(asr_model.change_attention_model, "rel_pos_local_attn", [256, 256])
            await asyncio.to_thread(asr_model.change_subsampling_conv_chunking_factor, 1)
            long_audio_settings_activated_for_session = True
            logger.debug(f"({request_id}) Session: Long audio settings (attention: rel_pos_local_attn, subsampling_factor: 1) applied.")
        except Exception as e_long:
            logger.warning(f"({request_id}) Session: Failed to apply long audio settings: {e_long}")
    else:
        logger.info(f"({request_id}) Session: Decision duration {decision_duration_s:.2f}s <= threshold {long_audio_threshold_config:.2f}s. Ensuring short audio settings.")
        try:
            await asyncio.to_thread(asr_model.change_attention_model, "rel_pos") # Default attention
            await asyncio.to_thread(asr_model.change_subsampling_conv_chunking_factor, -1) # Default subsampling
            logger.debug(f"({request_id}) Session: Short audio settings (attention: rel_pos, subsampling_factor: -1) ensured.")
        except Exception as e_short:
            logger.warning(f"({request_id}) Session: Failed to ensure short audio settings: {e_short}")
            
    # 4. Set model to the final target_operational_dtype (e.g., bfloat16) AFTER structural changes
    if next(asr_model.parameters()).dtype != target_operational_dtype:
        await asyncio.to_thread(asr_model.to, dtype=target_operational_dtype)
        logger.info(f"({request_id}) Session: Model set to final operational dtype: {target_operational_dtype} for ASR computation.")
    else:
        logger.debug(f"({request_id}) Session: Model already in final operational dtype: {target_operational_dtype}.")
            
    return long_audio_settings_activated_for_session


async def _revert_model_to_global_original_state(
    long_audio_settings_were_active_for_session: bool,
    session_processing_device: str, # The device the model was on during the session
    request_id: str = "req"
):
    """
    Reverts the ASR model to its globally original state (device and dtype) ONCE at the end of a session.
    If long audio settings were active, it reverts attention/subsampling to their defaults first.
    Clears CUDA cache if CUDA was used.

    Args:
        long_audio_settings_were_active_for_session: Flag indicating if long audio settings need reversion.
        session_processing_device: The device ("cuda" or "cpu") used during the session.
        request_id: Identifier for logging.
    """
    global asr_model, global_original_model_device_str, global_original_model_dtype_torch
    if asr_model is None:
        logger.error(f"({request_id}) ASR model is None in _revert_model_to_global_original_state.")
        return

    try:
        # This should be done while the model is still on its compute device and operational dtype,
        # before moving to the global original device/dtype, in case these changes expect that state.
        if long_audio_settings_were_active_for_session:
            logger.info(f"({request_id}) End of Session: Reverting long audio specific model settings (attention to rel_pos, subsampling to -1).")
            try:
                # Ensure model is float32 if structural changes expect it (safer)
                if next(asr_model.parameters()).dtype != torch.float32:
                    await asyncio.to_thread(asr_model.to, dtype=torch.float32)
                    logger.debug(f"({request_id}) End of Session: Model temporarily set to float32 for reverting structural changes.")

                await asyncio.to_thread(asr_model.change_attention_model, "rel_pos")
                await asyncio.to_thread(asr_model.change_subsampling_conv_chunking_factor, -1)
            
            except Exception as e_rev_long_specific:
                 logger.warning(f"({request_id}) End of Session: Failed to revert long-audio specific settings: {e_rev_long_specific}")

        # 2. ALWAYS revert model to its globally original device and dtype (from initial load)
        current_device_after_session_ops = str(next(asr_model.parameters()).device)
        current_dtype_after_session_ops = next(asr_model.parameters()).dtype

        if (current_device_after_session_ops != global_original_model_device_str or
            current_dtype_after_session_ops != global_original_model_dtype_torch):
            logger.info(f"({request_id}) End of Session: Reverting model to global original device ('{global_original_model_device_str}') and dtype ({global_original_model_dtype_torch}).")
            await asyncio.to_thread(asr_model.to, device=global_original_model_device_str, dtype=global_original_model_dtype_torch)
        else:
            logger.debug(f"({request_id}) End of Session: Model already on global original device/dtype. No change needed.")
            
        # 3. Clean up CUDA cache if CUDA was used during the session
        if session_processing_device == "cuda" and torch.cuda.is_available():
            await asyncio.to_thread(gc.collect)
            await asyncio.to_thread(torch.cuda.empty_cache)
            logger.debug(f"({request_id}) End of Session: CUDA cache cleared as '{session_processing_device}' was used.")
            
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
    l_thresh: Optional[float] = None
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

    Returns:
        A dictionary containing the validated configuration parameters.

    Raises:
        ValueError: If any parameter value is outside its allowed range.
    """
    config = {
        "chunk_length": c_len if c_len is not None else TRANSCRIBE_CHUNK_LEN,
        "chunk_overlap": c_ov if c_ov is not None else TRANSCRIBE_OVERLAP,
        "batch_size": b_size if b_size is not None else CHUNKING_BATCH_SIZE,
        "long_audio_threshold": l_thresh if l_thresh is not None else LONG_AUDIO_THRESHOLD_S
    }

    if not (0 < config["chunk_length"] <= 300): # Max 5 minutes chunk
        raise ValueError("chunk_length must be > 0 and <= 300 seconds.")
    if not (0 <= config["chunk_overlap"] < config["chunk_length"]):
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_length.")
    if not (1 <= config["batch_size"] <= 32): # Practical limit for batch size
        raise ValueError("batch_size must be between 1 and 32.")
    if not (0 <= config["long_audio_threshold"] <= 3600): # Max 1 hour threshold
        raise ValueError("long_audio_threshold must be >= 0 and <= 3600 seconds.")
        
    return config


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

    # Get common ASR config (chunk_length, overlap, batch_size, long_audio_threshold)
    asr_config = parse_request_config(
        client_cfg.get("chunk_length"),
        client_cfg.get("chunk_overlap"),
        client_cfg.get("batch_size"),
        client_cfg.get("long_audio_threshold")
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
    
    # ASR chunking parameters based on target 16kHz mono PCM from ffmpeg
    asr_chunk_len_s = client_config["chunk_length"]
    asr_chunk_overlap_s = client_config["chunk_overlap"]
    
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
        Consumer coroutine:
        1. Gets (audio_tensor, offset_s) from `chunk_queue`.
        2. Batches them for ASR.
        3. Performs transcription.
        4. Sends results back via WebSocket.
        """
        nonlocal accumulated_asr_processing_time_s, sent_segments_pcm
        
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
                "text": final_transcribed_text_pcm,
                "language": "en", # Assuming English, could be made configurable
                "transcription_time": round(accumulated_asr_processing_time_s, 3),
                "total_segments": len(sent_segments_pcm),
                "final_duration_processed_seconds": round(total_duration_processed_seconds_for_asr, 3),
                "csv_content": generate_csv_content(sent_segments_pcm), # Utility function
                "srt_content": generate_srt_content(sent_segments_pcm), # Utility function
                "streaming_mode": client_config.get("format", "unknown") # Reflect client-declared format
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


@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(
    file: UploadFile = File(...),
    chunk_length: Optional[float] = Query(None, description="Duration of audio chunks for ASR in seconds. Uses server default if not set."),
    chunk_overlap: Optional[float] = Query(None, description="Overlap between audio chunks in seconds. Uses server default if not set."),
    batch_size: Optional[int] = Query(None, description="Batch size for ASR model processing. Uses server default if not set."),
    long_audio_threshold: Optional[float] = Query(None, description="Threshold in seconds to apply long audio model settings. Uses server default if not set.")
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
        client_config_rest = parse_request_config(chunk_length, chunk_overlap, batch_size, long_audio_threshold)
        logger.info(f"({request_id}) REST: Parsed request config: {client_config_rest}")
    except ValueError as e_config:
        logger.warning(f"({request_id}) REST: Invalid request parameters: {e_config}")
        return JSONResponse(status_code=400, content={"error": f"Invalid request parameter: {str(e_config)}"})

    temp_audio_file_path: str = ""
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
                # Save uploaded file to a temporary location for processing by torchaudio
                # Using a NamedTemporaryFile ensures it's cleaned up even if errors occur.
                # delete=False is needed on Windows to allow torchaudio.load to open it by name.
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or ".unknown_audio_ext").suffix, prefix=f"rest_audio_{request_id}_") as tmp_file:
                    await asyncio.to_thread(shutil.copyfileobj, file.file, tmp_file)
                    temp_audio_file_path = tmp_file.name
                
                logger.info(f"({request_id}) REST: Uploaded file saved temporarily to '{temp_audio_file_path}'.")

                # Load and preprocess the entire audio file
                waveform_tensor, total_audio_duration_s = await load_and_preprocess_audio(
                    audio_source=temp_audio_file_path,
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
                    # Apply model settings (device, dtype, long/short audio attention) for this session
                    # For REST, the decision for long/short audio settings is based on the total file duration.
                    long_audio_settings_applied_this_session = await _apply_model_settings_for_session(
                        decision_duration_s=total_audio_duration_s,
                        target_processing_device=session_processing_device,
                        target_operational_dtype=session_target_operational_dtype,
                        long_audio_threshold_config=client_config_rest["long_audio_threshold"],
                        request_id=request_id
                    )
                    logger.info(f"({request_id}) REST: ASR model settings applied for session. Long audio specific settings active: {long_audio_settings_applied_this_session}.")

                    # Perform transcription on the entire waveform
                    # NeMo's `transcribe` can handle batching internally if a list of waveforms is provided.
                    # Here, we provide a list containing a single tensor.
                    hypotheses_list, asr_processing_time_s = await _perform_asr_transcription(
                        asr_model_instance=asr_model,
                        audio_input_list=[waveform_tensor], # List with one item for single file
                        batch_size_for_transcribe_call=client_config_rest["batch_size"], # Can be 1
                        num_asr_workers=NUM_WORKERS,
                        request_id=f"REST-{request_id}"
                    )

                    # Process raw hypotheses into structured segments
                    # For a single, non-chunked audio file, the offset is 0.0 for all hypotheses.
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
                        "language": "en", # Assuming English
                        "transcription_time_seconds": round(asr_processing_time_s, 3),
                        "total_request_time_server_seconds": total_server_processing_time_s,
                        "csv_content": generate_csv_content(segments),
                        "srt_content": generate_srt_content(segments),
                        "audio_duration_seconds": round(total_audio_duration_s, 3)
                    }
                    response_status_code = 200 # OK
                    logger.info(f"({request_id}) REST: Transcription successful. Duration: {total_audio_duration_s:.2f}s, ASR time: {asr_processing_time_s:.2f}s.")

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
        # Clean up the temporary audio file
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            try:
                await asyncio.to_thread(os.remove, temp_audio_file_path)
                logger.debug(f"({request_id}) REST: Successfully removed temporary file '{temp_audio_file_path}'.")
            except Exception as e_remove_temp:
                logger.warning(f"({request_id}) REST: Failed to remove temporary file '{temp_audio_file_path}': {e_remove_temp}")
        
        # Ensure file object from UploadFile is closed if FastAPI hasn't handled it.
        if hasattr(file, 'file') and file.file and not file.file.closed:
            await asyncio.to_thread(file.file.close)

        logger.info(f"({request_id}) REST request for file '{file.filename}' completed with status code {response_status_code if 'response_status_code' in locals() else 'unknown'}.")


@app.websocket("/v1/audio/transcriptions/ws_stream")
async def websocket_transcribe_endpoint_streaming(websocket: WebSocket):
    """
    Handles WebSocket connections for live/streaming audio transcription as its received.

    The client first sends a JSON configuration message. Then, it streams
    audio data in binary chunks. An "END" text message signals the end of the stream.
    The server uses `ffmpeg` to transcode incoming audio to a standard PCM format,
    then chunks and transcribes it, sending back `segments_batch` messages with
    intermediate results and a `final_transcription` message upon completion.

    Protocol:
    1. Client connects.
    2. Server accepts.
    3. Client sends JSON text message with configuration (see `parse_websocket_config`).
    4. Client sends audio data as binary messages.
    5. Client sends "END" text message to signal end of audio.
    6. Server sends `segments_batch` JSON messages with lists of transcribed segments.
    7. Server sends `final_transcription` JSON message with aggregated results.
    8. Connection is closed.

    Args:
        websocket: The WebSocket connection object provided by FastAPI.
    """
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode() # Unique ID for this session
    await websocket.accept()
    logger.info(f"({session_id}) WebSocket connection accepted for audio streaming (ffmpeg-based).")

    if not asr_model:
        logger.error(f"({session_id}) WS Stream: ASR model not available.")
        await websocket.send_json({"type": "error", "error": "ASR model not available. Service is initializing or encountered an error."})
        await websocket.close(code=1011) # 1011: Server error
        return

    session_processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    long_audio_settings_applied_this_session = False
    
    session_target_operational_dtype = torch.float32
    if session_processing_device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        session_target_operational_dtype = torch.bfloat16
    logger.info(f"({session_id}) WS Stream Session: Will use dtype {session_target_operational_dtype} on device {session_processing_device} for ASR computation.")

    client_config: Optional[dict] = None
    try:
        # 1. Receive and parse initial client configuration
        config_text = await asyncio.wait_for(websocket.receive_text(), timeout=20.0) # Timeout for config
        client_config = parse_websocket_config(json.loads(config_text))
        logger.info(f"({session_id}) WS Stream: Received valid client configuration: {client_config}")

        # Acquire lock for exclusive ASR model access
        async with model_access_lock:
            logger.debug(f"({session_id}) WS Stream: Acquired ASR model access lock.")
            try:
                # Apply model settings for this streaming session.
                # For streaming, decision for long/short audio is based on ASR chunk_length.
                long_audio_settings_applied_this_session = await _apply_model_settings_for_session(
                    decision_duration_s=client_config["chunk_length"], 
                    target_processing_device=session_processing_device,
                    target_operational_dtype=session_target_operational_dtype,
                    long_audio_threshold_config=client_config["long_audio_threshold"],
                    request_id=session_id
                )
                logger.info(f"({session_id}) WS Stream: ASR model settings applied for session. Long audio specific settings active: {long_audio_settings_applied_this_session}.")

                # Start the streaming PCM handling pipeline (producer/consumer with ffmpeg)
                await handle_streaming_pcm(websocket, session_id, session_processing_device, client_config)
            
            except Exception as e_locked_ws_processing:
                # Errors within the locked block (e.g., during handle_streaming_pcm or model setup)
                logger.error(f"({session_id}) WS Stream: Error during locked operation: {e_locked_ws_processing}", exc_info=True)
                if websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "error", "error": f"Server error during streaming: {str(e_locked_ws_processing)}"})
                    except Exception as e_send_err_on_locked_err:
                        logger.warning(f"({session_id}) WS Stream: Could not send error to client after locked operation error: {e_send_err_on_locked_err}")
            finally:
                logger.debug(f"({session_id}) WS Stream: Releasing ASR model lock and reverting model state.")
                # Always revert model state at the end of the locked block
                await _revert_model_to_global_original_state(
                    long_audio_settings_were_active_for_session=long_audio_settings_applied_this_session,
                    session_processing_device=session_processing_device,
                    request_id=f"{session_id}-stream_model_revert"
                )
                logger.info(f"({session_id}) WS Stream: ASR Model state reverted after session.")
    
    except WebSocketDisconnect:
        logger.info(f"({session_id}) WS Stream: WebSocket disconnected by client.")
    except asyncio.TimeoutError:
        logger.warning(f"({session_id}) WS Stream: Timeout occurred, likely waiting for initial client config.")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": "Timeout: No configuration received from client."})
            except Exception: pass # Ignore if send also fails
    except json.JSONDecodeError as e_json_decode:
        logger.warning(f"({session_id}) WS Stream: Failed to decode JSON configuration from client: {e_json_decode}")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": f"Invalid JSON configuration received: {str(e_json_decode)}"})
            except Exception: pass
    except ValueError as e_value_config: # From parse_websocket_config
        logger.warning(f"({session_id}) WS Stream: Invalid configuration parameters: {e_value_config}")
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": f"Invalid configuration: {str(e_value_config)}"})
            except Exception: pass
    except Exception as e_outer_ws_handler:
        # Catch-all for other unexpected errors in the WebSocket handler
        logger.error(f"({session_id}) WS Stream: Unhandled exception in WebSocket endpoint: {e_outer_ws_handler}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            try: await websocket.send_json({"type": "error", "error": "An unexpected server error occurred."})
            except Exception: pass
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            logger.info(f"({session_id}) WS Stream: Closing WebSocket connection (endpoint finally block).")
            try:
                await websocket.close(code=1000) # 1000: Normal Closure
            except Exception as e_close_ws:
                logger.warning(f"({session_id}) WS Stream: Error closing WebSocket in finally block: {e_close_ws}")
        logger.info(f"({session_id}) Streaming WebSocket session via ffmpeg ended.")


@app.websocket("/v1/audio/transcriptions/ws_upload")
async def websocket_transcribe_endpoint_full_file_upload(websocket: WebSocket):
    """
    Handles WebSocket connections for transcribing a full audio file uploaded via WebSocket.

    The client first sends a JSON configuration message. Then, it streams the
    entire audio file as binary messages, followed by an "END" text message.
    The server accumulates the entire file, then processes it by chunking the
    audio and performing ASR on these chunks. Intermediate `segments_batch`
    messages are sent, followed by a `final_transcription` message.

    Protocol:
    1. Client connects.
    2. Server accepts.
    3. Client sends JSON text message with configuration (see `parse_websocket_config`).
    4. Client sends all audio data as binary messages.
    5. Client sends "END" text message.
    6. Server sends `segments_batch` JSON messages with lists of transcribed segments from ASR batches.
    7. Server sends `final_transcription` JSON message with aggregated and deduplicated results.
    8. Connection is closed.

    Args:
        websocket: The WebSocket connection object provided by FastAPI.
    """
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    await websocket.accept()
    logger.info(f"({session_id}) WebSocket connection accepted for full file upload transcription.")

    if not asr_model:
        logger.error(f"({session_id}) WS Upload: ASR model not available.")
        await websocket.send_json({"type": "error", "error": "ASR model not available. Service is initializing or encountered an error."})
        await websocket.close(code=1011) # Server error
        return

    session_processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    long_audio_settings_applied_this_session = False
    final_response_payload: Optional[dict] = None # To store the final message content
    
    session_target_operational_dtype = torch.float32
    if session_processing_device == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        session_target_operational_dtype = torch.bfloat16
    logger.info(f"({session_id}) WS Upload Session: Will use dtype {session_target_operational_dtype} on device {session_processing_device} for ASR computation.")

    client_config: Optional[dict] = None
    try:
        # 1. Receive and parse initial client configuration
        config_text = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
        client_config = parse_websocket_config(json.loads(config_text))
        logger.info(f"({session_id}) WS Upload: Received valid client configuration: {client_config}")

        # Acquire lock for exclusive ASR model access
        async with model_access_lock:
            logger.debug(f"({session_id}) WS Upload: Acquired ASR model access lock.")
            try:
                # Apply model settings for this session.
                # For ws_upload, decision for long/short audio is based on ASR chunk_length,
                # as the full audio duration isn't known until all data is received.
                long_audio_settings_applied_this_session = await _apply_model_settings_for_session(
                    decision_duration_s=client_config["chunk_length"],
                    target_processing_device=session_processing_device,
                    target_operational_dtype=session_target_operational_dtype,
                    long_audio_threshold_config=client_config["long_audio_threshold"],
                    request_id=session_id
                )
                logger.debug(f"({session_id}) WS Upload: ASR model settings applied. Long audio specific settings active: {long_audio_settings_applied_this_session}.")

                # 2. Accumulate audio data from client
                accumulated_raw_audio_bytes = bytearray()
                logger.info(f"({session_id}) WS Upload: Waiting to receive audio file data...")
                while True:
                    if websocket.application_state != WebSocketState.CONNECTED:
                        # This can happen if client disconnects abruptly during transfer
                        raise WebSocketDisconnect(code=1001, reason="Client disconnected during file data transfer.")
                    
                    try:
                        # Timeout for each data chunk to detect stalled uploads
                        message = await asyncio.wait_for(websocket.receive(), timeout=60.0) 
                    except asyncio.TimeoutError:
                        logger.warning(f"({session_id}) WS Upload: Timeout waiting for audio data chunk from client.")
                        raise WebSocketDisconnect(code=1008, reason="Timeout waiting for file data from client.") # Policy Violation

                    if message.get("type") == "websocket.disconnect":
                        # Explicit disconnect message from FastAPI/Starlette
                        raise WebSocketDisconnect(code=message.get('code', 1000))

                    if 'text' in message and message['text']:
                        if message['text'].upper() == "END":
                            logger.info(f"({session_id}) WS Upload: 'END' signal received. Total bytes received: {len(accumulated_raw_audio_bytes)}.")
                            break # End of file transfer
                        else:
                            logger.warning(f"({session_id}) WS Upload: Received unexpected text message during data transfer: {message['text'][:100]}")
                    elif 'bytes' in message and message['bytes']:
                        accumulated_raw_audio_bytes.extend(message['bytes'])
                        # Log progress sparingly to avoid flooding logs for large files
                        if len(accumulated_raw_audio_bytes) % (1024 * 1024 * 5) < len(message['bytes']): # Log every ~5MB
                             logger.debug(f"({session_id}) WS Upload: Received {len(accumulated_raw_audio_bytes)} bytes so far...")
                    else:
                        logger.warning(f"({session_id}) WS Upload: Received unexpected message type: {message.get('type')}")

                if not accumulated_raw_audio_bytes:
                    logger.warning(f"({session_id}) WS Upload: No audio data received before 'END' signal.")
                    raise ValueError("No audio data was received for full file upload.")
                
                # 3. Process accumulated audio data
                logger.info(f"({session_id}) WS Upload: Processing {len(accumulated_raw_audio_bytes)} bytes of audio data.")
                waveform_full_upload_tensor, audio_duration_s_full_upload = await load_and_preprocess_audio(
                    audio_source=io.BytesIO(bytes(accumulated_raw_audio_bytes)), # Process from memory
                    target_sample_rate=MODEL_SAMPLE_RATE,
                    request_id=session_id
                )
                accumulated_raw_audio_bytes.clear() # Free memory

                if waveform_full_upload_tensor is None or audio_duration_s_full_upload == 0:
                    logger.error(f"({session_id}) WS Upload: Audio loading/preprocessing failed for uploaded data.")
                    raise ValueError("Full File Upload: Audio loading or preprocessing resulted in empty audio.")
                if asr_model is None: # Should be caught by lock, but defensive check
                    logger.critical(f"({session_id}) WS Upload: ASR Model became None during processing.")
                    raise RuntimeError("ASR Model is not available during full upload processing.")

                logger.info(f"({session_id}) WS Upload: Audio preprocessed. Duration: {audio_duration_s_full_upload:.2f}s. Creating ASR chunks...")
                
                # 4. Manually chunk the full waveform for batched ASR
                manual_asr_chunks_ws, manual_asr_offsets_ws = create_audio_chunks(
                    waveform=waveform_full_upload_tensor,
                    sample_rate=MODEL_SAMPLE_RATE,
                    chunk_len_s=client_config["chunk_length"],
                    overlap_s=client_config["chunk_overlap"]
                )
                
                all_raw_segments_from_asr_batches: List[dict] = []
                total_asr_processing_time_this_session_s: float = 0.0
                asr_batch_size_for_inference = client_config["batch_size"]

                logger.info(f"({session_id}) WS Upload: Processing {len(manual_asr_chunks_ws)} ASR chunks in batches of {asr_batch_size_for_inference}.")
                for batch_num, batch_start_idx in enumerate(range(0, len(manual_asr_chunks_ws), asr_batch_size_for_inference)):
                    if websocket.application_state != WebSocketState.CONNECTED:
                        logger.warning(f"({session_id}) WS Upload: Client disconnected during ASR batch processing. Aborting.")
                        break # Stop processing if client disconnected

                    current_batch_audio_tensors = manual_asr_chunks_ws[batch_start_idx : batch_start_idx + asr_batch_size_for_inference]
                    current_batch_time_offsets = manual_asr_offsets_ws[batch_start_idx : batch_start_idx + asr_batch_size_for_inference]
                    
                    if not current_batch_audio_tensors:
                        continue # Should not happen if range is correct

                    logger.debug(f"({session_id}) WS Upload: Processing ASR batch {batch_num + 1} "
                                 f"({len(current_batch_audio_tensors)} chunks).")
                    
                    hypotheses_for_batch, asr_batch_duration_s = await _perform_asr_transcription(
                        asr_model_instance=asr_model,
                        audio_input_list=current_batch_audio_tensors,
                        batch_size_for_transcribe_call=len(current_batch_audio_tensors),
                        num_asr_workers=NUM_WORKERS,
                        request_id=f"WS-FullUpload-{session_id}-b{batch_num}"
                    )
                    total_asr_processing_time_this_session_s += asr_batch_duration_s
                    
                    if hypotheses_for_batch:
                        segments_from_this_batch = _process_hypotheses_to_segments(
                            hypotheses_for_batch, current_batch_time_offsets, f"{session_id}-segproc_b{batch_num}"
                        )
                        if websocket.application_state == WebSocketState.CONNECTED and segments_from_this_batch:
                            # Sort segments within the batch by start time before sending
                            segments_from_this_batch.sort(key=lambda s: s.get("start", float('inf')))
                            logger.info(f"({session_id}) WS Upload: Sending {len(segments_from_this_batch)} segments from batch {batch_num + 1} to client.")
                            await websocket.send_json({"type": "segments_batch", "segments": segments_from_this_batch})
                        all_raw_segments_from_asr_batches.extend(segments_from_this_batch)
                    
                    # Clear CUDA cache periodically
                    if session_processing_device == "cuda" and torch.cuda.is_available():
                        await asyncio.to_thread(torch.cuda.empty_cache)
                
                # 5. Prepare and send final transcription if still connected
                if websocket.application_state == WebSocketState.CONNECTED:
                    logger.info(f"({session_id}) WS Upload: All ASR batches processed. Deduplicating {len(all_raw_segments_from_asr_batches)} raw segments.")
                    # Deduplicate segments from all batches
                    final_deduplicated_segments = _deduplicate_segments(
                        all_raw_segments_from_asr_batches,
                        client_config["chunk_overlap"] / 2.0 # Use a fraction of overlap as threshold
                    )
                    final_transcribed_text = " ".join(s['text'] for s in final_deduplicated_segments).strip()
                    
                    final_response_payload = {
                        "type": "final_transcription",
                        "text": final_transcribed_text,
                        "language": "en",
                        "transcription_time_seconds": round(total_asr_processing_time_this_session_s, 3),
                        "total_segments": len(final_deduplicated_segments),
                        "final_duration_processed_seconds": round(audio_duration_s_full_upload, 3),
                        "csv_content": generate_csv_content(final_deduplicated_segments),
                        "srt_content": generate_srt_content(final_deduplicated_segments)
                    }
                    logger.info(f"({session_id}) WS Upload: Final transcription prepared. Total segments: {len(final_deduplicated_segments)}.")
                else:
                    logger.info(f"({session_id}) WS Upload: Client disconnected before final transcription could be prepared.")

            except Exception as e_locked_full_upload_processing:
                logger.error(f"({session_id}) WS Upload: Error during locked operation: {e_locked_full_upload_processing}", exc_info=True)
                if websocket.application_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_json({"type": "error", "error": f"Server error during full file upload processing: {str(e_locked_full_upload_processing)}"})
                    except Exception: pass # Ignore if send also fails
            finally:
                logger.debug(f"({session_id}) WS Upload: Releasing ASR model lock and reverting model state.")
                await _revert_model_to_global_original_state(
                    long_audio_settings_were_active_for_session=long_audio_settings_applied_this_session,
                    session_processing_device=session_processing_device,
                    request_id=f"{session_id}-full_upload_model_revert"
                )
                logger.info(f"({session_id}) WS Upload: ASR Model state reverted after session.")
        
        # Send the final response if it was prepared and client is still connected
        if final_response_payload and websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json(final_response_payload)
            logger.info(f"({session_id}) WS Upload: Final transcription message sent to client.")
            
    except asyncio.TimeoutError: # For initial config wait
        logger.warning(f"({session_id}) WS Upload: Timeout waiting for initial client configuration.")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": "Timeout: No configuration received."})
    except json.JSONDecodeError as e_json:
        logger.warning(f"({session_id}) WS Upload: Failed to decode JSON configuration: {e_json}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": f"Invalid JSON configuration: {str(e_json)}"})
    except ValueError as e_val: # From parse_websocket_config or data validation
        logger.warning(f"({session_id}) WS Upload: Value error (config or data): {e_val}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": str(e_val)})
    except WebSocketDisconnect as e_ws_disconnect:
        logger.info(f"({session_id}) WS Upload: WebSocket disconnected. Reason: {e_ws_disconnect.reason} (Code: {e_ws_disconnect.code})")
    except Exception as e_outer_full_upload:
        logger.error(f"({session_id}) WS Upload: Unhandled exception in endpoint: {e_outer_full_upload}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": "An unexpected server error occurred."})
    finally:
        if websocket.application_state == WebSocketState.CONNECTED:
            logger.info(f"({session_id}) WS Upload: Closing WebSocket connection (endpoint finally block).")
            await websocket.close(code=1000) # Normal closure
        logger.info(f"({session_id}) Full file upload WebSocket session ended.")


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