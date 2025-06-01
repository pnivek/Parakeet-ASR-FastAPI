import gc
import os
import io
import time
import shutil
import tempfile
import base64
import asyncio
import logging
from pathlib import Path
import json
from typing import Optional, Tuple, List, Any 

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query, Request
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
from omegaconf import OmegaConf

from dotenv import load_dotenv

from utils import (
    generate_srt_content,
    generate_csv_content,
)

load_dotenv()

# Logging Configuration
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("parakeet-asr.main")
logger.info(f"Logging configured with level: {log_level_str}")

# App Configuration
REST_BATCH_SIZE = 1
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
MODEL_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8777))
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
LONG_AUDIO_THRESHOLD_S = float(os.getenv("LONG_AUDIO_THRESHOLD", 480.0))

# Chunking Configuration
TRANSCRIBE_CHUNK_LEN = float(os.getenv("TRANSCRIBE_CHUNK_LEN", 30.0))
TRANSCRIBE_OVERLAP = float(os.getenv("TRANSCRIBE_OVERLAP", 5.0))
CHUNKING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))

logger.info(
    f"Configuration loaded:\n"
    f"  Chunking: length={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s, batch={CHUNKING_BATCH_SIZE}\n"
    f"  App: rest_batch={REST_BATCH_SIZE}, workers={NUM_WORKERS}, "
    f"sample_rate={MODEL_SAMPLE_RATE}, port={PORT}\n"
    f"  Model: {ASR_MODEL_NAME}, long_audio_threshold={LONG_AUDIO_THRESHOLD_S}s"
)

# FastAPI App Setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Static Files Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index_page():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return HTMLResponse(f"UI not found at {index_path}.", status_code=404)

# Global Model Variables
asr_model: Optional[NeMoASRModelType] = None 
original_model_attention_config_dict: Optional[dict] = None 
original_model_subsampling_config_dict: Optional[dict] = None 
global_original_model_device_str: str = "cpu" 
global_original_model_dtype_torch: torch.dtype = torch.float32

try:
    logger.info(f"Loading ASR model: {ASR_MODEL_NAME}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.eval()
    
    # Store original model configuration for restoration
    if hasattr(asr_model, 'cfg') and hasattr(asr_model.cfg, 'encoder'):
        if hasattr(asr_model.cfg.encoder, 'attention'):
            original_model_attention_config_dict = OmegaConf.to_container(
                asr_model.cfg.encoder.attention, resolve=True
            )
        if hasattr(asr_model.cfg.encoder, 'conv_subsampling'):
            original_model_subsampling_config_dict = OmegaConf.to_container(
                asr_model.cfg.encoder.conv_subsampling, resolve=True
            )
    
    # Store original device and dtype for restoration
    global_original_model_device_str = str(next(asr_model.parameters()).device)
    global_original_model_dtype_torch = next(asr_model.parameters()).dtype
    
    logger.info(
        f"ASR model '{ASR_MODEL_NAME}' loaded. "
        f"Original device: {global_original_model_device_str}, "
        f"dtype: {global_original_model_dtype_torch}."
    )
except Exception as e:
    logger.critical(
        f"FATAL: Could not load ASR model '{ASR_MODEL_NAME}'. Error: {e}", 
        exc_info=True
    )
    asr_model = None

model_access_lock = asyncio.Lock()

async def load_and_preprocess_audio(
    audio_source: str | io.BytesIO, 
    target_sample_rate: int, 
    request_id: str = "req"
) -> Tuple[Optional[torch.Tensor], float]:
    """
    Load and preprocess audio from file path or BytesIO stream.
    
    Returns:
        Tuple of (waveform_tensor, audio_duration_seconds)
    """
    waveform_tensor: Optional[torch.Tensor] = None
    audio_duration_s: float = 0.0
    source_description = audio_source if isinstance(audio_source, str) else "BytesIO stream"
    
    try:
        wf, sr = await asyncio.to_thread(torchaudio.load, audio_source)
        
        # Resample if necessary
        if sr != target_sample_rate: 
            wf = await asyncio.to_thread(
                torchaudio.functional.resample, wf, sr, target_sample_rate
            )
        
        # Convert stereo to mono if needed
        if wf.shape[0] > 1: 
            wf = torch.mean(wf, dim=0, keepdim=True)
        
        waveform_tensor = wf.squeeze(0)
        audio_duration_s = waveform_tensor.shape[0] / target_sample_rate
        
        if waveform_tensor.numel() == 0: 
            return None, 0.0
            
        return waveform_tensor, audio_duration_s
        
    except Exception as e_load:
        logger.error(
            f"({request_id}): Failed to load/preprocess audio from {source_description}. "
            f"Error: {e_load}", 
            exc_info=True
        )
        return None, 0.0

async def convert_audio_chunk_to_target_pcm(
    audio_chunk_bytes: bytes, 
    client_format: str, 
    client_sample_rate: int,
    client_channels: int, 
    client_bytes_per_sample: int, 
    target_sample_rate: int,
    request_id: str = "req"
) -> Optional[bytes]:
    """
    Convert audio chunk from client format to target PCM format.
    
    Returns:
        PCM bytes in target format, or None if conversion failed
    """
    try:
        waveform: Optional[torch.Tensor] = None
        source_sr: int = client_sample_rate
        
        if not audio_chunk_bytes: 
            return None
        
        if client_format == "pcm":
            # Handle different PCM bit depths
            if client_bytes_per_sample == 2: 
                waveform = torch.frombuffer(audio_chunk_bytes, dtype=torch.int16).float() / 32768.0
            elif client_bytes_per_sample == 4: 
                waveform = torch.frombuffer(audio_chunk_bytes, dtype=torch.int32).float() / 2147483648.0
            elif client_bytes_per_sample == 1: 
                waveform = torch.frombuffer(audio_chunk_bytes, dtype=torch.uint8).float() / 128.0 - 1.0
            else: 
                return None
                
            if waveform.numel() == 0: 
                return None
                
            # Handle multi-channel audio
            if client_channels > 1:
                if waveform.numel() % client_channels != 0: 
                    return None
                try: 
                    waveform = waveform.view(-1, client_channels).transpose(0, 1)
                except RuntimeError: 
                    return None 
            else: 
                waveform = waveform.unsqueeze(0)
        else:
            # Handle other audio formats
            wf_temp, sr_temp = await asyncio.to_thread(
                torchaudio.load, 
                io.BytesIO(audio_chunk_bytes), 
                format=client_format if client_format not in ["raw", "pcm"] else None
            )
            waveform = wf_temp
            source_sr = sr_temp
            
            if waveform.numel() == 0: 
                return None
        
        # Ensure waveform is 2D
        if waveform.ndim == 1: 
            waveform = waveform.unsqueeze(0)
        
        # Resample if necessary
        if source_sr != target_sample_rate: 
            waveform = await asyncio.to_thread(
                torchaudio.functional.resample, waveform, source_sr, target_sample_rate
            )
        
        # Convert to mono
        if waveform.shape[0] > 1: 
            waveform = torch.mean(waveform, dim=0, keepdim=False) 
        elif waveform.ndim == 2 and waveform.shape[0] == 1: 
            waveform = waveform.squeeze(0) 
        
        if waveform.numel() == 0: 
            return None
        
        # Convert to 16-bit PCM
        pcm_int16 = (waveform.clamp(-1.0, 1.0) * 32767).to(torch.int16)
        return pcm_int16.numpy().tobytes()
        
    except Exception: 
        return None

async def _apply_model_settings_for_asr_call(
    audio_duration_s: float, 
    target_processing_device: str, 
    request_id: str = "req"
) -> bool:
    """
    Apply appropriate model settings based on audio duration and target device.
    
    Returns:
        bool: True if long audio settings were applied, False otherwise
    """
    global asr_model
    if asr_model is None: 
        return False
    
    long_audio_applied_flag = False
    current_device = str(next(asr_model.parameters()).device)
    
    # Move model to target device if needed
    if current_device != target_processing_device: 
        await asyncio.to_thread(asr_model.to, device=target_processing_device)
    
    # Ensure model is in float32 precision
    if next(asr_model.parameters()).dtype != torch.float32: 
        await asyncio.to_thread(asr_model.to, dtype=torch.float32)
    
    # Apply long audio optimizations for lengthy audio
    if audio_duration_s > LONG_AUDIO_THRESHOLD_S:
        try:
            await asyncio.to_thread(
                asr_model.change_attention_model, "rel_pos_local_attn", [256, 256]
            )
            await asyncio.to_thread(asr_model.change_subsampling_conv_chunking_factor, 1)
            long_audio_applied_flag = True
        except Exception: 
            pass
    
    # Use bfloat16 on CUDA if supported for better performance
    if (target_processing_device == "cuda" and 
        torch.cuda.is_available() and 
        torch.cuda.is_bf16_supported()):
        if next(asr_model.parameters()).dtype != torch.bfloat16: 
            await asyncio.to_thread(asr_model.to, dtype=torch.bfloat16)
    
    return long_audio_applied_flag

async def _revert_model_to_global_original_state(
    long_audio_applied_for_call: bool, 
    processing_device_this_call: str, 
    request_id: str = "req"
):
    """
    Revert model to its original state after processing.
    
    Args:
        long_audio_applied_for_call: Whether long audio settings were applied
        processing_device_this_call: Device used for this processing call
        request_id: Request ID for logging
    """
    global asr_model
    if asr_model is None: 
        return
    
    try:
        # Move model back to CPU first
        if str(next(asr_model.parameters()).device) != "cpu": 
            await asyncio.to_thread(asr_model.cpu)
        
        # Revert long audio settings if they were applied
        if long_audio_applied_for_call: 
            try:
                await asyncio.to_thread(asr_model.change_attention_model, "rel_pos") 
                await asyncio.to_thread(asr_model.change_subsampling_conv_chunking_factor, -1) 
            except Exception: 
                pass
        
        # Restore original device and dtype
        if (str(next(asr_model.parameters()).device) != global_original_model_device_str or 
            next(asr_model.parameters()).dtype != global_original_model_dtype_torch):
            await asyncio.to_thread(
                asr_model.to, 
                device=global_original_model_device_str, 
                dtype=global_original_model_dtype_torch
            )
        
        # Clean up CUDA memory if it was used
        if processing_device_this_call == "cuda" and torch.cuda.is_available():
            await asyncio.to_thread(gc.collect)
            await asyncio.to_thread(torch.cuda.empty_cache)
            
    except Exception: 
        pass

async def _perform_asr_transcription(
    asr_model_instance: NeMoASRModelType, 
    audio_input_list: List[torch.Tensor],
    batch_size_for_transcribe_call: int, 
    num_asr_workers: int, 
    request_id: str = "asr"
) -> Tuple[Optional[List[Hypothesis]], float]:
    """
    Perform ASR transcription on a list of audio tensors.
    
    Returns:
        Tuple of (hypotheses_list, transcription_duration_seconds)
    """
    if not audio_input_list or not asr_model_instance: 
        return None, 0.0
    
    hypotheses: Optional[List[Hypothesis]] = None
    duration: float = 0.0
    
    # Get model device and move audio to same device
    try: 
        model_device = next(asr_model_instance.parameters()).device
    except Exception as e_dev: 
        logger.error(f"({request_id}) Error getting model device: {e_dev}")
        raise
    
    try: 
        audio_on_device = [audio.to(model_device) for audio in audio_input_list]
    except Exception as e_mov: 
        logger.error(f"({request_id}) Error moving audio to device: {e_mov}")
        raise
    
    transcribe_call_start_time = time.time()
    try:
        raw_hypotheses = await asyncio.to_thread(
            asr_model_instance.transcribe, 
            audio=audio_on_device, 
            batch_size=batch_size_for_transcribe_call,
            num_workers=num_asr_workers, 
            return_hypotheses=True, 
            timestamps=True, 
            verbose=False
        )
        duration = round(time.time() - transcribe_call_start_time, 3)
        
        # Process raw hypotheses into standardized format
        if raw_hypotheses:
            if all(isinstance(h, Hypothesis) for h in raw_hypotheses): 
                hypotheses = raw_hypotheses
            elif all(isinstance(h_list, list) for h_list in raw_hypotheses): 
                hypotheses = [h_list[0] for h_list in raw_hypotheses if h_list and isinstance(h_list[0], Hypothesis)]
            elif isinstance(raw_hypotheses, Hypothesis): 
                hypotheses = [raw_hypotheses]
        
        return hypotheses, duration
        
    except Exception as e_transcribe: 
        logger.error(f"({request_id}) Error in asr_model.transcribe: {e_transcribe}", exc_info=True)
        raise

def _process_hypotheses_to_segments(
    batch_hypotheses: List[Hypothesis],
    batch_offsets_s: List[float],
    request_id: str = "seg_proc"
) -> List[dict]:
    """
    Converts a list of Hypothesis objects and their corresponding time offsets
    into a flat list of segment dictionaries with global timestamps.
    
    Args:
        batch_hypotheses: List of NeMo Hypothesis objects
        batch_offsets_s: List of time offsets in seconds for each hypothesis
        request_id: Request ID for logging
        
    Returns:
        List of segment dictionaries with global timestamps
    """
    all_segments = []
    
    if (not batch_hypotheses or not batch_offsets_s or 
        len(batch_hypotheses) != len(batch_offsets_s)):
        logger.warning(
            f"({request_id}) Mismatched hypotheses and offsets or empty input. "
            f"Hyps: {len(batch_hypotheses) if batch_hypotheses else 0}, "
            f"Offsets: {len(batch_offsets_s) if batch_offsets_s else 0}"
        )
        return all_segments

    for hyp_idx, hyp_obj in enumerate(batch_hypotheses):
        if hyp_obj is None:
            logger.debug(f"({request_id}) Hypothesis at index {hyp_idx} is None, skipping.")
            continue
        
        chunk_offset_s = batch_offsets_s[hyp_idx]
        
        # Process timestamp data if available
        if (hasattr(hyp_obj, "timestamp") and 
            hyp_obj.timestamp and 
            isinstance(hyp_obj.timestamp, dict)):
            
            for seg_meta in hyp_obj.timestamp.get("segment", []):
                seg_text = seg_meta.get("segment", "").strip()
                if not seg_text:
                    continue
                
                start_time = round(seg_meta.get("start", 0.0) + chunk_offset_s, 3)
                end_time = round(seg_meta.get("end", 0.0) + chunk_offset_s, 3)

                # Basic validation for segment times
                if start_time < 0 or end_time < start_time:
                    logger.warning(
                        f"({request_id}) Invalid segment times: start={start_time}, "
                        f"end={end_time} for text='{seg_text}'. "
                        f"Original: {seg_meta}, Offset: {chunk_offset_s}. Skipping segment."
                    )
                    continue

                new_segment = {
                    "start": start_time,
                    "end": end_time,
                    "text": seg_text,
                    # Standard Whisper-like fields
                    "seek": 0, 
                    "tokens": [], 
                    "temperature": 0.0, 
                    "avg_logprob": None, 
                    "compression_ratio": None, 
                    "no_speech_prob": None
                }
                all_segments.append(new_segment)
        else:
            logger.debug(
                f"({request_id}) Hypothesis at index {hyp_idx} "
                f"(offset {chunk_offset_s}s) has no valid timestamp data."
            )

    return all_segments

def _deduplicate_segments(
    raw_segments: List[dict], 
    overlap_threshold_seconds: float = 0.3
) -> List[dict]:
    """
    Deduplicates a list of segment dictionaries that may overlap.
    Sorts segments and merges/replaces based on start/end times and text length.
    Assigns final 'id' to segments.
    
    Args:
        raw_segments: List of segment dictionaries to deduplicate
        overlap_threshold_seconds: Overlap threshold for considering segments for merging
        
    Returns:
        List of deduplicated segments with assigned IDs
    """
    if not raw_segments:
        return []

    # Sort by start time, then by end time as secondary criterion
    try:
        # Ensure all segments have required keys for sorting
        for i, seg_check in enumerate(raw_segments):
            if 'start' not in seg_check or 'end' not in seg_check:
                logger.error(
                    f"Deduplication: Segment at index {i} missing 'start' or 'end': {seg_check}"
                )
        
        # Sort segments chronologically
        sorted_segments = sorted(
            raw_segments, 
            key=lambda s: (s.get('start', float('inf')), s.get('end', float('inf')))
        )
    except TypeError as e_sort:
        logger.error(
            f"Error sorting segments for deduplication: {e_sort}. "
            f"Segments: {raw_segments[:5]}..."
        )
        return []

    final_deduped_segments: List[dict] = []
    prev_seg_final: Optional[dict] = None

    for current_seg in sorted_segments:
        # Ensure essential keys are present
        if not all(k in current_seg for k in ["start", "end", "text"]):
            logger.warning(f"Deduplication: Skipping segment with missing keys: {current_seg}")
            continue

        if (prev_seg_final is None or 
            current_seg["start"] >= prev_seg_final["end"] - overlap_threshold_seconds):
            # No overlap or minimal overlap, treat as new segment
            current_seg["id"] = len(final_deduped_segments)
            final_deduped_segments.append(current_seg)
            prev_seg_final = current_seg
        else:
            # Segments overlap significantly, decide whether to merge or replace
            current_duration = current_seg["end"] - current_seg["start"]
            prev_duration = prev_seg_final["end"] - prev_seg_final["start"]

            # Check if current segment is better or extends the previous one
            is_better_or_extends = current_seg["end"] > prev_seg_final["end"] - overlap_threshold_seconds/2
            
            # Avoid replacing with a much shorter segment unless it extends significantly
            if (is_better_or_extends and 
                current_duration > prev_duration * 0.7):
                current_seg["id"] = prev_seg_final["id"]
                final_deduped_segments[-1] = current_seg
                prev_seg_final = current_seg

    return final_deduped_segments

def parse_request_config(
    chunk_length: Optional[float] = None, 
    chunk_overlap: Optional[float] = None,
    batch_size: Optional[int] = None, 
    long_audio_threshold: Optional[float] = None
) -> dict:
    """
    Parse and validate request configuration parameters.
    
    Returns:
        Dictionary with validated configuration values
    """
    config = {
        "chunk_length": chunk_length if chunk_length is not None else TRANSCRIBE_CHUNK_LEN,
        "chunk_overlap": chunk_overlap if chunk_overlap is not None else TRANSCRIBE_OVERLAP,
        "batch_size": batch_size if batch_size is not None else CHUNKING_BATCH_SIZE,
        "long_audio_threshold": (long_audio_threshold if long_audio_threshold is not None 
                               else LONG_AUDIO_THRESHOLD_S),
    }
    
    # Validation
    if not (0 < config["chunk_length"] <= 300): 
        raise ValueError("chunk_length must be > 0 and <= 300s")
    if not (0 <= config["chunk_overlap"] < config["chunk_length"]): 
        raise ValueError("chunk_overlap must be >= 0 and < chunk_length")
    if not (1 <= config["batch_size"] <= 32): 
        raise ValueError("batch_size must be 1-32")
    if not (0 <= config["long_audio_threshold"] <= 3600): 
        raise ValueError("long_audio_threshold must be 0-3600s")
    
    return config

def parse_websocket_config(client_config: dict) -> dict:
    """
    Parse and validate WebSocket client configuration.
    
    Args:
        client_config: Client configuration dictionary
        
    Returns:
        Dictionary with validated configuration values
    """
    required_fields = ["sample_rate", "channels", "bytes_per_sample", "format"]
    missing = [f for f in required_fields if f not in client_config]
    if missing: 
        raise ValueError(f"Missing required WebSocket fields: {', '.join(missing)}")
    
    # Parse base configuration
    config = parse_request_config(
        chunk_length=client_config.get("chunk_length"),
        chunk_overlap=client_config.get("chunk_overlap"),
        batch_size=client_config.get("batch_size"),
        long_audio_threshold=client_config.get("long_audio_threshold"),
    )
    
    # Add WebSocket-specific fields
    config.update({
        "sample_rate": int(client_config["sample_rate"]),
        "channels": int(client_config["channels"]),
        "bytes_per_sample": int(client_config["bytes_per_sample"]),
        "format": str(client_config["format"]).lower(),
    })
    
    return config

async def handle_streaming_pcm(
    websocket: WebSocket, 
    session_id: str, 
    processing_device: str, 
    client_config: dict,
):
    """
    Handle streaming PCM audio processing for WebSocket connections.
    
    Args:
        websocket: WebSocket connection
        session_id: Unique session identifier
        processing_device: Device to use for processing (cuda/cpu)
        client_config: Client audio configuration
    """
    # Initialize buffers and tracking variables
    pcm_buffer = bytearray()
    sent_segments_pcm = []
    prev_segment_pcm = None
    chunks_processed_count = 0
    total_samples_processed_pcm = 0
    
    # Extract configuration parameters
    chunk_length = client_config["chunk_length"]
    chunk_overlap = client_config["chunk_overlap"]
    samples_per_chunk = int(chunk_length * MODEL_SAMPLE_RATE)
    samples_per_stride = int((chunk_length - chunk_overlap) * MODEL_SAMPLE_RATE)
    bytes_per_sample_for_processing_chunks = 2
    bytes_per_chunk = samples_per_chunk * bytes_per_sample_for_processing_chunks
    bytes_per_stride = samples_per_stride * bytes_per_sample_for_processing_chunks
    
    # Client audio format parameters
    client_format = client_config["format"]
    client_sr = client_config["sample_rate"]
    client_channels = client_config["channels"]
    client_bps = client_config["bytes_per_sample"]
    
    # Check if format conversion is needed
    needs_format_conversion = (
        client_format != "pcm" or 
        client_sr != MODEL_SAMPLE_RATE or 
        client_channels != 1 or 
        client_bps != 2
    )
    
    # Format conversion setup
    convert_buffer = bytearray()
    convert_threshold_bytes = 0
    if needs_format_conversion:
        estimated_bytes_for_one_asr_chunk_duration = int(
            client_sr * client_channels * client_bps * chunk_length
        )
        convert_threshold_bytes = (
            estimated_bytes_for_one_asr_chunk_duration 
            if estimated_bytes_for_one_asr_chunk_duration > 0 
            else 32768
        )

    # Producer-consumer setup
    chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=client_config["batch_size"] * 2)
    producer_done_event = asyncio.Event()
    accumulated_asr_processing_time_s = 0.0

    async def pcm_producer():
        """Producer coroutine that reads WebSocket data and creates audio chunks."""
        nonlocal total_samples_processed_pcm, chunks_processed_count, convert_threshold_bytes, pcm_buffer
        
        try:
            while websocket.application_state == WebSocketState.CONNECTED:
                try: 
                    message = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                except asyncio.TimeoutError: 
                    break
                
                # Check for end signal
                if 'text' in message and message['text'] == "END": 
                    break
                
                # Process audio bytes
                if 'bytes' in message and message['bytes']:
                    if needs_format_conversion:
                        convert_buffer.extend(message['bytes'])
                        
                        # Convert chunks when buffer is large enough
                        while len(convert_buffer) >= convert_threshold_bytes and convert_threshold_bytes > 0:
                            try:
                                chunk_to_convert = bytes(convert_buffer[:convert_threshold_bytes])
                                convert_buffer[:] = convert_buffer[convert_threshold_bytes:]
                                
                                converted_pcm_bytes = await convert_audio_chunk_to_target_pcm(
                                    chunk_to_convert, client_format, client_sr, 
                                    client_channels, client_bps, MODEL_SAMPLE_RATE, session_id
                                )
                                
                                if converted_pcm_bytes: 
                                    pcm_buffer.extend(converted_pcm_bytes)
                                else: 
                                    if convert_threshold_bytes > 8192: 
                                        convert_threshold_bytes //= 2
                            except Exception:
                                if convert_threshold_bytes > 8192: 
                                    convert_threshold_bytes //= 2
                                else: 
                                    convert_buffer[:] = convert_buffer[1024:] 
                    else: 
                        pcm_buffer.extend(message['bytes'])
                
                # Create chunks from buffer
                while len(pcm_buffer) >= bytes_per_chunk:
                    chunk_bytes = bytes(pcm_buffer[:bytes_per_chunk])
                    tensor = torch.frombuffer(chunk_bytes, dtype=torch.int16).float() / 32768.0
                    offset = total_samples_processed_pcm / MODEL_SAMPLE_RATE
                    await chunk_queue.put((tensor, offset))
                    pcm_buffer[:] = pcm_buffer[bytes_per_stride:]
                    total_samples_processed_pcm += samples_per_stride
                    chunks_processed_count += 1
            
            # Handle remaining conversion buffer
            if needs_format_conversion and convert_buffer:
                try:
                    converted_pcm_bytes = await convert_audio_chunk_to_target_pcm(
                        bytes(convert_buffer), client_format, client_sr, 
                        client_channels, client_bps, MODEL_SAMPLE_RATE, session_id
                    )
                    if converted_pcm_bytes: 
                        pcm_buffer.extend(converted_pcm_bytes)
                except Exception: 
                    pass
                convert_buffer.clear()
            
            # Handle remaining PCM buffer
            if len(pcm_buffer) > bytes_per_sample_for_processing_chunks * MODEL_SAMPLE_RATE * 0.5:
                tensor_tail = torch.frombuffer(bytes(pcm_buffer), dtype=torch.int16).float() / 32768.0
                await chunk_queue.put((tensor_tail, total_samples_processed_pcm / MODEL_SAMPLE_RATE))
            pcm_buffer.clear()
            
        except WebSocketDisconnect: 
            pass
        except Exception: 
            pass
        finally: 
            producer_done_event.set()
            await chunk_queue.put(None)

    async def pcm_consumer():
        """Consumer coroutine that processes audio chunks and generates transcriptions."""
        # `prev_segment_pcm` is no longer needed here for on-the-fly stitching
        # `sent_segments_pcm` will still accumulate all segments for the final CSV/SRT.
        nonlocal accumulated_asr_processing_time_s, sent_segments_pcm 
        
        consumer_batch_size_cap = client_config["batch_size"] # Max ASR input tensors per ASR call
        
        while True:
            batch_audio_tensors, batch_offsets, first_item_is_sentinel = [], [], False
            
            # Collect a batch of ASR input audio tensors from the queue,
            # up to consumer_batch_size_cap.
            # This loop processes what's available, even if less than the cap.
            for _ in range(consumer_batch_size_cap):
                timeout = 0.02 if batch_audio_tensors else None # Wait indefinitely for the first item, short timeout for subsequent
                item = None
                
                try:
                    if producer_done_event.is_set() and chunk_queue.empty() and not batch_audio_tensors:
                        try: item = chunk_queue.get_nowait()
                        except asyncio.QueueEmpty: break
                    else: 
                        item = await asyncio.wait_for(chunk_queue.get(), timeout=timeout)
                except (asyncio.TimeoutError, asyncio.QueueEmpty): 
                    # Timeout occurred or queue empty after getting at least one item, or queue initially empty for non-first.
                    break 
                
                if item is None: # Sentinel value from producer
                    if not batch_audio_tensors: # This was the first item fetched, and it's the sentinel
                        first_item_is_sentinel = True
                    chunk_queue.task_done()
                    break # Stop trying to collect for this batch
                
                tensor, offset = item
                batch_audio_tensors.append(tensor)
                batch_offsets.append(offset)
                chunk_queue.task_done()
                
                # If producer is done and queue is now empty, process what we have
                if producer_done_event.is_set() and chunk_queue.empty(): 
                    break
            
            # If no audio tensors were collected (e.g., first item was sentinel or initial timeout),
            # check if we should exit or continue waiting.
            if not batch_audio_tensors:
                if first_item_is_sentinel or (producer_done_event.is_set() and chunk_queue.empty()): 
                    logger.debug(f"({session_id}) PCM Consumer: No more audio tensors to process. Exiting consumer.")
                    break # Exit the main while loop
                else: 
                    # This case should be rare if timeout for first item is None,
                    # unless producer_done_event is set and queue becomes empty.
                    logger.debug(f"({session_id}) PCM Consumer: No audio tensors in current attempt, but producer not done or queue not confirmed empty. Continuing.")
                    continue # Go back to waiting for items from the queue
            
            # Perform ASR transcription on the collected batch of audio tensors
            logger.info(f"({session_id}) PCM Consumer: Processing batch of {len(batch_audio_tensors)} ASR input tensor(s).")
            hyps_list, asr_call_duration = await _perform_asr_transcription(
                asr_model_instance=asr_model, 
                audio_input_list=batch_audio_tensors,
                batch_size_for_transcribe_call=len(batch_audio_tensors), # Pass all collected tensors to NeMo
                num_asr_workers=NUM_WORKERS,
                request_id=f"WS-PCM-{session_id}-b" # Log a batch ID if useful
            )
            accumulated_asr_processing_time_s += asr_call_duration

            if processing_device == "cuda" and torch.cuda.is_available(): 
                await asyncio.to_thread(torch.cuda.empty_cache)
            
            segments_generated_this_asr_call = []
            if hyps_list:
                segments_generated_this_asr_call = _process_hypotheses_to_segments(
                    hyps_list, batch_offsets, session_id # batch_offsets correspond to batch_audio_tensors
                )
                logger.info(f"({session_id}) PCM Consumer: ASR call yielded {len(segments_generated_this_asr_call)} segments.")


            # --- MODIFIED SEGMENT SENDING LOGIC ---
            if segments_generated_this_asr_call:
                if websocket.application_state == WebSocketState.CONNECTED:
                    # Segments are already chronologically ordered by _process_hypotheses_to_segments
                    # if hypotheses and offsets are ordered.
                    # No temporary IDs needed here unless client specifically expects them per intermediate batch.
                    await websocket.send_json({
                        "type": "segments_batch", # Mirroring full file upload message type
                        "segments": segments_generated_this_asr_call
                    })
                    logger.debug(f"({session_id}) PCM Consumer: Sent segments_batch with {len(segments_generated_this_asr_call)} segments.")
                
                # Accumulate all segments for the final SRT/CSV in the final_transcription message
                sent_segments_pcm.extend(segments_generated_this_asr_call)
            # --- END OF MODIFIED SEGMENT SENDING LOGIC ---
            
            # Check for completion (this condition is met if 'item is None' was processed,
            # and this was the last batch to clear out the queue after producer was done).
            if (producer_done_event.is_set() and 
                chunk_queue.empty() and 
                first_item_is_sentinel and # Ensures we recognized the sentinel
                not batch_audio_tensors): # And no actual data was in this final processing loop
                logger.debug(f"({session_id}) PCM Consumer: Post-processing sentinel check, exiting consumer.")
                break


    try:
        # Run producer and consumer concurrently
        await asyncio.gather(pcm_producer(), pcm_consumer())
        
        # Send final transcription
        final_text_pcm = " ".join(s["text"] for s in sent_segments_pcm).strip()
        if websocket.application_state == WebSocketState.CONNECTED:
            final_msg_pcm = {
                "type": "final_transcription",
                "text": final_text_pcm,
                "language": "en",
                "transcription_time": round(accumulated_asr_processing_time_s, 3),
                "total_segments": len(sent_segments_pcm),
                "final_duration_processed_seconds": round(total_samples_processed_pcm / MODEL_SAMPLE_RATE, 3),
                "csv_content": generate_csv_content(sent_segments_pcm),
                "srt_content": generate_srt_content(sent_segments_pcm),
                "streaming_mode": "pcm"
            }
            await websocket.send_json(final_msg_pcm)
            
    except Exception as e_pcm_pipeline:
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.send_json({"type": "error", "error": str(e_pcm_pipeline)})
    finally:
        if not producer_done_event.is_set(): 
            producer_done_event.set()
        try:
            chunk_queue.put_nowait(None) 
        except: 
            pass

def create_audio_chunks(
    waveform: torch.Tensor, 
    sample_rate: int = MODEL_SAMPLE_RATE, 
    chunk_len_s: float = TRANSCRIBE_CHUNK_LEN, 
    overlap_s: float = TRANSCRIBE_OVERLAP
):
    """
    Create overlapping audio chunks from a waveform tensor.
    
    Args:
        waveform: 1D audio tensor
        sample_rate: Audio sample rate
        chunk_len_s: Chunk length in seconds
        overlap_s: Overlap between chunks in seconds
        
    Returns:
        Tuple of (chunks_list, offsets_list)
    """
    assert waveform.ndim == 1
    total_dur = waveform.shape[0] / sample_rate
    stride = chunk_len_s - overlap_s
    
    if stride <= 0: 
        raise ValueError("Overlap must be < chunk length")
    
    chunks, offsets = [], []
    cur = 0.0
    
    while cur < total_dur:
        start_s = max(0.0, cur - overlap_s)
        end_s = min(total_dur, cur + chunk_len_s)
        s_idx, e_idx = int(start_s * sample_rate), int(end_s * sample_rate)
        
        if e_idx - s_idx <= 0: 
            if cur + stride >= total_dur: 
                break
            cur += stride
            continue
            
        chunk = waveform[s_idx:e_idx].clone()
        if chunk.numel(): 
            chunks.append(chunk)
            offsets.append(start_s) 
            
        if end_s >= total_dur: 
            break
        cur += stride
        if cur >= total_dur: 
            break
            
    return chunks, offsets

@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(
    file: UploadFile = File(...),
    chunk_length: Optional[float] = Query(None),
    chunk_overlap: Optional[float] = Query(None),
    batch_size: Optional[int] = Query(None),
    long_audio_threshold: Optional[float] = Query(None)
):
    """REST endpoint for audio transcription."""
    if not asr_model: 
        return JSONResponse(
            status_code=503,
            content={"error": "ASR model not available."}
        )
    
    request_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    
    try: 
        config = parse_request_config(chunk_length, chunk_overlap, batch_size, long_audio_threshold)
    except ValueError as e_cfg: 
        return JSONResponse(status_code=400, content={"error": str(e_cfg)})
    
    temp_uploaded_file_path: str = ""
    long_audio_settings_were_applied = False
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"
    request_start_time = time.time()
    response_content_dict: Optional[JSONResponse] = None

    try:
        async with model_access_lock:
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=Path(file.filename if file.filename else ".unk").suffix,
                    prefix=f"rest_{request_id}_"
                ) as tmp:
                    await asyncio.to_thread(shutil.copyfileobj, file.file, tmp)
                    temp_uploaded_file_path = tmp.name
                
                # Load and preprocess audio
                waveform_tensor, audio_duration_s = await load_and_preprocess_audio(
                    temp_uploaded_file_path, MODEL_SAMPLE_RATE, request_id
                )
                
                if waveform_tensor is None: 
                    response_content_dict = JSONResponse(
                        status_code=400,
                        content={"error": "Audio processing failed."}
                    )
                    raise ValueError("Audio loading failed")
                
                # Apply model settings
                long_audio_settings_were_applied = await _apply_model_settings_for_asr_call(
                    audio_duration_s, processing_device, request_id
                )
                
                if asr_model is None: 
                    response_content_dict = JSONResponse(
                        status_code=503,
                        content={"error": "ASR model unavailable."}
                    )
                    raise RuntimeError("ASR Model None")
                
                # Perform transcription
                hypotheses_list, transcribe_asr_call_duration = await _perform_asr_transcription(
                    asr_model, [waveform_tensor], config["batch_size"], NUM_WORKERS, f"REST-{request_id}"
                )
                
                # Process segments
                segments_result = []
                if hypotheses_list:
                    segments_result = _process_hypotheses_to_segments(
                        hypotheses_list, [0.0] * len(hypotheses_list), request_id
                    )
                
                text_result = " ".join(s['text'] for s in segments_result).strip()
                total_server_request_duration = round(time.time() - request_start_time, 3)
                
                response_payload = {
                    "text": text_result,
                    "segments": segments_result,
                    "language": "en",
                    "transcription_time": transcribe_asr_call_duration,
                    "total_request_time_server": total_server_request_duration,
                    "csv_content": generate_csv_content(segments_result),
                    "srt_content": generate_srt_content(segments_result)
                }
                response_content_dict = JSONResponse(content=response_payload)
                
            except Exception as e_locked:
                if response_content_dict is None: 
                    detail = str(e_locked)
                    if "CUDA out of memory" in detail: 
                        err_msg = "CUDA OOM."
                        response_content_dict = JSONResponse(
                            status_code=500,
                            content={"error": err_msg, "detail": detail}
                        )
                    elif "Audio loading failed" in detail: 
                        err_msg = "Audio load fail."
                        response_content_dict = JSONResponse(
                            status_code=400,
                            content={"error": err_msg, "detail": detail}
                        )
                    else: 
                        response_content_dict = JSONResponse(
                            status_code=500,
                            content={"error": "Server error.", "detail": detail}
                        )
            finally: 
                await _revert_model_to_global_original_state(
                    long_audio_settings_were_applied, processing_device, request_id
                )
                
        if response_content_dict: 
            return response_content_dict
        return JSONResponse(status_code=500, content={"error": "Internal server error."})
        
    except Exception as e_outer: 
        return JSONResponse(
            status_code=500,
            content={"error": "Unexpected server error.", "detail": str(e_outer)}
        )
    finally:
        if temp_uploaded_file_path and os.path.exists(temp_uploaded_file_path):
            try:
                await asyncio.to_thread(os.remove, temp_uploaded_file_path)
            except: 
                pass

@app.websocket("/v1/audio/transcriptions/ws")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio transcription.
    
    Supports two modes:
    1. Streaming PCM audio for real-time transcription
    2. Full file upload for batch processing
    """
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    await websocket.accept()
    
    # Check if ASR model is available
    if not asr_model: 
        await websocket.send_json({
            "type": "error", 
            "error": "ASR model not available."
        })
        await websocket.close(1011)
        return

    # Initialize session variables
    processing_device_ws = "cuda" if torch.cuda.is_available() else "cpu"
    long_audio_settings_applied_for_this_session = False
    ws_response_final_dict: Optional[dict] = None
    client_audio_fmt_parsed_from_config = "unknown"

    try:
        # Parse client configuration
        client_config = parse_websocket_config(
            json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=20.0))
        )
        client_audio_fmt_parsed_from_config = client_config["format"]
        
        async with model_access_lock:
            try:
                if client_config["format"] == "pcm":
                    # Handle streaming PCM audio
                    await handle_streaming_pcm(
                        websocket, session_id, processing_device_ws, client_config
                    )
                else:
                    # Handle full file upload mode
                    accumulated_raw_bytes_ws = bytearray()
                    
                    # Collect audio data
                    while True: 
                        if websocket.application_state != WebSocketState.CONNECTED: 
                            raise WebSocketDisconnect(1001, "Client disconnected")
                        
                        try:
                            message = await asyncio.wait_for(websocket.receive(), timeout=60.0) 
                            
                            if message.get("type") == "websocket.disconnect": 
                                raise WebSocketDisconnect(message.get('code', 1000))
                            
                            if 'text' in message and message['text']:
                                if message['text'].upper() == "END": 
                                    break 
                                elif client_config.get("format") == "base64": 
                                    try:
                                        accumulated_raw_bytes_ws.extend(
                                            base64.b64decode(message['text'])
                                        )
                                    except: 
                                        pass
                            elif 'bytes' in message and message['bytes']: 
                                accumulated_raw_bytes_ws.extend(message['bytes'])
                                
                        except asyncio.TimeoutError: 
                            raise WebSocketDisconnect(1008, "Timeout before END")
                    
                    # Validate collected audio data
                    if not accumulated_raw_bytes_ws:
                        if websocket.application_state == WebSocketState.CONNECTED: 
                            await websocket.send_json({
                                "type": "error", 
                                "error": "No audio data."
                            })
                        raise ValueError("No audio for WS Full File Upload")
                    
                    # Load and preprocess audio
                    waveform_ws_tensor, audio_duration_s_ws = await load_and_preprocess_audio(
                        io.BytesIO(bytes(accumulated_raw_bytes_ws)), MODEL_SAMPLE_RATE, session_id
                    )
                    accumulated_raw_bytes_ws.clear()
                    
                    if waveform_ws_tensor is None: 
                        raise ValueError("WS Full File: Audio loading failed.")
                    
                    # Apply model settings for long audio if needed
                    long_audio_settings_applied_for_this_session = await _apply_model_settings_for_asr_call(
                        audio_duration_s_ws, processing_device_ws, session_id
                    )
                    
                    if asr_model is None: 
                        raise RuntimeError("ASR Model None.")
                    
                    # Create audio chunks for processing
                    manual_chunks_ws, manual_offsets_ws = create_audio_chunks(
                        waveform_ws_tensor, MODEL_SAMPLE_RATE, 
                        client_config["chunk_length"], client_config["chunk_overlap"]
                    )
                    
                    all_raw_segments_from_batches = []
                    asr_total_processing_time_this_mode = 0.0
                    batch_size_cfg = client_config["batch_size"]
                    
                    # Process chunks in batches
                    for batch_start_idx in range(0, len(manual_chunks_ws), batch_size_cfg):
                        if websocket.application_state != WebSocketState.CONNECTED: 
                            break
                        
                        current_batch_tensors = manual_chunks_ws[
                            batch_start_idx:batch_start_idx + batch_size_cfg
                        ]
                        current_batch_offsets = manual_offsets_ws[
                            batch_start_idx:batch_start_idx + batch_size_cfg
                        ]
                        
                        if not current_batch_tensors: 
                            continue
                        
                        # Perform ASR transcription for this batch
                        hypotheses_batch, batch_duration = await _perform_asr_transcription(
                            asr_model, current_batch_tensors, len(current_batch_tensors), 
                            NUM_WORKERS, f"WS-Full-{session_id}-b{batch_start_idx//batch_size_cfg}"
                        )
                        asr_total_processing_time_this_mode += batch_duration
                        
                        # Process segments from this batch
                        if hypotheses_batch:
                            segments_from_this_batch = _process_hypotheses_to_segments(
                                hypotheses_batch, current_batch_offsets, session_id
                            )
                            
                            if (websocket.application_state == WebSocketState.CONNECTED and 
                                segments_from_this_batch):
                                # Sort segments chronologically and assign temporary IDs
                                segments_from_this_batch.sort(key=lambda s: s["start"])
                                for idx_s, seg_s in enumerate(segments_from_this_batch): 
                                    seg_s["id"] = idx_s 
                                
                                await websocket.send_json({
                                    "type": "segments_batch",
                                    "segments": segments_from_this_batch
                                })
                                all_raw_segments_from_batches.extend(segments_from_this_batch)
                        
                        # Clean up CUDA memory if used
                        if processing_device_ws == "cuda" and torch.cuda.is_available(): 
                            await asyncio.to_thread(torch.cuda.empty_cache)
                    
                    # Deduplicate and finalize segments
                    final_deduped_segments = _deduplicate_segments(all_raw_segments_from_batches)
                    final_text = " ".join(s['text'] for s in final_deduped_segments).strip()
                    
                    ws_response_final_dict = {
                        "type": "final_transcription",
                        "text": final_text,
                        "language": "en",
                        "transcription_time": round(asr_total_processing_time_this_mode, 3),
                        "total_segments": len(final_deduped_segments),
                        "final_duration_processed_seconds": round(audio_duration_s_ws, 3),
                        "csv_content": generate_csv_content(final_deduped_segments),
                        "srt_content": generate_srt_content(final_deduped_segments)
                    }
                    
            except Exception as e_locked_ws: 
                if websocket.application_state == WebSocketState.CONNECTED:
                    detail = str(e_locked_ws)
                    err_type = "CUDA OOM" if "CUDA out of memory" in detail else "Server error"
                    await websocket.send_json({
                        "type": "error", 
                        "error": f"{err_type}: {detail}"
                    })
            finally: 
                await _revert_model_to_global_original_state(
                    long_audio_settings_applied_for_this_session, 
                    processing_device_ws, 
                    session_id
                )
        
        # Send final response for non-PCM modes
        if (client_audio_fmt_parsed_from_config != "pcm" and 
            ws_response_final_dict and 
            websocket.application_state == WebSocketState.CONNECTED):
            await websocket.send_json(ws_response_final_dict)
            
    except asyncio.TimeoutError: 
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.send_json({
                "type": "error", 
                "error": "Config timeout."
            })
    except json.JSONDecodeError as e_json:
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.send_json({
                "type": "error", 
                "error": str(e_json)
            })
    except ValueError as e_val: 
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.send_json({
                "type": "error", 
                "error": str(e_val)
            })
    except WebSocketDisconnect: 
        pass
    except Exception as e_outer: 
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.send_json({
                "type": "error", 
                "error": "Unhandled exception."
            })
    finally:
        if websocket.application_state == WebSocketState.CONNECTED: 
            await websocket.close(1000)

if __name__ == "__main__":
    import uvicorn
    if not asr_model: 
        logger.critical(f"ASR Model ('{ASR_MODEL_NAME}') failed. Server cannot start.")
    else: 
        uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)