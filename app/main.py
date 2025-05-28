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

from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import torchaudio  # faster audio I/O for long files
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models.asr_model import ASRModel as NeMoASRModelType
from omegaconf import OmegaConf # Keep for config manipulation

from dotenv import load_dotenv

from utils import (
    generate_srt_content,
    generate_csv_content,
    apply_model_settings_for_request,
    revert_model_settings_after_request,
    global_original_model_device_str,
    global_original_model_dtype_torch,
)

load_dotenv()

# --- Logging Configuration ---
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("parakeet-asr.main")
logger.info(f"Logging configured with level: {log_level_str}")

# --- App Configuration ---
REST_BATCH_SIZE = 1
NUM_WORKERS = int(os.getenv("NUM_WORKERS", 0))
MODEL_SAMPLE_RATE = 16000
PORT = int(os.getenv("PORT", 8777))
ASR_MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
LONG_AUDIO_THRESHOLD_S = float(os.getenv("LONG_AUDIO_THRESHOLD", 480.0))

# --- Chunking Configuration ---
TRANSCRIBE_CHUNK_LEN = float(os.getenv("TRANSCRIBE_CHUNK_LEN", 30.0))
TRANSCRIBE_OVERLAP = float(os.getenv("TRANSCRIBE_OVERLAP", 5.0))
CHUNKING_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1))

# Log configuration
logger.info(
    f"Configuration loaded:\n"
    f"  Chunking: length={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s, batch={CHUNKING_BATCH_SIZE}\n"
    f"  App: rest_batch={REST_BATCH_SIZE}, workers={NUM_WORKERS}, "
    f"sample_rate={MODEL_SAMPLE_RATE}, port={PORT}\n"
    f"  Model: {ASR_MODEL_NAME}, long_audio_threshold={LONG_AUDIO_THRESHOLD_S}s"
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Static Files ---
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, "static")

# Ensure directory exists – Dockerfile creates it but be defensive for local runs
os.makedirs(static_dir, exist_ok=True)
logger.info(f"Static files directory set to: {static_dir}")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index_page():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path) if os.path.exists(index_path) else \
           HTMLResponse(f"UI not found at {index_path}.", status_code=404)

# --- ASR Model Loading & Global State ---
asr_model: NeMoASRModelType = None
original_model_attention_config_dict: dict = None # Store as plain dict
original_model_subsampling_config_dict: dict = None # Store as plain dict
global_original_model_device_str: str = "cpu" # Default to CPU if model fails to load
global_original_model_dtype_torch: torch.dtype = torch.float32 # Default

try:
    logger.info(f"Loading ASR model: {ASR_MODEL_NAME}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=ASR_MODEL_NAME)
    asr_model.preprocessor.featurizer.dither = 0.0
    asr_model.eval()
    
    # Store original configurations precisely
    if hasattr(asr_model, 'cfg') and hasattr(asr_model.cfg, 'encoder'):
        if hasattr(asr_model.cfg.encoder, 'attention'):
            # Convert OmegaConf to plain dict for storage to avoid issues with live OmegaConf objects
            original_model_attention_config_dict = OmegaConf.to_container(asr_model.cfg.encoder.attention, resolve=True)
        if hasattr(asr_model.cfg.encoder, 'conv_subsampling'):
            original_model_subsampling_config_dict = OmegaConf.to_container(asr_model.cfg.encoder.conv_subsampling, resolve=True)

    global_original_model_device_str = str(next(asr_model.parameters()).device)
    global_original_model_dtype_torch = next(asr_model.parameters()).dtype
    logger.info(f"ASR model '{ASR_MODEL_NAME}' loaded. Original device: {global_original_model_device_str}, dtype: {global_original_model_dtype_torch}. Configs stored if found.")
except Exception as e:
    logger.critical(f"FATAL: Could not load ASR model '{ASR_MODEL_NAME}'. Error: {e}", exc_info=True)
    asr_model = None

model_access_lock = asyncio.Lock()


def create_audio_chunks(waveform: torch.Tensor, sample_rate: int = MODEL_SAMPLE_RATE, chunk_len_s: float = TRANSCRIBE_CHUNK_LEN, overlap_s: float = TRANSCRIBE_OVERLAP):
    """Split 1-D mono waveform into overlapping chunks and return (chunks, offsets_s)."""
    assert waveform.ndim == 1, "waveform must be 1-D mono"
    total_dur = waveform.shape[0] / sample_rate
    stride = chunk_len_s - overlap_s
    if stride <= 0:
        raise ValueError("OVERLAP must be smaller than CHUNK_LENGTH")

    chunks, offsets = [], []
    cur = 0.0
    while cur < total_dur:
        start_s = max(0.0, cur - overlap_s)
        end_s = min(total_dur, cur + chunk_len_s)
        s_idx = int(start_s * sample_rate)
        e_idx = int(end_s * sample_rate)
        if s_idx >= e_idx:
            break
        chunk = waveform[s_idx:e_idx].clone()
        if chunk.numel():
            chunks.append(chunk)
            offsets.append(start_s)
        cur += stride
    return chunks, offsets

# --- REST Endpoint ---
@app.post("/v1/audio/transcriptions")
async def transcribe_endpoint_rest(file: UploadFile = File(...)):
    if not asr_model:
        return JSONResponse(status_code=503, content={"error": "ASR model not available."})

    request_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    logger.info(f"REST ({request_id}): Received request for '{file.filename}'.")

    temp_uploaded_file_path: str = ""
    long_audio_settings_were_applied = False # For this request
    device_dtype_was_changed_from_global = False # For this request
    
    processing_device_for_this_req = "cuda" if torch.cuda.is_available() else "cpu"
    request_processing_start_time = time.time() 

    try:
        async with model_access_lock: 
            logger.debug(f"REST ({request_id}): Acquired model_access_lock.")
            
            # 1. Save & Preprocess audio (torchaudio)
            file_suffix = Path(file.filename).suffix if file.filename else ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix, prefix=f"rest_up_{request_id}_") as tmp_upload:
                await asyncio.to_thread(shutil.copyfileobj, file.file, tmp_upload)
                temp_uploaded_file_path = tmp_upload.name

            try:
                waveform, sr_orig = await asyncio.to_thread(torchaudio.load, temp_uploaded_file_path)
                if sr_orig != MODEL_SAMPLE_RATE: # resample if needed: model expects 16kHz sample rate
                    waveform = await asyncio.to_thread(torchaudio.functional.resample, waveform, sr_orig, MODEL_SAMPLE_RATE)

                # Convert to mono 1-D tensor: model expects mono audio
                if waveform.ndim > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                elif waveform.ndim == 2 and waveform.shape[0] == 1:
                    waveform = waveform.squeeze(0)

                audio_duration_s = waveform.shape[-1] / MODEL_SAMPLE_RATE # duration in seconds = samples / sample rate
                logger.info(f"REST ({request_id}): torchaudio loaded '{file.filename}'. Duration: {audio_duration_s:.2f}s.")
            
            except Exception as ta_err:
                logger.error(f"REST ({request_id}): torchaudio failed to decode audio: {ta_err}", exc_info=True)
                return JSONResponse(status_code=400, content={"error": "Unsupported or corrupt audio file.", "detail": str(ta_err)})

            # 2. Apply model settings (device, dtype, long audio)
            long_audio_settings_were_applied, device_dtype_was_changed_from_global = await apply_model_settings_for_request(
                asr_model, audio_duration_s, processing_device_for_this_req, LONG_AUDIO_THRESHOLD_S, request_id
            )

            # 3. Transcribe using NeMo's high-level API
            logger.info(f"REST ({request_id}): Starting NeMo transcription for '{temp_uploaded_file_path}'.")
            transcribe_call_start_time = time.time()
            hypotheses = await asyncio.to_thread(
                asr_model.transcribe,
                audio=[waveform],
                batch_size=REST_BATCH_SIZE,
                num_workers=NUM_WORKERS,
                return_hypotheses=True,
                timestamps=True
            )
            transcribe_call_duration = round(time.time() - transcribe_call_start_time, 3)
            logger.info(f"REST ({request_id}): NeMo transcription finished in {transcribe_call_duration}s.")

            # 4. Process results
            text_result, segments_result = "", []
            if hypotheses and isinstance(hypotheses, list) and len(hypotheses) > 0:
                hypothesis = hypotheses[0]
                text_result = hypothesis.text.strip() if hasattr(hypothesis, 'text') else ""
                if hasattr(hypothesis, 'timestamp') and hypothesis.timestamp and 'segment' in hypothesis.timestamp:
                    for i, seg_meta in enumerate(hypothesis.timestamp['segment']):
                        segments_result.append({
                            "id": i, "start": round(seg_meta["start"], 3), "end": round(seg_meta["end"], 3),
                            "text": seg_meta.get("segment", "").strip(),
                            "seek": 0, "tokens": [], "temperature": 0.0, 
                            "avg_logprob": None, "compression_ratio": None, "no_speech_prob": None
                        })
            if not text_result and segments_result:
                text_result = " ".join(s['text'] for s in segments_result).strip()

            csv_content = generate_csv_content(segments_result)
            srt_content = generate_srt_content(segments_result)
            
            total_request_duration = round(time.time() - request_processing_start_time, 3)
            response = {
                "text": text_result, "segments": segments_result, "language": "en",
                "transcription_time": transcribe_call_duration, # More accurate: actual ASR time
                "total_request_time_server": total_request_duration, # Includes overhead
                "csv_content": csv_content, "srt_content": srt_content
            }
            return JSONResponse(content=response)

    except RuntimeError as e_rt: 
        if "CUDA out of memory" in str(e_rt): # Specific OOM
            logger.error(f"REST ({request_id}): CUDA OOM: {e_rt}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": "CUDA out of memory.", "detail": str(e_rt)})
        logger.error(f"REST ({request_id}): Runtime error: {e_rt}", exc_info=True) # Other runtime errors
        return JSONResponse(status_code=500, content={"error": "Server runtime error during transcription.", "detail": str(e_rt)})
    
    except Exception as e_main:
        logger.error(f"REST ({request_id}): General error for '{file.filename}': {e_main}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": "Unexpected server error.", "detail": str(e_main)})
    
    finally:
        # Revert model settings and clean up resources
        await revert_model_settings_after_request(
            asr_model, long_audio_settings_were_applied, device_dtype_was_changed_from_global, 
            processing_device_for_this_req, request_id
        )
        for p in [temp_uploaded_file_path]:
            if p and os.path.exists(p):
                try: await asyncio.to_thread(os.remove, p)
                except Exception as e_del: logger.error(f"REST ({request_id}): Error deleting temp file {p}: {e_del}")
        logger.debug(f"REST ({request_id}): Releasing model_access_lock and cleanup complete.")


# --- WebSocket Endpoint ---
@app.websocket("/v1/audio/transcriptions/ws")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    session_id = base64.urlsafe_b64encode(os.urandom(6)).decode()
    logger.info(f"WS ({session_id}): Connection from {websocket.client.host}:{websocket.client.port}")
    await websocket.accept()

    if not asr_model:
        await websocket.send_json({"type": "error", "error": "ASR model not available."})
        await websocket.close(code=1011); return

    accumulated_raw_bytes = bytearray()
    temp_uploaded_file_path: str = ""
    long_audio_settings_were_applied_ws = False
    device_dtype_was_changed_from_global_ws = False
    processing_device_for_this_req_ws = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        raw_config_msg = await asyncio.wait_for(websocket.receive_text(), timeout=20.0)
        client_config = json.loads(raw_config_msg)
        client_sr = client_config.get("sample_rate", MODEL_SAMPLE_RATE)
        client_ch = client_config.get("channels", 1)
        client_bps = client_config.get("bytes_per_sample", 2)
        client_audio_fmt = client_config.get("format", "binary").lower()
        logger.info(f"WS ({session_id}): Config: SR={client_sr}, Ch={client_ch}, BPS={client_bps}, Format={client_audio_fmt}")

        # 1. Accumulate audio data from client
        while True: 
            if websocket.application_state != WebSocketState.CONNECTED: break
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=60.0)
                if 'bytes' in message and message['bytes']:
                    accumulated_raw_bytes.extend(message['bytes'])
                elif 'text' in message and message['text']:
                    if message['text'].upper() == "END":
                        logger.info(f"WS ({session_id}): END signal. Total bytes: {len(accumulated_raw_bytes)}.")
                        break
                    elif client_audio_fmt == "base64": accumulated_raw_bytes.extend(base64.b64decode(message['text']))
            except asyncio.TimeoutError: logger.warning(f"WS ({session_id}): Receive timeout, assuming end of stream."); break
            except WebSocketDisconnect: logger.info(f"WS ({session_id}): Client disconnected during accumulation."); break
        
        if not accumulated_raw_bytes:
            await websocket.send_json({"type": "error", "error": "No audio data received by server."}); return

        # 2. Process accumulated audio under lock
        async with model_access_lock:
            logger.debug(f"WS ({session_id}): Acquired model_access_lock for processing.")

            # torchaudio decode & chunking pipeline
            waveform_ws, sr_orig_ws = await asyncio.to_thread(torchaudio.load, io.BytesIO(bytes(accumulated_raw_bytes)))
            accumulated_raw_bytes.clear()

            if sr_orig_ws != MODEL_SAMPLE_RATE:
                waveform_ws = await asyncio.to_thread(torchaudio.functional.resample, waveform_ws, sr_orig_ws, MODEL_SAMPLE_RATE)
            
            # convert to mono 1-D tensor
            if waveform_ws.ndim > 1 and waveform_ws.shape[0] > 1:
                waveform_ws = waveform_ws.mean(dim=0)
            elif waveform_ws.ndim == 2 and waveform_ws.shape[0] == 1:
                waveform_ws = waveform_ws.squeeze(0)

            audio_duration_s_ws = waveform_ws.shape[-1] / MODEL_SAMPLE_RATE
            logger.info(f"WS ({session_id}): Loaded stream via torchaudio. Duration: {audio_duration_s_ws:.2f}s.")

            # Create overlapping chunks
            chunks_ws, offsets_ws = create_audio_chunks(waveform_ws)
            logger.info(f"WS ({session_id}): Created {len(chunks_ws)} chunks (len={TRANSCRIBE_CHUNK_LEN}s, overlap={TRANSCRIBE_OVERLAP}s)")

            # Apply model settings (device/dtype/long-audio)
            long_audio_settings_were_applied_ws, device_dtype_was_changed_from_global_ws = await apply_model_settings_for_request(
                asr_model, audio_duration_s_ws, processing_device_for_this_req_ws, LONG_AUDIO_THRESHOLD_S, session_id
            )

            sent_segments = []
            prev_segment = None
            asr_start_ws = time.time()  # Start timing pure ASR processing

            for batch_start in range(0, len(chunks_ws), CHUNKING_BATCH_SIZE):
                batch_end = min(batch_start + CHUNKING_BATCH_SIZE, len(chunks_ws))
                batch_chunks = chunks_ws[batch_start:batch_end]
                batch_offsets = offsets_ws[batch_start:batch_end]

                hyps_batch = await asyncio.to_thread(
                    asr_model.transcribe,
                    audio=batch_chunks,
                    batch_size=len(batch_chunks),
                    return_hypotheses=True,
                    timestamps=True,
                    verbose=False,
                    num_workers=NUM_WORKERS,
                )

                for idx, hyp in enumerate(hyps_batch):
                    offset = batch_offsets[idx]
                    if hasattr(hyp, "timestamp") and hyp.timestamp and "segment" in hyp.timestamp:
                        for seg_meta in hyp.timestamp["segment"]:
                            seg_text = seg_meta.get("segment", "").strip()
                            if not seg_text:
                                continue
                            seg = {
                                "start": round(seg_meta["start"] + offset, 3),
                                "end": round(seg_meta["end"] + offset, 3),
                                "text": seg_text,
                                "seek": 0,
                                "tokens": [],
                                "temperature": 0.0,
                                "avg_logprob": None,
                                "compression_ratio": None,
                                "no_speech_prob": None,
                            }

                            if prev_segment is None or seg["start"] >= prev_segment["end"] - 0.3:
                                seg["id"] = len(sent_segments)
                                if websocket.application_state == WebSocketState.CONNECTED:
                                    await websocket.send_json({"type": "segment", **seg})
                                sent_segments.append(seg)
                                prev_segment = seg
                            else:
                                curr_dur = seg["end"] - seg["start"]
                                prev_dur = prev_segment["end"] - prev_segment["start"]
                                if curr_dur > prev_dur:
                                    seg["id"] = prev_segment["id"]
                                    if websocket.application_state == WebSocketState.CONNECTED:
                                        await websocket.send_json({"type": "segment", **seg})
                                    sent_segments[-1] = seg
                                    prev_segment = seg

            # Calculate timing metrics
            asr_end_ws = time.time()
            transcribe_call_duration_ws = round(asr_end_ws - asr_start_ws, 3)  # Pure ASR time
            
            # Prepare final transcription text
            text_ws = " ".join(s["text"] for s in sent_segments).strip()

            if websocket.application_state == WebSocketState.CONNECTED:
                csv_ws = generate_csv_content(sent_segments)
                srt_ws = generate_srt_content(sent_segments)
                
                final_response = {
                    "type": "final_transcription",
                    "text": text_ws,
                    "language": "en",
                    "transcription_time": transcribe_call_duration_ws,  # Pure ASR time only
                    "total_segments": len(sent_segments),
                    "final_duration_processed_seconds": round(audio_duration_s_ws, 3),
                    "csv_content": csv_ws,
                    "srt_content": srt_ws,
                }
                    
                await websocket.send_json(final_response)
                
            logger.info(f"WS ({session_id}): Streaming processing complete. Segments: {len(sent_segments)}. "
                       f"ASR time: {transcribe_call_duration_ws}s")

    except RuntimeError as e_rt_ws: # Specific for runtime errors like CUDA OOM
        if "CUDA out of memory" in str(e_rt_ws):
            logger.error(f"WS ({session_id}): CUDA OOM: {e_rt_ws}", exc_info=True)
            if websocket.application_state == WebSocketState.CONNECTED: await websocket.send_json({"type":"error", "error":f"CUDA out of memory: {e_rt_ws}"})
        else:
            logger.error(f"WS ({session_id}): Runtime error: {e_rt_ws}", exc_info=True)
            if websocket.application_state == WebSocketState.CONNECTED: await websocket.send_json({"type":"error", "error":f"Server runtime error: {e_rt_ws}"})
            
    except Exception as e_ws_outer:
        logger.error(f"WS ({session_id}): Outer error: {e_ws_outer}", exc_info=True)
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "error": "Unhandled server exception."})
            
    finally:
        # Revert model settings and clean up resources
        await revert_model_settings_after_request(
            asr_model, long_audio_settings_were_applied_ws, device_dtype_was_changed_from_global_ws,
            processing_device_for_this_req_ws, session_id
        )
        if temp_uploaded_file_path and os.path.exists(temp_uploaded_file_path):
            try: await asyncio.to_thread(os.remove, temp_uploaded_file_path)
            except Exception as e_del_ws: logger.error(f"WS ({session_id}): Error deleting temp WAV: {e_del_ws}")
        
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.close(code=1000)
        logger.info(f"WS ({session_id}): Connection closed and handler finished.")

if __name__ == "__main__":
    import uvicorn
    if not asr_model:
        logger.critical(f"ASR Model ('{ASR_MODEL_NAME}') failed to load. Server cannot start.")
    else:
        logger.info(f"Starting Uvicorn server on host 0.0.0.0, port {PORT}. ASR Model: {ASR_MODEL_NAME}")
        uvicorn.run("main:app", host="0.0.0.0", port=PORT, workers=1)