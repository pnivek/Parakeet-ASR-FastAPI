"""
StreamingPrevBatchedEngine — NVIDIA's blessed streaming pattern (NeMo PR #9106).

This is the Option C engine. It replaces the BatchedFrameASRTDT-based
`StreamingTdtEngine` in main.py with the API NeMo's reference script
(examples/asr/asr_chunked_inference/rnnt/speech_to_text_streaming_infer_rnnt.py)
uses:

    StreamingBatchedAudioBuffer   ← FIFO audio buffer with explicit ContextSize
    asr_model(input_signal=...)   ← encoder forward, eager (graph captures decoder only)
    decoding_computer(            ← TDT decoder, captured-graph friendly
        x=encoder_output[:, left:],
        out_len=...,
        prev_batched_state=state,
    )                             ← returns (chunk_hyps, _, state)
    current_hyps.merge_(chunk_hyps)
    asr_model.decoding.compute_rnnt_timestamps(hyp)  ← public timestamp API

No reach into leading-underscore internals. State threading is explicit.
No accumulator list to grow unbounded. This is the API NVIDIA shipped FOR
streaming, in contrast to BatchedFrameASRTDT which they shipped for
offline buffered chunking and which we were misusing for live.

Design notes vs the old StreamingTdtEngine:

  - The decoder runs on encoder_output[:, left_context:] (i.e., chunk +
    right context). Out_len=chunk-frames so it emits tokens ONLY for the
    chunk portion, treating right context as lookahead. No middle-token
    merge needed — the decoder is told exactly what to decode.

  - Per-token timestamps are encoder-frame indices inside `chunk_hyps.timestamps`.
    Map to global audio time via `chunk_index * chunk_secs + frame * stride_s`.

  - Sentence-bounded emission policy is the same as the old engine for
    UX continuity (segments end on '.', '!', '?'). The plan calls out
    `compute_rnnt_timestamps` for word/segment timestamps as a follow-up
    once the engine is proven.

  - All CUDA-touching calls must run on the dedicated executor — caller
    (main.py) is responsible for that, same as the old engine.
"""

from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from nemo.collections.asr.parts.utils.rnnt_utils import BatchedHyps
from nemo.collections.asr.parts.utils.streaming_utils import (
    ContextSize,
    StreamingBatchedAudioBuffer,
)


def _make_divisible_by(value: int, factor: int) -> int:
    """Round `value` down to the nearest multiple of `factor`. Mirrors NeMo's util."""
    return (value // factor) * factor


class StreamingPrevBatchedEngine:
    """
    Drives `decoding_computer(prev_batched_state=...)` chunk-by-chunk over
    a `StreamingBatchedAudioBuffer`. Designed for batch_size=1 live streams
    (each WS connection owns one engine instance).

    Lifecycle:
        feed_float32(samples)   — repeat as PCM arrives
        flush()                 — exactly once at EOF
        pop_committed_segments() — call after feed/flush to drain emissions
        pop_final_segments()    — call after flush to flush trailing partial

    Threading: not async-safe. Call from a single coroutine under the
    server's model_access_lock. CUDA work should be dispatched via the
    server's dedicated _asr_executor.
    """

    def __init__(
        self,
        asr_model_instance,
        chunk_secs: float,
        left_context_secs: float,
        right_context_secs: float,
        request_id: str = "stream-v2",
    ):
        self.asr_model = asr_model_instance
        self.request_id = request_id
        self.chunk_secs = float(chunk_secs)
        self.left_context_secs = float(left_context_secs)
        self.right_context_secs = float(right_context_secs)

        # Read invariants from the model config (mirrors the reference script).
        model_cfg = asr_model_instance._cfg
        self.sample_rate = int(model_cfg.preprocessor["sample_rate"])
        feature_stride_sec = float(model_cfg.preprocessor["window_stride"])
        features_per_sec = 1.0 / feature_stride_sec
        encoder_subsampling_factor = int(asr_model_instance.encoder.subsampling_factor)

        # NeMo's `make_divisible_by` so each encoder frame maps to an integer
        # number of audio samples.
        features_frame2audio_samples = _make_divisible_by(
            int(self.sample_rate * feature_stride_sec),
            factor=encoder_subsampling_factor,
        )
        self.encoder_frame2audio_samples = (
            features_frame2audio_samples * encoder_subsampling_factor
        )
        self.encoder_stride_s = self.encoder_frame2audio_samples / self.sample_rate

        # Context in encoder frames + the same in audio samples. We pass
        # context_samples to StreamingBatchedAudioBuffer; encoder_context is
        # used to slice the encoder output after each step.
        self.encoder_context = ContextSize(
            left=int(left_context_secs * features_per_sec / encoder_subsampling_factor),
            chunk=int(chunk_secs * features_per_sec / encoder_subsampling_factor),
            right=int(right_context_secs * features_per_sec / encoder_subsampling_factor),
        )
        self.context_samples = ContextSize(
            left=self.encoder_context.left * self.encoder_frame2audio_samples,
            chunk=self.encoder_context.chunk * self.encoder_frame2audio_samples,
            right=self.encoder_context.right * self.encoder_frame2audio_samples,
        )
        # Theoretical emission lag = (chunk + right) seconds; we round-trip
        # the audio-sample form to a float-seconds form for the public attr.
        self.emission_lag_secs = (
            (self.context_samples.chunk + self.context_samples.right) / self.sample_rate
        )

        # Captured-graph decoder. Lives on the global asr_model so its graph
        # persists across requests (dual-decoder swap in main.py keeps the
        # right computer active).
        self.decoding_computer = asr_model_instance.decoding.decoding.decoding_computer
        self.tokenizer = asr_model_instance.tokenizer

        # Streaming state — re-init by reset()
        self._device = asr_model_instance.device
        self._dtype = next(asr_model_instance.parameters()).dtype
        self._reset_streaming_state()

        # Sentence-emission state — same policy as the legacy engine for UX continuity
        self._next_seg_id: int = 0
        self._sentence_buffer_ids: List[int] = []
        self._sentence_buffer_start: Optional[float] = None
        self._sentence_buffer_last_t: float = 0.0

        # Index into `_committed_token_times` of tokens already emitted
        self._tokens_emitted_through: int = 0

    def _reset_streaming_state(self) -> None:
        # Buffer holds float32 audio samples; encoder runs eager so dtype
        # cast happens inside the model.
        self.buffer = StreamingBatchedAudioBuffer(
            batch_size=1,
            context_samples=self.context_samples,
            dtype=torch.float32,
            device=self._device,
        )
        self.state = None  # decoder state, threaded across chunks via prev_batched_state
        self.current_batched_hyps: Optional[BatchedHyps] = None
        # Buffered float32 PCM not yet committed to a chunk step
        self._pcm_buffer = torch.zeros(0, dtype=torch.float32, device=self._device)
        # Audio-time stream of (token_id, seconds) — only the NEW part each
        # pop call uses, so this can be appended to incrementally without
        # re-walking history.
        self._committed_token_times: List[Tuple[int, float]] = []
        self._chunk_index: int = 0
        self._asr_time_s: float = 0.0
        self._eof_flushed: bool = False

    # --------- Public feed/flush ----------

    def feed_float32(self, samples_f32: np.ndarray) -> None:
        """Append float32 PCM in [-1, 1]; advance the engine for any complete chunks.

        Per NeMo's reference loop: the FIRST chunk needs `chunk + right` samples
        to prime the buffer's right context, then every subsequent chunk is
        just `chunk` samples. Trying to feed `chunk` samples on iter 0 trips
        the assertion `self.right == expected_context.right` inside
        StreamingBatchedAudioBuffer because the right side starts unprimed.
        """
        if samples_f32 is None or samples_f32.size == 0:
            return
        if self._eof_flushed:
            raise RuntimeError("feed_float32 called after flush()")
        tensor = torch.from_numpy(np.ascontiguousarray(samples_f32, dtype=np.float32)).to(self._device)
        self._pcm_buffer = torch.cat([self._pcm_buffer, tensor], dim=0)
        chunk_samples = self.context_samples.chunk
        right_samples = self.context_samples.right
        while True:
            needed = (chunk_samples + right_samples) if self._chunk_index == 0 else chunk_samples
            if self._pcm_buffer.numel() < needed:
                break
            chunk = self._pcm_buffer[:needed]
            self._pcm_buffer = self._pcm_buffer[needed:]
            self._step_one_chunk(chunk, is_last_chunk=False)

    def flush(self) -> None:
        """Drain any residual PCM as the LAST chunk so the right context closes out."""
        if self._eof_flushed:
            return
        residual = self._pcm_buffer
        self._pcm_buffer = torch.zeros(0, dtype=torch.float32, device=self._device)
        self._step_one_chunk(residual, is_last_chunk=True)
        self._eof_flushed = True

    # --------- Public pop ----------

    def pop_committed_segments(self) -> List[dict]:
        """Return segments (sentence-bounded) for tokens committed since last pop."""
        new_tokens = self._committed_token_times[self._tokens_emitted_through:]
        self._tokens_emitted_through = len(self._committed_token_times)
        return self._consume_tokens_into_segments(new_tokens, flush_partial=False)

    def pop_final_segments(self) -> List[dict]:
        """At EOF — flush any in-progress sentence as the last segment."""
        committed = self.pop_committed_segments()
        if self._sentence_buffer_ids:
            text = self.tokenizer.ids_to_text(self._sentence_buffer_ids).strip()
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

    @property
    def asr_time_s(self) -> float:
        return self._asr_time_s

    def reset(self) -> None:
        """Release buffer + decoder state. Safe to re-feed afterwards."""
        self._reset_streaming_state()
        self._next_seg_id = 0
        self._sentence_buffer_ids = []
        self._sentence_buffer_start = None
        self._sentence_buffer_last_t = 0.0
        self._tokens_emitted_through = 0

    # --------- Internals ----------

    def _step_one_chunk(self, chunk_samples: torch.Tensor, is_last_chunk: bool) -> None:
        """One iteration of the NeMo reference loop, specialised for batch_size=1.

        Wrapping the whole body in `torch.inference_mode()` matters: chunk_hyps
        returned from `decoding_computer` is an inference tensor; merge_'s
        in-place scatter on `current_batched_hyps.transcript` only works if
        we're still inside the same inference-mode context."""
        t0 = time.time()
        chunk_len = chunk_samples.shape[0]
        device = self._device

        with torch.inference_mode():
            audio_batch = chunk_samples.unsqueeze(0)  # [1, T]
            chunk_lengths_batch = torch.tensor([chunk_len], dtype=torch.long, device=device)
            is_last_chunk_batch = torch.tensor([is_last_chunk], dtype=torch.bool, device=device)

            self.buffer.add_audio_batch_(
                audio_batch,
                audio_lengths=chunk_lengths_batch,
                is_last_chunk=is_last_chunk,
                is_last_chunk_batch=is_last_chunk_batch,
            )

            # Encode the whole buffer (eager — graphs capture decoder only).
            with torch.amp.autocast(device.type, dtype=self._dtype):
                encoder_output, encoder_output_len = self.asr_model(
                    input_signal=self.buffer.samples,
                    input_signal_length=self.buffer.context_size_batch.total(),
                )
            encoder_output = encoder_output.transpose(1, 2)  # [B, T, C]
            # The captured CUDA graph was warmed up during FULL transcribes
            # under NeMo's own autocast and baked in bf16 inputs for the
            # joint's project_encoder Linear. Force the model's pinned dtype
            # so the captured weights see matching inputs.
            encoder_output = encoder_output.to(dtype=self._dtype)

            # Slice off the left context — we don't want to redecode tokens already
            # emitted in earlier chunks.
            encoder_context = self.buffer.context_size.subsample(factor=self.encoder_frame2audio_samples)
            encoder_context_batch = self.buffer.context_size_batch.subsample(factor=self.encoder_frame2audio_samples)
            encoder_output = encoder_output[:, encoder_context.left:]

            # Decode just the chunk frames (right context is lookahead the decoder
            # uses but does NOT emit tokens for, unless it's the very last chunk
            # where we want to drain everything).
            if is_last_chunk:
                out_len = encoder_output_len - encoder_context_batch.left
            else:
                out_len = encoder_context_batch.chunk

            with torch.amp.autocast(device.type, dtype=self._dtype):
                chunk_hyps, _, self.state = self.decoding_computer(
                    x=encoder_output,
                    out_len=out_len,
                    prev_batched_state=self.state,
                )

            # Merge into running hypothesis (for end-of-stream final transcript).
            if self.current_batched_hyps is None:
                self.current_batched_hyps = chunk_hyps
            else:
                self.current_batched_hyps.merge_(chunk_hyps)

            # Map token timestamps → global audio seconds.
            #
            # IMPORTANT: chunk_hyps.timestamps is NOT chunk-local. When
            # prev_batched_state is non-None, decoding_computer's
            # _fix_timestamps_for_iterative_decoding (tdt_label_looping.py:1360)
            # shifts the timestamps by prev_batched_state.decoded_lengths —
            # i.e., the returned values are CUMULATIVE encoder frame indices
            # from the start of the stream. For chunk 0 there's no shift, so
            # the same formula `frame * stride_s` gives chunk-local audio time
            # (which equals stream-time-from-zero for chunk 0). Use the
            # cumulative interpretation uniformly.
            n_new = int(chunk_hyps.current_lengths[0].item())
            if n_new > 0:
                new_ids = chunk_hyps.transcript[0, :n_new].detach().cpu().tolist()
                new_frame_idx = chunk_hyps.timestamps[0, :n_new].detach().cpu().tolist()
                stride_s = self.encoder_stride_s
                for tid, f in zip(new_ids, new_frame_idx):
                    t_s = float(f) * stride_s
                    if t_s < 0.0:
                        t_s = 0.0
                    self._committed_token_times.append((int(tid), t_s))

        self._chunk_index += 1
        self._asr_time_s += time.time() - t0

    def _consume_tokens_into_segments(
        self,
        new_tokens: List[Tuple[int, float]],
        flush_partial: bool,
    ) -> List[dict]:
        """Group new (id, time) pairs into sentence-bounded segment dicts.
        Carries running sentence state across pop calls so a sentence spanning
        many chunks emits exactly once when its terminal '.!?' lands."""
        segments: List[dict] = []
        tokenizer = self.tokenizer
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
