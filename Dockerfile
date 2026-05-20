# --- Stage 1: build the React/Vite client. -----------------------------
# Output lands in /static (vite.config.ts sets build.outDir='../static',
# relative to WORKDIR=/web). Stage 2 copies it into /app/static.
# Done first because there are no GPU/torch deps and rebuilds take ~5s.
FROM node:20-alpine AS webbuild
WORKDIR /web
COPY app/web/package.json app/web/package-lock.json* ./
RUN npm ci --no-audit --no-fund
COPY app/web/ ./
RUN npm run build

# --- Stage 2: the actual ASR runtime. ----------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Swap the arm64 Ubuntu mirror. The default ports.ubuntu.com (Canonical's
# non-amd64 mirror) has had recurring connectivity issues from this builder.
# mirror.us.leaseweb.net carries a complete noble ubuntu-ports tree and
# is consistently reachable. (mirrors.kernel.org redirects to an edge
# host that 404s on ubuntu-ports/noble, so it's NOT a working alternate.)
# Idempotent — guarded by the sources file existing.
RUN if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://ports.ubuntu.com/ubuntu-ports|http://mirror.us.leaseweb.net/ubuntu-ports|g' \
            /etc/apt/sources.list.d/ubuntu.sources; \
    fi

# Install system deps
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    torch==2.11.0 \
    torchaudio==2.11.0 \
    torchcodec==0.11.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    fastapi==0.115.12 \
    websockets==15.0.1 \
    uvicorn==0.34.2 \
    python-multipart==0.0.20 \
    python-dotenv==1.1.1

RUN pip install --no-cache-dir nemo_toolkit[asr]==2.7.3

# Silero VAD — voice activity detection on the streaming PCM path. Used by
# handle_streaming_pcm to gate silent windows out of the engine queue. The
# package bundles the ONNX/jit model so no extra download is needed at
# runtime.
RUN pip install --no-cache-dir silero-vad==5.1.2

COPY app/ /app/
# Strip the client source tree from the runtime image — it was already
# compiled in the webbuild stage; we only need the bundled output.
RUN rm -rf /app/web
# vite.config.ts writes to '../static' relative to /web, so the build
# output lands in /static inside the webbuild stage.
COPY --from=webbuild /static /app/static

EXPOSE 8777

CMD ["python3", "main.py"]
