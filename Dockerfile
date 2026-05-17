FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

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

RUN pip install --no-cache-dir nemo_toolkit[asr]==2.2.1

COPY app/ /app/
RUN mkdir -p /app/static
RUN if [ -f /app/index.html ]; then cp /app/index.html /app/static/; fi

EXPOSE 8777

CMD ["python3", "main.py"]
