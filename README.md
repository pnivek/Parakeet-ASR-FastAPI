# Parakeet ASR

A high-performance Automatic Speech Recognition (ASR) server built with NVIDIA's NeMo Parakeet model. This application provides both REST API and WebSocket interfaces for transcribing audio files to text, with an interactive web UI for easy use.

## 🚀 Features

- **State-of-the-art ASR**: Powered by NVIDIA's parakeet-tdt-0.6b-v2 model.
- **Multiple APIs**:
  - REST API for robust file transcriptions, including improved handling for long audio files up to 3 hours.
  - WebSocket API for chunk-based, live transcription of audio files with no file limits.
- **Versatile Output**: Transcriptions available as plain text, segment lists, CSV, and SRT formats.
- **Interactive Web UI**: Built-in browser interface for easy testing, segment playback, and result downloads.
- **Docker Ready**: Easy deployment using Docker containers.
- **Configurable**: Multiple environment variables to tune performance.

## 📋 Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended for optimal performance)
- Docker (optional, but recommended for deployment)
- Dependencies listed in `app/requirements.txt`

## 🛠️ Installation

### Using Docker (Recommended) (Mac/Linux/Windows)

The easiest way to run Parakeet ASR is using Docker:

```bash
# Clone the repository
git clone https://github.com/pnivek/Parakeet-ASR-FastAPI.git
cd Parakeet-ASR-FastAPI # Updated directory name if it changed

# Build the Docker image
docker build -t parakeet-asr .

# Run the container
# Ensure your Docker setup allows GPU access if you have an NVIDIA GPU
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8777:8777 parakeet-asr
```

### Manual Installation (Mac/Linux)

If you prefer to run without Docker:

```bash
# Clone the repository
git clone https://github.com/pnivek/Parakeet-ASR-FastAPI.git
cd Parakeet-ASR-FastAPI # Updated directory name if it changed

# Install dependencies (ensure you have build tools for packages that need compilation)
# It's recommended to use a virtual environment
# python -m venv venv
# source venv/bin/activate
pip install -r app/requirements.txt

# Run the application
cd app
python main.py
```

## ⚙️ Configuration

The application can be configured using environment variables or a `.env` file in the `app` directory:

| Variable | Description | Default |
|----------|-------------|---------|
| `BATCH_SIZE` | ASR batch size during inference (primarily for REST endpoint with long audio) | 1 |
| `NUM_WORKERS` | Number of workers for NeMo's `transcribe` method | 0 |
| `TRANSCRIBE_CHUNK_LEN` | Audio chunk length in seconds (Websocket chunked inference) | 30 |
| `TRANSCRIBE_OVERLAP` | Overlap between chunks in seconds (Websocket chunked inference) | 5 |
| `LONG_AUDIO_THRESHOLD` | length threshold before using long audio strategy | 480 |
| `PORT` | Server port | 8777 |
| `LOG_LEVEL` | Logging level (e.g., INFO, DEBUG) | INFO |

## 🔌 API Documentation

> **Note:** Both the REST and WebSocket APIs require the entire audio file to be uploaded to the server. The WebSocket API streams back transcription segments as they are processed, but does not support true partial audio upload or real-time audio streaming from the client. The difference is in how results are returned: REST returns all results at once after processing, while WebSocket returns segments as they are transcribed.

### REST API

The REST API uses a similar inference strategy implemented here: https://huggingface.co/spaces/nvidia/parakeet-tdt-0.6b-v2

#### Transcribe Audio

```
POST /v1/audio/transcriptions
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: Form with an audio file attached as `file` (e.g., WAV, MP3, FLAC, OGG).

**Response (Example):**
```json
{
  "text": "The complete transcribed text. This is the first segment. This is the second segment.",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.55,
      "text": "This is the first segment.",
      "seek": 0,
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": null,
      "compression_ratio": null,
      "no_speech_prob": null
    },
    {
      "id": 1,
      "start": 2.56,
      "end": 5.2,
      "text": "This is the second segment.",
      "seek": 0,
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": null,
      "compression_ratio": null,
      "no_speech_prob": null
    }
  ],
  "language": "en",
  "transcription_time": 1.234,
  "csv_content": "Start (s),End (s),Segment\\n0.000,2.550,This is the first segment.\\n2.560,5.200,This is the second segment.\\n",
  "srt_content": "1\\n00:00:00,000 --> 00:00:02,550\\nThis is the first segment.\\n\\n2\\n00:00:02,560 --> 00:00:05,200\\nThis is the second segment.\\n\\n"
}
```

### WebSocket API

The WebSocket API processes audio in chunks, allowing for real-time streaming of large audio files and receiving transcribed segments as they become available.

> **Important:** The WebSocket API requires the full audio file to be uploaded (in chunks or as a stream), just like the REST API. It does not support continuous, live audio streaming from the client.
Connect to `/v1/audio/transcriptions/ws` endpoint:

1.  **Connection**: Establish a WebSocket connection.
2.  **Configuration (Optional but Recommended)**: Send a JSON message with audio configuration details:
    ```json
    {
      "sample_rate": 16000, // Client's audio sample rate
      "channels": 1,        // Client's audio channels
      "format": "binary"    // "binary" for raw bytes, "base64" for base64 encoded strings
    }
    ```
    If not sent, the server may assume defaults or try to infer from the data, but providing it is more robust.
3.  **Audio Data**: Send audio data in binary chunks (if `format: "binary"`) or as base64 encoded text messages (if `format: "base64"`).
4.  **End Signal**: After sending all audio data, send a text message "END" to signal the end of the audio stream.
5.  **Receiving Results (Segments)**: As audio chunks are processed, the server sends back JSON messages for each transcribed segment. These messages typically look like:
    ```json
    {
      "id": 0,                   // Segment sequence ID
      "start": 0.0,              // Start time of the segment in seconds
      "end": 2.55,               // End time of the segment in seconds
      "text": "Segment text",    // Transcribed text for this segment
      "type": "segment"          // Indicates this is an intermediate segment message
                                 // (Actual key might be 'type': 'segment_transcription' or similar based on server implementation)
    }
    ```
6.  **Final Result**: After all audio is processed and the "END" signal is received, the server sends a final summary message. This message includes the full aggregated text, total processing time, and the complete transcription in CSV and SRT formats:
    ```json
    {
      "type": "final_transcription",
      "text": "The complete transcribed text...",
      "language": "en",
      "transcription_time": 10.567,
      "total_segments": 50,
      "final_duration_processed_seconds": 120.5,
      "csv_content": "Start (s),End (s),Segment\\n...",
      "srt_content": "1\\n00:00:00,000 --> ...\\n..."
    }
    ```

## 🖥️ Web Interface

A user-friendly web interface is available at the root URL (`/`) when the server is running. This interface allows for easy interaction and testing of the ASR service:

-   Upload audio files for transcription.
-   Choose between REST API (full file upload) and WebSocket API (chunked processing for streaming simulation).
-   View transcription results, including the full text and timing information.
-   **Displays transcription segments in an interactive table.**
-   **Allows playback of individual audio segments by clicking on rows in the table.**
-   **Provides download options for the full transcription in CSV and SRT formats.**
-   Includes a debug mode for viewing detailed logs and message exchanges.

## 🛠️ Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/pnivek/Parakeet-ASR-FastAPI.git
cd Parakeet-ASR-FastAPI # Updated directory name

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Install dependencies
pip install -r app/requirements.txt

# Run with debug logging (from the 'app' directory)
cd app
LOG_LEVEL=DEBUG python main.py
```

## 🔍 Troubleshooting

Common issues:

-   **Model Loading Errors**: Ensure you have enough GPU memory. If using CPU, transcription will be significantly slower.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

## 🙏 Acknowledgements

-   [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) for the Parakeet ASR model
-   [FastAPI](https://fastapi.tiangolo.com/) for the web framework
-   [PyTorch](https://pytorch.org/) and [torchaudio](https://pytorch.org/audio) for audio processing
-   The developers of all other libraries listed in `requirements.txt`.
