# Voice-to-Image

Speak a description, get an AI-generated image. Optionally cast to Chromecast.

## Pipeline

1. **Record** — Browser microphone via Web Audio API
2. **Transcribe** — faster-whisper (speech-to-text)
3. **Refine** — Ollama LLM converts speech to image prompt
4. **Generate** — Stable Diffusion creates the image
5. **Cast** — (optional) Display on Chromecast via catt

## Requirements

- Python 3.12+
- CUDA GPU (16GB+ VRAM recommended for high quality mode)
- Ollama running locally with llama3.2 model
- uv package manager

## Setup

```bash
# Install Ollama and pull model
ollama pull llama3.2

# Install dependencies
uv sync

# Run the server
uv run python server.py
```

Server runs on port 8765.

## Web UI

```bash
uv run python webui/app.py
```

Web UI runs on port 8766. Access from any device on local network.

**Note:** For remote mic access (not localhost), Chrome requires HTTPS or a flag:
```
chrome://flags/#unsafely-treat-insecure-origin-as-secure
```
Add your server URL (e.g., `http://192.168.1.43:8766`) and restart Chrome.

## Quality Presets

| Preset | Whisper | LLM | Image Model | Steps | VRAM |
|--------|---------|-----|-------------|-------|------|
| Lite | tiny | llama3.2:1b | sd-turbo | 4 | ~4GB |
| Standard | base | llama3.2 | sd-turbo | 4 | ~6GB |
| High | base | llama3.2 | sdxl-turbo | 4 | ~10GB |
| Ultra | medium | llama3.2 | SDXL | 30 | ~24GB |
| Flux | base | llama3.2 | FLUX.1-schnell | 4 | ~24GB* |

*FLUX automatically uses multi-GPU parallelism if available (2x 12GB works), or CPU offload on single GPU (slower).

## Image Sizes

All 16:9 aspect ratio:
- 640x360 (Lite)
- 768x432 (Standard)
- 1024x576 (HD)
- 1280x720 (720p)

## API Endpoints

- `POST /pipeline` — Full pipeline (audio file → image)
- `POST /transcribe` — Audio → text
- `POST /refine` — Text → image prompt
- `POST /generate` — Prompt → image
- `POST /cast` — Cast image to Chromecast
- `POST /unload` — Free GPU memory
- `GET /health` — Status check

## Chromecast

Requires `catt` (installed via dependencies). Set your device name in `server.py`:

```python
DEFAULT_CHROMECAST = "Living Room TV"
```

---

