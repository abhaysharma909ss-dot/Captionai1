# CaptionFlow AI
AI video caption editor using Faster-Whisper (multilingual Hindi/English speech recognition) and FFmpeg (subtitle burn-in).

## Run
Requires Python 3.10+, FFmpeg and FFprobe.
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000`. The first transcription downloads the Whisper model. Default model is `small`; set `WHISPER_MODEL=base` for lower memory or `medium` for better accuracy with more resources.

## Deploy
Build with the included Dockerfile on a Docker-capable host that supports CPU workloads and writable storage. Set `WHISPER_MODEL=small`, `WHISPER_DEVICE=cpu`, `WHISPER_COMPUTE_TYPE=int8`, and `MAX_UPLOAD_MB=500`. Free tiers change and may sleep, cap memory, disk or request duration. Static-only hosting cannot run this backend. Test a short clip first.

## Endpoints
- `GET /api/health`
- `POST /api/transcribe` multipart `video`, query `language=auto|hi|en`
- `POST /api/render` JSON `job_id`, `captions`, and styling options
- `GET /api/download/{job_id}`

## Notes
Files are stored in `work/<job_id>`. Add cleanup, authentication, rate limiting, persistent storage and a background job queue before public production use. This is a functional starter, not a hardened multi-user production service. No API key is required; the open-source model downloads on first use.
