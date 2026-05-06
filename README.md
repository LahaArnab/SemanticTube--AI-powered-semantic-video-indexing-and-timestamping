# YouTube Timestamp Search

Semantic timestamp search for YouTube videos that returns a URL jumping directly to the best timestamp.

## Features

- YouTube Data API v3 search (find top relevant videos)
- Downloads audio locally via `yt-dlp`
- Transcribes locally via `faster-whisper` (CPU by default; CUDA optional)
- Chunked semantic search using sentence-transformers + FAISS
- Returns timestamp jump links like `https://www.youtube.com/watch?v=VIDEO_ID&t=125s`

## Setup (Local)

1) Create a virtual environment

```bash
python -m venv .venv
```

2) Activate it

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

3) Install dependencies

```bash
pip install -r requirements.txt
```

4) Configure environment variables

```bash
copy .env.example .env
```

Set `YOUTUBE_API_KEY` in `.env`.

5) Run the API

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## Streamlit UI

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Create `.env`

```bash
copy .env.example .env
```

3) Run Streamlit

```bash
streamlit run streamlit_app.py
```

The UI will be available at http://localhost:8501.

Usage:

- Enter a question, then click **Search** (the app searches YouTube automatically)

## API Usage

POST `/search`

```json
{
  "query": "how to use python for machine learning",
  "top_k_videos": 5,
  "top_k_matches": 5
}
```

Response example:

```json
{
  "best_match": {
    "video_id": "abc123",
    "video_title": "Machine Learning with Python",
    "timestamp": 210,
    "video_url": "https://www.youtube.com/watch?v=abc123&t=210s",
    "matched_text": "...",
    "confidence": 0.6217
  },
  "top_matches": [
    {
      "video_id": "abc123",
      "video_title": "Machine Learning with Python",
      "timestamp": 210,
      "video_url": "https://www.youtube.com/watch?v=abc123&t=210s",
      "matched_text": "...",
      "confidence": 0.6217
    }
  ]
}
```

## Docker

1) Create `.env`

```bash
copy .env.example .env
```

2) Build and run

```bash
docker compose up --build
```

## Notes

- If a video can't be downloaded/transcribed, it will be skipped.
- `confidence` is cosine similarity (higher is better).
- Transcription requires downloading audio for each video, so it can take a while (cached on disk under `.cache/`).

## Cookies / Proxy (for yt-dlp)

Some networks/IPs hit consent/age-restrictions or require a proxy for downloading.

If downloads suddenly start failing with messages like **"Please sign in"** or **"confirm you’re not a bot"**, upgrade `yt-dlp` (YouTube changes frequently):

```bash
python -m pip install -U yt-dlp
```

- **Cookies**: Export your own YouTube cookies to a Netscape/Mozilla cookies file (often named `cookies.txt`) and set `TRANSCRIPT_COOKIES_PATH`.
  - If you place `cookies.txt` in the project root, it will be auto-detected (no env var needed).
  - Keep this file private (it can contain session tokens).
  - On Windows, use forward slashes for paths, e.g. `TRANSCRIPT_COOKIES_PATH=D:/Projects/Yutube timestep/cookies.txt`.
- **Proxy** (optional): If you must route requests through a corporate proxy, set `TRANSCRIPT_PROXY` (applies to both http/https), or set `TRANSCRIPT_PROXY_HTTP` / `TRANSCRIPT_PROXY_HTTPS`.

## Local Transcription

Transcription is performed locally via `faster-whisper`.

Common `.env` settings:

```bash
LOCAL_ASR_MODEL=base
LOCAL_ASR_DEVICE=cpu
LOCAL_ASR_COMPUTE_TYPE=int8
```

Notes:

- You may need `ffmpeg` installed and available on your PATH (depending on the audio format yt-dlp downloads).
- For NVIDIA GPUs, try `LOCAL_ASR_DEVICE=cuda` with `LOCAL_ASR_COMPUTE_TYPE=float16`.
- Only use this on videos you have permission to process; respect YouTube Terms of Service and copyright.
