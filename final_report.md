(# YouTube Timestamp Search — Project Report

## Overview

This project is a complete, production-ready system for **semantic search of YouTube videos with timestamp jumping**. Given a natural language query, it finds the most relevant YouTube videos, transcribes their audio, semantically matches transcript segments to the query, and returns direct timestamped YouTube URLs to the best answers.

---

## Key Features

- **YouTube Data API v3**: Finds the most relevant videos for any query.
- **Local ASR Transcription**: Downloads audio using `yt-dlp` and transcribes with `faster-whisper` (no reliance on YouTube’s transcript endpoint, so it works even when transcripts are unavailable or rate-limited).
- **Semantic Search**: Splits transcripts into short chunks, generates embeddings with `sentence-transformers`, and uses FAISS for fast similarity search.
- **Timestamped Results**: Returns clickable YouTube URLs that jump directly to the most relevant moment in the video.
- **Streamlit UI**: Simple web interface for interactive search and debugging.
- **FastAPI Backend**: Clean, documented API for programmatic access.
- **Docker Support**: Easy deployment with Docker and Docker Compose.

---

## Architecture & Workflow

1. **User Query**: The user enters a natural language question.
2. **Video Search**: The system uses the YouTube Data API to find the top N relevant videos.
3. **Audio Download**: For each video, audio is downloaded using `yt-dlp`. If YouTube blocks downloads, the system supports cookies and proxy configuration.
4. **Transcription**: Audio is transcribed locally using `faster-whisper` (supports CPU and CUDA for GPU acceleration).
5. **Chunking & Embedding**: The transcript is split into short segments (default: 8 seconds). Each chunk is embedded using a transformer model.
6. **Semantic Matching**: The user query is embedded and matched against all transcript chunks using FAISS (vector similarity search).
7. **Result Formatting**: The best matches are returned as timestamped YouTube URLs, with confidence scores and matched text.

---

## Project Structure

- `main.py` — FastAPI backend (API endpoints)
- `streamlit_app.py` — Streamlit UI for interactive search
- `utils.py` — Core logic: YouTube search, download, ASR, chunking, embedding, matching
- `config.py` — Environment/config management
- `requirements.txt` — All dependencies (see below)
- `.env.example` — Example environment variables
- `Dockerfile` / `docker-compose.yml` — Containerization support

---

## Setup & Installation

1. **Clone the repository**
2. **Create a virtual environment**
	```bash
	python -m venv .venv
	.venv\Scripts\Activate.ps1  # On Windows
	```
3. **Install dependencies**
	```bash
	pip install -r requirements.txt
	```
4. **Configure environment**
	- Copy `.env.example` to `.env`
	- Set your `YOUTUBE_API_KEY` (get from Google Cloud Console)
	- (Optional) Export YouTube cookies to `cookies.txt` if you hit download/sign-in issues
5. **Run the backend**
	```bash
	uvicorn main:app --reload
	```
6. **Run the Streamlit UI**
	```bash
	streamlit run streamlit_app.py
	```

---

## How to Use

**Web UI:**
- Open http://localhost:8501
- Enter a question (e.g., "how to use python for machine learning")
- The app will search YouTube, transcribe, and show timestamped results

**API:**
- POST to `/search` with JSON:
  ```json
  {
	 "query": "how to use python for machine learning",
	 "top_k_videos": 5,
	 "top_k_matches": 3
  }
  ```
- Response includes best match and top matches with URLs, timestamps, and confidence scores

---

## Learning Value

This project is an excellent resource for learning about:

- Integrating multiple AI/ML and web technologies (ASR, embeddings, vector search, API integration)
- Building robust, production-ready Python applications
- Handling real-world issues (rate limits, download blocks, cookies, proxies)
- Clean code structure and environment management
- Deploying with Docker

---

## Troubleshooting

- If downloads fail with "Please sign in" or "not a bot", upgrade `yt-dlp` and/or export fresh YouTube cookies
- See README for more details on cookies/proxy setup
- Use debug mode in Streamlit for detailed error messages

---

## Requirements

See `requirements.txt` for all dependencies. Key packages:

- fastapi
- uvicorn
- google-api-python-client
- sentence-transformers
- faiss-cpu
- python-dotenv
- numpy
- streamlit
- yt-dlp
- faster-whisper

---

## Author & License

Created by a senior AI engineer. For educational and research use only. Respect YouTube’s Terms of Service.
)
