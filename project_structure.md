# Project File Responsibilities and Flow

This document explains the purpose, responsibilities, and main tasks of each file in the SemanticTube project. It also describes the overall flow of the application.

---

## Project Structure Overview

```
.
├── main.py
├── streamlit_app.py
├── utils.py
├── config.py
├── requirements.txt
├── .env / .env.example
├── Dockerfile
├── docker-compose.yml
├── README.md
├── final_report.md
├── instruction.md
└── ... (other files/folders)
```

---

## File Descriptions

### 1. `main.py`
- **Type:** FastAPI backend
- **Responsibility:**
  - Exposes the main API endpoint `/search`.
  - Handles incoming search requests (query, top_k_videos, top_k_matches).
  - Orchestrates the workflow: YouTube search → audio download → transcription → semantic search → result formatting.
  - Returns best matches with timestamped YouTube URLs.
- **Key Tasks:**
  - Receives POST requests, validates input.
  - Calls helper functions from `utils.py`.
  - Handles errors (e.g., missing API key, no results).

### 2. `streamlit_app.py`
- **Type:** Streamlit web UI
- **Responsibility:**
  - Provides an interactive web interface for users.
  - Lets users enter queries, configure options, and view results.
  - Shows debug info, errors, and allows cache clearing.
- **Key Tasks:**
  - Calls the same core logic as the backend for consistency.
  - Displays progress/status for each step (search, download, transcribe, match).
  - Presents results with clickable timestamped URLs and video previews.

### 3. `utils.py`
- **Type:** Core logic/helpers
- **Responsibility:**
  - Implements all core pipeline steps as reusable functions.
  - Handles YouTube search, audio download (yt-dlp), local ASR transcription (faster-whisper), transcript chunking, embedding, and semantic matching.
- **Key Tasks:**
  - Robustly extracts video IDs from URLs.
  - Downloads audio, handles cookies/proxy, retries with different strategies.
  - Transcribes audio, splits into chunks, generates embeddings, and performs FAISS similarity search.
  - Formats results as timestamped URLs.

### 4. `config.py`
- **Type:** Configuration loader
- **Responsibility:**
  - Loads environment variables from `.env` or system environment.
  - Provides a single `settings` object for all config (API keys, model, chunk size, cookies, proxy, etc).
- **Key Tasks:**
  - Handles path resolution for cookies and cache directories.
  - Makes configuration available to all modules.

### 5. `requirements.txt`
- **Type:** Dependency list
- **Responsibility:**
  - Lists all Python packages required to run the project.
- **Key Tasks:**
  - Ensures reproducible environments for development and deployment.

### 6. `.env` / `.env.example`
- **Type:** Environment variables
- **Responsibility:**
  - Stores sensitive and configurable settings (API keys, model, chunk size, etc).
- **Key Tasks:**
  - `.env.example` provides a template for users to create their own `.env`.

### 7. `Dockerfile`
- **Type:** Container build file
- **Responsibility:**
  - Defines how to build a Docker image for the backend API.
- **Key Tasks:**
  - Installs system and Python dependencies, sets up the app user, exposes port 8000.

### 8. `docker-compose.yml`
- **Type:** Multi-container orchestration
- **Responsibility:**
  - Defines how to run the API (and optionally other services) together.
- **Key Tasks:**
  - Maps ports, loads environment variables, builds the image.

### 9. `README.md`
- **Type:** Project documentation
- **Responsibility:**
  - Explains project purpose, features, setup, usage, troubleshooting, and credits.

### 10. `final_report.md`
- **Type:** Project summary/report
- **Responsibility:**
  - Provides a comprehensive overview for learning and review.

### 11. `instruction.md`
- **Type:** Project requirements/spec
- **Responsibility:**
  - Documents the original requirements and design goals for the project.

---

## Project Flow (Step-by-Step)

1. **User submits a query** (via Streamlit UI or API).
2. **YouTube Search:**
   - The system uses the YouTube Data API to find the top N relevant videos.
3. **Audio Download:**
   - For each video, audio is downloaded using yt-dlp.
   - If YouTube blocks downloads, cookies/proxy are used as needed.
4. **Transcription:**
   - Audio is transcribed locally using faster-whisper (ASR model).
5. **Chunking & Embedding:**
   - The transcript is split into short segments (default: 8 seconds).
   - Each chunk is embedded using a transformer model.
6. **Semantic Matching:**
   - The user query is embedded and matched against all transcript chunks using FAISS (vector similarity search).
7. **Result Formatting:**
   - The best matches are returned as timestamped YouTube URLs, with confidence scores and matched text.
8. **Results are displayed** in the UI or returned via API.

---

## Summary Table

| File                | Main Responsibility                                  | Short Description                                  |
|---------------------|------------------------------------------------------|----------------------------------------------------|
| main.py             | FastAPI backend, API orchestration                   | Handles API requests and main workflow             |
| streamlit_app.py    | Streamlit UI                                         | Interactive web interface for users                |
| utils.py            | Core logic/helpers                                   | Implements all pipeline steps                      |
| config.py           | Configuration loader                                 | Loads and manages environment/config               |
| requirements.txt    | Dependency list                                      | Lists all required Python packages                 |
| .env / .env.example | Environment variables                                | Stores API keys and settings                       |
| Dockerfile          | Docker build                                         | Builds container image for backend                 |
| docker-compose.yml  | Docker orchestration                                 | Runs API and services together                     |
| README.md           | Documentation                                        | Setup, usage, troubleshooting, credits             |
| final_report.md     | Project summary/report                               | Overview for learning and review                   |
| instruction.md      | Project requirements/spec                            | Original requirements and design goals             |

---

This document should help any new developer or reviewer quickly understand the structure, responsibilities, and flow of the SemanticTube project.
