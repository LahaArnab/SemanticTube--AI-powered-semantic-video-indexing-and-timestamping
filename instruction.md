You are a senior AI engineer. Build a complete working project that performs **semantic YouTube search with timestamp jumping**.

## 🎯 Goal

Given a user query, the system must:

1. Find the most relevant YouTube video
2. Extract transcript
3. Identify the exact timestamp where the answer is discussed
4. Return a clickable YouTube URL that jumps directly to that timestamp

Example output:
https://www.youtube.com/watch?v=VIDEO_ID&t=125s

---

## 🧠 System Design Requirements

### 1. Input

* User query (natural language)

### 2. Video Retrieval

* Use YouTube Data API v3
* Search top 5 relevant videos
* Extract video IDs

### 3. Transcript Extraction

* Use youtube-transcript-api
* Fetch full transcript for each video
* Handle errors if transcript unavailable

### 4. Semantic Search (Core Logic)

* Convert transcript into chunks (5–10 sec segments)
* Generate embeddings using:

  * sentence-transformers (all-MiniLM-L6-v2)
* Store embeddings in FAISS

### 5. Query Matching

* Convert user query to embedding
* Perform similarity search
* Return best matching transcript chunk

### 6. Timestamp Extraction

* Extract timestamp from matched chunk
* Convert into seconds

### 7. Final Output

* Generate URL:
  https://www.youtube.com/watch?v={video_id}&t={timestamp}s

---

## 🧱 Tech Stack

* Python
* FastAPI (backend API)
* FAISS (vector DB)
* sentence-transformers (embeddings)
* youtube-transcript-api
* Google API (YouTube Data API)

---

## 📦 Deliverables (IMPORTANT)

Write FULL working code:

1. requirements.txt
2. main.py (FastAPI backend)
3. utils.py (helper functions)
4. config file for API keys
5. clear step-by-step setup instructions

---

## ⚙️ Functional Requirements

* Handle multiple videos → choose best match
* Skip videos without transcripts
* Optimize for speed (limit chunk size)
* Clean transcript text before embedding
* Return top 1 best result

---

## 🚀 Bonus Features (Include if possible)

* Return top 3 timestamps
* Confidence score
* Highlight matched text
* Simple frontend (optional Streamlit)

---

## 🧪 Example

Input:
"how to use python for machine learning"

Output:
{
"video_title": "...",
"video_url": "https://www.youtube.com/watch?v=abc123&t=210s",
"timestamp": 210,
"matched_text": "In this part we use Python libraries like sklearn..."
}

---

## ⚠️ Constraints

* Code must be clean, modular, production-ready
* Avoid unnecessary complexity
* Use environment variables for API keys
* Must be runnable locally

---

Now generate the FULL code step-by-step.
