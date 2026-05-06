from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import settings
from utils import (
    fetch_transcript_local_asr_with_debug,
    find_best_matches,
    format_video_url,
    search_youtube_videos,
)

app = FastAPI(title="YouTube Timestamp Search")
model = SentenceTransformer(settings.MODEL_NAME)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k_videos: Optional[int] = None
    top_k_matches: Optional[int] = None


class Match(BaseModel):
    video_id: str
    video_title: str
    timestamp: int
    video_url: str
    matched_text: str
    confidence: float


class SearchResponse(BaseModel):
    best_match: Match
    top_matches: List[Match]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    if not settings.YOUTUBE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing YOUTUBE_API_KEY")

    top_k_videos = payload.top_k_videos or settings.TOP_K_VIDEOS
    top_k_matches = payload.top_k_matches or settings.TOP_K_MATCHES

    try:
        videos = search_youtube_videos(
            api_key=settings.YOUTUBE_API_KEY,
            query=payload.query,
            max_results=top_k_videos,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    if not videos:
        raise HTTPException(status_code=404, detail="No videos found")

    results: List[Match] = []

    for video in videos:
        transcript, _debug = fetch_transcript_local_asr_with_debug(
            video_id=video["video_id"],
            model_name=settings.LOCAL_ASR_MODEL,
            device=settings.LOCAL_ASR_DEVICE,
            compute_type=settings.LOCAL_ASR_COMPUTE_TYPE,
            cache_dir=settings.LOCAL_ASR_CACHE_DIR,
            audio_dir=settings.LOCAL_ASR_AUDIO_DIR,
            cookies_path=settings.TRANSCRIPT_COOKIES_PATH,
            proxies=settings.TRANSCRIPT_PROXIES,
        )
        if not transcript:
            continue

        matches = find_best_matches(
            model=model,
            query=payload.query,
            transcript=transcript,
            chunk_seconds=settings.CHUNK_SECONDS,
            max_chunks=settings.MAX_CHUNKS_PER_VIDEO,
            top_k_matches=top_k_matches,
        )
        for match in matches:
            timestamp = int(match["start"])
            results.append(
                Match(
                    video_id=video["video_id"],
                    video_title=video.get("title", ""),
                    timestamp=timestamp,
                    video_url=format_video_url(video["video_id"], timestamp),
                    matched_text=match["text"],
                    confidence=round(match["confidence"], 4),
                )
            )

    if not results:
        raise HTTPException(status_code=404, detail="No matches found")

    results.sort(key=lambda item: item.confidence, reverse=True)
    best = results[0]
    top = results[: max(1, top_k_matches)]

    return SearchResponse(best_match=best, top_matches=top)
