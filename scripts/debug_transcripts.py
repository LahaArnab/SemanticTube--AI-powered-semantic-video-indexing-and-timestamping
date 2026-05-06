import argparse
import json
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings
from utils import (
    fetch_transcript_local_asr_with_debug,
    find_best_matches,
    format_video_url,
    search_youtube_videos,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug local ASR (yt-dlp + faster-whisper)")
    parser.add_argument("query", help="YouTube search query (also used for semantic matching)")
    parser.add_argument("--videos", type=int, default=settings.TOP_K_VIDEOS)
    parser.add_argument("--top", type=int, default=settings.TOP_K_MATCHES)
    parser.add_argument("--chunk-seconds", type=int, default=settings.CHUNK_SECONDS)
    parser.add_argument("--model", default=settings.LOCAL_ASR_MODEL)
    parser.add_argument("--device", default=settings.LOCAL_ASR_DEVICE)
    parser.add_argument("--compute-type", default=settings.LOCAL_ASR_COMPUTE_TYPE)
    args = parser.parse_args()

    if not settings.YOUTUBE_API_KEY:
        print("Missing YOUTUBE_API_KEY in environment (.env)")
        sys.exit(2)

    try:
        videos = search_youtube_videos(
            api_key=settings.YOUTUBE_API_KEY,
            query=args.query,
            max_results=int(args.videos),
        )
    except Exception as e:
        print(str(e))
        sys.exit(3)
    print(f"Found videos: {len(videos)}")

    embedder = SentenceTransformer(settings.MODEL_NAME)

    for v in videos:
        video_id = v.get("video_id")
        title = v.get("title")
        if not video_id:
            continue

        transcript, dbg = fetch_transcript_local_asr_with_debug(
            video_id=video_id,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cache_dir=settings.LOCAL_ASR_CACHE_DIR,
            audio_dir=settings.LOCAL_ASR_AUDIO_DIR,
            cookies_path=settings.TRANSCRIPT_COOKIES_PATH,
            proxies=settings.TRANSCRIPT_PROXIES,
        )

        ok = bool(transcript)
        print("-")
        print(f"{title} ({video_id})")
        print(f"transcript_ok={ok} entries={(len(transcript) if transcript else 0)}")
        print(json.dumps({k: dbg.get(k) for k in ["audio_path", "cache_hit", "timings", "error_type", "error"]}, indent=2))

        if ok:
            matches = find_best_matches(
                model=embedder,
                query=args.query,
                transcript=transcript,
                chunk_seconds=int(args.chunk_seconds),
                max_chunks=settings.MAX_CHUNKS_PER_VIDEO,
                top_k_matches=int(args.top),
            )
            for m in matches:
                ts = int(m["start"])
                print(
                    f"  - {ts}s | {round(float(m['confidence']), 4)} | {format_video_url(video_id, ts)}\n"
                    f"    {m['text']}"
                )


if __name__ == "__main__":
    main()
