import streamlit as st
from sentence_transformers import SentenceTransformer
from pathlib import Path

from config import settings
from utils import (
    fetch_transcript_local_asr_with_debug,
    find_best_matches,
    format_video_url,
    search_youtube_videos,
)

st.set_page_config(page_title="YouTube Timestamp Search", layout="wide")


@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(settings.MODEL_NAME)


@st.cache_data(ttl=60 * 60, show_spinner=False)
def cached_fetch_transcript_local_asr(
    video_id: str,
    model_name: str,
    device: str,
    compute_type: str,
    cache_dir: str,
    audio_dir: str,
    cookies_path: str | None,
    proxies_items: tuple[tuple[str, str], ...] | None,
):
    proxies = dict(proxies_items) if proxies_items else None
    return fetch_transcript_local_asr_with_debug(
        video_id=video_id,
        model_name=model_name,
        device=device,
        compute_type=compute_type,
        cache_dir=cache_dir,
        audio_dir=audio_dir,
        cookies_path=cookies_path,
        proxies=proxies,
    )


def run_search_with_steps(
    query: str,
    top_k_videos: int,
    top_k_matches: int,
    chunk_seconds: int,
    asr_model: str,
    asr_device: str,
    asr_compute_type: str,
    debug: bool,
) -> list[dict]:
    if not settings.YOUTUBE_API_KEY:
        st.error("Missing YOUTUBE_API_KEY in .env")
        return []

    model = load_model()

    with st.status("Step 1/3: Searching YouTube", expanded=debug) as s1:
        try:
            videos = search_youtube_videos(
                api_key=settings.YOUTUBE_API_KEY,
                query=query,
                max_results=int(top_k_videos),
            )
        except Exception as e:
            s1.update(label="Step 1/3: YouTube search failed", state="error")
            st.error(str(e))
            return []
        if not videos:
            s1.update(label="Step 1/3: No videos found", state="error")
            st.warning("No videos found.")
            return []

        st.toast(f"Found {len(videos)} videos")
        with st.popover("Show extracted videos"):
            st.dataframe(
                [
                    {
                        "title": v.get("title", ""),
                        "video_id": v.get("video_id", ""),
                        "channel": v.get("channel_title", ""),
                    }
                    for v in videos
                ],
                use_container_width=True,
            )
        s1.update(label=f"Step 1/3: Found {len(videos)} videos", state="complete")

    results: list[dict] = []
    transcript_debug_rows: list[dict] = []
    transcript_debug_map: dict[str, dict] = {}

    with st.status("Step 2/3: Downloading audio + transcribing", expanded=debug) as s2:
        progress = st.progress(0, text="Processing videos...")
        for i, video in enumerate(videos, start=1):
            video_id = video.get("video_id", "")
            title = video.get("title", "")
            progress.progress(i / len(videos), text=f"Video {i}/{len(videos)}: {title or video_id}")

            transcript, debug_info = cached_fetch_transcript_local_asr(
                video_id=video_id,
                model_name=asr_model,
                device=asr_device,
                compute_type=asr_compute_type,
                cache_dir=settings.LOCAL_ASR_CACHE_DIR,
                audio_dir=settings.LOCAL_ASR_AUDIO_DIR,
                cookies_path=settings.TRANSCRIPT_COOKIES_PATH,
                proxies_items=(
                    tuple(sorted(settings.TRANSCRIPT_PROXIES.items()))
                    if settings.TRANSCRIPT_PROXIES
                    else None
                ),
            )

            ok = bool(transcript)
            transcript_debug_map[video_id] = debug_info
            timings = debug_info.get("timings") or {}
            error_message = (debug_info.get("error") or "").strip()
            if len(error_message) > 240:
                error_message = error_message[:240] + "…"
            transcript_debug_rows.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "transcript_ok": ok,
                    "entries": len(transcript) if transcript else 0,
                    "cache_hit": bool(debug_info.get("cache_hit")),
                    "download_s": timings.get("download_seconds"),
                    "transcribe_s": timings.get("transcribe_seconds"),
                    "error": debug_info.get("error_type"),
                    "error_message": error_message or None,
                }
            )

            if not transcript:
                continue

            hits = find_best_matches(
                model=model,
                query=query,
                transcript=transcript,
                chunk_seconds=int(chunk_seconds),
                max_chunks=settings.MAX_CHUNKS_PER_VIDEO,
                top_k_matches=int(top_k_matches),
            )
            for hit in hits:
                ts = int(hit["start"])
                results.append(
                    {
                        "video_id": video_id,
                        "video_title": title,
                        "timestamp": ts,
                        "video_url": format_video_url(video_id, ts),
                        "matched_text": hit["text"],
                        "confidence": round(hit["confidence"], 4),
                    }
                )

        progress.empty()
        ok_count = sum(1 for row in transcript_debug_rows if row.get("transcript_ok"))
        s2.update(
            label=f"Step 2/3: Transcribed {ok_count}/{len(videos)} videos",
            state="complete" if ok_count else "error",
        )

        if debug:
            st.subheader("Transcription summary")
            st.dataframe(transcript_debug_rows, use_container_width=True)

    if debug and transcript_debug_map:
        st.subheader("Transcription debug details")
        video_id_to_title = {
            row["video_id"]: row.get("title") or row["video_id"] for row in transcript_debug_rows
        }
        selected_video_id = st.selectbox(
            "Select a video",
            options=list(transcript_debug_map.keys()),
            format_func=lambda vid: f"{video_id_to_title.get(vid, vid)} ({vid})",
        )
        if selected_video_id:
            st.json(transcript_debug_map[selected_video_id])

    with st.status("Step 3/3: Ranking matches", expanded=debug) as s3:
        if not results:
            s3.update(label="Step 3/3: No matches", state="error")
            st.warning("No semantic matches found in the processed videos.")
            return []

        results.sort(key=lambda item: item["confidence"], reverse=True)
        s3.update(label=f"Step 3/3: Ranked {len(results)} matches", state="complete")

    return results


def render_results(results: list[dict], top_k_matches: int) -> None:
    best = results[0]
    top = results[: max(1, top_k_matches)]

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Best Match")
        st.write(f"**{best.get('video_title', '')}**")
        st.write(f"Timestamp: {best['timestamp']}s")
        st.write(f"Confidence: {best['confidence']}")
        st.write(best["matched_text"])
        st.markdown(f"[Open on YouTube]({best['video_url']})")

        st.subheader("Top Matches")
        for match in top:
            st.markdown(
                f"- **{match.get('video_title', '')}** | {match['timestamp']}s | "
                f"{match['confidence']} | [Open]({match['video_url']})"
            )
            st.caption(match["matched_text"])

    with right:
        st.subheader("Recommended Video")
        st.video(
            f"https://www.youtube.com/watch?v={best['video_id']}",
            start_time=int(best["timestamp"]),
        )


st.title("YouTube Timestamp Search")

query = st.text_input("Ask a question", placeholder="e.g. how to use python for machine learning")
debug_mode = st.toggle("Debug mode", value=True, help="Show step-by-step progress and ASR debug details")

if debug_mode:
    if st.button("Clear transcript cache"):
        cached_fetch_transcript_local_asr.clear()
        st.success("Cleared cached transcript results")
    cookies_path = settings.TRANSCRIPT_COOKIES_PATH
    st.caption(f"TRANSCRIPT_COOKIES_PATH: {cookies_path or '(not set)'}")
    if cookies_path:
        if Path(cookies_path).exists():
            st.caption("Cookies file: found")
        else:
            st.error("Cookies file not found at TRANSCRIPT_COOKIES_PATH")
    st.caption(f"TRANSCRIPT_PROXIES set: {bool(settings.TRANSCRIPT_PROXIES)}")
    st.caption(
        f"faster-whisper default: {settings.LOCAL_ASR_MODEL} | "
        f"{settings.LOCAL_ASR_DEVICE}/{settings.LOCAL_ASR_COMPUTE_TYPE}"
    )

col1, col2, col3 = st.columns(3)
with col1:
    top_k_videos = st.number_input(
        "Top videos",
        min_value=1,
        max_value=10,
        value=max(1, int(settings.TOP_K_VIDEOS)),
        step=1,
    )

with col2:
    top_k_matches = st.number_input(
        "Top matches",
        min_value=1,
        max_value=5,
        value=min(5, max(1, settings.TOP_K_MATCHES)),
        step=1,
    )
with col3:
    chunk_seconds = st.number_input(
        "Chunk seconds",
        min_value=5,
        max_value=15,
        value=settings.CHUNK_SECONDS,
        step=1,
    )

col6, col7, col8 = st.columns(3)
with col6:
    asr_model = st.selectbox(
        "ASR model (faster-whisper)",
        options=["tiny", "base", "small"],
        index=(
            ["tiny", "base", "small"].index(settings.LOCAL_ASR_MODEL)
            if settings.LOCAL_ASR_MODEL in {"tiny", "base", "small"}
            else 1
        ),
        help="Smaller models are faster but less accurate.",
    )

with col7:
    asr_device = st.selectbox(
        "Device",
        options=["cpu", "cuda"],
        index=1 if settings.LOCAL_ASR_DEVICE == "cuda" else 0,
        help="Use 'cuda' only if you have an NVIDIA GPU + CUDA support.",
    )
with col8:
    asr_compute_type = st.selectbox(
        "Compute type",
        options=["int8", "float16", "int8_float16"],
        index=(
            ["int8", "float16", "int8_float16"].index(settings.LOCAL_ASR_COMPUTE_TYPE)
            if settings.LOCAL_ASR_COMPUTE_TYPE in {"int8", "float16", "int8_float16"}
            else 0
        ),
        help="CPU: int8 is usually best. CUDA: float16 is common.",
    )

if st.button("Search", type="primary"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Running semantic search..."):
            results = run_search_with_steps(
                query=query.strip(),
                top_k_videos=int(top_k_videos),
                top_k_matches=int(top_k_matches),
                chunk_seconds=int(chunk_seconds),
                asr_model=str(asr_model),
                asr_device=str(asr_device),
                asr_compute_type=str(asr_compute_type),
                debug=bool(debug_mode),
            )
        if results:
            render_results(results, int(top_k_matches))
