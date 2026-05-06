import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import faiss
import numpy as np


def extract_youtube_video_id(url_or_id: str) -> Optional[str]:
    """Extract a YouTube video id from a URL or return it if already an id.

    Accepts:
    - 11-char video ids
    - Full URLs (watch/youtu.be/shorts/embed/live/v)
    - URLs without scheme (e.g. 'www.youtube.com/watch?v=...')
    - IDs embedded in surrounding text (e.g. markdown bullets)
    """

    raw = (url_or_id or "").strip()
    if not raw:
        return None

    # Common copy/paste wrappers.
    raw = raw.strip().strip("<>\"'()[]{}")

    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw

    # If the user pasted a URL without a scheme, urlparse treats it as a path.
    candidate_url = raw
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", raw):
        lowered = raw.lower()
        if lowered.startswith(("www.", "youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be")):
            candidate_url = "https://" + raw

    try:
        parsed = urlparse(candidate_url)
    except Exception:
        parsed = None

    if parsed:
        host = (parsed.netloc or "").lower()
        path = (parsed.path or "").strip("/")

        if "youtu.be" in host:
            candidate = path.split("/")[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                return candidate

        if any(h in host for h in ["youtube.com", "youtube-nocookie.com"]):
            if path == "watch":
                qs = parse_qs(parsed.query or "")
                v = (qs.get("v", [None]) or [None])[0]
                if v and re.fullmatch(r"[A-Za-z0-9_-]{11}", v):
                    return v
            for prefix in ("shorts/", "embed/", "live/", "v/"):
                if path.startswith(prefix):
                    candidate = path.split("/", 1)[1].split("/")[0]
                    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                        return candidate

    # Fallback: try to find an ID inside the string.
    patterns = [
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/(?:watch\?v=|shorts/|embed/|live/|v/))([A-Za-z0-9_-]{11})",
        r"(?:\bv=)([A-Za-z0-9_-]{11})\b",
    ]
    for pat in patterns:
        m = re.search(pat, raw)
        if m:
            return m.group(1)

    if "youtu" in raw.lower():
        m = re.search(r"(?<![A-Za-z0-9_-])([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])", raw)
        if m:
            return m.group(1)

    return None


def search_youtube_videos(
    api_key: str,
    query: str,
    max_results: int = 5,
) -> List[Dict]:
    """Search YouTube for videos using the Data API v3 and return basic metadata.

    Note: This does NOT fetch transcripts. Transcript text is produced locally via ASR.
    """

    if not api_key:
        return []

    q = (query or "").strip()
    if not q:
        return []

    try:
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
        response = (
            youtube.search()
            .list(
                part="snippet",
                q=q,
                type="video",
                maxResults=max_results,
            )
            .execute()
        )
    except Exception as e:
        msg = str(e) or type(e).__name__
        if len(msg) > 800:
            msg = msg[:800] + "…"
        raise RuntimeError(f"YouTube API search failed: {msg}") from e

    results: List[Dict] = []
    seen: set[str] = set()

    for item in response.get("items", []) or []:
        video_id = ((item.get("id") or {}).get("videoId") or "").strip()
        if not video_id or video_id in seen:
            continue
        seen.add(video_id)
        snippet = item.get("snippet") or {}
        results.append(
            {
                "video_id": video_id,
                "title": snippet.get("title", "") or "",
                "description": snippet.get("description", "") or "",
                "channel_title": snippet.get("channelTitle", "") or "",
                "published_at": snippet.get("publishedAt", "") or "",
            }
        )

    return results


def _coalesce_proxy(proxies: Optional[dict]) -> Optional[str]:
    if not proxies:
        return None
    return proxies.get("https") or proxies.get("http")


def _cookiefile_probably_has_youtube(cookiefile: str) -> bool:
    try:
        p = Path(cookiefile)
        if not p.exists() or not p.is_file():
            return False
        sample = p.read_text(encoding="utf-8", errors="ignore")
        lowered = sample.lower()
        return "youtube.com" in lowered or "accounts.google.com" in lowered
    except Exception:
        return False


def download_youtube_audio(
    video_id: str,
    output_dir: str,
    cookies_path: Optional[str] = None,
    proxies: Optional[dict] = None,
) -> str:
    """Download audio for a YouTube video to a stable local file path.

    This is used as a fallback when transcript endpoints are blocked.
    """

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in existing:
        if p.is_file() and p.stat().st_size > 0:
            return str(p)

    url = f"https://www.youtube.com/watch?v={video_id}"

    cookiefile = None
    if cookies_path:
        p = Path(cookies_path)
        if p.exists() and p.is_file():
            cookiefile = str(p)

    cookiefile_has_youtube = bool(cookiefile) and _cookiefile_probably_has_youtube(cookiefile)

    base_ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_dir / f"{video_id}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "cookiefile": cookiefile,
        "proxy": _coalesce_proxy(proxies),
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 30,
    }

    attempts: list[dict] = [
        {"label": "default", "extra": {}},
        {
            "label": "player_client=android",
            "extra": {"extractor_args": {"youtube": {"player_client": ["android"]}}},
        },
        {
            "label": "player_client=ios",
            "extra": {"extractor_args": {"youtube": {"player_client": ["ios"]}}},
        },
        {
            "label": "player_client=tv",
            "extra": {"extractor_args": {"youtube": {"player_client": ["tv"]}}},
        },
        {
            "label": "player_client=mweb",
            "extra": {"extractor_args": {"youtube": {"player_client": ["mweb"]}}},
        },
        {
            "label": "player_client=web",
            "extra": {"extractor_args": {"youtube": {"player_client": ["web"]}}},
        },
    ]

    last_error: Optional[BaseException] = None
    last_error_text: str = ""
    sign_in_error_text: str = ""

    for attempt in attempts:
        ydl_opts = dict(base_ydl_opts)
        ydl_opts.update(attempt.get("extra") or {})

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded = Path(ydl.prepare_filename(info))

            if downloaded.exists() and downloaded.stat().st_size > 0:
                return str(downloaded)

            candidates = sorted(
                out_dir.glob(f"{video_id}.*"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for c in candidates:
                if c.is_file() and c.stat().st_size > 0 and not c.name.endswith((".part", ".ytdl")):
                    return str(c)

        except Exception as e:
            last_error = e
            last_error_text = str(e) or type(e).__name__
            lowered = last_error_text.lower()
            if not sign_in_error_text and (
                "sign in" in lowered
                or "login" in lowered
                or "confirm" in lowered
                or "not a bot" in lowered
                or "bot" in lowered
            ):
                sign_in_error_text = last_error_text
            for tmp in list(out_dir.glob(f"{video_id}.*.part")) + list(out_dir.glob(f"{video_id}.*.ytdl")):
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass

            continue

    if last_error is not None:
        error_text_for_user = sign_in_error_text or last_error_text or type(last_error).__name__
        lowered = error_text_for_user.lower()
        hint = ""
        if "sign in" in lowered or "login" in lowered or "confirm" in lowered or "bot" in lowered:
            if cookiefile and not cookiefile_has_youtube:
                hint = (
                    " Your cookies file exists but does not appear to include youtube/google cookies."
                    " Export YouTube cookies (cookies.txt) from your browser and point TRANSCRIPT_COOKIES_PATH to it."
                )
            elif not cookiefile:
                hint = (
                    " This video likely requires YouTube sign-in/consent."
                    " Export YouTube cookies (cookies.txt) from your browser and set TRANSCRIPT_COOKIES_PATH."
                )
            else:
                hint = (
                    " This video likely requires YouTube sign-in/consent."
                    " Try re-exporting fresh cookies and updating TRANSCRIPT_COOKIES_PATH."
                )
        elif "requested format is not available" in lowered and (not cookiefile or not cookiefile_has_youtube):
            hint = (
                " This can happen when YouTube blocks anonymous downloads."
                " Try exporting YouTube cookies (cookies.txt) and setting TRANSCRIPT_COOKIES_PATH."
            )

        raise DownloadError(
            f"Failed to download audio for video_id={video_id}. {error_text_for_user}{hint}"
        ) from last_error

    candidates = sorted(out_dir.glob(f"{video_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])

    raise DownloadError(f"Failed to download audio for video_id={video_id} (no output file was created)")


@lru_cache(maxsize=4)
def _get_faster_whisper_model(model_name: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe_audio_faster_whisper(
    audio_path: str,
    model_name: str,
    device: str,
    compute_type: str,
) -> List[Dict]:
    model = _get_faster_whisper_model(model_name, device, compute_type)

    segments, _info = model.transcribe(audio_path)
    transcript: List[Dict] = []
    for seg in segments:
        text = clean_text(getattr(seg, "text", "") or "")
        if not text:
            continue
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", start) or start)
        duration = max(0.0, end - start)
        transcript.append({"text": text, "start": start, "duration": duration})
    return transcript


def fetch_transcript_local_asr_with_debug(
    video_id: str,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    cache_dir: str = ".cache/local_asr",
    audio_dir: str = ".cache/audio",
    cookies_path: Optional[str] = None,
    proxies: Optional[dict] = None,
) -> tuple[Optional[List[Dict]], Dict]:
    debug: Dict = {
        "video_id": video_id,
        "source": "local_asr",
        "cookies_path_set": bool(cookies_path),
        "proxies_set": bool(proxies),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "cache_hit": False,
        "cache_path": None,
        "audio_path": None,
        "timings": {},
        "error_type": None,
        "error": None,
    }

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"{video_id}.{model_name}.json"
    debug["cache_path"] = str(cache_path)

    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached:
                debug["cache_hit"] = True
                return cached, debug
        except Exception:
            pass

    try:
        t0 = time.time()
        audio_path = download_youtube_audio(
            video_id=video_id,
            output_dir=audio_dir,
            cookies_path=cookies_path,
            proxies=proxies,
        )
        debug["timings"]["download_seconds"] = round(time.time() - t0, 3)
        debug["audio_path"] = audio_path
    except Exception as e:
        debug["error_type"] = type(e).__name__
        debug["error"] = str(e)
        return None, debug

    try:
        t1 = time.time()
        transcript = transcribe_audio_faster_whisper(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            compute_type=compute_type,
        )
        debug["timings"]["transcribe_seconds"] = round(time.time() - t1, 3)
    except Exception as e:
        debug["error_type"] = type(e).__name__
        debug["error"] = str(e)
        return None, debug

    if transcript:
        try:
            cache_path.write_text(json.dumps(transcript, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return transcript, debug

    return None, debug


def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_transcript(
    transcript: List[Dict],
    chunk_seconds: int = 8,
    max_chunks: int = 300,
) -> List[Dict]:
    chunks: List[Dict] = []
    current_texts: List[str] = []
    chunk_start: Optional[float] = None
    chunk_end: Optional[float] = None

    for entry in transcript:
        start = float(entry.get("start", 0.0))
        duration = float(entry.get("duration", 0.0))
        end = start + duration
        text = clean_text(entry.get("text", ""))
        if not text:
            continue

        if chunk_start is None:
            chunk_start = start
        chunk_end = end
        current_texts.append(text)

        if chunk_end - chunk_start >= chunk_seconds:
            chunks.append(
                {
                    "start": float(chunk_start),
                    "end": float(chunk_end),
                    "text": clean_text(" ".join(current_texts)),
                }
            )
            current_texts = []
            chunk_start = None
            chunk_end = None

        if len(chunks) >= max_chunks:
            break

    if current_texts and chunk_start is not None and chunk_end is not None:
        chunks.append(
            {
                "start": float(chunk_start),
                "end": float(chunk_end),
                "text": clean_text(" ".join(current_texts)),
            }
        )

    return chunks


def build_embeddings(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> (np.ndarray, np.ndarray):
    return index.search(query_embedding, top_k)


def format_video_url(video_id: str, timestamp_seconds: int) -> str:
    return f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds}s"


def find_best_matches(
    model,
    query: str,
    transcript: List[Dict],
    chunk_seconds: int,
    max_chunks: int,
    top_k_matches: int,
) -> List[Dict]:
    chunks = chunk_transcript(
        transcript,
        chunk_seconds=chunk_seconds,
        max_chunks=max_chunks,
    )
    if not chunks:
        return []

    texts = [chunk["text"] for chunk in chunks]
    embeddings = build_embeddings(model, texts)
    index = build_faiss_index(embeddings)

    query_embedding = build_embeddings(model, [query])
    scores, indices = search_index(index, query_embedding, top_k=top_k_matches)

    results = []
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "start": int(chunk["start"]),
                "end": int(chunk["end"]),
                "text": chunk["text"],
                "confidence": float(scores[0][rank]),
            }
        )

    return results
