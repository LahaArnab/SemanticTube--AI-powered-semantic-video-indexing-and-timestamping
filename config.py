import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _strip_quotes(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and ((value[0] == value[-1]) and value[0] in {"\"", "'"}):
        value = value[1:-1].strip()
    return value


class Settings:
    YOUTUBE_API_KEY = _strip_quotes(
        os.getenv("YOUTUBE_API_KEY")
        or os.getenv("YOUTUBE_DATA_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    )
    MODEL_NAME = _strip_quotes(os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))
    CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "8"))
    TOP_K_VIDEOS = int(os.getenv("TOP_K_VIDEOS", "5"))
    TOP_K_MATCHES = int(os.getenv("TOP_K_MATCHES", "3"))
    MAX_CHUNKS_PER_VIDEO = int(os.getenv("MAX_CHUNKS_PER_VIDEO", "300"))
    TRANSCRIPT_COOKIES_PATH = _strip_quotes(os.getenv("TRANSCRIPT_COOKIES_PATH", "")) or None
    if TRANSCRIPT_COOKIES_PATH:
        cookies_path = Path(TRANSCRIPT_COOKIES_PATH)
        if not cookies_path.is_absolute():
            cookies_path = Path(__file__).resolve().parent / cookies_path
        TRANSCRIPT_COOKIES_PATH = str(cookies_path)
    else:
        default_cookies = Path(__file__).resolve().parent / "cookies.txt"
        if default_cookies.exists():
            TRANSCRIPT_COOKIES_PATH = str(default_cookies)

    _TRANSCRIPT_PROXY = _strip_quotes(os.getenv("TRANSCRIPT_PROXY", ""))
    _TRANSCRIPT_PROXY_HTTP = _strip_quotes(os.getenv("TRANSCRIPT_PROXY_HTTP", ""))
    _TRANSCRIPT_PROXY_HTTPS = _strip_quotes(os.getenv("TRANSCRIPT_PROXY_HTTPS", ""))
    _TRANSCRIPT_PROXIES_JSON = _strip_quotes(os.getenv("TRANSCRIPT_PROXIES", ""))

    TRANSCRIPT_PROXIES = None
    if _TRANSCRIPT_PROXIES_JSON:
        try:
            parsed = json.loads(_TRANSCRIPT_PROXIES_JSON)
            if isinstance(parsed, dict):
                cleaned = {str(k): str(v) for k, v in parsed.items() if v}
                TRANSCRIPT_PROXIES = cleaned or None
        except Exception:
            TRANSCRIPT_PROXIES = None
    elif _TRANSCRIPT_PROXY_HTTP or _TRANSCRIPT_PROXY_HTTPS:
        proxies = {}
        if _TRANSCRIPT_PROXY_HTTP:
            proxies["http"] = _TRANSCRIPT_PROXY_HTTP
        if _TRANSCRIPT_PROXY_HTTPS:
            proxies["https"] = _TRANSCRIPT_PROXY_HTTPS
        TRANSCRIPT_PROXIES = proxies or None
    elif _TRANSCRIPT_PROXY:
        TRANSCRIPT_PROXIES = {"http": _TRANSCRIPT_PROXY, "https": _TRANSCRIPT_PROXY}
    LOCAL_ASR_MODEL = _strip_quotes(os.getenv("LOCAL_ASR_MODEL", "base"))
    LOCAL_ASR_DEVICE = _strip_quotes(os.getenv("LOCAL_ASR_DEVICE", "cpu"))
    LOCAL_ASR_COMPUTE_TYPE = _strip_quotes(os.getenv("LOCAL_ASR_COMPUTE_TYPE", "int8"))

    LOCAL_ASR_CACHE_DIR = _strip_quotes(os.getenv("LOCAL_ASR_CACHE_DIR", ".cache/local_asr"))
    if LOCAL_ASR_CACHE_DIR:
        cache_dir = Path(LOCAL_ASR_CACHE_DIR)
        if not cache_dir.is_absolute():
            cache_dir = Path(__file__).resolve().parent / cache_dir
        LOCAL_ASR_CACHE_DIR = str(cache_dir)

    LOCAL_ASR_AUDIO_DIR = _strip_quotes(os.getenv("LOCAL_ASR_AUDIO_DIR", ".cache/audio"))
    if LOCAL_ASR_AUDIO_DIR:
        audio_dir = Path(LOCAL_ASR_AUDIO_DIR)
        if not audio_dir.is_absolute():
            audio_dir = Path(__file__).resolve().parent / audio_dir
        LOCAL_ASR_AUDIO_DIR = str(audio_dir)


settings = Settings()
