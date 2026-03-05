import os
import io
import re
import asyncio
import socket
import logging
import ipaddress
from urllib.parse import urlparse
from functools import lru_cache
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google.cloud import texttospeech_v1beta1 as tts
import trafilatura
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Article TTS API")

# ---------------------------------------------------------------------------
# Config — set these as Cloud Run environment variables
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]       # OAuth Web Client ID
ALLOWED_EMAIL    = os.environ["ALLOWED_EMAIL"]           # your Google account email
GCP_PROJECT      = os.environ["GCP_PROJECT"]             # GCP project ID
GCP_LOCATION     = os.environ.get("GCP_LOCATION", "us-central1")
FRONTEND_ORIGIN  = os.environ.get("FRONTEND_ORIGIN", "*")  # Your PWA origin e.g. https://tts.yourdomain.com

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Article-Title"],
)

# ---------------------------------------------------------------------------
# Auth — verify Google ID token and check it's your email
# ---------------------------------------------------------------------------
security = HTTPBearer()

def verify_google_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        id_info = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10,
        )
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired Google token")

    email = id_info.get("email")
    if email != ALLOWED_EMAIL:
        logger.warning(f"Unauthorized email attempt: {email}")
        raise HTTPException(status_code=403, detail="You are not authorized to use this service")

    return id_info

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ExtractRequest(BaseModel):
    url: HttpUrl

class TTSRequest(BaseModel):
    url: HttpUrl
    voice_name: str = "en-US-Chirp3-HD-Charon"
    speaking_rate: float = Field(default=1.0, ge=0.25, le=4.0)
    pitch: float = Field(default=0.0, ge=-20.0, le=20.0)

# ---------------------------------------------------------------------------
# SSRF protection — block requests to private/internal IPs
# ---------------------------------------------------------------------------
def _validate_url_safe(url: str) -> None:
    """Reject URLs that resolve to private/internal IP addresses."""
    parsed = urlparse(str(url))
    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Invalid URL")
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="Only HTTP(S) URLs are allowed")
    try:
        resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Could not resolve hostname")
    for _, _, _, _, sockaddr in resolved:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise HTTPException(status_code=400, detail="URLs pointing to internal addresses are not allowed")

# ---------------------------------------------------------------------------
# TTS client singleton
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_tts_client() -> tts.TextToSpeechClient:
    return tts.TextToSpeechClient()

# ---------------------------------------------------------------------------
# Article extraction helper
# ---------------------------------------------------------------------------
async def fetch_article_text(url: str) -> tuple[str, str]:
    """Returns (title, body_text)"""
    _validate_url_safe(url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        resp = await client.get(str(url), headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        html = resp.text

    result = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
        output_format="txt",
    )
    metadata = trafilatura.extract_metadata(html)
    title = metadata.title if metadata and metadata.title else "Article"

    if not result or len(result.strip()) < 100:
        raise HTTPException(status_code=422, detail="Could not extract readable text from this URL")

    return title, result.strip()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(req: ExtractRequest, _=Depends(verify_google_token)):
    """Preview extracted article text before converting to audio."""
    title, text = await fetch_article_text(str(req.url))
    word_count = len(text.split())
    return {
        "title": title,
        "text": text,
        "word_count": word_count,
        "estimated_minutes": max(1, round(word_count / 150)),  # ~150 wpm
    }


@app.post("/tts")
async def text_to_speech(req: TTSRequest, _=Depends(verify_google_token)):
    """Extract article and stream back MP3 audio."""
    title, text = await fetch_article_text(str(req.url))

    # Vertex AI TTS has a 5000 byte limit per request — chunk if needed
    MAX_BYTES = 4800
    chunks = _chunk_text(text, MAX_BYTES)
    logger.info(f"Converting '{title}' — {len(chunks)} chunk(s)")

    client = _get_tts_client()
    audio_parts = []

    for chunk in chunks:
        synthesis_input = tts.SynthesisInput(text=chunk)
        voice = tts.VoiceSelectionParams(
            language_code="en-US",
            name=req.voice_name,
        )
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.MP3,
            speaking_rate=req.speaking_rate,
            pitch=req.pitch,
        )
        response = await asyncio.to_thread(
            client.synthesize_speech,
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        audio_parts.append(response.audio_content)

    combined = b"".join(audio_parts)
    safe_title = _safe_filename(title)
    # Sanitize header value: strip control characters to prevent header injection
    header_title = re.sub(r'[\r\n\x00]', '', title)

    return StreamingResponse(
        io.BytesIO(combined),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_title}.mp3"',
            "X-Article-Title": header_title,
            "Content-Length": str(len(combined)),
        },
    )


def _chunk_text(text: str, max_bytes: int) -> list[str]:
    """Split text into chunks that fit within Vertex TTS byte limit."""
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.replace("\n", " "))
    chunks, current = [], ""
    for sentence in sentences:
        candidate = current + " " + sentence if current else sentence
        if len(candidate.encode("utf-8")) > max_bytes:
            if current:
                chunks.append(current.strip())
            # Handle single sentence exceeding limit — split on word boundaries
            if len(sentence.encode("utf-8")) > max_bytes:
                words = sentence.split()
                part = ""
                for word in words:
                    test = part + " " + word if part else word
                    if len(test.encode("utf-8")) > max_bytes:
                        if part:
                            chunks.append(part.strip())
                        part = word
                    else:
                        part = test
                current = part
            else:
                current = sentence
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_bytes]]


def _safe_filename(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
