import os
import io
import logging
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
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
    voice_name: str = "en-US-Chirp3-HD-Charon"   # good Chirp 3 HD default
    speaking_rate: float = 1.0                     # 0.25 – 4.0
    pitch: float = 0.0                             # -20.0 – 20.0 semitones

# ---------------------------------------------------------------------------
# Article extraction helper
# ---------------------------------------------------------------------------
async def fetch_article_text(url: str) -> tuple[str, str]:
    """Returns (title, body_text)"""
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
        "estimated_minutes": round(word_count / 150),  # ~150 wpm
    }


@app.post("/tts")
async def text_to_speech(req: TTSRequest, _=Depends(verify_google_token)):
    """Extract article and stream back MP3 audio."""
    title, text = await fetch_article_text(str(req.url))

    # Vertex AI TTS has a 5000 byte limit per request — chunk if needed
    MAX_BYTES = 4800
    chunks = _chunk_text(text, MAX_BYTES)
    logger.info(f"Converting '{title}' — {len(chunks)} chunk(s)")

    client = tts.TextToSpeechClient()
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
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
        )
        audio_parts.append(response.audio_content)

    combined = b"".join(audio_parts)

    return StreamingResponse(
        io.BytesIO(combined),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{_safe_filename(title)}.mp3"',
            "X-Article-Title": title,
            "Content-Length": str(len(combined)),
        },
    )


def _chunk_text(text: str, max_bytes: int) -> list[str]:
    """Split text into chunks that fit within Vertex TTS byte limit."""
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""
    for sentence in sentences:
        candidate = current + sentence + ". "
        if len(candidate.encode("utf-8")) > max_bytes:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_bytes]]


def _safe_filename(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
