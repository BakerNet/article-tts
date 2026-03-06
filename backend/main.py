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
from google.api_core.exceptions import InvalidArgument
from google.cloud import texttospeech_v1beta1 as tts
from lxml import etree
import trafilatura
import httpx
from lxml import html as lxml_html
from lxml.html import tostring as lxml_tostring

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Article TTS API")

# ---------------------------------------------------------------------------
# Config — set these as Cloud Run environment variables
# ---------------------------------------------------------------------------
GOOGLE_CLIENT_ID = os.environ["GOOGLE_CLIENT_ID"]       # OAuth Web Client ID
ALLOWED_EMAILS   = {e.strip().lower() for e in os.environ["ALLOWED_EMAILS"].split(",")}  # comma-separated list
FRONTEND_ORIGIN  = os.environ["FRONTEND_ORIGIN"]            # Your PWA origin e.g. https://tts.yourdomain.com

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
    if email.lower() not in ALLOWED_EMAILS:
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
    voice_name: str = "en-US-Chirp3-HD-Aoede"
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
# HTML pre-processing
# ---------------------------------------------------------------------------
def _merge_article_elements(html: str) -> str:
    """Merge sibling <article> elements into one so trafilatura sees all content.

    Some sites (e.g. Discord blog) split a single post across multiple <article>
    tags under a common parent.  We only merge articles that share the same
    grandparent to avoid pulling in unrelated "related posts" widgets.
    """
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return html
    articles = tree.xpath("//article")
    if len(articles) <= 1:
        return html

    # Group articles by grandparent element
    from collections import defaultdict
    groups: dict[int, list] = defaultdict(list)
    for a in articles:
        parent = a.getparent()
        grandparent = parent.getparent() if parent is not None else parent
        groups[id(grandparent)].append(a)

    # Find the largest group with more than one article
    best = max(groups.values(), key=lambda g: sum(len(a.text_content() or "") for a in g))
    if len(best) <= 1:
        return html

    # If one article already dominates (>70% of total text), trafilatura will
    # extract it fine — merging would only add junk like "related posts".
    sizes = [len(a.text_content() or "") for a in best]
    total = sum(sizes)
    if total > 0 and max(sizes) / total > 0.7:
        return html

    merged_inner = "".join(lxml_tostring(a, encoding="unicode") for a in best)
    return f"<html><body><article>{merged_inner}</article></body></html>"


def _replace_code_blocks(html: str) -> str:
    """Replace <pre> and standalone <code> blocks with a notice for TTS."""
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return html

    replacement = "A code block was excluded from this reading."

    # Replace <pre> elements (which usually wrap <code> blocks)
    for el in tree.xpath("//pre"):
        placeholder = etree.Element("p")
        placeholder.text = replacement
        el.getparent().replace(el, placeholder)

    # Replace remaining standalone <code> blocks that aren't inline.
    # Inline <code> (inside a <p>, <span>, <li>, etc.) is kept as-is since
    # it's typically just a word or short phrase.
    for el in tree.xpath("//code"):
        parent = el.getparent()
        if parent is not None and parent.tag in ("p", "span", "li", "td", "th", "a", "em", "strong", "h1", "h2", "h3", "h4", "h5", "h6"):
            continue
        placeholder = etree.Element("p")
        placeholder.text = replacement
        parent.replace(el, placeholder)

    return lxml_tostring(tree, encoding="unicode")


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

    # Some sites (e.g. Discord blog) split content across multiple <article>
    # elements. Trafilatura only extracts the first one, so merge them.
    html = _merge_article_elements(html)
    html = _replace_code_blocks(html)

    result = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False,
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

    voice = tts.VoiceSelectionParams(
        language_code="en-US",
        name=req.voice_name,
    )
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.MP3,
        speaking_rate=req.speaking_rate,
        pitch=req.pitch,
    )

    for i, chunk in enumerate(chunks):
        try:
            response = await asyncio.to_thread(
                client.synthesize_speech,
                input=tts.SynthesisInput(text=chunk),
                voice=voice,
                audio_config=audio_config,
            )
            audio_parts.append(response.audio_content)
        except InvalidArgument as e:
            error_msg = str(e)
            if "sentences that are too long" in error_msg:
                # Re-split this chunk more aggressively and retry
                logger.warning(f"Chunk {i} had sentences too long, re-splitting: {error_msg}")
                sub_text = _split_long_sentences(chunk, max_sentence_chars=200)
                sub_chunks = _chunk_text(sub_text, MAX_BYTES)
                for sub_chunk in sub_chunks:
                    try:
                        response = await asyncio.to_thread(
                            client.synthesize_speech,
                            input=tts.SynthesisInput(text=sub_chunk),
                            voice=voice,
                            audio_config=audio_config,
                        )
                        audio_parts.append(response.audio_content)
                    except InvalidArgument:
                        logger.error(f"Chunk still too long after re-split, skipping: {sub_chunk[:80]}...")
                        continue
            else:
                raise

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


def _split_long_sentences(text: str, max_sentence_chars: int = 400) -> str:
    """Break long sentences at natural punctuation so TTS doesn't reject them.

    Google Cloud TTS rejects individual sentences that are too long, even if
    the overall request is within the byte limit.  This function inserts
    sentence-ending periods at natural break-points (semicolons, colons,
    em-dashes, commas before conjunctions) so every "sentence" stays short.
    """
    # Split into existing sentences first
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    result: list[str] = []

    for sentence in raw_sentences:
        if len(sentence) <= max_sentence_chars:
            result.append(sentence)
            continue

        # Try splitting at natural break-points, ordered by preference
        fragments = _split_at_break_points(sentence, max_sentence_chars)
        result.extend(fragments)

    return " ".join(result)


def _split_at_break_points(sentence: str, max_chars: int) -> list[str]:
    """Split a single long sentence at natural punctuation boundaries."""
    # Break-point patterns ordered by preference (strongest to weakest)
    break_patterns = [
        r';\s+',                          # semicolons
        r':\s+',                          # colons
        r'\s*[—–]\s*',                    # em/en-dashes
        r',\s+(?=and |or |but |so |yet )',  # comma before conjunction
        r',\s+',                          # any comma
    ]

    fragments: list[str] = []
    remaining = sentence

    for pattern in break_patterns:
        if len(remaining) <= max_chars:
            break
        new_remaining_parts: list[str] = []
        for part in [remaining] if not new_remaining_parts else new_remaining_parts:
            if len(part) <= max_chars:
                new_remaining_parts.append(part)
                continue
            pieces = re.split(pattern, part)
            new_remaining_parts.extend(pieces)

        # Reassemble pieces that fit, adding periods to create sentence breaks
        merged: list[str] = []
        current = ""
        for piece in new_remaining_parts:
            piece = piece.strip()
            if not piece:
                continue
            if not current:
                current = piece
            elif len(current) + 1 + len(piece) <= max_chars:
                current = current + " " + piece
            else:
                # End the current fragment with a period if it doesn't have one
                if current and current[-1] not in ".!?":
                    current = current.rstrip(",;:—–- ") + "."
                merged.append(current)
                current = piece
        if current:
            merged.append(current)

        if merged and all(len(m) <= max_chars for m in merged):
            fragments = merged
            remaining = ""
            break
        elif merged:
            # Some fragments still too long — continue with next pattern
            fragments = [m for m in merged if len(m) <= max_chars]
            remaining_parts = [m for m in merged if len(m) > max_chars]
            remaining = " ".join(remaining_parts) if remaining_parts else ""

    # Last resort: force-split on word boundaries
    if remaining:
        words = remaining.split()
        current = ""
        for word in words:
            test = current + " " + word if current else word
            if len(test) > max_chars:
                if current:
                    if current[-1] not in ".!?":
                        current = current.rstrip(",;:—–- ") + "."
                    fragments.append(current)
                current = word
            else:
                current = test
        if current:
            fragments.append(current)

    return fragments if fragments else [sentence]


def _chunk_text(text: str, max_bytes: int) -> list[str]:
    """Split text into chunks that fit within Vertex TTS byte limit."""
    # First, break up any long sentences that TTS would reject
    text = _split_long_sentences(text)

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
    if not chunks:
        encoded = text.encode("utf-8")[:max_bytes]
        return [encoded.decode("utf-8", errors="ignore").strip()]
    return chunks


def _safe_filename(title: str) -> str:
    return "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
