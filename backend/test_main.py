import os
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Set required env vars before importing main
os.environ.setdefault("GOOGLE_CLIENT_ID", "test-client-id")
os.environ.setdefault("ALLOWED_EMAILS", "test@example.com")
os.environ.setdefault("FRONTEND_ORIGIN", "http://localhost:3000")

from fastapi.testclient import TestClient
from fastapi import HTTPException
from google.api_core.exceptions import InvalidArgument

from main import (
    app,
    _chunk_text,
    _safe_filename,
    _validate_url_safe,
    _merge_article_elements,
    _replace_code_blocks,
    fetch_article_text,
    verify_google_token,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def client():
    """TestClient with auth dependency overridden."""
    app.dependency_overrides[verify_google_token] = lambda: {"email": "test@example.com"}
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def unauthed_client():
    """TestClient without auth override — tests real auth rejection."""
    app.dependency_overrides.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------
class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "Hello world. This is a test."
        chunks = _chunk_text(text, 4800)
        assert chunks == [text]

    def test_splits_on_sentence_boundaries(self):
        # Each sentence is ~25 bytes; max_bytes=40 should force splits
        text = "First sentence here now. Second sentence here now. Third sentence here now."
        chunks = _chunk_text(text, 40)
        assert len(chunks) >= 2
        # All original text is preserved
        joined = " ".join(chunks)
        assert "First sentence" in joined
        assert "Third sentence" in joined

    def test_respects_byte_limit(self):
        text = ". ".join(f"Sentence number {i}" for i in range(100))
        chunks = _chunk_text(text, 200)
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 200

    def test_handles_single_long_sentence(self):
        # A single sentence exceeding the limit should be split on words
        text = "word " * 500
        chunks = _chunk_text(text, 100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 100

    def test_empty_text_returns_chunk(self):
        chunks = _chunk_text("", 4800)
        assert len(chunks) >= 1

    def test_multibyte_characters(self):
        text = "Héllo wörld. Ünïcödé tëxt. Möre séntëncés."
        chunks = _chunk_text(text, 30)
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 30

    def test_newlines_replaced(self):
        text = "Line one.\nLine two.\nLine three."
        chunks = _chunk_text(text, 4800)
        # Newlines are replaced with spaces
        for chunk in chunks:
            assert "\n" not in chunk


# ---------------------------------------------------------------------------
# _safe_filename
# ---------------------------------------------------------------------------
class TestSafeFilename:
    def test_normal_title(self):
        assert _safe_filename("My Article Title") == "My Article Title"

    def test_special_characters_replaced(self):
        result = _safe_filename('Hello <World> "Test"')
        assert "<" not in result
        assert ">" not in result
        assert '"' not in result

    def test_truncates_at_80(self):
        long_title = "A" * 100
        assert len(_safe_filename(long_title)) == 80

    def test_preserves_hyphens_underscores(self):
        assert _safe_filename("my-article_2024") == "my-article_2024"


# ---------------------------------------------------------------------------
# _validate_url_safe (SSRF protection)
# ---------------------------------------------------------------------------
class TestValidateUrlSafe:
    def test_rejects_private_ip(self):
        with patch("main.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("192.168.1.1", 80)),
        ]):
            with pytest.raises(HTTPException) as exc_info:
                _validate_url_safe("http://internal.example.com/secret")
            assert exc_info.value.status_code == 400

    def test_rejects_loopback(self):
        with patch("main.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("127.0.0.1", 80)),
        ]):
            with pytest.raises(HTTPException) as exc_info:
                _validate_url_safe("http://localhost/secret")
            assert exc_info.value.status_code == 400

    def test_rejects_non_http_scheme(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_url_safe("ftp://example.com/file")
        assert exc_info.value.status_code == 400
        assert "HTTP(S)" in exc_info.value.detail

    def test_rejects_unresolvable_host(self):
        import socket
        with patch("main.socket.getaddrinfo", side_effect=socket.gaierror):
            with pytest.raises(HTTPException) as exc_info:
                _validate_url_safe("http://doesnotexist.invalid/page")
            assert exc_info.value.status_code == 400

    def test_allows_public_ip(self):
        with patch("main.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("93.184.216.34", 80)),
        ]):
            # Should not raise
            _validate_url_safe("http://example.com/page")


# ---------------------------------------------------------------------------
# _merge_article_elements
# ---------------------------------------------------------------------------
class TestMergeArticleElements:
    def test_no_articles_unchanged(self):
        html = "<html><body><p>Hello</p></body></html>"
        assert _merge_article_elements(html) == html

    def test_single_article_unchanged(self):
        html = "<html><body><article><p>Content</p></article></body></html>"
        result = _merge_article_elements(html)
        assert "Content" in result

    def test_multiple_articles_merged(self):
        html = """<html><body><div><div>
            <article><p>Part one content that is long enough.</p></article>
            <article><p>Part two content that is long enough.</p></article>
        </div></div></body></html>"""
        result = _merge_article_elements(html)
        assert "Part one" in result
        assert "Part two" in result

    def test_dominant_article_not_merged(self):
        # One article has >70% of text — should not merge
        html = """<html><body><div><div>
            <article><p>{}</p></article>
            <article><p>Short</p></article>
        </div></div></body></html>""".format("Long content. " * 50)
        result = _merge_article_elements(html)
        # Should return original html (not merged)
        assert "Short" in result


# ---------------------------------------------------------------------------
# _replace_code_blocks
# ---------------------------------------------------------------------------
class TestReplaceCodeBlocks:
    def test_replaces_pre_blocks(self):
        html = "<html><body><pre><code>x = 1</code></pre><p>Text</p></body></html>"
        result = _replace_code_blocks(html)
        assert "x = 1" not in result
        assert "code block was excluded" in result

    def test_keeps_inline_code(self):
        html = "<html><body><p>Use <code>pip install</code> to install.</p></body></html>"
        result = _replace_code_blocks(html)
        assert "pip install" in result

    def test_replaces_standalone_code(self):
        html = "<html><body><div><code>standalone code block</code></div></body></html>"
        result = _replace_code_blocks(html)
        assert "standalone code block" not in result


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
class TestAuth:
    def test_extract_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/extract", json={"url": "http://example.com"})
        assert resp.status_code in (401, 403)

    def test_tts_requires_auth(self, unauthed_client):
        resp = unauthed_client.post("/tts", json={"url": "http://example.com"})
        assert resp.status_code in (401, 403)

    def test_health_no_auth_needed(self, unauthed_client):
        resp = unauthed_client.get("/health")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Extract endpoint
# ---------------------------------------------------------------------------
class TestExtractEndpoint:
    def test_extract_success(self, client):
        fake_text = "This is a long enough article body. " * 10
        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("Test Title", fake_text)):
            resp = client.post("/extract", json={"url": "http://example.com/article"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["title"] == "Test Title"
        assert data["text"] == fake_text
        assert data["word_count"] > 0
        assert "estimated_minutes" in data

    def test_extract_invalid_url(self, client):
        resp = client.post("/extract", json={"url": "not-a-url"})
        assert resp.status_code == 422

    def test_extract_extraction_failure(self, client):
        with patch("main.fetch_article_text", new_callable=AsyncMock, side_effect=HTTPException(status_code=422, detail="Could not extract")):
            resp = client.post("/extract", json={"url": "http://example.com/empty"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# TTS endpoint
# ---------------------------------------------------------------------------
class TestTTSEndpoint:
    def test_tts_streams_audio(self, client):
        fake_text = "This is a test article. " * 20
        mock_response = MagicMock()
        mock_response.audio_content = b"\xff\xfb\x90\x00" * 100  # fake MP3 bytes

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("Test", fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            resp = client.post("/tts", json={"url": "http://example.com/article"})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/mpeg"
        assert "X-Article-Title" in resp.headers
        assert len(resp.content) > 0

    def test_tts_custom_voice_params(self, client):
        fake_text = "Short article content. " * 10
        mock_response = MagicMock()
        mock_response.audio_content = b"\xff\xfb\x90\x00"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("Test", fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            resp = client.post("/tts", json={
                "url": "http://example.com/article",
                "voice_name": "en-US-Chirp3-HD-Leda",
                "speaking_rate": 1.5,
                "pitch": 2.0,
            })

        assert resp.status_code == 200

    def test_tts_invalid_speaking_rate(self, client):
        resp = client.post("/tts", json={
            "url": "http://example.com/article",
            "speaking_rate": 10.0,  # exceeds max 4.0
        })
        assert resp.status_code == 422

    def test_tts_content_disposition_header(self, client):
        fake_text = "Article content here. " * 10
        mock_response = MagicMock()
        mock_response.audio_content = b"\xff\xfb\x90\x00"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("My Article", fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            resp = client.post("/tts", json={"url": "http://example.com/article"})

        assert "My Article.mp3" in resp.headers["content-disposition"]


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------
class TestRequestValidation:
    def test_tts_pitch_out_of_range(self, client):
        resp = client.post("/tts", json={
            "url": "http://example.com",
            "pitch": -25.0,
        })
        assert resp.status_code == 422

    def test_extract_missing_url(self, client):
        resp = client.post("/extract", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# verify_google_token
# ---------------------------------------------------------------------------
class TestVerifyGoogleToken:
    def test_valid_token_allowed_email(self):
        fake_id_info = {"email": "test@example.com", "sub": "123"}
        with patch("main.id_token.verify_oauth2_token", return_value=fake_id_info):
            creds = MagicMock()
            creds.credentials = "fake-token"
            result = verify_google_token(creds)
        assert result["email"] == "test@example.com"

    def test_invalid_token_raises_401(self):
        with patch("main.id_token.verify_oauth2_token", side_effect=ValueError("bad token")):
            creds = MagicMock()
            creds.credentials = "bad-token"
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token(creds)
            assert exc_info.value.status_code == 401

    def test_unauthorized_email_raises_403(self):
        fake_id_info = {"email": "hacker@evil.com", "sub": "456"}
        with patch("main.id_token.verify_oauth2_token", return_value=fake_id_info):
            creds = MagicMock()
            creds.credentials = "valid-token"
            with pytest.raises(HTTPException) as exc_info:
                verify_google_token(creds)
            assert exc_info.value.status_code == 403


# ---------------------------------------------------------------------------
# fetch_article_text
# ---------------------------------------------------------------------------
class TestFetchArticleText:
    @pytest.mark.anyio
    async def test_successful_extraction(self):
        article_html = "<html><body><article><p>{}</p></article></body></html>".format(
            "This is a real article with enough content. " * 10
        )
        mock_resp = MagicMock()
        mock_resp.text = article_html
        mock_resp.raise_for_status = MagicMock()

        with patch("main._validate_url_safe"), \
             patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            title, text = await fetch_article_text("http://example.com/article")
        assert len(text) > 100

    @pytest.mark.anyio
    async def test_short_content_raises_422(self):
        article_html = "<html><body><p>Short</p></body></html>"
        mock_resp = MagicMock()
        mock_resp.text = article_html
        mock_resp.raise_for_status = MagicMock()

        with patch("main._validate_url_safe"), \
             patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await fetch_article_text("http://example.com/empty")
            assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# TTS InvalidArgument retry behavior
# ---------------------------------------------------------------------------
class TestTTSRetryBehavior:
    def test_tts_retries_on_long_sentences(self, client):
        """When TTS returns 'sentences that are too long', it should re-split and retry."""
        fake_text = "A very long sentence. " * 20
        good_response = MagicMock()
        good_response.audio_content = b"\xff\xfb\x90\x00" * 10

        mock_client = MagicMock()
        # First call raises InvalidArgument, subsequent calls succeed
        mock_client.synthesize_speech.side_effect = [
            InvalidArgument("sentences that are too long"),
            good_response,
            good_response,
            good_response,
            good_response,
        ]

        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("Test", fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            resp = client.post("/tts", json={"url": "http://example.com/article"})

        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_tts_raises_on_non_sentence_invalid_argument(self, client):
        """InvalidArgument errors NOT about long sentences should propagate."""
        fake_text = "Some article text. " * 10
        mock_client = MagicMock()
        mock_client.synthesize_speech.side_effect = InvalidArgument("some other error")

        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=("Test", fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            # StreamingResponse raises during iteration, so the error propagates
            with pytest.raises(InvalidArgument):
                client.post("/tts", json={"url": "http://example.com/article"})


# ---------------------------------------------------------------------------
# Header injection protection
# ---------------------------------------------------------------------------
class TestHeaderSanitization:
    def test_title_with_newlines_sanitized(self, client):
        fake_text = "Article content here. " * 10
        mock_response = MagicMock()
        mock_response.audio_content = b"\xff\xfb\x90\x00"

        mock_client = MagicMock()
        mock_client.synthesize_speech.return_value = mock_response

        malicious_title = "Good Title\r\nX-Injected: evil"
        with patch("main.fetch_article_text", new_callable=AsyncMock, return_value=(malicious_title, fake_text)), \
             patch("main._get_tts_client", return_value=mock_client):
            resp = client.post("/tts", json={"url": "http://example.com/article"})

        assert resp.status_code == 200
        header_val = resp.headers["X-Article-Title"]
        assert "\r" not in header_val
        assert "\n" not in header_val


# ---------------------------------------------------------------------------
# SSRF edge cases
# ---------------------------------------------------------------------------
class TestSSRFEdgeCases:
    def test_rejects_link_local(self):
        with patch("main.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("169.254.1.1", 80)),
        ]):
            with pytest.raises(HTTPException) as exc_info:
                _validate_url_safe("http://metadata.example.com/")
            assert exc_info.value.status_code == 400

    def test_rejects_ipv6_loopback(self):
        with patch("main.socket.getaddrinfo", return_value=[
            (None, None, None, None, ("::1", 80, 0, 0)),
        ]):
            with pytest.raises(HTTPException) as exc_info:
                _validate_url_safe("http://example.com/")
            assert exc_info.value.status_code == 400

    def test_rejects_empty_hostname(self):
        with pytest.raises(HTTPException) as exc_info:
            _validate_url_safe("http:///no-host")
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# _chunk_text edge cases
# ---------------------------------------------------------------------------
class TestChunkTextEdgeCases:
    def test_all_text_preserved(self):
        """Concatenated chunks should contain all original words."""
        text = ". ".join(f"Sentence number {i} with some words" for i in range(50))
        chunks = _chunk_text(text, 200)
        joined = " ".join(chunks)
        for i in range(50):
            assert f"Sentence number {i}" in joined

    def test_single_word_exceeding_limit(self):
        text = "a" * 200
        chunks = _chunk_text(text, 100)
        # Single word can't be split further, so it stays as one chunk
        assert len(chunks) >= 1

    def test_very_small_limit(self):
        text = "Hello. World."
        chunks = _chunk_text(text, 10)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk.encode("utf-8")) <= 10 or len(chunk.split()) == 1


# ---------------------------------------------------------------------------
# fetch_article_text: favor_recall extracts all articles
# ---------------------------------------------------------------------------
class TestFavorRecallExtraction:
    @pytest.mark.anyio
    async def test_multi_article_extracts_all_sections(self):
        """Extraction with multiple <article> elements should return content from all of them."""
        sections = [
            f"<article><p>Section {i} has unique content about topic-{i} that is long enough to matter. " * 3 + "</p></article>"
            for i in range(1, 8)
        ]
        html = f"<html><body>{''.join(sections)}</body></html>"

        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()

        with patch("main._validate_url_safe"), \
             patch("main.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_resp
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            title, text = await fetch_article_text("http://example.com/multi-article")

        # Content from early AND late sections should be present
        assert "topic-1" in text
        assert "topic-5" in text
        assert "topic-7" in text
