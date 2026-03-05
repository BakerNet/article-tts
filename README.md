# Listener — Article TTS

Personal article-to-audio tool using Vertex AI Chirp 3 HD voices. Paste a URL, get an MP3.

## Architecture

```
PWA (any static host)  →  FastAPI on Cloud Run  →  Vertex AI TTS
                       ↑
              Google OAuth2 ID tokens (only your email allowed)
```

---

## Prerequisites

- GCP project with billing enabled
- `gcloud` CLI installed and authed
- Python 3.12+ (for local dev)
- A static file host for the frontend (Firebase Hosting, Cloudflare Pages, or any CDN — even GitHub Pages works)

---

## Step 1 — Enable GCP APIs

```bash
gcloud services enable \
  run.googleapis.com \
  texttospeech.googleapis.com \
  cloudbuild.googleapis.com
```

---

## Step 2 — Create a Google OAuth Web Client

1. Go to **GCP Console → APIs & Services → Credentials**
2. Click **Create Credentials → OAuth client ID**
3. Application type: **Web application**
4. Name: `Listener PWA`
5. Authorised JavaScript origins: add your PWA's URL (e.g. `https://listener.yourdomain.com`)
6. Click **Create** — copy the **Client ID** (looks like `xxxx.apps.googleusercontent.com`)

---

## Step 3 — Deploy the backend to Cloud Run

```bash
cd backend

# Build and deploy in one step
gcloud run deploy article-tts \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars "GOOGLE_CLIENT_ID=YOUR_CLIENT_ID_HERE,ALLOWED_EMAIL=you@gmail.com,GCP_PROJECT=your-project-id,FRONTEND_ORIGIN=https://listener.yourdomain.com" \
  --memory 512Mi \
  --timeout 120
```

Copy the Cloud Run URL from the output (e.g. `https://article-tts-xxxx-uc.a.run.app`).

> **Note on `--allow-unauthenticated`**: Cloud Run is open to the internet but the
> FastAPI app enforces Google OAuth — only your email can call it. This is fine for
> personal use. If you want double-layered security, remove this flag and use Cloud Run's
> built-in IAM auth instead (more complex setup).

### Grant Vertex AI access to the Cloud Run service account

```bash
# Find the service account Cloud Run is using
gcloud run services describe article-tts --region us-central1 --format='value(spec.template.spec.serviceAccountName)'

# Grant it Vertex AI user role (replace SA_EMAIL with the above output)
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:SA_EMAIL" \
  --role="roles/aiplatform.user"
```

---

## Step 4 — Configure and deploy the frontend

Edit `frontend/index.html` and replace the two placeholders:

```
REPLACE_WITH_YOUR_GOOGLE_CLIENT_ID  →  your OAuth client ID
REPLACE_WITH_YOUR_CLOUD_RUN_URL     →  your Cloud Run URL
```

Also add the service worker registration snippet just before `</body>`:

```html
<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js');
}
</script>
```

### Deploy to Firebase Hosting (simplest GCP-native option)

```bash
npm install -g firebase-tools
firebase login
firebase init hosting   # select your GCP project, set public dir to "frontend"
firebase deploy
```

### Or deploy to Cloudflare Pages (free, fast)

```bash
npx wrangler pages deploy frontend --project-name listener
```

---

## Step 5 — Add to Android home screen

1. Open Chrome on your Android phone
2. Navigate to your PWA URL
3. Sign in with Google
4. Tap the **⋮** menu → **Add to Home screen**
5. Done — it'll appear as a standalone app icon

---

## Local development

```bash
cd backend
pip install -r requirements.txt

export GOOGLE_CLIENT_ID="your-client-id"
export ALLOWED_EMAIL="you@gmail.com"
export GCP_PROJECT="your-project-id"
export FRONTEND_ORIGIN="http://localhost:3000"

# Uses your local gcloud credentials for Vertex AI
gcloud auth application-default login
uvicorn main:app --reload --port 8080
```

For the frontend, just serve the directory:

```bash
cd frontend
python -m http.server 3000
```

---

## Voice options (Chirp 3 HD)

| Voice name                     | Gender | Character          |
|-------------------------------|--------|--------------------|
| `en-US-Chirp3-HD-Charon`      | M      | Deep, authoritative|
| `en-US-Chirp3-HD-Aoede`       | F      | Warm, narrative    |
| `en-US-Chirp3-HD-Fenrir`      | M      | Energetic          |
| `en-US-Chirp3-HD-Kore`        | F      | Clear, bright      |

---

## Estimated costs

Chirp 3 HD pricing (as of early 2026): ~$0.000160 per character.
An average 1500-word article ≈ 8000 characters ≈ **~$0.013 per article**.

Cloud Run: effectively free for personal use (generous free tier).

---

## Troubleshooting

**"Could not extract readable text"** — some sites block scrapers. Try the Reader Mode URL or a cached version.

**401 errors** — your Google ID token may have expired (they last 1 hour). Refresh the page.

**Audio cuts off** — the chunking logic splits on sentences. Very long articles are split into multiple TTS calls and concatenated; if a chunk boundary lands badly, adjust `MAX_BYTES` in `main.py`.
