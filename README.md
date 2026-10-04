# KENAI OMEGA — Web Frontend + Music Engine API

A combined HTML page + Vercel-hosted API for two things:

1. **Music Engine** — fully working. Generates original songs (drums, bass,
   chords, motif-developed melody, formant-synthesized vocals) across 11
   genres, entirely offline, in under a couple of seconds.
2. **Chat endpoint** — a **stub only**. See the "What's real vs. what's a
   stub" section below before you assume this talks to your actual OMEGA
   backend — it doesn't, yet.

## File structure

```
.
├── api/
│   └── index.py          # Flask app — Vercel auto-detects the `app` object
├── lib/
│   └── ai/music/          # The actual music engine (theory, drums, melody,
│                          # vocals, arrangement) — pure Python + numpy/scipy
├── public/
│   └── index.html         # The combined frontend (Music Engine + Chat UI)
├── vercel.json             # Routes /api/* to the Flask app, everything
│                          # else to public/
├── requirements.txt        # flask, numpy, scipy
└── .gitignore
```

## What's real vs. what's a stub

**Real and tested:** `/api/generate-music`. This calls the actual
`KenaiMusicEngine` — the same engine, vectorized with numpy/scipy so a
16-32 bar song renders in well under a second instead of the multi-minute
stalls an earlier, non-vectorized version hit. Verified end-to-end
(HTTP → Flask → engine → valid WAV) before this zip was put together.

**Stub only:** `/api/chat`. I do not have your actual KENAI OMEGA
backend source code — only your project notes describing its architecture
(Omega Kernel → Identity/Context/Guardian → Nexus Memory → Brain → Model
Fabric → ...). I never received the real Python files, so I can't include
real OMEGA logic here. `chat_stub_endpoint()` in `api/index.py` returns a
canned response in the same envelope shape your notes describe
(`kenai.response.v1`), clearly labeled as a stub, so the frontend has
something real to call.

## Why your real OMEGA backend can't just live on Vercel too

Per your own project notes, OMEGA's backend depends on:
- persistent SQLite (WAL mode) for memory, sessions, and audit trails
- subprocess-based code sandboxing with RSS memory polling
- long-running rate limiting (Guardian) backed by that same SQLite

None of that survives on Vercel's serverless model: the filesystem is
ephemeral and reset on every cold start, there's no persistent process
between requests, and function execution is time-limited (10s on the free
Hobby tier, 60s on Pro). A genuinely stateful backend like this needs a
host with a persistent filesystem and long-running process — a small VPS,
Render, Railway, Fly.io, or simply your Termux device itself kept
reachable (e.g. via Tailscale/ngrok).

**To wire the real backend in once it's hosted somewhere persistent:**
replace the body of `chat_stub_endpoint()` in `api/index.py` with an HTTP
call to that host's `/api/v1/chat` endpoint (matching the contract your
notes already describe), forward the response through, and remove the
stub labeling in `public/index.html`. Everything else — routing, response
envelope shape, the frontend UI — is already built to make that a small,
contained change.

## Deploying

1. Push this whole folder to a new GitHub repo.
2. Go to [vercel.com/new](https://vercel.com/new), import that repo.
3. Vercel will detect `vercel.json` and `requirements.txt` automatically —
   no build command needed, no environment variables required for the
   music engine to work out of the box.
4. Deploy. Your music generator will be live at
   `https://<your-project>.vercel.app/`.

## Local testing before you deploy

```bash
pip install -r requirements.txt
python3 -c "
import sys; sys.path.insert(0, 'api'); sys.path.insert(0, 'lib')
from api.index import app
app.run(port=5055)
"
```
Then open `public/index.html` directly, or serve it alongside the API and
visit `http://localhost:5055/`.

## API reference

| Endpoint | Method | Body | Returns |
|---|---|---|---|
| `/api/generate-music` | POST | `{prompt, genre?, bars?, vocals?, seed?}` | `audio/wav` binary |
| `/api/genres` | GET | — | `{success, genres: [...]}` |
| `/api/chat` | POST | `{message}` | `kenai.response.v1` envelope (**stub**) |
| `/api/health` | GET | — | `{success, status}` |

`bars` is clamped to 4–32 server-side to keep renders comfortably inside
Vercel's execution time limit on every plan tier.
