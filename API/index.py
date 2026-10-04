"""
KENAI combined API for Vercel's Python serverless runtime.

Two endpoints:
  POST /api/generate-music  -> fully working, wraps the real KENAI Music Engine
  POST /api/chat            -> STUB ONLY, see the big comment block below

Vercel auto-detects a WSGI `app` object exported from this file when
api/index.py (or any file under /api) defines one — no extra config needed
beyond vercel.json routing everything under /api/* here.
"""

import os
import sys
import time
import uuid

from flask import Flask, request, jsonify, Response

# Make the sibling `lib/` directory (containing `ai/music/...`) importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))

from ai.music import KenaiMusicEngine, GENRES, detect_genre_from_prompt  # noqa: E402

app = Flask(__name__)

MAX_BARS = 32  # keep renders comfortably inside Vercel's execution time limit


@app.route("/api/generate-music", methods=["POST"])
def generate_music_endpoint():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "Missing 'prompt' in request body."}), 400

    genre = body.get("genre")
    if genre and genre not in GENRES:
        return jsonify({
            "success": False,
            "error": f"Unknown genre '{genre}'.",
            "valid_genres": sorted(GENRES.keys()),
        }), 400

    bars = int(body.get("bars", 16))
    bars = max(4, min(MAX_BARS, bars))
    vocals = bool(body.get("vocals", True))
    seed = body.get("seed")
    seed = int(seed) if seed is not None else None

    try:
        t0 = time.time()
        engine = KenaiMusicEngine(seed=seed)
        audio_bytes = engine.generate(prompt, bars=bars, genre=genre, vocals=vocals, seed=seed)
        elapsed = time.time() - t0
    except Exception as e:
        return jsonify({"success": False, "error": f"Generation failed: {e}"}), 500

    resolved_genre = genre or detect_genre_from_prompt(prompt)
    return Response(
        audio_bytes,
        mimetype="audio/wav",
        headers={
            "Content-Disposition": f'inline; filename="{uuid.uuid4().hex}.wav"',
            "X-Kenai-Genre": resolved_genre,
            "X-Kenai-Bars": str(bars),
            "X-Kenai-Render-Seconds": f"{elapsed:.2f}",
        },
    )


@app.route("/api/genres", methods=["GET"])
def genres_endpoint():
    return jsonify({"success": True, "genres": sorted(GENRES.keys())})


# -----------------------------------------------------------------------
# CHAT STUB — NOT the real KENAI OMEGA kernel
# -----------------------------------------------------------------------
# The real OMEGA backend (Omega Kernel -> Identity/Context/Guardian ->
# Nexus Memory -> Brain -> Model Fabric -> ... ) depends on:
#   - persistent SQLite (WAL mode) for memory/sessions/audit
#   - subprocess-based code sandboxing
#   - long-running rate limiting (Guardian)
# None of that fits Vercel's serverless model: the filesystem is
# ephemeral and reset per invocation, there's no persistent process
# between requests, and execution is time-limited.
#
# This stub exists so the frontend has something real to call and so
# the response shape matches your documented kenai.response.v1 envelope.
# To go live, replace the body of this function with a call into your
# actual Kernel (e.g. an HTTP call to a persistently-hosted OMEGA
# instance on a VPS/Render/Railway/Fly.io/your Termux device itself,
# or — if you port the Kernel to be stateless — a direct in-process
# call). Everything else on this page (routing, envelope shape,
# frontend) is built to make that swap a one-function change.
# -----------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat_stub_endpoint():
    body = request.get_json(silent=True) or {}
    message = (body.get("message") or "").strip()

    envelope = {
        "contract": "kenai.response.v1",
        "success": True,
        "data": {
            "reply": (
                "This is the KENAI chat STUB, not the real Omega Kernel. "
                "Wire api/index.py's chat_stub_endpoint() up to your actual "
                "backend (hosted somewhere persistent, not this serverless "
                "function) to get real responses."
                + (f" You said: \u201c{message}\u201d" if message else "")
            ),
            "stub": True,
        },
        "error": None,
        "meta": {
            "request_id": uuid.uuid4().hex,
            "timestamp": time.time(),
        },
    }
    return jsonify(envelope)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"success": True, "status": "ok"})
  
