"""
KENAI Music Engine - Music Theory & Audio Utilities (vectorized)
===================================================================
Scales, chords, progressions, genre definitions, and low-level
waveform / envelope / mixing utilities used by every instrument.
All synthesis is vectorized with numpy — no per-sample Python loops.
"""

import os
import math
import random
import struct
import wave
import io
from typing import List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

SAMPLE_RATE = 44100

# ============================================================================
# MUSIC THEORY
# ============================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "minor_pentatonic": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}

CHORD_TYPES = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dom7": [0, 4, 7, 10],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
    "dim": [0, 3, 6],
    "add9": [0, 4, 7, 14],
}

PROGRESSIONS = {
    "reggae": [0, 3, 4, 0],
    "dancehall": [0, 5, 3, 4],
    "pop": [0, 3, 4, 0],
    "rnb": [0, 5, 3, 4],
    "lofi": [0, 3, 4, 3],
    "afrobeat": [0, 4, 5, 3],
    "trap": [0, 5, 3, 4],
    "rock": [0, 3, 4, 0],
    "ambient": [0, 3, 0, 4],
    "electronic": [0, 5, 3, 4],
    "hiphop": [0, 5, 3, 4],
}

GENRES = {
    "reggae": {
        "tempo": 75, "scale": "major", "progression": "reggae",
        "drum_pattern": "one_drop", "root_note": "C", "root_octave": 3,
        "instruments": ["drums", "bass", "skank", "organ", "melody"],
    },
    "dancehall": {
        "tempo": 95, "scale": "minor", "progression": "dancehall",
        "drum_pattern": "dancehall", "root_note": "A", "root_octave": 3,
        "instruments": ["drums", "bass", "skank", "synth", "melody"],
    },
    "lofi": {
        "tempo": 80, "scale": "major", "progression": "lofi",
        "drum_pattern": "lofi", "root_note": "F", "root_octave": 3,
        "instruments": ["drums", "bass", "chords", "melody"],
    },
    "hiphop": {
        "tempo": 90, "scale": "minor", "progression": "hiphop",
        "drum_pattern": "boom_bap", "root_note": "D", "root_octave": 2,
        "instruments": ["drums", "bass_808", "chords", "melody"],
    },
    "trap": {
        "tempo": 140, "scale": "minor", "progression": "trap",
        "drum_pattern": "trap", "root_note": "F", "root_octave": 2,
        "instruments": ["drums", "bass_808", "synth", "melody"],
    },
    "pop": {
        "tempo": 112, "scale": "major", "progression": "pop",
        "drum_pattern": "pop", "root_note": "G", "root_octave": 3,
        "instruments": ["drums", "bass", "chords", "melody"],
    },
    "rnb": {
        "tempo": 85, "scale": "major", "progression": "rnb",
        "drum_pattern": "rnb", "root_note": "E", "root_octave": 3,
        "instruments": ["drums", "bass", "chords", "melody"],
    },
    "afrobeat": {
        "tempo": 115, "scale": "major", "progression": "afrobeat",
        "drum_pattern": "afrobeat", "root_note": "D", "root_octave": 3,
        "instruments": ["drums", "bass", "guitar", "melody"],
    },
    "ambient": {
        "tempo": 60, "scale": "major", "progression": "ambient",
        "drum_pattern": "none", "root_note": "C", "root_octave": 3,
        "instruments": ["pads", "melody"],
    },
    "electronic": {
        "tempo": 128, "scale": "minor", "progression": "electronic",
        "drum_pattern": "four_floor", "root_note": "A", "root_octave": 2,
        "instruments": ["drums", "bass", "synth", "melody"],
    },
    "rock": {
        "tempo": 130, "scale": "minor", "progression": "rock",
        "drum_pattern": "rock", "root_note": "E", "root_octave": 2,
        "instruments": ["drums", "bass", "guitar", "melody"],
    },
}


def note_to_freq(note: str, octave: int = 4) -> float:
    semitone = NOTE_NAMES.index(note)
    midi = semitone + (octave + 1) * 12
    return 440.0 * (2 ** ((midi - 69) / 12))


def scale_degree_to_freq(root_freq: float, scale_name: str, degree: int) -> float:
    scale = SCALES.get(scale_name, SCALES["major"])
    semitones = scale[degree % len(scale)]
    octave_shift = (degree // len(scale)) * 12
    return root_freq * (2 ** ((semitones + octave_shift) / 12))


def detect_genre_from_prompt(prompt: str) -> str:
    text = prompt.lower()
    for genre in GENRES:
        if genre in text or genre.replace("hop", " hop") in text:
            return genre
    if any(w in text for w in ["chill", "relax", "study", "sleep", "lofi", "lo-fi"]):
        return "lofi"
    if any(w in text for w in ["sad", "melancholy", "dark", "atmospheric"]):
        return "ambient"
    if any(w in text for w in ["energy", "dance", "party", "edm", "club"]):
        return "electronic"
    if any(w in text for w in ["love", "smooth", "soul"]):
        return "rnb"
    if any(w in text for w in ["island", "sun", "beach", "irie"]):
        return "reggae"
    return "pop"


def detect_mood_params(prompt: str) -> dict:
    text = prompt.lower()
    tempo_mult = 1.0
    energy = 0.7
    if any(w in text for w in ["fast", "energetic", "hype", "upbeat"]):
        tempo_mult = 1.12
        energy = 0.9
    if any(w in text for w in ["slow", "chill", "calm", "relax", "sad"]):
        tempo_mult = 0.9
        energy = 0.5
    return {"tempo_mult": tempo_mult, "energy": energy}


# ============================================================================
# AUDIO UTILITIES (all vectorized — numpy is a hard requirement, not optional)
# ============================================================================
def _require_numpy():
    if not HAS_NUMPY:
        raise RuntimeError(
            "numpy is required for KENAI Music Engine audio synthesis "
            "(pip install numpy, or in Termux: pip install numpy --break-system-packages)"
        )


def make_sine(freq, duration, volume=0.5, sr=SAMPLE_RATE, phase=0.0):
    _require_numpy()
    n = max(1, int(sr * duration))
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t + phase) * volume


def make_saw(freq, duration, volume=0.5, sr=SAMPLE_RATE):
    _require_numpy()
    n = max(1, int(sr * duration))
    period = sr / freq if freq > 0 else 1
    t = np.arange(n)
    w = 2 * ((t % period) / period) - 1
    return w * volume


def make_square(freq, duration, volume=0.5, sr=SAMPLE_RATE, pulse_width=0.5):
    _require_numpy()
    n = max(1, int(sr * duration))
    period = sr / freq if freq > 0 else 1
    t = np.arange(n)
    return np.where((t % period) < period * pulse_width, volume, -volume)


def make_triangle(freq, duration, volume=0.5, sr=SAMPLE_RATE):
    _require_numpy()
    n = max(1, int(sr * duration))
    period = sr / freq if freq > 0 else 1
    t = np.arange(n)
    phase = (t % period) / period
    val = np.where(phase < 0.25, 4 * phase,
          np.where(phase < 0.75, 2 - 4 * phase, 4 * phase - 4))
    return val * volume


def make_noise(duration, volume=0.3, sr=SAMPLE_RATE):
    _require_numpy()
    n = max(1, int(sr * duration))
    return np.random.uniform(-volume, volume, n)


def blend(a, b, mix: float):
    n = min(len(a), len(b))
    return a[:n] * (1 - mix) + b[:n] * mix


def envelope_adsr(samples, attack=0.01, decay=0.1, sustain=0.7, release=0.2, sr=SAMPLE_RATE):
    _require_numpy()
    n = len(samples)
    a = max(1, int(sr * attack))
    d = max(1, int(sr * decay))
    r = max(1, int(sr * release))
    s_len = max(0, n - a - d - r)

    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s_len, sustain),
        np.linspace(sustain, 0, r),
    ])
    if len(env) < n:
        env = np.concatenate([env, np.zeros(n - len(env))])
    env = env[:n]
    return samples * env


def envelope_percussive(samples, decay=0.3, sr=SAMPLE_RATE):
    _require_numpy()
    n = len(samples)
    t = np.arange(n) / sr
    return samples * np.exp(-t / decay)


def _comb_filter(x, delay_samples: int, decay: float):
    """
    Efficient comb filter: y[n] = x[n] + decay * y[n - delay_samples].

    A naive lfilter with a sparse-but-huge denominator (order = delay_samples)
    costs O(n * delay_samples) because lfilter's direct-form solver doesn't
    exploit the sparsity. For a 50ms delay at 44.1kHz that's a ~2200-tap
    filter applied densely — the actual cause of multi-minute render times
    on longer songs in an earlier version of this engine.

    Fix: decompose the signal into `delay_samples` interleaved phases
    (polyphase decomposition). Within each phase the recursion is a plain
    order-1 IIR, and all phases are filtered together in one batched,
    vectorized scipy call. O(n) instead of O(n * delay_samples).
    """
    from scipy.signal import lfilter

    n = len(x)
    if delay_samples <= 0 or delay_samples >= n:
        return x.copy()

    pad = (-n) % delay_samples
    x_padded = np.concatenate([x, np.zeros(pad)]) if pad else x
    phases = x_padded.reshape(-1, delay_samples).T  # (delay_samples, n_frames)

    filtered_phases = lfilter([1.0], [1.0, -decay], phases, axis=1)

    y_padded = filtered_phases.T.reshape(-1)
    return y_padded[:n]


def apply_reverb(samples, mix=0.3, decay=0.5, sr=SAMPLE_RATE):
    """Multi-tap Schroeder-style reverb, built from efficient polyphase comb filters."""
    _require_numpy()
    delays_ms = [29.7, 37.1, 41.3, 43.5, 47.1, 53.3]
    wet = np.array(samples, dtype=np.float64)
    for delay_ms in delays_ms:
        delay_samples = int(sr * delay_ms / 1000)
        wet = _comb_filter(wet, delay_samples, decay)
    return samples * (1 - mix) + wet * mix


def apply_lowpass(samples, cutoff_ratio=0.3):
    _require_numpy()
    from scipy.signal import lfilter
    if len(samples) == 0:
        return samples
    alpha = max(0.001, min(1.0, cutoff_ratio))
    return lfilter([alpha], [1.0, -(1.0 - alpha)], samples)


def mix_tracks(*tracks, gains=None):
    _require_numpy()
    tracks = [t for t in tracks if len(t) > 0]
    if not tracks:
        return np.array([])
    if gains is None:
        gains = [1.0] * len(tracks)
    max_len = max(len(t) for t in tracks)
    mixed = np.zeros(max_len)
    for track, g in zip(tracks, gains):
        mixed[:len(track)] += track * g
    peak = np.max(np.abs(mixed)) if mixed.size else 1.0
    if peak > 1.0:
        mixed = mixed / peak * 0.95
    return mixed


def pad_to(samples, length: int):
    _require_numpy()
    if len(samples) >= length:
        return samples[:length]
    return np.concatenate([samples, np.zeros(length - len(samples))])


def overlay(base, addition, start_sample: int):
    _require_numpy()
    end = start_sample + len(addition)
    if end > len(base):
        base = np.concatenate([base, np.zeros(end - len(base))])
    base[start_sample:end] += addition
    return base


def stereo_widen(mono, width=0.35):
    _require_numpy()
    L = mono * (1.0 + width * 0.5)
    R = mono * (1.0 - width * 0.5)
    stereo = np.empty(len(mono) * 2)
    stereo[0::2] = L
    stereo[1::2] = R
    return stereo


def write_wav(samples, filename=None, sr=SAMPLE_RATE, channels=2) -> bytes:
    _require_numpy()
    arr = np.asarray(samples, dtype=np.float64)
    clamped = np.clip(arr, -1.0, 1.0)
    int_samples = (clamped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframesraw(int_samples.tobytes())

    audio_bytes = buf.getvalue()
    if filename:
        d = os.path.dirname(filename)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(filename, "wb") as f:
            f.write(audio_bytes)
    return audio_bytes


def silence(duration: float, sr=SAMPLE_RATE):
    _require_numpy()
    return np.zeros(int(sr * duration))
