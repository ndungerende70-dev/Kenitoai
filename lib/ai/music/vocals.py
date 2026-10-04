"""
KENAI Music Engine - Formant Vocal Synthesis (vectorized)
=============================================================
Synthetic "voice-like" texture: a buzzy glottal-pulse source shaped
by formant resonances. Not a real singer, but more voice-like than
a bare sine/saw tone. Harmonic source is vectorized with numpy; the
recursive resonant filters run through scipy's compiled lfilter.
"""

import random
from typing import List

import numpy as np
from scipy.signal import lfilter

from .theory import SAMPLE_RATE, scale_degree_to_freq, silence, envelope_adsr

VOWEL_FORMANTS = {
    "ah": (700, 1220),
    "ee": (270, 2290),
    "oo": (300, 870),
    "oh": (500, 1000),
    "eh": (530, 1840),
}

N_HARMONICS = 12


def _glottal_pulse(freq: float, duration: float, sr=SAMPLE_RATE):
    n = max(1, int(sr * duration))
    t = np.arange(n) / sr
    harmonics = np.arange(1, N_HARMONICS + 1)
    hfreqs = freq * harmonics
    valid = hfreqs < (sr / 2)
    harmonics = harmonics[valid]
    hfreqs = hfreqs[valid]
    if len(harmonics) == 0:
        return np.zeros(n)

    amps = 1.0 / harmonics
    phases = 2 * np.pi * np.outer(hfreqs, t)
    signal = (amps[:, None] * np.sin(phases)).sum(axis=0)

    peak = np.max(np.abs(signal)) or 1.0
    return signal / peak


def _resonate(samples, center_freq: float, bandwidth: float, sr=SAMPLE_RATE):
    if len(samples) == 0:
        return samples
    r = np.exp(-np.pi * bandwidth / sr)
    theta = 2 * np.pi * center_freq / sr
    a1 = 2 * r * np.cos(theta)
    a2 = -r * r
    out = lfilter([1.0], [1.0, -a1, -a2], samples)
    peak = np.max(np.abs(out)) or 1.0
    return out / peak


def synth_vowel(freq: float, duration: float, vowel: str = "ah", volume: float = 0.4, sr=SAMPLE_RATE):
    source = _glottal_pulse(freq, duration, sr)
    f1, f2 = VOWEL_FORMANTS.get(vowel, VOWEL_FORMANTS["ah"])
    band1 = _resonate(source, f1, bandwidth=80, sr=sr)
    band2 = _resonate(source, f2, bandwidth=120, sr=sr)
    n = min(len(band1), len(band2))
    mixed = (band1[:n] * 0.65 + band2[:n] * 0.35) * volume

    vib_rate = 5.5
    t = np.arange(n) / sr
    mixed = mixed * (1.0 + 0.03 * np.sin(2 * np.pi * vib_rate * t))

    mixed = envelope_adsr(mixed, attack=0.03, decay=0.08, sustain=0.75, release=0.18, sr=sr)
    return mixed


def render_vocal_line(
    melodic_degrees: List[int], root_freq: float, scale_name: str, beat_dur: float, bars: int,
    notes_per_bar: int = 2, volume: float = 0.4, vowels=("ah", "oh", "ee"),
):
    note_dur = (beat_dur * 4) / notes_per_bar
    chunks = []
    for bar in range(bars):
        for i in range(notes_per_bar):
            degree = melodic_degrees[(bar * notes_per_bar + i) % len(melodic_degrees)]
            freq = scale_degree_to_freq(root_freq, scale_name, degree)
            if random.random() < 0.15:
                chunks.append(silence(note_dur))
                continue
            vowel = random.choice(vowels)
            note = synth_vowel(freq, note_dur * 0.9, vowel=vowel, volume=volume, sr=SAMPLE_RATE)
            chunks.append(note)
            chunks.append(silence(note_dur * 0.1))
    return np.concatenate(chunks) if chunks else np.array([])
