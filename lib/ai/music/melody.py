"""
KENAI Music Engine - Melody, Bass, Chords (vectorized)
==========================================================
Motif-based melody generation (transpose/invert/retrograde across
sections), plus bass line and chord/pad synthesis.
"""

import random
from typing import List

import numpy as np

from .theory import (
    CHORD_TYPES, scale_degree_to_freq,
    make_sine, make_saw, make_square, make_triangle,
    envelope_adsr, envelope_percussive, silence,
)


def generate_motif(length=4, max_step=3) -> List[int]:
    motif = [0]
    for _ in range(length - 1):
        step = random.choice(range(-max_step, max_step + 1))
        motif.append(motif[-1] + step)
    return motif


def transpose_motif(motif: List[int], amount: int) -> List[int]:
    return [n + amount for n in motif]


def invert_motif(motif: List[int]) -> List[int]:
    root = motif[0]
    return [root - (n - root) for n in motif]


def retrograde_motif(motif: List[int]) -> List[int]:
    return list(reversed(motif))


def develop_motif(motif: List[int], section: str) -> List[int]:
    if section == "verse":
        return motif
    if section == "chorus":
        return transpose_motif(motif, 2)
    if section == "bridge":
        return invert_motif(motif)
    if section == "outro":
        return retrograde_motif(motif)
    return motif


def render_melody(
    motif: List[int], root_freq: float, scale_name: str, beat_dur: float, bars: int,
    notes_per_bar: int = 4, volume: float = 0.35, waveform: str = "triangle",
):
    wave_fn = {"sine": make_sine, "saw": make_saw, "square": make_square, "triangle": make_triangle}[waveform]
    note_dur = (beat_dur * 4) / notes_per_bar
    chunks = []

    for bar in range(bars):
        local_motif = motif if bar % 4 < 2 else transpose_motif(motif, random.choice([-2, 0, 2]))
        for i in range(notes_per_bar):
            degree = local_motif[i % len(local_motif)]
            freq = scale_degree_to_freq(root_freq, scale_name, degree)
            if random.random() < 0.08:
                chunks.append(silence(note_dur))
                continue
            note = wave_fn(freq, note_dur * 0.92, volume)
            note = envelope_adsr(note, attack=0.01, decay=0.05, sustain=0.6, release=0.15)
            chunks.append(note)
            chunks.append(silence(note_dur * 0.08))
    return np.concatenate(chunks) if chunks else np.array([])


def render_bass(
    progression: List[int], root_freq: float, scale_name: str, beat_dur: float, bars: int,
    style: str = "sustained", volume: float = 0.45, is_808: bool = False,
):
    chunks = []
    bar_dur = beat_dur * 4
    for bar in range(bars):
        degree = progression[bar % len(progression)]
        freq = scale_degree_to_freq(root_freq, scale_name, degree) / 2

        if style == "pulsed" or is_808:
            for _ in range(4):
                note = make_sine(freq, beat_dur * 0.85, volume)
                note = envelope_percussive(note, decay=0.35 if is_808 else 0.5)
                chunks.append(note)
                chunks.append(silence(beat_dur * 0.15))
        elif style == "skank":
            for _ in range(4):
                chunks.append(silence(beat_dur * 0.5))
                note = make_sine(freq, beat_dur * 0.4, volume * 0.8)
                note = envelope_percussive(note, decay=0.15)
                chunks.append(note)
                chunks.append(silence(beat_dur * 0.1))
        else:
            note = make_triangle(freq, bar_dur * 0.95, volume)
            note = envelope_adsr(note, attack=0.02, decay=0.15, sustain=0.75, release=0.2)
            chunks.append(note)
            chunks.append(silence(bar_dur * 0.05))
    return np.concatenate(chunks) if chunks else np.array([])


def _mix_chord_notes(notes):
    length = max(len(c) for c in notes)
    mixed = np.zeros(length)
    for c in notes:
        mixed[:len(c)] += c
    return mixed


def render_chords(
    progression: List[int], root_freq: float, scale_name: str, beat_dur: float, bars: int,
    chord_quality_cycle: List[str] = None, volume: float = 0.22, style: str = "pad",
):
    if chord_quality_cycle is None:
        chord_quality_cycle = ["major", "minor", "minor", "major"]
    chunks = []
    bar_dur = beat_dur * 4

    for bar in range(bars):
        degree = progression[bar % len(progression)]
        quality = chord_quality_cycle[bar % len(chord_quality_cycle)]
        chord_intervals = CHORD_TYPES.get(quality, CHORD_TYPES["major"])
        chord_root = scale_degree_to_freq(root_freq, scale_name, degree)

        if style == "stab":
            for _ in range(4):
                chunks.append(silence(beat_dur * 0.5))
                chord_notes = [
                    make_square(chord_root * (2 ** (iv / 12)), beat_dur * 0.4, volume / len(chord_intervals))
                    for iv in chord_intervals
                ]
                mixed = _mix_chord_notes(chord_notes)
                mixed = envelope_percussive(mixed, decay=0.12)
                chunks.append(mixed)
                chunks.append(silence(beat_dur * 0.1))
        else:
            chord_notes = [
                make_triangle(chord_root * (2 ** (iv / 12)), bar_dur * 0.95, volume / len(chord_intervals))
                for iv in chord_intervals
            ]
            mixed = _mix_chord_notes(chord_notes)
            mixed = envelope_adsr(mixed, attack=0.08, decay=0.2, sustain=0.7, release=0.3)
            chunks.append(mixed)
            chunks.append(silence(bar_dur * 0.05))
    return np.concatenate(chunks) if chunks else np.array([])
