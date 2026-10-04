"""
KENAI Music Engine v6.1 (vectorized)
=======================================
Arranges theory + drums + melody/bass/chords + vocals into a full
song: intro -> verse -> chorus -> verse -> chorus -> bridge -> chorus
-> outro, mixed to a stereo WAV. All synthesis is vectorized with
numpy/scipy, which is what makes render time scale predictably with
song length instead of stalling on longer renders.
"""

import random
from typing import Optional, List

import numpy as np

from .theory import (
    GENRES, PROGRESSIONS, SAMPLE_RATE,
    note_to_freq, detect_genre_from_prompt, detect_mood_params,
    mix_tracks, apply_reverb, stereo_widen, write_wav, silence,
)
from .drums import DrumSynth
from .melody import generate_motif, develop_motif, render_melody, render_bass, render_chords
from .vocals import render_vocal_line

DEFAULT_STRUCTURE = [
    ("intro", 2), ("verse", 4), ("chorus", 4), ("verse", 4),
    ("chorus", 4), ("bridge", 2), ("chorus", 4), ("outro", 2),
]


class KenaiMusicEngine:
    def __init__(self, sample_rate: int = SAMPLE_RATE, humanize: bool = True, seed: Optional[int] = None):
        self.sr = sample_rate
        self.drum_synth = DrumSynth(sr=sample_rate, humanize=humanize)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate(
        self, prompt: str, bars: int = 16, genre: Optional[str] = None,
        output_path: Optional[str] = None, vocals: bool = True, seed: Optional[int] = None,
    ) -> bytes:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        genre_name = genre or detect_genre_from_prompt(prompt)
        genre_cfg = GENRES.get(genre_name, GENRES["pop"])
        mood = detect_mood_params(prompt)

        tempo = genre_cfg["tempo"] * mood["tempo_mult"]
        beat_dur = 60.0 / tempo
        scale_name = genre_cfg["scale"]
        progression = PROGRESSIONS[genre_cfg["progression"]]
        root_freq = note_to_freq(genre_cfg["root_note"], genre_cfg["root_octave"])
        drum_pattern = genre_cfg["drum_pattern"]
        instruments = genre_cfg["instruments"]

        structure = self._scale_structure(bars)
        motif = generate_motif(length=4, max_step=3)

        section_chunks = []
        for name, section_bars in structure:
            section_chunks.append(self._render_section(
                section_name=name, section_bars=section_bars, motif=motif,
                genre_cfg=genre_cfg, progression=progression, scale_name=scale_name,
                root_freq=root_freq, beat_dur=beat_dur, drum_pattern=drum_pattern,
                instruments=instruments, use_vocals=vocals,
            ))

        song = np.concatenate(section_chunks) if section_chunks else np.array([])

        song = apply_reverb(song, mix=0.18, decay=0.35, sr=self.sr)
        peak = np.max(np.abs(song)) if song.size else 1.0
        if peak > 0.98:
            song = song / peak * 0.95
        stereo = stereo_widen(song, width=0.3)

        return write_wav(stereo, filename=output_path, sr=self.sr, channels=2)

    def _scale_structure(self, target_bars: int) -> List[tuple]:
        base_total = sum(b for _, b in DEFAULT_STRUCTURE)
        if target_bars <= 0:
            return DEFAULT_STRUCTURE
        ratio = max(0.5, target_bars / base_total)
        return [(name, max(1, round(b * ratio))) for name, b in DEFAULT_STRUCTURE]

    def _render_section(
        self, section_name, section_bars, motif, genre_cfg, progression,
        scale_name, root_freq, beat_dur, drum_pattern, instruments, use_vocals,
    ):
        section_motif_degrees = develop_motif(motif, section_name)
        tracks, gains = [], []

        if drum_pattern != "none" and section_name != "intro":
            drums = self.drum_synth.build_pattern(drum_pattern, beat_dur, bars=section_bars)
            tracks.append(drums); gains.append(0.9)
        elif drum_pattern != "none" and section_name == "intro":
            drums = self.drum_synth.build_pattern(drum_pattern, beat_dur, bars=section_bars)
            tracks.append(drums * 0.3); gains.append(0.5)

        if "bass_808" in instruments:
            bass = render_bass(progression, root_freq, scale_name, beat_dur, section_bars,
                                style="pulsed", is_808=True, volume=0.5)
            tracks.append(bass); gains.append(0.85)
        elif "bass" in instruments:
            style = "skank" if genre_cfg.get("drum_pattern") in ("one_drop", "dancehall") else "sustained"
            bass = render_bass(progression, root_freq, scale_name, beat_dur, section_bars,
                                style=style, volume=0.45)
            tracks.append(bass); gains.append(0.8)

        if "skank" in instruments:
            chords = render_chords(progression, root_freq, scale_name, beat_dur, section_bars, style="stab")
            tracks.append(chords); gains.append(0.6)
        elif "chords" in instruments or "pads" in instruments or "synth" in instruments:
            chords = render_chords(progression, root_freq, scale_name, beat_dur, section_bars, style="pad")
            tracks.append(chords); gains.append(0.55)

        if section_name != "intro" or "pads" in instruments:
            melody = render_melody(section_motif_degrees, root_freq, scale_name, beat_dur, section_bars,
                                    notes_per_bar=4, volume=0.3)
            tracks.append(melody); gains.append(0.65)

        if use_vocals and section_name in ("verse", "chorus", "bridge"):
            vocal_degrees = [d + (4 if section_name == "chorus" else 0) for d in section_motif_degrees]
            vocal = render_vocal_line(vocal_degrees, root_freq, scale_name, beat_dur, section_bars,
                                       notes_per_bar=2, volume=0.42 if section_name == "chorus" else 0.35)
            tracks.append(vocal); gains.append(0.9 if section_name == "chorus" else 0.75)

        if not tracks:
            return silence(beat_dur * 4 * section_bars, sr=self.sr)

        return mix_tracks(*tracks, gains=gains)


def generate_music(prompt: str, bars: int = 16, genre: Optional[str] = None,
                    output_path: Optional[str] = None, vocals: bool = True,
                    seed: Optional[int] = None) -> bytes:
    engine = KenaiMusicEngine()
    return engine.generate(prompt, bars=bars, genre=genre, output_path=output_path,
                            vocals=vocals, seed=seed)
