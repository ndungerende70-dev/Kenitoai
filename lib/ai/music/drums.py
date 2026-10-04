"""
KENAI Music Engine - Drum Synthesis (vectorized)
====================================================
Synthesized drum hits and genre-specific pattern builders, with light
humanization. All waveform math is vectorized with numpy.
"""

import random
import numpy as np

from .theory import SAMPLE_RATE


class DrumSynth:
    def __init__(self, sr=SAMPLE_RATE, humanize=True):
        self.sr = sr
        self.humanize = humanize

    def _jit(self, amount=0.03):
        return random.uniform(-amount, amount) if self.humanize else 0.0

    def kick(self, freq_start=150, freq_end=50, duration=0.2, volume=0.8):
        n = max(1, int(self.sr * duration))
        vol = volume * (1 + self._jit(0.06))
        t = np.arange(n) / self.sr
        freq = freq_start + (freq_end - freq_start) * (np.arange(n) / n)
        env = vol * np.exp(-t * 8)
        phase = 2 * np.pi * np.cumsum(freq) / self.sr
        return np.sin(phase) * env

    def kick_808(self, volume=0.9, duration=0.35):
        n = max(1, int(self.sr * duration))
        vol = volume * (1 + self._jit(0.05))
        t = np.arange(n) / self.sr
        freq = 180 + (38 - 180) * (np.arange(n) / n) ** 0.7
        phase = 2 * np.pi * np.cumsum(freq) / self.sr
        val = np.sin(phase)
        val = np.tanh(val * 3) * 0.7
        return val * vol * np.exp(-t * 4.5)

    def kick_dancehall(self, volume=0.85):
        return self.kick(120, 45, 0.25, volume)

    def snare(self, volume=0.6):
        dur = 0.15
        n = int(self.sr * dur)
        vol = volume * (1 + self._jit(0.08))
        t = np.arange(n) / self.sr
        tone = np.sin(2 * np.pi * 200 * t) * 0.4
        noise = np.random.uniform(-1, 1, n) * 0.6
        env = vol * np.exp(-t * 12)
        return (tone + noise) * env

    def hihat(self, duration=0.06, volume=0.3):
        n = int(self.sr * duration)
        vol = volume * (1 + self._jit(0.1))
        t = np.arange(n) / self.sr
        return np.random.uniform(-1, 1, n) * vol * np.exp(-t * 20)

    def open_hat(self, volume=0.25):
        return self.hihat(0.3, volume)

    def rimshot(self, volume=0.5):
        dur = 0.05
        n = int(self.sr * dur)
        t = np.arange(n) / self.sr
        tone = np.sin(2 * np.pi * 800 * t) * 0.6
        noise = np.random.uniform(-1, 1, n) * 0.4
        return (tone + noise) * volume * np.exp(-t * 25)

    def clap(self, volume=0.4):
        dur = 0.12
        n = int(self.sr * dur)
        bursts = [0, int(n * 0.1), int(n * 0.2)]
        idx = np.arange(n)
        env = np.zeros(n)
        for b in bursts:
            burst_env = np.clip(1.0 - (idx - b) / (n * 0.4), 0.0, None)
            burst_env[idx < b] = 0.0
            env = np.maximum(env, burst_env)
        return np.random.uniform(-1, 1, n) * volume * env

    def shaker(self, duration=0.08, volume=0.15):
        n = int(self.sr * duration)
        i = np.arange(n)
        return np.random.uniform(-1, 1, n) * volume * (1 - i / n)

    def conga(self, freq=200, volume=0.5):
        dur = 0.2
        n = int(self.sr * dur)
        t = np.arange(n) / self.sr
        return np.sin(2 * np.pi * freq * t) * volume * np.exp(-t * 8)

    def _slot(self, hits, slot_len):
        slot = np.zeros(slot_len)
        for h in hits:
            n = min(len(h), slot_len)
            slot[:n] += h[:n]
        return slot

    def build_pattern(self, pattern_type: str, beat_dur: float, bars: int = 8):
        steps_per_beat = 4 if pattern_type in ("trap",) else 1
        slot_dur = beat_dur / steps_per_beat
        slot_len = int(self.sr * slot_dur)
        chunks = []

        for bar in range(bars):
            for beat in range(4):
                for step in range(steps_per_beat):
                    hits = self._pattern_step(pattern_type, beat, step, steps_per_beat)
                    chunks.append(self._slot(hits, slot_len))

        if not chunks:
            return np.zeros(int(self.sr * beat_dur * 4 * bars))
        return np.concatenate(chunks)

    def _pattern_step(self, pattern_type, beat, step, steps_per_beat):
        hits = []

        if pattern_type == "one_drop":
            if beat == 2:
                hits.append(self.kick(volume=0.75))
            if beat in (1, 3):
                hits.append(self.rimshot(0.45))
            hits.append(self.hihat(0.05, 0.2))

        elif pattern_type == "dancehall":
            if beat in (0, 2):
                hits.append(self.kick_dancehall())
            if beat in (1, 3):
                hits.append(self.clap())
            hits.append(self.hihat(0.05, 0.22))

        elif pattern_type == "boom_bap":
            if beat in (0, 2):
                hits.append(self.kick(100, 40, 0.2, 0.9))
            if beat in (1, 3):
                hits.append(self.snare(0.65))
            hits.append(self.hihat(0.05, 0.25))

        elif pattern_type == "trap":
            if beat in (0, 2) and step == 0:
                hits.append(self.kick_808())
            if beat == 2 and step == 0:
                hits.append(self.snare(0.7))
            hat_vol = 0.2 if step % 2 == 0 else 0.12
            hits.append(self.hihat(0.03, hat_vol))

        elif pattern_type == "four_floor":
            hits.append(self.kick(100, 40, 0.15, 0.9))
            if beat in (1, 3):
                hits.append(self.snare(0.55))
                hits.append(self.open_hat(0.2))
            else:
                hits.append(self.hihat(0.05, 0.25))

        elif pattern_type == "afrobeat":
            if beat in (0, 1, 3):
                hits.append(self.kick(130, 50, 0.18, 0.8))
            if beat == 2:
                hits.append(self.snare(0.5))
            if beat in (0, 2):
                hits.append(self.conga(200, 0.4))
            hits.append(self.hihat(0.06, 0.22))

        elif pattern_type == "lofi":
            if beat in (0, 2):
                hits.append(self.kick(100, 45, 0.18, 0.6))
            if beat in (1, 3):
                hits.append(self.snare(0.35))
            hits.append(self.hihat(0.04, 0.15))
            hits.append(self.shaker(0.06, 0.08))

        elif pattern_type == "rnb":
            if beat in (0, 2):
                hits.append(self.kick(105, 42, 0.18, 0.7))
            if beat in (1, 3):
                hits.append(self.snare(0.45))
            hits.append(self.hihat(0.05, 0.18))

        elif pattern_type == "rock":
            if beat in (0, 2):
                hits.append(self.kick(140, 55, 0.2, 0.9))
            if beat in (1, 3):
                hits.append(self.snare(0.75))
            hits.append(self.hihat(0.05, 0.3))

        elif pattern_type == "none":
            pass

        else:
            if beat in (0, 2):
                hits.append(self.kick())
            if beat in (1, 3):
                hits.append(self.snare())
            hits.append(self.hihat())

        return hits
