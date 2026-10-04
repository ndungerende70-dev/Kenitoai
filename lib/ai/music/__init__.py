"""
KENAI Music Engine v6.1
Procedural music generation with formant vocals.
11 genres, motif-based melodies, genre-specific grooves, fully
vectorized (numpy + scipy) for fast, consistent render times.

Public API:
    from ai.music import KenaiMusicEngine, generate_music
"""

from .engine import KenaiMusicEngine, generate_music
from .theory import GENRES, detect_genre_from_prompt

__all__ = ["KenaiMusicEngine", "generate_music", "GENRES", "detect_genre_from_prompt"]
__version__ = "6.1.0"
