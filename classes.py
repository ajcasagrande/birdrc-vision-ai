# Copyright (C) 2025 Anthony Casagrande
# AGPL-3.0 license

from dataclasses import dataclass

@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    fps: float
    codec: str
