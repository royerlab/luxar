"""Encoding-mode resolution shared across gsplat command groups."""

from __future__ import annotations

from typing import Any


def _resolve_encoding_mode(mode: str) -> Any:
    """Convert a string encoding mode to an EncodingMode enum value."""
    from luxar.encoding import EncodingMode

    _ENCODING_MAP = {
        "auto": EncodingMode.AUTO,
        "precision": EncodingMode.PRECISION,
        "memory": EncodingMode.MEMORY,
    }
    return _ENCODING_MAP[mode]
