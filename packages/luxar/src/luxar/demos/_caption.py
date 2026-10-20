"""One caption format for every bundled demo, plus compression-ratio wording."""

from __future__ import annotations

import math
from typing import Mapping, Optional, Protocol

from luxar.core.citation import CITATION_REF_MAX_LENGTH, validate_citation


class _TextScene(Protocol):
    def add_text(
        self, text: str, position: tuple[float, float], **kwargs: object
    ) -> object:
        """Add a text overlay."""


def format_demo_caption(caption: str, citation: Optional[Mapping[str, str]]) -> str:
    """Append the compact reference using ``•`` as the caption field separator."""
    validated = validate_citation(citation)
    if validated is None:
        return caption
    reference = validated.get("ref", validated["short"])
    if len(reference) > CITATION_REF_MAX_LENGTH:
        raise ValueError(
            "citation.ref is required when citation.short exceeds "
            f"{CITATION_REF_MAX_LENGTH} characters"
        )
    if caption.endswith(reference):
        return caption
    return f"{caption} • {reference}"


def add_demo_caption(
    scene: _TextScene,
    caption: str,
    citation: Optional[Mapping[str, str]],
) -> None:
    """Add the standard bottom-right demo caption, including attribution."""
    scene.add_text(
        format_demo_caption(caption, citation),
        position=(0.98, 0.97),
        font_size=0.015,
        anchor="bottom-right",
        color="rgba(200,200,200,0.5)",
    )


_DECIMAL_UNITS = (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3))


def compression_ratio(raw_bytes: int, compressed_bytes: int) -> float:
    """How many times smaller ``compressed_bytes`` is than ``raw_bytes``."""
    if raw_bytes <= 0 or compressed_bytes <= 0:
        raise ValueError(
            f"byte counts must be positive, got {raw_bytes} and {compressed_bytes}"
        )
    return raw_bytes / compressed_bytes


def format_ratio(ratio: float) -> str:
    """A ratio as a caption figure: ``4.2x``, ``42x``, ``315x``, ``12,300x``.

    One decimal below 10, a whole number below 1,000, three significant
    figures from there on: the inputs are measured sizes, so more digits would
    claim a precision the comparison does not have.
    """
    if round(ratio, 1) < 10:
        return f"{round(ratio, 1):.1f}x"
    whole = round(ratio)
    if whole < 1000:
        return f"{whole:,}x"
    digits = len(str(whole)) - 3
    return f"{round(whole, -digits):,}x"


def format_bytes(n_bytes: int) -> str:
    """A byte count in decimal units, three significant figures (``34.9 GB``).

    Decimal (1 GB = 10^9 bytes), the convention data repositories publish
    file sizes in, so a quoted size matches the one on the record.
    """
    for unit, scale in _DECIMAL_UNITS:
        # 0.9995: a value that rounds up to 1000 reads "1 GB", not "1e+03 MB".
        if n_bytes >= scale * 0.9995:
            value = float(f"{n_bytes / scale:.3g}")  # 3 significant figures
            if value >= 1000:  # only in the largest unit
                return f"{value:,.0f} {unit}"
            decimals = max(0, 2 - math.floor(math.log10(value)))
            return f"{value:.{decimals}f} {unit}"
    return f"{n_bytes} B"


def compression_caption(raw_bytes: int, scene_bytes: int, raw_label: str) -> str:
    """Caption figure: ``"111 MB, ≈315x smaller than the raw CZI"``.

    Both sizes are ON DISK: ``raw_bytes`` is the source file as published,
    ``scene_bytes`` the final scene the demo serves (sum of its file sizes).
    """
    ratio = format_ratio(compression_ratio(raw_bytes, scene_bytes))
    return f"{format_bytes(scene_bytes)}, ≈{ratio} smaller than the {raw_label}"


def compression_note(raw_bytes: int, scene_bytes: int, raw_label: str) -> str:
    """Description sentence: ``"111 MB on disk, from the 34.9 GB raw CZI (≈315x
    smaller)"``. Same on-disk sizes as :func:`compression_caption`."""
    ratio = format_ratio(compression_ratio(raw_bytes, scene_bytes))
    return (
        f"{format_bytes(scene_bytes)} on disk, from the "
        f"{format_bytes(raw_bytes)} {raw_label} (≈{ratio} smaller)"
    )
