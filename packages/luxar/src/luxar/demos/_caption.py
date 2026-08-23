"""One caption format for every bundled demo."""

from __future__ import annotations

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
