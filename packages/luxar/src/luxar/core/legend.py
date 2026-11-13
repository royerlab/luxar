"""The one definition of a scene's colour legend.

A legend says what the colours in a scene mean: one entry per colour, with
its hex code, a plain colour name and what it stands for. It is written to the
store's root attributes (``legend``) so the explanation travels with the data,
and ``luxar export`` prints it in the exported folder's README. Like the
citation (:mod:`luxar.core.citation`), it is validated once, here, rather than
trusted wherever it is read.

The hex colours display-encode authored linear [0, 1] values as sRGB. On screen a colour can also
be scaled by a layer's intensity, the exposure and tone mapping, and summed
where points overlap, so the legend's job is to name the hue: ``note`` is the
place to say what else changes the look.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

#: ``#rrggbb``; stored upper-case.
_HEX = re.compile(r"#[0-9A-Fa-f]{6}")


#: Length caps. A legend is printed -- README.txt, COLOURS.pdf, the panel
#: sheet's key -- so it is kept to what a reader can take in at a glance, and
#: no single field can push a printed layout off its page.
MAX_ENTRIES = 64
_MAX_CHARS = {
    "legend title": 120,
    "legend label": 80,
    "legend color_name": 40,
    "legend description": 400,
    "legend note": 1000,
}


def _single_line(where: str, text: Any, *, required: bool = True) -> str:
    """A stripped single line of printable text (see ``citation``'s reasoning)."""
    if not isinstance(text, str):
        kind = "a non-empty string" if required else "a string"
        raise ValueError(f"{where} must be {kind}, got {type(text).__name__}")
    if required and not text.strip():
        raise ValueError(f"{where} must be a non-empty string")
    limit = _MAX_CHARS[where]
    if len(text.strip()) > limit:
        raise ValueError(
            f"{where} is {len(text.strip())} characters; at most {limit} fit "
            "the printed legend"
        )
    bad = {ch for ch in text if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp")}
    if bad:
        raise ValueError(
            f"{where} must be a single line of printable text; it contains "
            + ", ".join(sorted("U+%04X" % ord(ch) for ch in bad))
        )
    return text.strip()


def hex_color(rgb: Iterable[float]) -> str:
    """sRGB ``#RRGGBB`` for an authored linear RGB triple (clamped to [0, 1])."""
    channels = [float(c) for c in rgb]
    if len(channels) != 3:
        raise ValueError(f"expected an RGB triple, got {len(channels)} values")
    linear = [min(1.0, max(0.0, c)) for c in channels]
    srgb = [
        c / 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055 for c in linear
    ]
    return "#" + "".join(f"{round(c * 255):02X}" for c in srgb)


@dataclass(frozen=True)
class LegendEntry:
    """One colour and what it means."""

    #: What the colour stands for, e.g. ``"Proteobacteria"``.
    label: str
    #: ``#RRGGBB``.
    color: str
    #: A plain colour name a reader can say, e.g. ``"lime green"``.
    color_name: str
    #: Optional detail: what falls under the label.
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _single_line("legend label", self.label))
        if not isinstance(self.color, str) or not _HEX.fullmatch(self.color):
            raise ValueError(f"legend color must be #RRGGBB, got {self.color!r}")
        object.__setattr__(self, "color", self.color.upper())
        object.__setattr__(
            self, "color_name", _single_line("legend color_name", self.color_name)
        )
        object.__setattr__(
            self,
            "description",
            _single_line("legend description", self.description, required=False),
        )

    def to_dict(self) -> dict[str, str]:
        """The stored form; ``description`` only when there is one."""
        out = {"label": self.label, "color": self.color, "color_name": self.color_name}
        if self.description:
            out["description"] = self.description
        return out


@dataclass(frozen=True)
class Legend:
    """What a scene's colours mean: a title, the entries, and an optional note."""

    title: str
    entries: tuple[LegendEntry, ...]
    #: Anything else a reader needs to read the colours (dimming, highlights).
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", _single_line("legend title", self.title))
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("a legend needs at least one entry")
        if len(entries) > MAX_ENTRIES:
            raise ValueError(
                f"a legend has at most {MAX_ENTRIES} entries, got {len(entries)}"
            )
        if not all(isinstance(entry, LegendEntry) for entry in entries):
            raise ValueError("legend entries must be LegendEntry objects")
        labels = [entry.label for entry in entries]
        if len(set(labels)) != len(labels):
            raise ValueError("legend labels must be unique")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(
            self, "note", _single_line("legend note", self.note, required=False)
        )

    def to_dict(self) -> dict[str, Any]:
        """The stored form (the ``legend`` root attribute); ``note`` only when set."""
        out: dict[str, Any] = {
            "title": self.title,
            "entries": [entry.to_dict() for entry in self.entries],
        }
        if self.note:
            out["note"] = self.note
        return out

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Legend":
        """Read a stored ``legend``; raises ``ValueError`` if it is malformed."""
        if not isinstance(value, Mapping):
            raise ValueError("legend must be an object")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("legend.entries must be a list")
        entries = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping):
                raise ValueError("each legend entry must be an object")
            entries.append(
                LegendEntry(
                    label=raw.get("label"),  # type: ignore[arg-type]
                    color=raw.get("color"),  # type: ignore[arg-type]
                    color_name=raw.get("color_name"),  # type: ignore[arg-type]
                    description=raw.get("description", ""),
                )
            )
        return cls(
            title=value.get("title"),  # type: ignore[arg-type]
            entries=tuple(entries),
            note=value.get("note", ""),
        )


def validate_legend(value: Optional[Legend]) -> Optional[dict[str, Any]]:
    """The stored form of ``value``: ``None`` (no legend) or its dict."""
    if value is None:
        return None
    if not isinstance(value, Legend):
        raise ValueError("legend must be a luxar.core.legend.Legend")
    return value.to_dict()
