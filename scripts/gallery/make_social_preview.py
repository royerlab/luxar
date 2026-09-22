#!/usr/bin/env python3
"""Compose the repository's social-preview banner (1280x640) from four scene panels.

The banner is the image GitHub shows when the repository link is shared
(Settings -> Social preview) and the header of the root README. It is four
portrait panels of real Luxar scenes side by side, darkened towards the lower
left, with the wordmark, a tagline and the four geometry types over the scrim.

Each panel source is one of:

* ``key:<demo-id>``   the still of that demo's gallery tile, resolved through
  ``scripts/gallery/media-manifest.json`` (first frame of the animated WebP);
* ``url:<https://...>`` any image URL;
* ``file:<path>``     a local image.

Sources are cropped to the panel's aspect (cover fit) around a horizontal
anchor, centred by default; append ``@<0..1>`` to a source to take the crop
from its left (``@0``) or right (``@1``) side instead, e.g. ``file:frame.png@0.2``.
The defaults reproduce the shipped banner; pass ``--panel`` four times to
change the line-up and ``--scale 2`` for a 2560x1280 master.

    hatch run python scripts/gallery/make_social_preview.py -o social-preview.png
    hatch run python scripts/gallery/make_social_preview.py \\
        --panel key:gsplats_3d_cells3d_multichannel \\
        --panel file:laniakea.png --panel file:drosophila.png \\
        --panel key:dmri_tractography -o banner.png

The output is deliberately not committed: it is uploaded to the data host by
content hash (``data.luxarviewer.dev/media/<sha256[:16]>.png``) for the README
and to GitHub's Social preview by hand. Only Pillow and NumPy are needed.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
MEDIA_MANIFEST = REPO_ROOT / "scripts/gallery/media-manifest.json"
USER_AGENT = "LuxarSocialPreview/1.0"

# Cells3D (gallery tile), Cosmicflows-4 Laniakea (viewer frame, right-hand
# crop), Drosophila gastrulation (viewer frame), HCP tractography (gallery
# tile). The two viewer frames are hosted by content hash beside the tiles.
DEFAULT_PANELS = (
    "key:gsplats_3d_cells3d_multichannel",
    "url:https://data.luxarviewer.dev/media/2bc40436c253b2be.png@0.9",
    "url:https://data.luxarviewer.dev/media/023168eeb5de26d0.png",
    "key:dmri_tractography",
)
TITLE = "Luxar"
TAGLINE = "n-dimensional scientific data, compiled and explored in the browser"
TYPES = "points  ·  lines  ·  gaussian splats  ·  meshes"
# Bold and regular faces are separate lists: a bold-only file must never be
# picked for the regular weight (it would set the tagline in bold and overrun the
# scrim on a machine without the macOS fonts).
BOLD_FONTS = (
    ("/System/Library/Fonts/Helvetica.ttc", 1),
    ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
)
REGULAR_FONTS = (
    ("/System/Library/Fonts/Helvetica.ttc", 0),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
)


def split_source(source: str) -> tuple[str, float]:
    """Separate an optional ``@anchor`` suffix from a panel source."""
    base, at, anchor = source.rpartition("@")
    if at and base and anchor.replace(".", "", 1).isdigit():
        return base, min(1.0, max(0.0, float(anchor)))
    return source, 0.5


def _load_bytes(source: str) -> bytes:
    kind, _, ref = source.partition(":")
    if kind == "file":
        return Path(ref).expanduser().read_bytes()
    if kind == "key":
        manifest = json.loads(MEDIA_MANIFEST.read_text())
        tile = manifest["tiles"].get(ref, {}).get("webp")
        if tile is None:
            raise SystemExit(f"no gallery still for demo {ref!r} in {MEDIA_MANIFEST}")
        url = f"{manifest['base_url']}/{tile['key']}"
    elif kind == "url":
        url = ref
    else:
        raise SystemExit(
            f"panel source must start with key:, url: or file: ({source!r})"
        )
    if not url.startswith("https://"):
        raise SystemExit(f"panel URLs must be https:// ({url!r})")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    # The scheme is pinned to https just above, which is what B310 asks for.
    with urlopen(request, timeout=60) as response:  # nosec B310
        data: bytes = response.read()
    return data


def load_panel(source: str) -> Image.Image:
    """Return the first frame of a panel source as RGB."""
    image = Image.open(io.BytesIO(_load_bytes(source)))
    image.seek(0)
    return image.convert("RGB")


def cover_crop(
    image: Image.Image, width: int, height: int, anchor: float = 0.5
) -> Image.Image:
    """Scale to cover ``width x height`` and crop around a horizontal anchor (0 left, 1 right)."""
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = round((resized.width - width) * anchor)
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path, index in BOLD_FONTS if bold else REGULAR_FONTS:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    return ImageFont.load_default()


def compose(panels: list[tuple[Image.Image, float]], scale: int = 1) -> Image.Image:
    width, height = 1280 * scale, 640 * scale
    canvas = Image.new("RGB", (width, height), (8, 9, 12))
    panel_width = width // len(panels)
    for index, (panel, anchor) in enumerate(panels):
        canvas.paste(
            cover_crop(panel, panel_width, height, anchor), (index * panel_width, 0)
        )

    # Scrim: nothing at the top, near-opaque at the lower left where the text
    # sits, fading out to the right so the artwork stays bright.
    ys = np.linspace(0, 1, height)[:, None]
    xs = np.linspace(0, 1, width)[None, :]
    vertical = np.clip((ys - 0.42) / 0.45, 0, 1) ** 1.5
    horizontal = np.clip((0.80 - xs) / 0.34, 0, 1) ** 0.9
    mask = np.clip(0.20 * vertical + 0.88 * vertical * horizontal, 0, 1)[..., None]
    base = np.asarray(canvas, dtype=np.float32)
    dark = np.array([4, 5, 9], dtype=np.float32)
    canvas = Image.fromarray((base * (1 - mask) + dark * mask).astype(np.uint8))

    draw = ImageDraw.Draw(canvas)
    s = scale
    draw.text(
        (72 * s, height - 230 * s), TITLE, font=font(108 * s), fill=(255, 255, 255)
    )
    draw.text(
        (78 * s, height - 104 * s),
        TAGLINE,
        font=font(29 * s, False),
        fill=(212, 219, 229),
    )
    draw.text(
        (78 * s, height - 60 * s), TYPES, font=font(24 * s, False), fill=(134, 178, 226)
    )
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--panel",
        action="append",
        help="panel source (key:, url: or file:), four times",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        help="1 for 1280x640 (GitHub's size), 2 for a 2560x1280 master",
    )
    parser.add_argument("-o", "--output", type=Path, default=Path("social-preview.png"))
    args = parser.parse_args(argv)

    sources = args.panel or list(DEFAULT_PANELS)
    if len(sources) != 4:
        parser.error("exactly four --panel sources are needed")
    panels = []
    for source in sources:
        base, anchor = split_source(source)
        panels.append((load_panel(base), anchor))
    image = compose(panels, scale=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, "PNG", optimize=True)
    print(
        f"{image.width}x{image.height} -> {args.output} ({args.output.stat().st_size / 1024:.0f} KB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
