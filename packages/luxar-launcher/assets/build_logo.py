"""Render the 🌌 emoji to a 1024×1024 PNG used as the Luxar logo.

Mirrors the inline-SVG favicon used by ``packages/luxar-viewer/index.html``.
The output PNG is committed to the repo so the build does not need a
runtime emoji-rendering pipeline. Re-run only if you want to refresh the
logo.

Apple Color Emoji is a bitmap-strike font (native size 160 px). Pillow
with ``embedded_color=True`` reads those strikes and we upscale once with
LANCZOS to 1024 px.

Usage:
    hatch run python packages/luxar-launcher/assets/build_logo.py
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EMOJI = "\U0001F30C"  # 🌌
# Apple Color Emoji ships bitmap strikes at fixed sizes (20, 32, 40, 48,
# 64, 96, 160). Pillow refuses any other pixel size with "invalid pixel
# size", so we render at the largest native strike (160) then upscale.
NATIVE_FONT_PIXEL = 160
TARGET_SIZE = 1024
FONT_PATH = "/System/Library/Fonts/Apple Color Emoji.ttc"


def render_emoji(size: int = TARGET_SIZE) -> Image.Image:
    font = ImageFont.truetype(FONT_PATH, NATIVE_FONT_PIXEL)
    canvas = Image.new(
        "RGBA",
        (NATIVE_FONT_PIXEL, NATIVE_FONT_PIXEL),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((0, 0), EMOJI, font=font, embedded_color=True)
    return canvas.resize((size, size), Image.LANCZOS)


def main() -> None:
    if platform.system() != "Darwin":
        sys.exit(
            "build_logo.py: macOS-only — relies on Apple Color Emoji "
            "(/System/Library/Fonts/Apple Color Emoji.ttc). Re-run on a "
            "Mac to refresh luxar-logo.png; the rendered file is checked "
            "in so non-macOS contributors don't need to regenerate it."
        )
    # Output directly into the package-bundled asset dir so the rendered
    # PNG ships with `pip install luxar`. The script's own location (in
    # luxar-launcher/) is intentionally NOT under the package.
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "packages" / "luxar" / "src" / "luxar" / "cli" / "_launcher_assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    img = render_emoji(TARGET_SIZE)
    out = out_dir / "luxar-logo.png"
    img.save(out, "PNG")
    print(f"Wrote {out} ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
