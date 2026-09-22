#!/usr/bin/env python3
"""Cut the README's looping animations (WebP) from the release screen recordings.

The README embeds two animations, both served from ``data.luxarviewer.dev/media``
by content hash rather than committed to git:

* ``hero``        the Drosophila gastrulation recording (social clip ``d01``),
* ``quickstart``  two commands typed in a terminal, then the ``cloud`` demo
                  playing in the viewer (social clip ``d36``).

The source clips are the release social kit's 1080p H.264 clips, which live on
the shared drive (``Shared drives/royerlab/Projects/luxar/social_media_release/
clips/x_threads_linkedin_1080p/``); set ``LUXAR_SOCIAL_KIT`` to that folder or
pass ``--clips-dir``. Each clip carries a 72 px caption band at the bottom that
is cropped off. Output is a looping WebP at 12 fps, sized for a README column.

    hatch run python scripts/gallery/make_readme_animations.py -o out/            # both
    hatch run python scripts/gallery/make_readme_animations.py hero -o out/        # one
    hatch run python scripts/gallery/publish_media.py out/*.webp                    # host them

Needs ffmpeg with libwebp on PATH.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404: shells out to ffmpeg with a fixed argv
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_KIT = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-loic.royer@czbiohub.org/Shared drives/royerlab/Projects/luxar"
    / "social_media_release/clips/x_threads_linkedin_1080p"
)
CAPTION_BAND_PX = (
    72  # the clips are 1920x1080 with a caption band below a 1920x1008 picture
)


@dataclass(frozen=True)
class Animation:
    name: str
    clip: str
    width: int
    fps: int = 12
    quality: int = 62


ANIMATIONS = {
    "hero": Animation(
        "hero-drosophila", "luxar_d01_drosophila_gastrulation.mp4", 1000, quality=60
    ),
    "quickstart": Animation("quickstart-cloud", "luxar_d36_quickstart_cloud.mp4", 1200),
}


def encode(source: Path, out: Path, spec: Animation) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    vf = (
        f"crop=1920:{1080 - CAPTION_BAND_PX}:0:0,fps={spec.fps},"
        f"scale={spec.width}:-2:flags=lanczos"
    )
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vf", vf,
        "-loop", "0", "-c:v", "libwebp", "-quality", str(spec.quality),
        "-compression_level", "6", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)  # nosec B603, B607: fixed argv, tool from PATH


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "which",
        nargs="*",
        choices=sorted(ANIMATIONS),
        help="which animations (default: all)",
    )
    parser.add_argument(
        "--clips-dir",
        type=Path,
        default=Path(os.environ.get("LUXAR_SOCIAL_KIT", DEFAULT_KIT)),
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("readme-animations"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.which or sorted(ANIMATIONS):
        spec = ANIMATIONS[key]
        source = args.clips_dir / spec.clip
        if not source.is_file():
            raise SystemExit(
                f"source clip not found: {source} (set LUXAR_SOCIAL_KIT or --clips-dir)"
            )
        out = args.out / f"{spec.name}.webp"
        encode(source, out, spec)
        print(
            f"{out} ({out.stat().st_size / 1e6:.1f} MB, {spec.width} px wide, {spec.fps} fps)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
