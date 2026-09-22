#!/usr/bin/env python3
"""Cut the README's looping animations (WebP) from the release screen recordings.

The README embeds five animations, all served from ``data.luxarviewer.dev/media``
by content hash rather than committed to git:

* ``hero``        the Drosophila gastrulation recording (social clip ``d01``),
* ``quickstart``  two commands typed in a terminal, then the ``cloud`` demo
                  playing in the viewer (social clip ``d36``),
* ``volume``      a light-sheet stack in napari, ``luxar gsplat fit`` running in
                  a terminal with its PSNR curve, then the fitted splats in the
                  viewer: four excerpts of Supplementary Video 1's master,
* ``lod``         the six ``lod --recipe`` topologies built from one Tribolium
                  fit, side by side and up close: excerpts of Supplementary
                  Video 7's master,
* ``ndnav``       the Dimension Navigation panel playing the time axis of the
                  C. elegans recording: an excerpt of Supplementary Video 12.

The sources are the release social kit's 1080p H.264 clips on the shared drive
(``Shared drives/royerlab/Projects/luxar/social_media_release/clips/
x_threads_linkedin_1080p/``; ``LUXAR_SOCIAL_KIT`` or ``--clips-dir``), which
carry a 72 px caption band that is cropped off, and the uncarded supplementary
video masters (``luxar-paper/supp_videos/final/uncarded/``;
``LUXAR_SUPP_VIDEOS`` or ``--supp-dir``), from which timed excerpts are
concatenated. Output is a looping WebP sized for a README column.

    hatch run python scripts/gallery/make_readme_animations.py -o out/            # all
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
DEFAULT_SUPP = Path.home() / "workspace/python/luxar-paper/supp_videos/final/uncarded"
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
    source: str = (
        "social"  # "social" (caption band cropped) or "supp" (uncarded master)
    )
    segments: tuple[
        tuple[float, float], ...
    ] = ()  # (start s, duration s); empty = whole clip


ANIMATIONS = {
    "hero": Animation(
        "hero-drosophila", "luxar_d01_drosophila_gastrulation.mp4", 1000, quality=60
    ),
    "quickstart": Animation("quickstart-cloud", "luxar_d36_quickstart_cloud.mp4", 1200),
    "volume": Animation(
        "volume-to-scene",
        "SuppVideo01_volume_to_scene.mp4",
        1000,
        fps=10,
        quality=48,
        source="supp",
        # napari stack; fit console with the PSNR curve; the script; the viewer opens
        segments=((6.0, 5.0), (24.0, 8.0), (67.0, 3.0), (82.0, 9.0)),
    ),
    "lod": Animation(
        "lod-recipes",
        "SuppVideo07_lod_recipes.mp4",
        1000,
        fps=10,
        quality=48,
        source="supp",
        # the six recipes side by side; the levels column swapping coarse to fine;
        # the tiles column; the adaptive column close up; back to the row
        segments=((0.0, 5.0), (12.0, 6.0), (45.0, 5.0), (99.0, 6.0), (118.0, 4.0)),
    ),
    "ndnav": Animation(
        "nd-navigation",
        "SuppVideo12_viewer_interface.mp4",
        1000,
        fps=10,
        quality=48,
        source="supp",
        # the Dimension Navigation panel plays the time axis of the C. elegans
        # recording: a handful of nuclei become a full embryo with its tracks
        segments=((74.0, 20.0),),
    ),
}


def encode(source: Path, out: Path, spec: Animation) -> None:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    crop = f"crop=1920:{1080 - CAPTION_BAND_PX}:0:0," if spec.source == "social" else ""
    finish = f"{crop}fps={spec.fps},scale={spec.width}:-2:flags=lanczos"
    if spec.segments:
        parts = "".join(
            f"[0:v]trim=start={start}:duration={dur},setpts=PTS-STARTPTS[s{k}];"
            for k, (start, dur) in enumerate(spec.segments)
        )
        inputs = "".join(f"[s{k}]" for k in range(len(spec.segments)))
        graph = f"{parts}{inputs}concat=n={len(spec.segments)}:v=1:a=0,{finish}[out]"
        video = ["-filter_complex", graph, "-map", "[out]"]
    else:
        video = ["-vf", finish]
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source), *video,
        "-loop", "0", "-c:v", "libwebp", "-quality", str(spec.quality),
        "-compression_level", "6", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True)  # nosec B603: fixed argv, no shell


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
    parser.add_argument(
        "--supp-dir",
        type=Path,
        default=Path(os.environ.get("LUXAR_SUPP_VIDEOS", DEFAULT_SUPP)),
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("readme-animations"))
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    for key in args.which or sorted(ANIMATIONS):
        spec = ANIMATIONS[key]
        source = (
            args.supp_dir if spec.source == "supp" else args.clips_dir
        ) / spec.clip
        if not source.is_file():
            raise SystemExit(
                f"source clip not found: {source} "
                "(set LUXAR_SOCIAL_KIT / LUXAR_SUPP_VIDEOS or --clips-dir / --supp-dir)"
            )
        out = args.out / f"{spec.name}.webp"
        encode(source, out, spec)
        print(
            f"{out} ({out.stat().st_size / 1e6:.1f} MB, {spec.width} px wide, {spec.fps} fps)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
