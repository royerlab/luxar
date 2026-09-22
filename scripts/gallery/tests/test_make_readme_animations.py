"""Tests for the README animation cutter.

The encode itself needs ffmpeg and the release social-kit clips, which live on
the shared drive, so the reproducibility check is skipped where either is
missing (CI). Where it runs, it asserts the hero encode is byte-identical to the
hosted object: content-addressing would give a different encode a different URL
rather than break anything, but a silently drifting claim is worse than a red
test.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "make_readme_animations.py"
SPEC = importlib.util.spec_from_file_location("make_readme_animations", SCRIPT)
assert SPEC and SPEC.loader
anim = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = anim
SPEC.loader.exec_module(anim)

HOSTED_HERO_SHA16 = "4a0c1a68d18e092e"


def test_every_animation_names_its_clip_and_a_readme_sized_width() -> None:
    for key, spec in anim.ANIMATIONS.items():
        assert spec.clip.endswith(".mp4"), key
        assert 800 <= spec.width <= 1400, key
        assert spec.fps == 12, key


def test_missing_clip_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="source clip not found"):
        anim.main(["hero", "--clips-dir", str(tmp_path), "-o", str(tmp_path / "out")])


@pytest.mark.slow
def test_hero_encode_reproduces_the_hosted_object(tmp_path: Path) -> None:
    clip = anim.DEFAULT_KIT / anim.ANIMATIONS["hero"].clip
    if shutil.which("ffmpeg") is None or not clip.is_file():
        pytest.skip("needs ffmpeg and the social-kit clips (shared drive)")
    anim.main(["hero", "-o", str(tmp_path)])
    digest = hashlib.sha256(
        (tmp_path / "hero-drosophila.webp").read_bytes()
    ).hexdigest()
    assert digest[:16] == HOSTED_HERO_SHA16, (
        "the hero encode no longer matches the hosted object; publish the new encode "
        "under its own hash and update HOSTED_HERO_SHA16"
    )
