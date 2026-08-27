#!/usr/bin/env python3
"""Report committed README gallery tiles that predate their render inputs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path("scripts/gallery/manifest.json")
DEMOS_DIR = Path("packages/luxar/src/luxar/demos")
SHADING_PATH = Path("packages/luxar/src/luxar/shading")
SHADING_PATHSPECS = (
    f"{SHADING_PATH.as_posix()}/*.py",
    f":(exclude){SHADING_PATH.as_posix()}/tests/**",
)
TILES_PATHSPEC = "docs/images/readme/gallery/*.webp"

_BLAME_HEADER = re.compile(r"^([0-9a-f^]+) \d+ \d+(?: \d+)?$")


class StalenessError(RuntimeError):
    """The report could not be computed reliably."""


@dataclass(frozen=True)
class CommitStamp:
    sha: str
    committed_at: datetime


@dataclass(frozen=True)
class LineRange:
    start: int
    stop: int


@dataclass(frozen=True)
class TileStatus:
    demo_id: str
    tile: CommitStamp
    inputs: dict[str, CommitStamp]
    stale_inputs: tuple[str, ...]


def stale_inputs(tile: CommitStamp, inputs: dict[str, CommitStamp]) -> list[str]:
    """Return render-input labels committed strictly after ``tile``."""
    return [
        label
        for label, stamp in inputs.items()
        if stamp.committed_at > tile.committed_at
    ]


def manifest_entry_line_ranges(text: str) -> dict[str, LineRange]:
    """Locate each current ``demos`` object without mistaking nested braces for boundaries."""
    key = re.search(r'"demos"\s*:', text)
    if key is None:
        raise StalenessError("gallery manifest has no 'demos' array")
    cursor = text.find("[", key.end())
    if cursor < 0:
        raise StalenessError("gallery manifest 'demos' value is not an array")
    cursor += 1

    decoder = json.JSONDecoder()
    ranges: dict[str, LineRange] = {}
    while True:
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        if cursor >= len(text) or text[cursor] == "]":
            break
        start = cursor
        entry, consumed = decoder.raw_decode(text[cursor:])
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise StalenessError(
                "every gallery manifest entry must be an object with a string id"
            )
        cursor += consumed
        demo_id = entry["id"]
        if demo_id in ranges:
            raise StalenessError(f"duplicate gallery manifest id: {demo_id}")
        ranges[demo_id] = LineRange(
            start=text.count("\n", 0, start) + 1,
            stop=text.count("\n", 0, cursor) + 1,
        )
    return ranges


class GalleryHistory:
    """Read the narrow Git history inputs that can invalidate committed gallery tiles."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def _git(self, *args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.strip()
                if isinstance(exc, subprocess.CalledProcessError)
                else str(exc)
            )
            raise StalenessError(f"git {' '.join(args)} failed: {detail}") from exc

    def _require_full_history(self) -> None:
        if self._git("rev-parse", "--is-shallow-repository") == "true":
            raise StalenessError(
                "gallery tile staleness requires full Git history; fetch with --unshallow first"
            )

    def _last_commit_for_pathspecs(self, *pathspecs: str) -> CommitStamp:
        output = self._git("log", "-1", "--format=%H%x00%cI", "--", *pathspecs)
        if not output:
            raise StalenessError(f"no commit history for {', '.join(pathspecs)}")
        sha, committed_at = output.split("\0", 1)
        return CommitStamp(sha=sha, committed_at=datetime.fromisoformat(committed_at))

    def _last_commit(self, path: Path) -> CommitStamp:
        return self._last_commit_for_pathspecs(path.as_posix())

    def _manifest_entry_commit(self, line_range: LineRange) -> CommitStamp:
        output = self._git(
            "blame",
            "--line-porcelain",
            f"-L{line_range.start},{line_range.stop}",
            "--",
            MANIFEST_PATH.as_posix(),
        )
        stamps: list[CommitStamp] = []
        current_sha: str | None = None
        for line in output.splitlines():
            header = _BLAME_HEADER.match(line)
            if header:
                current_sha = header.group(1).lstrip("^")
            elif line.startswith("committer-time ") and current_sha:
                timestamp = int(line.removeprefix("committer-time "))
                stamps.append(
                    CommitStamp(
                        sha=current_sha,
                        committed_at=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                    )
                )
        if not stamps:
            raise StalenessError(
                f"no blame history for manifest lines {line_range.start}-{line_range.stop}"
            )
        return max(stamps, key=lambda stamp: stamp.committed_at)

    def _tracked_tiles(self) -> list[Path]:
        output = self._git("ls-files", "--", TILES_PATHSPEC)
        tiles = [Path(line) for line in output.splitlines() if line]
        if not tiles:
            raise StalenessError("no committed README gallery tiles found")
        return sorted(tiles)

    def tile_statuses(self) -> list[TileStatus]:
        self._require_full_history()
        manifest_text = (self.repo_root / MANIFEST_PATH).read_text()
        manifest = json.loads(manifest_text)
        entries = {entry["id"]: entry for entry in manifest["demos"]}
        ranges = manifest_entry_line_ranges(manifest_text)
        shading = self._last_commit_for_pathspecs(*SHADING_PATHSPECS)

        statuses: list[TileStatus] = []
        for tile_path in self._tracked_tiles():
            demo_id = tile_path.stem
            entry: dict[str, Any] | None = entries.get(demo_id)
            if entry is None or demo_id not in ranges:
                raise StalenessError(
                    f"committed tile {demo_id!r} has no manifest entry"
                )
            inputs = {
                "luxar.shading": shading,
            }
            script = entry.get("script")
            if script is not None:
                if not isinstance(script, str):
                    raise StalenessError(
                        f"manifest script for {demo_id!r} is not a string or null"
                    )
                inputs["demo generator"] = self._last_commit(DEMOS_DIR / script)
            inputs["manifest entry"] = self._manifest_entry_commit(ranges[demo_id])
            tile = self._last_commit(tile_path)
            statuses.append(
                TileStatus(
                    demo_id=demo_id,
                    tile=tile,
                    inputs=inputs,
                    stale_inputs=tuple(stale_inputs(tile, inputs)),
                )
            )
        return statuses


def _format_status(status: TileStatus) -> str:
    tile_stamp = _format_stamp(status.tile)
    if not status.stale_inputs:
        return f"CURRENT {status.demo_id}: tile {tile_stamp}"
    details = ", ".join(
        f"{label} {_format_stamp(status.inputs[label])}"
        for label in status.stale_inputs
    )
    return f"STALE {status.demo_id}: tile {tile_stamp}; newer inputs: {details}"


def _format_stamp(stamp: CommitStamp) -> str:
    timestamp = stamp.committed_at.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )
    return f"{timestamp.replace('+00:00', 'Z')} ({stamp.sha[:8]})"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    try:
        statuses = GalleryHistory(args.repo_root).tile_statuses()
    except (OSError, KeyError, json.JSONDecodeError, StalenessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for status in statuses:
        print(_format_status(status))
    stale_count = sum(bool(status.stale_inputs) for status in statuses)
    print(
        f"\nGallery tile staleness: {stale_count} stale, {len(statuses) - stale_count} current"
    )
    print(
        "Report only — inspect stale tiles and regenerate them when their render changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
