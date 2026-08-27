#!/usr/bin/env python3
"""Report committed README gallery tiles that predate their render inputs."""

from __future__ import annotations

import argparse
import ast
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
TILES_DIR = Path("docs/images/readme/gallery")
GLOBAL_INPUT_PATHSPECS = {
    "dataset generator": ("scripts/gallery/generate_gallery_datasets.py",),
    "gallery capture": (
        "packages/luxar-viewer/src/tests/screenshots/generate-gallery.spec.ts",
        "packages/luxar-viewer/src/tests/screenshots/orbit-axis.ts",
        "packages/luxar-viewer/playwright.gallery.config.ts",
    ),
    "exposure policy": (
        "packages/luxar-viewer/src/tests/screenshots/exposure-policy.ts",
    ),
    "crop policy": ("packages/luxar-viewer/src/tests/screenshots/crop-policy.ts",),
}

_BLAME_HEADER = re.compile(r"^([0-9a-f]+) \d+ \d+(?: \d+)?$")


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
    global_input_labels: tuple[str, ...]
    stale_inputs: tuple[str, ...]


@dataclass(frozen=True)
class UnknownStatus:
    demo_id: str
    reason: str


@dataclass(frozen=True)
class GalleryReport:
    global_inputs: dict[str, CommitStamp]
    statuses: tuple[TileStatus | UnknownStatus, ...]


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
        start_line = text.count("\n", 0, start) + 1
        stop_line = text.count("\n", 0, cursor) + 1
        closing_line_start = text.rfind("\n", 0, cursor - 1) + 1
        if text[closing_line_start:cursor].strip() == "}":
            stop_line -= 1
        ranges[demo_id] = LineRange(
            start=start_line,
            stop=max(start_line, stop_line),
        )
    return ranges


def imports_luxar_shading(source: str) -> bool:
    """Return whether a demo directly imports the public shading package."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "luxar.shading" or alias.name.startswith("luxar.shading.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "luxar.shading" or (
                node.module is not None and node.module.startswith("luxar.shading.")
            ):
                return True
            if node.module == "luxar" and any(
                alias.name == "shading" for alias in node.names
            ):
                return True
    return False


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

    def _require_tracked_head_files(self, label: str, *pathspecs: str) -> None:
        candidates = self._git(
            "ls-files", "--cached", "--with-tree=HEAD", "--", *pathspecs
        ).splitlines()
        tracked = (
            self._git("ls-tree", "-r", "--name-only", "HEAD", "--", *candidates)
            if candidates
            else ""
        )
        if not tracked:
            raise StalenessError(
                f"configured gallery input {label!r} has no tracked files at HEAD: "
                f"{', '.join(pathspecs)}"
            )

    def _last_commit(self, path: Path) -> CommitStamp:
        return self._last_commit_for_pathspecs(path.as_posix())

    def _manifest_entry_commit(self, line_range: LineRange) -> CommitStamp:
        output = self._git(
            "blame",
            "--line-porcelain",
            f"-L{line_range.start},{line_range.stop}",
            "HEAD",
            "--",
            MANIFEST_PATH.as_posix(),
        )
        stamps: list[CommitStamp] = []
        current_sha: str | None = None
        for line in output.splitlines():
            header = _BLAME_HEADER.match(line)
            if header:
                current_sha = header.group(1)
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

    def _head_text(self, path: Path) -> str | None:
        tracked = self._git("ls-tree", "--name-only", "HEAD", "--", path.as_posix())
        if not tracked:
            return None
        return self._git("show", f"HEAD:{path.as_posix()}")

    def _tracked_tile_media(self) -> list[tuple[str, tuple[Path, ...]]]:
        output = self._git(
            "ls-tree", "-r", "--name-only", "HEAD", "--", TILES_DIR.as_posix()
        )
        media = [
            Path(line)
            for line in output.splitlines()
            if Path(line).suffix in {".webp", ".webm"}
        ]
        if not media:
            raise StalenessError("no committed README gallery tiles found")
        by_demo: dict[str, list[Path]] = {}
        for path in media:
            by_demo.setdefault(path.stem, []).append(path)
        return [
            (demo_id, tuple(sorted(paths)))
            for demo_id, paths in sorted(by_demo.items())
        ]

    def report(self) -> GalleryReport:
        self._require_full_history()
        for label, pathspecs in GLOBAL_INPUT_PATHSPECS.items():
            self._require_tracked_head_files(label, *pathspecs)
        self._require_tracked_head_files("luxar.shading", *SHADING_PATHSPECS)
        manifest_text = self._git("show", f"HEAD:{MANIFEST_PATH.as_posix()}")
        manifest = json.loads(manifest_text)
        entries = {entry["id"]: entry for entry in manifest["demos"]}
        ranges = manifest_entry_line_ranges(manifest_text)
        global_inputs = {
            label: self._last_commit_for_pathspecs(*pathspecs)
            for label, pathspecs in GLOBAL_INPUT_PATHSPECS.items()
        }
        shading_input = self._last_commit_for_pathspecs(*SHADING_PATHSPECS)

        statuses: list[TileStatus | UnknownStatus] = []
        for demo_id, media_paths in self._tracked_tile_media():
            entry: dict[str, Any] | None = entries.get(demo_id)
            if entry is None or demo_id not in ranges:
                statuses.append(
                    UnknownStatus(
                        demo_id=demo_id,
                        reason=f"committed tile {demo_id!r} has no manifest entry",
                    )
                )
                continue
            try:
                inputs = dict(global_inputs)
                script = entry.get("script")
                if script is not None:
                    if not isinstance(script, str):
                        raise StalenessError(
                            f"manifest script for {demo_id!r} is not a string or null"
                        )
                    script_path = DEMOS_DIR / script
                    inputs["demo generator"] = self._last_commit(script_path)
                    source = self._head_text(script_path)
                    if source is not None and imports_luxar_shading(source):
                        inputs["luxar.shading"] = shading_input
                inputs["manifest entry"] = self._manifest_entry_commit(ranges[demo_id])
                tile = min(
                    (self._last_commit(path) for path in media_paths),
                    key=lambda stamp: stamp.committed_at,
                )
            except (SyntaxError, StalenessError) as exc:
                statuses.append(UnknownStatus(demo_id=demo_id, reason=str(exc)))
                continue
            statuses.append(
                TileStatus(
                    demo_id=demo_id,
                    tile=tile,
                    inputs=inputs,
                    global_input_labels=tuple(global_inputs),
                    stale_inputs=tuple(stale_inputs(tile, inputs)),
                )
            )
        return GalleryReport(global_inputs=global_inputs, statuses=tuple(statuses))

    def tile_statuses(self) -> list[TileStatus | UnknownStatus]:
        return list(self.report().statuses)


def _format_status(status: TileStatus) -> str:
    tile_stamp = _format_stamp(status.tile)
    if not status.stale_inputs:
        return f"CURRENT {status.demo_id}: tile {tile_stamp}"
    stale_per_tile = [
        label
        for label in status.stale_inputs
        if label not in status.global_input_labels
    ]
    stale_global = [
        label for label in status.stale_inputs if label in status.global_input_labels
    ]
    details = []
    if stale_global:
        details.append(f"newer global inputs: {', '.join(stale_global)}")
    per_tile_details = ", ".join(
        f"{label} {_format_stamp(status.inputs[label])}" for label in stale_per_tile
    )
    if per_tile_details:
        details.append(f"newer per-tile inputs: {per_tile_details}")
    return f"STALE {status.demo_id}: tile {tile_stamp}; {'; '.join(details)}"


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
        report = GalleryHistory(args.repo_root).report()
    except (OSError, KeyError, json.JSONDecodeError, StalenessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    global_details = "; ".join(
        f"{label} {_format_stamp(stamp)}"
        for label, stamp in report.global_inputs.items()
    )
    print(f"global inputs: {global_details}\n")
    for status in report.statuses:
        if isinstance(status, UnknownStatus):
            print(f"UNKNOWN {status.demo_id}: {status.reason}")
        else:
            print(_format_status(status))
    known_statuses = [
        status for status in report.statuses if isinstance(status, TileStatus)
    ]
    unknown_count = len(report.statuses) - len(known_statuses)
    stale_count = sum(bool(status.stale_inputs) for status in known_statuses)
    print(
        f"\nGallery tile staleness: {stale_count} stale, "
        f"{len(known_statuses) - stale_count} current, {unknown_count} unknown"
    )
    print(
        "Report only — inspect stale tiles and regenerate them when their render changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
