#!/usr/bin/env python3
"""Report committed README gallery tile staleness and media sizes."""

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
MEDIA_MANIFEST_PATH = Path("scripts/gallery/media-manifest.json")
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
MEBIBYTE = 1024 * 1024
GALLERY_MEDIA_WARNING_BYTES = 20 * MEBIBYTE
GALLERY_MEDIA_LIMIT_BYTES = 25 * MEBIBYTE
LARGEST_MEDIA_COUNT = 5
MAX_LFS_POINTER_BYTES = 1024

_BLAME_HEADER = re.compile(r"^([0-9a-f]+) \d+ \d+(?: \d+)?$")
_LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1\n"
_LFS_POINTER_SIZE = re.compile(rb"^size ([0-9]+)$", re.MULTILINE)


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
class GalleryMedia:
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class TileStatus:
    demo_id: str
    tile: CommitStamp
    media: tuple[GalleryMedia, ...]
    inputs: dict[str, CommitStamp]
    global_input_labels: tuple[str, ...]
    stale_inputs: tuple[str, ...]


@dataclass(frozen=True)
class UnknownStatus:
    demo_id: str
    reason: str
    media: tuple[GalleryMedia, ...]


@dataclass(frozen=True)
class GalleryReport:
    global_inputs: dict[str, CommitStamp]
    statuses: tuple[TileStatus | UnknownStatus, ...]
    media: tuple[GalleryMedia, ...]


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


def media_manifest_entry_line_ranges(text: str) -> dict[str, LineRange]:
    """Locate each top-level demo entry in the media manifest ``tiles`` object."""
    key = re.search(r'"tiles"\s*:', text)
    if key is None:
        raise StalenessError("gallery media manifest has no 'tiles' object")
    cursor = text.find("{", key.end())
    if cursor < 0:
        raise StalenessError("gallery media manifest 'tiles' value is not an object")
    cursor += 1

    decoder = json.JSONDecoder()
    ranges: dict[str, LineRange] = {}
    while True:
        while cursor < len(text) and (text[cursor].isspace() or text[cursor] == ","):
            cursor += 1
        if cursor >= len(text) or text[cursor] == "}":
            break
        start = cursor
        demo_id, consumed = decoder.raw_decode(text[cursor:])
        if not isinstance(demo_id, str):
            raise StalenessError(
                "every gallery media manifest tile key must be a string"
            )
        cursor += consumed
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != ":":
            raise StalenessError(
                f"gallery media manifest tile {demo_id!r} has no object value"
            )
        cursor += 1
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        entry, consumed = decoder.raw_decode(text[cursor:])
        if not isinstance(entry, dict):
            raise StalenessError(
                f"gallery media manifest tile {demo_id!r} is not an object"
            )
        cursor += consumed
        if demo_id in ranges:
            raise StalenessError(f"duplicate gallery media manifest id: {demo_id}")
        ranges[demo_id] = LineRange(
            start=text.count("\n", 0, start) + 1,
            stop=text.count("\n", 0, cursor) + 1,
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


def direct_demo_helper_modules(source: str) -> tuple[str, ...]:
    """Return directly imported private modules under ``luxar.demos``."""
    tree = ast.parse(source)
    modules = {
        node.module.removeprefix("luxar.demos.")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module is not None
        and node.module.startswith("luxar.demos._")
    }
    return tuple(sorted(modules))


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

    def _git_bytes(self, *args: str, input_bytes: bytes | None = None) -> bytes:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=self.repo_root,
                check=True,
                capture_output=True,
                input=input_bytes,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = (
                exc.stderr.decode(errors="replace").strip()
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
        exclusions = tuple(
            pathspec for pathspec in pathspecs if pathspec.startswith(":(exclude)")
        )
        for pathspec in pathspecs:
            if pathspec in exclusions:
                continue
            candidates = self._git(
                "ls-files",
                "--cached",
                "--with-tree=HEAD",
                "--",
                pathspec,
                *exclusions,
            ).splitlines()
            tracked = (
                self._git("ls-tree", "-r", "--name-only", "HEAD", "--", *candidates)
                if candidates
                else ""
            )
            if not tracked:
                raise StalenessError(
                    f"configured gallery input {label!r} has no tracked files at HEAD: "
                    f"{pathspec}"
                )

    def _last_commit(self, path: Path) -> CommitStamp:
        return self._last_commit_for_pathspecs(path.as_posix())

    def _entry_commit(self, path: Path, line_range: LineRange) -> CommitStamp:
        output = self._git(
            "blame",
            "--line-porcelain",
            f"-L{line_range.start},{line_range.stop}",
            "HEAD",
            "--",
            path.as_posix(),
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
                f"no blame history for {path} lines {line_range.start}-{line_range.stop}"
            )
        return max(stamps, key=lambda stamp: stamp.committed_at)

    def _head_text(self, path: Path) -> str | None:
        tracked = self._git("ls-tree", "--name-only", "HEAD", "--", path.as_posix())
        if not tracked:
            return None
        return self._git("show", f"HEAD:{path.as_posix()}")

    def _small_blob_contents(self, object_ids: Sequence[str]) -> dict[str, bytes]:
        unique_ids = tuple(dict.fromkeys(object_ids))
        if not unique_ids:
            return {}
        output = self._git_bytes(
            "cat-file",
            "--batch",
            input_bytes=("\n".join(unique_ids) + "\n").encode(),
        )
        blobs: dict[str, bytes] = {}
        cursor = 0
        for expected_id in unique_ids:
            header_end = output.find(b"\n", cursor)
            if header_end < 0:
                raise StalenessError("truncated git cat-file batch header")
            header = output[cursor:header_end].split()
            if len(header) == 2 and header[1] == b"missing":
                raise StalenessError(f"Git object is unavailable: {expected_id}")
            if len(header) != 3:
                raise StalenessError(
                    f"unexpected git cat-file batch header for {expected_id}"
                )
            object_id, object_type, size_text = header
            blob_size = int(size_text)
            cursor = header_end + 1
            blob = output[cursor : cursor + blob_size]
            cursor += blob_size
            if output[cursor : cursor + 1] != b"\n":
                raise StalenessError("truncated git cat-file batch object")
            cursor += 1
            decoded_id = object_id.decode()
            if decoded_id != expected_id or object_type != b"blob":
                raise StalenessError(
                    f"unexpected git cat-file batch object for {expected_id}"
                )
            blobs[decoded_id] = blob
        return blobs

    def _media_size(
        self,
        object_id: str,
        blob_size: int,
        small_blobs: dict[str, bytes],
    ) -> int:
        blob = small_blobs.get(object_id)
        if blob is None:
            return blob_size
        if not blob.startswith(_LFS_POINTER_HEADER):
            return blob_size
        match = _LFS_POINTER_SIZE.search(blob)
        if match is None:
            raise StalenessError(f"invalid Git LFS pointer object: {object_id}")
        return int(match.group(1))

    def _published_tile_media(
        self, text: str
    ) -> list[tuple[str, tuple[GalleryMedia, ...]]]:
        """Tiles as described by the committed media manifest."""
        try:
            manifest = json.loads(text)
        except json.JSONDecodeError as exc:
            raise StalenessError(
                f"{MEDIA_MANIFEST_PATH} is not valid JSON: {exc}"
            ) from exc
        tiles = manifest.get("tiles") or {}
        if not tiles:
            raise StalenessError("the gallery media manifest describes no tiles")
        by_demo: list[tuple[str, tuple[GalleryMedia, ...]]] = []
        for demo_id in sorted(tiles):
            media = tuple(
                sorted(
                    (
                        GalleryMedia(
                            path=Path(entry["key"]), size_bytes=int(entry["bytes"])
                        )
                        for entry in tiles[demo_id].values()
                    ),
                    key=lambda item: item.path,
                )
            )
            if media:
                by_demo.append((demo_id, media))
        return by_demo

    def _demo_code_inputs(
        self,
        demo_id: str,
        script: Any,
        shading_input: CommitStamp,
    ) -> dict[str, CommitStamp]:
        if script is None:
            return {}
        if not isinstance(script, str):
            raise StalenessError(
                f"manifest script for {demo_id!r} is not a string or null"
            )
        script_path = DEMOS_DIR / script
        inputs = {"demo generator": self._last_commit(script_path)}
        source = self._head_text(script_path)
        if source is None:
            return inputs
        for module in direct_demo_helper_modules(source):
            helper_path = DEMOS_DIR.joinpath(*module.split(".")).with_suffix(".py")
            if self._head_text(helper_path) is None:
                raise StalenessError(f"demo helper {module!r} is not tracked at HEAD")
            inputs[f"demo helper {module}"] = self._last_commit(helper_path)
        if imports_luxar_shading(source):
            inputs["luxar.shading"] = shading_input
        return inputs

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

        media_manifest_text = self._head_text(MEDIA_MANIFEST_PATH)
        if media_manifest_text is None:
            raise StalenessError(
                f"{MEDIA_MANIFEST_PATH} is not committed — the gallery media manifest "
                "is what records the published tiles"
            )
        media_ranges = media_manifest_entry_line_ranges(media_manifest_text)
        tracked_media = self._published_tile_media(media_manifest_text)
        statuses: list[TileStatus | UnknownStatus] = []
        for demo_id, media in tracked_media:
            entry: dict[str, Any] | None = entries.get(demo_id)
            if entry is None or demo_id not in ranges:
                statuses.append(
                    UnknownStatus(
                        demo_id=demo_id,
                        reason=f"published tile {demo_id!r} has no manifest entry",
                        media=media,
                    )
                )
                continue
            try:
                inputs = dict(global_inputs)
                inputs.update(
                    self._demo_code_inputs(
                        demo_id,
                        entry.get("script"),
                        shading_input,
                    )
                )
                inputs["manifest entry"] = self._entry_commit(
                    MANIFEST_PATH, ranges[demo_id]
                )
                tile = self._entry_commit(MEDIA_MANIFEST_PATH, media_ranges[demo_id])
            except (SyntaxError, StalenessError) as exc:
                statuses.append(
                    UnknownStatus(demo_id=demo_id, reason=str(exc), media=media)
                )
                continue
            statuses.append(
                TileStatus(
                    demo_id=demo_id,
                    tile=tile,
                    media=media,
                    inputs=inputs,
                    global_input_labels=tuple(global_inputs),
                    stale_inputs=tuple(stale_inputs(tile, inputs)),
                )
            )
        return GalleryReport(
            global_inputs=global_inputs,
            statuses=tuple(statuses),
            media=tuple(item for _, media in tracked_media for item in media),
        )

    def tile_statuses(self) -> list[TileStatus | UnknownStatus]:
        return list(self.report().statuses)


def _format_status(status: TileStatus) -> str:
    tile_stamp = _format_stamp(status.tile)
    if not status.stale_inputs:
        return (
            f"CURRENT {status.demo_id}: tile {tile_stamp}; "
            f"media: {_format_tile_media(status.media)}"
        )
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
    details.append(f"media: {_format_tile_media(status.media)}")
    return f"STALE {status.demo_id}: tile {tile_stamp}; {'; '.join(details)}"


def _format_stamp(stamp: CommitStamp) -> str:
    timestamp = stamp.committed_at.astimezone(timezone.utc).isoformat(
        timespec="seconds"
    )
    return f"{timestamp.replace('+00:00', 'Z')} ({stamp.sha[:8]})"


def _format_media_size(size_bytes: int) -> str:
    return f"{size_bytes * 100 // MEBIBYTE / 100:.2f} MiB"


def _media_flag(media: GalleryMedia) -> str:
    if media.size_bytes >= GALLERY_MEDIA_LIMIT_BYTES:
        return " [OVER LIMIT]"
    if media.size_bytes >= GALLERY_MEDIA_WARNING_BYTES:
        return " [WARNING]"
    return ""


def _format_tile_media(media: tuple[GalleryMedia, ...]) -> str:
    details = []
    for item in media:
        flag = _media_flag(item)
        exact_size = f" ({item.size_bytes:,} bytes)" if flag else ""
        details.append(
            f"{item.path.name} {_format_media_size(item.size_bytes)}{exact_size}{flag}"
        )
    return ", ".join(details)


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
            print(
                f"UNKNOWN {status.demo_id}: {status.reason}; "
                f"media: {_format_tile_media(status.media)}"
            )
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
    total_bytes = sum(item.size_bytes for item in report.media)
    print(
        f"Gallery media: {_format_media_size(total_bytes)} total across "
        f"{len(report.media)} files"
    )
    print("Largest gallery media:")
    for item in sorted(
        report.media,
        key=lambda media: (-media.size_bytes, media.path.name),
    )[:LARGEST_MEDIA_COUNT]:
        print(
            f"{_format_media_size(item.size_bytes):>11} "
            f"({item.size_bytes:,} bytes)  "
            f"{item.path.name}{_media_flag(item)}"
        )
    warning_count = sum(
        GALLERY_MEDIA_WARNING_BYTES <= item.size_bytes < GALLERY_MEDIA_LIMIT_BYTES
        for item in report.media
    )
    over_limit_count = sum(
        item.size_bytes >= GALLERY_MEDIA_LIMIT_BYTES for item in report.media
    )
    warning_label = "warning" if warning_count == 1 else "warnings"
    over_limit_label = "file" if over_limit_count == 1 else "files"
    print(
        f"Gallery media limits: {warning_count} {warning_label} "
        f"({_format_media_size(GALLERY_MEDIA_WARNING_BYTES)} <= size < "
        f"{_format_media_size(GALLERY_MEDIA_LIMIT_BYTES)}), "
        f"{over_limit_count} over-limit {over_limit_label} "
        f"(>= {_format_media_size(GALLERY_MEDIA_LIMIT_BYTES)})"
    )
    print(
        "Report only — inspect stale tiles and regenerate them when their render changed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
