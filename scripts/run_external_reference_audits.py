"""Run every external-reference audit and write one non-gating summary.

Most legs observe third parties over the network. One does not: the record
attribution audit compares two copies of the same provenance claim that both
live in this repository, and belongs here because the live record text it
measures against is authored on Zenodo, so a finding cannot be fixed by a
commit and must not gate one.

The shared report vocabulary is deliberately small:

``PASS``
    The audit completed without findings.
``NOTICE``
    The audit completed, but some references still require a human check.
``WARNING``
    The audit found stale/broken references or a third-party request failed.
``ERROR``
    The audit could not run because its command or configuration is broken.

The command always exits zero. These checks measure things this repository does
not own, so their findings belong in the report rather than in the required CI
gate.
"""

from __future__ import annotations

import html
import os
import re

# The commands are fixed below and are never passed through a shell.
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Mapping, Sequence


class Level(IntEnum):
    PASS = 0
    NOTICE = 1
    WARNING = 2
    ERROR = 3


@dataclass(frozen=True)
class Audit:
    name: str
    command: tuple[str, ...]
    required_env: tuple[str, ...] = ()
    parse_levels: bool = False


@dataclass(frozen=True)
class Result:
    audit: Audit
    level: Level
    detail: str
    output: str


AUDITS = (
    Audit("Documentation links", ("make", "check-docs-external-links")),
    Audit("Demo click-throughs", ("make", "check-demo-links"), parse_levels=True),
    Audit("Zenodo record snapshots", ("make", "check-zenodo-snapshots")),
    Audit("Record attribution", ("make", "check-record-attribution")),
    Audit(
        "Zenodo manifest pins",
        ("make", "check-zenodo-live"),
        required_env=("ZENODO_TOKEN",),
    ),
    Audit("Hosted gallery media", ("make", "check-gallery-media")),
)
MAX_SUMMARY_OUTPUT_CHARS = 20_000
REPO_ROOT = Path(__file__).resolve().parents[1]
_SENSITIVE_ENV = frozenset(name for audit in AUDITS for name in audit.required_env)

_DEMO_LEVELS = {
    "OK": Level.PASS,
    "HUMAN": Level.NOTICE,
    "STALE": Level.WARNING,
    "FAIL": Level.WARNING,
    "ERROR": Level.WARNING,
    "CONFIG": Level.ERROR,
}
_LEVEL_PATTERN = re.compile(
    rf"^\[({'|'.join(re.escape(level) for level in _DEMO_LEVELS)})]", re.MULTILINE
)
_COMMAND_ERROR_MARKERS = (
    # A producer that reports [CONFIG] could not run its comparison at all,
    # which is a broken configuration rather than a finding about a reference.
    "[CONFIG]",
    "No rule to make target",
    "Traceback (most recent call last):",
    "command not found",
    "Error 127",
)
_REJECTED_CREDENTIAL_MARKERS = (
    "Zenodo returned HTTP 401",
    "Zenodo returned HTTP 403",
)


def _level_from_output(output: str) -> Level | None:
    levels = [_DEMO_LEVELS[match] for match in _LEVEL_PATTERN.findall(output)]
    return max(levels) if levels else None


def _captured_output(*parts: str | bytes | None) -> str:
    decoded = []
    for part in parts:
        if isinstance(part, bytes):
            part = part.decode("utf-8", errors="replace")
        if part:
            decoded.append(part.rstrip())
    return "\n".join(decoded)


def run_audit(
    audit: Audit,
    *,
    env: Mapping[str, str] = os.environ,
    timeout_seconds: int = 900,
) -> Result:
    missing = [name for name in audit.required_env if not env.get(name)]
    if missing:
        names = ", ".join(missing)
        return Result(audit, Level.NOTICE, f"not configured; leg skipped: {names}", "")

    try:
        command_env = dict(env)
        for name in _SENSITIVE_ENV.difference(audit.required_env):
            command_env.pop(name, None)
        # Every argv comes from the constant AUDITS registry.
        completed = subprocess.run(
            audit.command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=command_env,
            cwd=REPO_ROOT,
        )  # nosec B603
    except subprocess.TimeoutExpired as error:
        output = _captured_output(error.stdout, error.stderr)
        return Result(
            audit,
            Level.WARNING,
            f"timed out after {timeout_seconds} seconds",
            output,
        )
    except OSError as error:
        return Result(audit, Level.ERROR, str(error), "")

    output = _captured_output(completed.stdout, completed.stderr)
    if completed.returncode != 0:
        command_error = any(marker in output for marker in _COMMAND_ERROR_MARKERS)
        rejected_credentials = bool(audit.required_env) and any(
            marker in output for marker in _REJECTED_CREDENTIAL_MARKERS
        )
        level = Level.ERROR if command_error or rejected_credentials else Level.WARNING
        return Result(audit, level, f"exited {completed.returncode}", output)

    level = _level_from_output(output) if audit.parse_levels else Level.PASS
    if level is None:
        return Result(
            audit,
            Level.ERROR,
            "completed with invalid report levels",
            output,
        )
    details = {
        Level.PASS: "completed without findings",
        Level.NOTICE: "completed with human checks",
        Level.WARNING: "completed with findings",
        Level.ERROR: "completed with configuration findings",
    }
    return Result(audit, level, details[level], output)


def render_summary(results: Sequence[Result]) -> str:
    worst_level = max((result.level for result in results), default=Level.PASS)
    lines = [
        "# External reference audits",
        "",
        "These external-reference checks are report-only and never gate merges.",
        "",
        f"**Worst level: {worst_level.name}**",
        "",
        "| Audit | Level | Detail |",
        "| --- | --- | --- |",
    ]
    for result in results:
        detail = result.detail.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {result.audit.name} | **{result.level.name}** | {detail} |")
    for result in results:
        output = result.output
        if len(output) > MAX_SUMMARY_OUTPUT_CHARS:
            omitted = len(output) - MAX_SUMMARY_OUTPUT_CHARS
            output = f"... {omitted} earlier characters omitted ...\n{output[-MAX_SUMMARY_OUTPUT_CHARS:]}"
        lines.extend(
            [
                "",
                f"<details><summary>{result.audit.name}: {result.level.name}</summary>",
                "",
                "<pre>",
                html.escape(output or "(no command output)"),
                "</pre>",
                "</details>",
            ]
        )
    return "\n".join(lines) + "\n"


def render_annotations(results: Sequence[Result]) -> str:
    lines = []
    for result in results:
        if result.level is Level.PASS:
            continue
        command = "error" if result.level is Level.ERROR else "warning"
        message = f"{result.audit.name}: {result.level.name} - {result.detail}"
        message = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        lines.append(f"::{command} title=External reference audit::{message}")
    return "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    results = [run_audit(audit) for audit in AUDITS]
    summary = render_summary(results)
    print(render_annotations(results), end="")
    print(summary, end="")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        try:
            with Path(summary_path).open("a") as summary_file:
                summary_file.write(summary)
        except OSError as error:
            print(f"Could not write GitHub summary: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
