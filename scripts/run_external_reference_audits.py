"""Run every network-backed reference audit and write one non-gating summary.

The shared report vocabulary is deliberately small:

``PASS``
    The audit completed without findings.
``NOTICE``
    The audit completed, but some references still require a human check.
``WARNING``
    The audit found stale/broken references or a third-party request failed.
``ERROR``
    The audit could not run because its command or configuration is broken.

The command always exits zero. These checks observe third parties, so their
findings belong in the report rather than in the required CI gate.
"""

from __future__ import annotations

import html
import os
import re
import subprocess
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
    Audit(
        "Zenodo manifest pins",
        ("hatch", "run", "python", "scripts/zenodo_migration_audit.py", "--live"),
        required_env=("ZENODO_TOKEN",),
    ),
)
MAX_SUMMARY_OUTPUT_CHARS = 20_000
_SENSITIVE_ENV = frozenset(name for audit in AUDITS for name in audit.required_env)

_DEMO_LEVELS = {
    "OK": Level.PASS,
    "HUMAN": Level.NOTICE,
    "STALE": Level.WARNING,
    "FAIL": Level.WARNING,
    "ERROR": Level.WARNING,
    "CONFIG": Level.ERROR,
}
_LEVEL_PATTERN = re.compile(r"^\[([A-Z]+)]", re.MULTILINE)


def _level_from_output(output: str) -> Level:
    levels = [
        _DEMO_LEVELS.get(match, Level.ERROR) for match in _LEVEL_PATTERN.findall(output)
    ]
    return max(levels, default=Level.ERROR)


def run_audit(
    audit: Audit,
    *,
    env: Mapping[str, str] = os.environ,
    timeout_seconds: int = 900,
) -> Result:
    missing = [name for name in audit.required_env if not env.get(name)]
    if missing:
        names = ", ".join(missing)
        return Result(audit, Level.ERROR, f"missing required environment: {names}", "")

    try:
        command_env = dict(env)
        for name in _SENSITIVE_ENV.difference(audit.required_env):
            command_env.pop(name, None)
        completed = subprocess.run(
            audit.command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=command_env,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return Result(audit, Level.ERROR, str(error), "")

    output = "\n".join(
        part.rstrip() for part in (completed.stdout, completed.stderr) if part
    )
    if completed.returncode != 0:
        missing_target = "No rule to make target" in output
        level = Level.ERROR if missing_target else Level.WARNING
        return Result(audit, level, f"exited {completed.returncode}", output)

    level = _level_from_output(output) if audit.parse_levels else Level.PASS
    details = {
        Level.PASS: "completed without findings",
        Level.NOTICE: "completed with human checks",
        Level.WARNING: "completed with findings",
        Level.ERROR: "completed with invalid report levels",
    }
    return Result(audit, level, details[level], output)


def render_summary(results: Sequence[Result]) -> str:
    lines = [
        "# External reference audits",
        "",
        "These network-backed checks are report-only and never gate merges.",
        "",
        "| Audit | Level | Detail |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(
            f"| {result.audit.name} | **{result.level.name}** | {result.detail} |"
        )
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


def main() -> int:
    results = [run_audit(audit) for audit in AUDITS]
    summary = render_summary(results)
    print(summary, end="")
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(summary_path).open("a") as summary_file:
            summary_file.write(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
