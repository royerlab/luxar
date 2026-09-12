#!/usr/bin/env python3
"""Generate the remote-control wire constants for Python, TypeScript and Go.

``control-contract/contract.yaml`` is the single source of truth for everything
a control hub and a control client must agree on: roles, query-parameter names,
the close code for a refused handshake, the JSON-RPC error codes, and the limits
that bound what an untrusted peer can make the relay hold. This script projects
it into:

    packages/luxar/src/luxar/cli/_control_contract.py      (constants)
    packages/luxar-viewer/src/config/control-contract.ts   (const + types)
    packages/luxar-launcher/control_contract.go            (consts)

A separate contract from ``format-contract/contract.yaml`` on purpose: that one
describes the DATA format, this one a network protocol, and merging them would
mean a change to either re-generating both.

Why generate at all, rather than write three small files by hand: the third
implementation lives in a **shipped native binary**. Nobody rebuilds the
launcher when the Python hub's close code or pending cap changes, so a
hand-maintained copy would drift silently and the drift would show up as a
kiosk that fails to attach in front of an audience. ``--check`` is the gate that
makes that impossible.

Usage::

    python scripts/gen_control_contract.py           # regenerate all three
    python scripts/gen_control_contract.py --check   # drift gate (CI)

``--check`` regenerates in memory and diffs against the committed files; it
exits 0 when they match, 1 on drift (printing a unified diff), 2 on a
read/parse error.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "control-contract" / "contract.yaml"
PY_OUT = REPO / "packages/luxar/src/luxar/cli/_control_contract.py"
TS_OUT = REPO / "packages/luxar-viewer/src/config/control-contract.ts"
GO_OUT = REPO / "packages/luxar-launcher/control_contract.go"

_EDIT_HINT = (
    "edit control-contract/contract.yaml, then run `hatch run gen-control-contract`"
)

_BANNER = (
    "GENERATED FILE — DO NOT EDIT.\n"
    "\n"
    "Projected from control-contract/contract.yaml by\n"
    "scripts/gen_control_contract.py. Edit the YAML and regenerate; a drift\n"
    "gate (`hatch run check-control-contract`) fails the build if this file\n"
    "and the contract disagree."
)


# ─────────────────────────────── reading ────────────────────────────────────


def _section(contract: Dict[str, Any], name: str) -> Dict[str, Any]:
    value = contract.get(name)
    if not isinstance(value, dict) or not value:
        raise TypeError(f"contract section {name!r} must be a non-empty mapping")
    return value


def _int_items(contract: Dict[str, Any], *path: str) -> List[tuple[str, int]]:
    """``(key, value)`` pairs from a nested section, every value an int."""
    node: Any = contract
    for step in path:
        node = _section(node if isinstance(node, dict) else {}, step)
    items = []
    for key, value in node.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{'.'.join(path)}.{key} must be an int, got {value!r}")
        items.append((str(key), value))
    return items


def _str_items(contract: Dict[str, Any], *path: str) -> List[tuple[str, str]]:
    """``(key, value)`` pairs from a nested section, every value a string."""
    node: Any = contract
    for step in path:
        node = _section(node if isinstance(node, dict) else {}, step)
    items = []
    for key, value in node.items():
        if not isinstance(value, str) or not value:
            raise TypeError(
                f"{'.'.join(path)}.{key} must be a non-empty string, got {value!r}"
            )
        items.append((str(key), value))
    return items


# ─────────────────────────────── Python ─────────────────────────────────────


def render_python(c: Dict[str, Any]) -> str:
    lines = ['"""' + _BANNER + '"""', "", "from __future__ import annotations", ""]

    lines.append("# Roles a socket may declare as ?role=.")
    for key, value in _str_items(c, "roles"):
        lines.append(f'ROLE_{key.upper()} = "{value}"')
    roles = ", ".join(f"ROLE_{k.upper()}" for k, _ in _str_items(c, "roles"))
    lines += ["ROLES = frozenset({" + roles + "})", ""]

    lines.append("# Query-parameter names on the socket URL.")
    for key, value in _str_items(c, "query_params"):
        lines.append(f'PARAM_{key.upper()} = "{value}"')
    lines.append("")

    lines.append("# WebSocket close codes.")
    for key, value in _int_items(c, "close_codes"):
        lines.append(f"CLOSE_{key.upper()} = {value}")
    lines.append("")

    lines.append("# JSON-RPC.")
    lines.append(f'JSONRPC_VERSION = "{c["jsonrpc"]["version"]}"')
    lines.append(f'EVENT_METHOD = "{c["jsonrpc"]["event_method"]}"')
    for key, value in _int_items(c, "jsonrpc", "error_codes"):
        lines.append(f"{key.upper()} = {value}")
    lines.append("")

    lines.append("# Limits bounding what an untrusted peer can make us hold.")
    for key, value in _int_items(c, "limits"):
        lines.append(f"{key.upper()} = {value}")
    lines.append("")

    return "\n".join(lines)


# ───────────────────────────── TypeScript ───────────────────────────────────


def render_typescript(c: Dict[str, Any]) -> str:
    lines = ["/**", *[f" * {ln}".rstrip() for ln in _BANNER.split("\n")], " */", ""]

    lines.append("/** Roles a socket may declare as `?role=`. */")
    lines.append("export const CONTROL_ROLES = {")
    for key, value in _str_items(c, "roles"):
        lines.append(f"  {key}: '{value}',")
    lines += ["} as const;", ""]
    lines += [
        "export type ControlRole = (typeof CONTROL_ROLES)[keyof typeof CONTROL_ROLES];",
        "",
    ]

    lines.append("/** Query-parameter names on the socket URL. */")
    lines.append("export const CONTROL_PARAMS = {")
    for key, value in _str_items(c, "query_params"):
        lines.append(f"  {key}: '{value}',")
    lines += ["} as const;", ""]

    lines.append("/** WebSocket close codes. */")
    for key, value in _int_items(c, "close_codes"):
        lines.append(f"export const CLOSE_{key.upper()} = {value};")
    lines.append("")

    lines.append("/** JSON-RPC. */")
    lines.append(f"export const JSONRPC_VERSION = '{c['jsonrpc']['version']}';")
    lines.append(f"export const EVENT_METHOD = '{c['jsonrpc']['event_method']}';")
    for key, value in _int_items(c, "jsonrpc", "error_codes"):
        lines.append(f"export const {key.upper()} = {value};")
    lines.append("")

    lines.append("/** Limits bounding what an untrusted peer can make us hold. */")
    for key, value in _int_items(c, "limits"):
        lines.append(f"export const {key.upper()} = {value};")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────── Go ────────────────────────────────────────


def _go_name(key: str) -> str:
    """``max_frame_bytes`` -> ``MaxFrameBytes`` (Go exported camel case)."""
    return "".join(part.capitalize() for part in key.split("_"))


def _go_const_block(entries: List[tuple[str, str]]) -> List[str]:
    """A gofmt-shaped ``const (...)`` block.

    gofmt aligns the ``=`` of consecutive single-line specs within a block, so
    the generator has to produce that alignment itself — otherwise every
    regeneration leaves the file one ``gofmt -w`` away from its committed form,
    and the launcher's format check fails on generated output nobody edited.
    """
    width = max(len(name) for name, _ in entries)
    return (
        ["const ("]
        + [f"\t{name.ljust(width)} = {literal}" for name, literal in entries]
        + [")"]
    )


def render_go(c: Dict[str, Any]) -> str:
    lines = [
        *[f"// {ln}".rstrip() for ln in _BANNER.split("\n")],
        "",
        "package main",
        "",
    ]

    blocks = [
        (
            "// Roles a socket may declare as ?role=.",
            [(f"Role{_go_name(k)}", f'"{v}"') for k, v in _str_items(c, "roles")],
        ),
        (
            "// Query-parameter names on the socket URL.",
            [
                (f"Param{_go_name(k)}", f'"{v}"')
                for k, v in _str_items(c, "query_params")
            ],
        ),
        (
            "// WebSocket close codes.",
            [(f"Close{_go_name(k)}", str(v)) for k, v in _int_items(c, "close_codes")],
        ),
        (
            "// JSON-RPC.",
            [
                ("JSONRPCVersion", f'"{c["jsonrpc"]["version"]}"'),
                ("EventMethod", f'"{c["jsonrpc"]["event_method"]}"'),
                *[
                    (f"Code{_go_name(k)}", str(v))
                    for k, v in _int_items(c, "jsonrpc", "error_codes")
                ],
            ],
        ),
        (
            "// Limits bounding what an untrusted peer can make us hold.",
            [(_go_name(k), str(v)) for k, v in _int_items(c, "limits")],
        ),
    ]

    for comment, entries in blocks:
        lines.append(comment)
        lines.extend(_go_const_block(entries))
        lines.append("")

    return "\n".join(lines)


# ────────────────────────────── driver ──────────────────────────────────────


def load_contract() -> Dict[str, Any]:
    data = yaml.safe_load(CONTRACT.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{CONTRACT} must contain a top-level mapping")
    return data


def _diff(path: Path, expected: str) -> List[str]:
    actual = path.read_text() if path.exists() else ""
    if actual == expected:
        return []
    return list(
        difflib.unified_diff(
            actual.splitlines(keepends=True),
            expected.splitlines(keepends=True),
            fromfile=f"{path} (committed)",
            tofile=f"{path} (regenerated)",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the committed files match the contract (no writes); "
        "exit 1 on drift",
    )
    args = parser.parse_args()

    try:
        contract = load_contract()
        outputs = {
            PY_OUT: render_python(contract),
            TS_OUT: render_typescript(contract),
            GO_OUT: render_go(contract),
        }
    except (OSError, TypeError, KeyError, yaml.YAMLError) as exc:
        print(f"error: cannot project {CONTRACT}: {exc}", file=sys.stderr)
        return 2

    if args.check:
        drifted = False
        for path, expected in outputs.items():
            diff = _diff(path, expected)
            if diff:
                drifted = True
                print(f"drift: {path.relative_to(REPO)}", file=sys.stderr)
                sys.stderr.writelines(diff)
        if drifted:
            print(
                f"\ncontrol-contract projections are stale. To fix, {_EDIT_HINT} "
                "and commit the regenerated files.",
                file=sys.stderr,
            )
            return 1
        print(f"✅ control-contract projections match ({len(outputs)} files)")
        return 0

    for path, expected in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected)
        print(f"wrote {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
