#!/usr/bin/env python3
"""Generate the demo site's stable ``/d/<demo-key>`` routes as a Pages ``_redirects``.

Every URL the gallery emits carries a dated data prefix, so anything durable that
links to one — the README, a paper, an issue, a message — breaks on the next
deploy. It breaks *silently*: the origin answers a miss with ``200 text/html``
rather than 404, so a reader gets a blank viewer and nothing goes red.

``/d/<demo-key>`` is an identity; the dated prefix is a location. This maps the
former onto the latter and is regenerated at deploy, so the mapping updates
itself and cannot drift from what is served.

A static ``_redirects`` is deliberate: Cloudflare Pages serves it without putting
a Function on the request path.

Two things here are load-bearing and easy to get wrong in a rewrite:

1. **A demo key is not always its store name.** Eight manifest entries differ
   (``cosmicflows_laniakea`` → ``cosmicflows_laniakea_full``, ``nd_transforms`` →
   ``nd_transforms_bench``, and six more). A mechanical ``id == store`` mapping
   emits routes to stores that do not exist, each rendering a blank viewer with
   HTTP 200. Routes are therefore emitted for *both* spellings.
2. **The README contract.** Tile titles in the root README link to these routes,
   so a key losing its route is a broken public link. ``--check-contract``
   fails the build naming any README-linked key without a route. Its source is
   ``media-manifest.json`` rather than the ``docs/images/readme/gallery/``
   basenames, because those files no longer exist — a guard reading them would
   pass on an empty set, which is worse than failing.

There is deliberately no ``/d/*`` catch-all: an unknown key should not quietly
land on the gallery, because that hides a typo behind a page that looks fine.

Usage::

    gen_redirects.py --prefix 2026-09-02 --live-stores stores.txt -o _redirects
    rclone lsf r2:luxar-demos/data/2026-09-02 --dirs-only | \\
        gen_redirects.py --prefix 2026-09-02 --live-stores - -o _redirects
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "scripts/gallery/manifest.json"
MEDIA_MANIFEST_PATH = REPO_ROOT / "scripts/gallery/media-manifest.json"
README_PATH = REPO_ROOT / "README.md"
DEFAULT_DATA_HOST = "https://data.luxarviewer.dev"
VIEWER_PATH = "/viewer/index.html"


class RouteError(RuntimeError):
    """Routes could not be generated, or would break a documented link."""


def store_of(entry: dict) -> str | None:
    """Store name for a manifest entry, from ``dataset`` — never from ``id``."""
    dataset = entry.get("dataset") or ""
    base = Path(dataset).name
    return base.removesuffix(".luxar.zarr") if base else None


def normalise_stores(names: Iterable[str]) -> set[str]:
    """Accept bare names, trailing slashes, or ``*.luxar.zarr`` spellings."""
    out: set[str] = set()
    for raw in names:
        name = raw.strip().rstrip("/")
        if not name:
            continue
        out.add(name.removesuffix(".luxar.zarr"))
    return out


def build_routes(
    demos: Sequence[dict],
    live_stores: set[str],
    prefix: str,
    data_host: str = DEFAULT_DATA_HOST,
) -> tuple[list[str], set[str], list[tuple[str, str]]]:
    """Return ``(lines, routed_keys, skipped)``.

    ``skipped`` holds ``(key, store)`` for manifest entries with no live store —
    reported rather than silently dropped, since that is how a missing tile hides.
    """
    lines: list[str] = []
    routed: set[str] = set()
    skipped: list[tuple[str, str]] = []

    for entry in sorted(demos, key=lambda d: d.get("id") or ""):
        key = entry.get("id")
        store = store_of(entry)
        if not key or not store:
            continue
        if store not in live_stores:
            skipped.append((key, store))
            continue
        target = f"{VIEWER_PATH}?src={data_host}/data/{prefix}/{store}.luxar.zarr"
        for route_key in (key, store) if store != key else (key,):
            if route_key in routed:
                raise RouteError(f"duplicate route path: /d/{route_key}")
            lines.append(f"/d/{route_key}  {target}  302")
            routed.add(route_key)
    return lines, routed, skipped


def readme_linked_keys(
    media_manifest: Path | None = None, readme: Path | None = None
) -> set[str]:
    """Keys linked at ``/d/`` by the README and its gallery tile manifest.

    Resolved at call time, not bound as a default argument: a module-level
    default would be captured at import and could never be redirected, which
    silently pins the contract check to one path.
    """
    media_path = media_manifest if media_manifest is not None else MEDIA_MANIFEST_PATH
    readme_path = readme if readme is not None else README_PATH
    keys: set[str] = set()
    found_source = False
    if media_path.exists():
        found_source = True
        keys.update(json.loads(media_path.read_text()).get("tiles", {}))
    if readme_path.exists():
        found_source = True
        keys.update(re.findall(r"/d/([\w-]+)", readme_path.read_text()))
    if not found_source:
        raise RouteError(
            f"README contract sources not found: {media_path} and {readme_path}"
        )
    if not keys:
        raise RouteError("README contract sources contain no /d/<demo-key> links")
    return keys


def render(lines: Sequence[str], prefix: str) -> str:
    header = [
        "# Stable per-demo routes. GENERATED by scripts/gallery/gen_redirects.py",
        "# at deploy time — do not hand-edit; it is rewritten with the current",
        "# data prefix on every publish.",
        f"# prefix: {prefix}",
        "#",
        "# /d/<demo-key> is an IDENTITY; the dated prefix is a LOCATION. Link to",
        "# the former from anything durable so the link survives the next deploy.",
        "",
    ]
    return "\n".join([*header, *lines]) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--prefix", required=True, help="dated data prefix, e.g. 2026-09-02"
    )
    parser.add_argument(
        "--live-stores",
        required=True,
        help="file listing stores live at --prefix, or '-' for stdin",
    )
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--data-host", default=DEFAULT_DATA_HOST)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--media-manifest", type=Path, default=MEDIA_MANIFEST_PATH)
    parser.add_argument("--readme", type=Path, default=README_PATH)
    parser.add_argument(
        "--check-contract",
        action="store_true",
        help="fail if a README-linked demo key has no route",
    )
    args = parser.parse_args(argv)

    text = (
        sys.stdin.read()
        if args.live_stores == "-"
        else Path(args.live_stores).read_text()
    )
    live = normalise_stores(text.splitlines())
    if not live:
        raise RouteError(
            "no live stores supplied; refusing to emit an empty route table"
        )

    demos = json.loads(args.manifest.read_text())["demos"]
    lines, routed, skipped = build_routes(demos, live, args.prefix, args.data_host)
    if not lines:
        raise RouteError("no routes generated; check --prefix and --live-stores")

    linked: set[str] = set()
    if args.check_contract:
        linked = readme_linked_keys(args.media_manifest, args.readme)
        missing = sorted(linked - routed)
        if missing:
            raise RouteError(
                "README links these demo keys but they have no route, so each would "
                "render a blank viewer with HTTP 200: " + ", ".join(missing)
            )

    args.output.write_text(render(lines, args.prefix))
    print(f"{len(lines)} routes for {len(routed)} keys -> {args.output}")
    if linked:
        print(f"README-linked keys checked: {len(linked)}")
    for key, store in skipped:
        print(f"  skipped (no live store): {key} -> {store}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except RouteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
