#!/usr/bin/env python3
"""Self-Contained Demo: CAIDA AS Topology — the Internet as a Graph.

Visualize the Internet's routing backbone: ~80k Autonomous Systems (ASes)
as Points and ~350k BGP relationships (provider-customer + peer) as Lines,
laid out in 3D from graph structure alone. The hierarchical core-periphery
structure of the Internet — tier-1 backbone in the middle, long tail of
stub networks at the edges — emerges naturally from a good graph layout.

================================================================================
WHAT IS THE AS GRAPH?
================================================================================

Every network connected to the Internet runs a Border Gateway Protocol (BGP)
daemon that announces which IP prefixes it can reach. The routing fabric
is made of ~80,000 such networks, called Autonomous Systems (ASes), and
the relationships between them. CAIDA infers these relationships from
observed BGP tables and publishes monthly snapshots. Two relationship
kinds matter here:

    Provider → Customer  (directed): one AS pays another to transport
        its traffic. Tier-1 backbones sit atop the hierarchy; everyone
        else is someone's customer.

    Peer ⇄ Peer  (symmetric): two ASes exchange traffic for free,
        usually at an Internet Exchange Point (IXP).

Directly observing this structure is one of the few ways to see the
Internet as a *thing* rather than a collection of websites. The famous
Opte project renderings are visualizations of this same graph.

DATASET AT A GLANCE
-------------------
- ~80k ASes (nodes) with org names + country codes from CAIDA as2org
- ~350k BGP relationships (edges), ~15% peers, ~85% provider-customer
- No native spatial coordinates — layout computed from graph structure
- Tier-1 detection: ASes with no upstream provider (~10 globally,
  matches the canonical list of "Internet backbone" networks)

LAYOUT
------
Three-stage pipeline tuned for aesthetics at this scale:
    1. Spectral embedding: top-50 smallest-eigenvalue eigenvectors of
       the normalized Laplacian → 50-D embedding where modular and
       hierarchical structure is preserved.
    2. UMAP 3D with graph-tuned parameters (n_neighbors=30, min_dist=0.05,
       metric='cosine', spread=2.0) — keeps clusters tight, disperses
       them enough for readability.
    3. PCA-realign so the longest axis is X — gives a consistent,
       pleasing orientation regardless of UMAP's random init.

COLOR / STRUCTURE
-----------------
Two runtime-switchable color views:

    - "Community" — Louvain communities on the undirected graph. These
      correlate roughly with regional carrier groups, industry verticals,
      and geographic neighborhoods.
    - "Country"   — ASN's country code from CAIDA as2org. The US,
      Brazil, Russia, China, and European carriers each form their own
      visually-distinct cloud.

Edges are colored and styled by RELATIONSHIP TYPE (the structural
dividing line on the Internet):

    - Provider → Customer  : warm amber, WIDTH-TAPERED (thick at
      provider, thin at customer) to make direction readable at a glance.
    - Peer ⇄ Peer         : cool cyan, uniform width — the symmetry
      mirrors the relationship.

Node radii scale with log(degree) and tier-1 ASes get an extra bump so
the backbone pops visually.

HOVER INFO
----------
- Node: AS number, org name, country, degree, tier-1 callout, community
- Edge: AS_A (org, country) ↔ AS_B (org, country), relationship kind
        with narrative ("Tier-1 peering" / "Tier-1 transit" / etc.)

DATA SOURCES
------------
- CAIDA AS Relationships (serial-2):
    https://publicdata.caida.org/datasets/as-relationships/serial-2/
- CAIDA AS→Organization mapping:
    https://publicdata.caida.org/datasets/as-organizations/
- AS-Rank portal:
    https://asrank.caida.org/

The first run downloads ~6 MB of compressed snapshots (tens of MB once
decompressed) and computes the layout (~2-3 min at this graph size).
Later runs make at most one discovery check a week, and none at all while
the memo is fresh — everything else is served from the cache directory
(``~/.cache/luxar/caida/`` by default; ``--cache-dir`` overrides it):

    - ``snapshots.json`` memoizes WHICH CAIDA snapshot pair is current, so
      the two directory-listing GETs that discover it are skipped for a
      week (``--refresh-snapshots`` re-checks immediately).
    - The two snapshot files themselves (``*.as-rel2.txt.bz2`` /
      ``*.as-org2info.txt.gz``).
    - ``pipeline_<rel>_<org>_v1.pkl`` — the whole deterministic derived
      chain (parse → largest connected component → tier-1 detection →
      Louvain communities → degrees), ~12 s of recompute otherwise
      (``--recompute-pipeline`` rebuilds it).
    - ``layout3d_<node-hash>_e<n-edges>_v1.pkl`` — the 3D coordinates.

Invalidation is by content, not by timestamp: the pipeline cache is keyed
on the two snapshot FILENAMES and the layout on the node list plus the
edge count it was computed from, so a new monthly CAIDA release downloads
once and recomputes both derived artifacts once, while an older snapshot's
caches stay valid rather than being clobbered. Superseded snapshots (raw
files plus their derived pipeline bundles) are pruned to the newest
``--keep-snapshots`` releases on every run.

Two things are deliberately never deleted. The layout pickles (~1 MB each)
have no release date to key them by, so they are not pruned — running this
demo monthly for years accumulates a few tens of MB, and
``luxar demo cache clear caida`` wipes the lot. And a ``layout_3d.npz``
left by an older version of this demo is left alone, because that filename
belongs to ``demo_huri_interactome``'s live layout cache and
``--cache-dir`` may well point both demos at one shared directory.

Usage:
    python -m luxar.demos.demo_caida_as_topology
    python -m luxar.demos.demo_caida_as_topology --no-serve
    python -m luxar.demos.demo_caida_as_topology --max-edges 80000
    python -m luxar.demos.demo_caida_as_topology --recompute-layout
    python -m luxar.demos.demo_caida_as_topology --recompute-pipeline
    python -m luxar.demos.demo_caida_as_topology --refresh-snapshots
    python -m luxar.demos.demo_caida_as_topology --keep-snapshots 4
"""

from __future__ import annotations

DEMO_META = {
    "key": "caida_as_topology",
    "title": "CAIDA AS Topology",
    "description": "The Internet's AS graph: ~80k autonomous systems and ~350k BGP links in 3D.",
    "category": "networks",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 6,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["caida"],
    "outputs": ["caida_as_topology"],
}

import bz2
import gzip
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import (
    cache_computed,
    launch_viewer,
    parse_int_arg,
    parse_path_arg,
    require_module,
)
from luxar.demos._graph_common import (
    build_adjacency,
    build_community_legend,
    compute_communities,
    download_file,
    nodes_hash,
)
from luxar.utils._umap_utils import build_legend_html, get_categorical_color
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "luxar" / "caida"

AS_REL_INDEX_URL = "https://publicdata.caida.org/datasets/as-relationships/serial-2/"
AS_ORG_INDEX_URL = "https://publicdata.caida.org/datasets/as-organizations/"

# The dated snapshot filenames, in one place: used to validate a memo entry, to
# recognise a cached snapshot file when pruning, and (wrapped in ``href="…"``) to
# scrape the directory listings.
AS_REL_NAME_PATTERN = re.compile(r"\d{8}\.as-rel2\.txt\.bz2")
AS_ORG_NAME_PATTERN = re.compile(r"\d{8}\.as-org2info\.txt\.gz")
AS_REL_FILE_PATTERN = re.compile(rf'href="({AS_REL_NAME_PATTERN.pattern})"')
AS_ORG_FILE_PATTERN = re.compile(rf'href="({AS_ORG_NAME_PATTERN.pattern})"')

# The as-org2info section markers. CAIDA writes them with NO space after the
# colon ("# format:org_id|changed|org_name|country|source"); mirrored/older
# copies use "# format: org_id". Accept both — matching neither leaves every AS
# labelled "unknown"/"??", which is silent (the demo still renders) and was
# exactly the state of the "Country" color view before this was made tolerant.
ORG_SECTION_PATTERN = re.compile(r"format:\s*org_id\b")
AUT_SECTION_PATTERN = re.compile(r"format:\s*aut\b")

# Snapshot discovery memo: which snapshot pair was current, and when we checked.
# Discovery costs two directory-listing GETs and CAIDA publishes monthly, so
# re-checking once a week is ample — and it lets a warm run be fully offline.
SNAPSHOT_MEMO_FILENAME = "snapshots.json"
SNAPSHOT_DISCOVERY_TTL_SECONDS = 7 * 24 * 60 * 60

# Cache versions for the two derived artifacts (bump to invalidate).
PIPELINE_CACHE_VERSION = 1
LAYOUT_CACHE_VERSION = 1

# How many snapshot releases to keep on disk (raw files + derived bundles).
DEFAULT_KEEP_SNAPSHOTS = 2

# Tuned for aesthetic quality at 80k-node scale
N_SPECTRAL_DIMS = 50
UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.05
UMAP_SPREAD = 2.0
UMAP_METRIC = "cosine"

# Visual caps
DEFAULT_MAX_EDGES = 150_000

# Edge relationship colors
COLOR_PROVIDER_CUSTOMER: tuple[float, float, float] = (1.00, 0.70, 0.25)  # warm amber
COLOR_PEER: tuple[float, float, float] = (0.30, 0.85, 1.00)  # cool cyan


# -----------------------------------------------------------------------------
# Download / cache
# -----------------------------------------------------------------------------


def _find_latest_file(index_url: str, pattern: re.Pattern) -> str:
    """Hit the CAIDA directory listing and return the newest matching filename."""
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    matches = pattern.findall(r.text)
    if not matches:
        raise RuntimeError(f"No files matching {pattern.pattern} found at {index_url}")
    return sorted(matches)[-1]


def _load_snapshot_memo(cache_dir: Path) -> dict | None:
    """Return the validated snapshot memo, or ``None`` if there isn't a usable one.

    A missing, unreadable, non-JSON or structurally-invalid memo reads as absent
    (discovery is then attempted) rather than raising — this is a cache hint, not
    data. Both filenames must match the dated snapshot patterns, so a
    hand-edited or truncated memo can never be turned into a bogus download URL,
    and ``checked_at`` must be a finite number (``json`` happily parses ``NaN`` /
    ``Infinity``, either of which would defeat the freshness comparison).
    """
    path = cache_dir / SNAPSHOT_MEMO_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    rel_name = raw.get("as_rel")
    org_name = raw.get("as_org")
    checked_at = raw.get("checked_at")
    if not isinstance(rel_name, str) or not AS_REL_NAME_PATTERN.fullmatch(rel_name):
        return None
    if not isinstance(org_name, str) or not AS_ORG_NAME_PATTERN.fullmatch(org_name):
        return None
    if isinstance(checked_at, bool) or not isinstance(checked_at, (int, float)):
        return None
    if not math.isfinite(checked_at):
        return None
    return {"as_rel": rel_name, "as_org": org_name, "checked_at": float(checked_at)}


def _snapshot_present(path: Path) -> bool:
    """True if ``path`` is a usable cached snapshot (present and non-empty).

    Matches :func:`download_file`'s own cache-hit rule, so the memo shortcut and
    the downloader agree on what "already have it" means.
    """
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:  # pragma: no cover - racing removal
        return False


def _newest_complete_local_pair(cache_dir: Path) -> tuple[str, str] | None:
    """Newest complete ``(as_rel, as_org)`` pair already on disk, or ``None``.

    The two series are published on different cadences, so the newest present
    file of each is chosen INDEPENDENTLY — exactly as a live discovery round
    would. Used by the offline fallback when the memo names a file we do not
    actually have (an interrupted first run leaves that state, since the memo is
    written before the downloads).
    """

    def newest(pattern: re.Pattern) -> str | None:
        names = sorted(
            p.name
            for p in cache_dir.glob("*")
            if pattern.fullmatch(p.name) and _snapshot_present(p)
        )
        return names[-1] if names else None

    rel_name = newest(AS_REL_NAME_PATTERN)
    org_name = newest(AS_ORG_NAME_PATTERN)
    if rel_name is None or org_name is None:
        return None
    return rel_name, org_name


def _save_snapshot_memo(
    cache_dir: Path, rel_name: str, org_name: str, now: float
) -> None:
    """Record the discovered snapshot pair, written atomically.

    Best-effort, like the pruner: a cache directory that is readable but not
    writable (someone else's, or read-only) still serves a perfectly good run, so
    a failed memo write warns instead of aborting after the downloads are done.
    """
    path = cache_dir / SNAPSHOT_MEMO_FILENAME
    payload = {"as_rel": rel_name, "as_org": org_name, "checked_at": float(now)}
    tmp = path.with_suffix(".json.tmp")
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        aprint(f"  ⚠ Could not write {SNAPSHOT_MEMO_FILENAME} ({exc})")
        aprint("    Discovery will run again next time; nothing else is affected.")


def _memo_shortcut_applies(memo: dict, cache_dir: Path, *, now: float) -> bool:
    """True when the memo may stand in for a fresh discovery round.

    Requires a memo that is neither stale nor stamped in the FUTURE (a skewed
    clock on a fresh VM, or a restored cache, would otherwise pin one snapshot
    pair for years), and — since skipping discovery is only a win if there is
    nothing left to fetch — both of its snapshot files already on disk. An
    interrupted first run leaves exactly that state: the memo is written before
    the downloads, so a missing file must fall through to discovery rather than
    hand a raw connection error back from :func:`download_file`.
    """
    age = now - memo["checked_at"]
    if not 0 <= age < SNAPSHOT_DISCOVERY_TTL_SECONDS:
        return False
    return _snapshot_present(cache_dir / memo["as_rel"]) and _snapshot_present(
        cache_dir / memo["as_org"]
    )


def ensure_data(
    cache_dir: Path, *, refresh: bool = False, now: float | None = None
) -> tuple[Path, Path]:
    """Fetch latest AS-rel + AS-org2info files. Returns local paths.

    Discovery (two directory-listing GETs) is memoized in
    ``snapshots.json`` for :data:`SNAPSHOT_DISCOVERY_TTL_SECONDS`, so a warm run
    makes no network request at all — :func:`download_file` is already a no-op
    for a file that is present. ``refresh=True`` forces a re-check; see
    :func:`_memo_shortcut_applies` for the rest of the conditions. When discovery
    fails (offline, HTTP error, or an index with no matching file) an existing
    memo — even a stale one — is used instead, WITHOUT restamping it, so the next
    run re-checks. If the memo is absent, or names a file we do not actually
    have, the newest complete pair on disk is used (see
    :func:`_newest_complete_local_pair`): what decides an offline run is whether
    the DATA is cached, not whether a memo happens to exist. The error propagates
    only when there is no complete pair at all — a genuine first run.

    ``now`` (seconds, defaults to :func:`time.time`) is injectable for tests.
    """
    if now is None:
        now = time.time()
    memo = _load_snapshot_memo(cache_dir)

    if (
        memo is not None
        and not refresh
        and _memo_shortcut_applies(memo, cache_dir, now=now)
    ):
        rel_name = memo["as_rel"]
        org_name = memo["as_org"]
        age_days = (now - memo["checked_at"]) / 86400.0
        with asection("Discovering latest CAIDA snapshots"):
            aprint(
                f"  Skipped (memoized {age_days:.1f} days ago; re-checked every "
                f"{SNAPSHOT_DISCOVERY_TTL_SECONDS / 86400:.0f} days, "
                f"or with --refresh-snapshots)"
            )
            aprint(f"  AS relationships: {rel_name}")
            aprint(f"  AS organizations: {org_name}")
    else:
        with asection("Discovering latest CAIDA snapshots"):
            try:
                rel_name = _find_latest_file(AS_REL_INDEX_URL, AS_REL_FILE_PATTERN)
                org_name = _find_latest_file(AS_ORG_INDEX_URL, AS_ORG_FILE_PATTERN)
            except (requests.RequestException, RuntimeError) as exc:
                aprint(f"  ⚠ Snapshot discovery failed: {exc}")
                memo_pair: tuple[str, str] | None = None
                if memo is not None:
                    candidate = (memo["as_rel"], memo["as_org"])
                    if _snapshot_present(
                        cache_dir / candidate[0]
                    ) and _snapshot_present(cache_dir / candidate[1]):
                        memo_pair = candidate

                if memo_pair is not None:
                    rel_name, org_name = memo_pair
                    aprint(
                        f"    Falling back to the memoized pair: "
                        f"{rel_name} / {org_name}"
                    )
                else:
                    # No memo, or one naming files we do not actually have (an
                    # interrupted first run leaves that). What decides the run is
                    # whether a COMPLETE pair is on disk, not whether a memo
                    # happens to exist — every user who ran this demo before the
                    # memo existed has the former and not the latter.
                    local_pair = _newest_complete_local_pair(cache_dir)
                    if local_pair is None:
                        aprint(
                            "    No complete snapshot pair in the cache to fall "
                            "back on — the first run of this demo needs network "
                            "access to CAIDA."
                        )
                        raise
                    rel_name, org_name = local_pair
                    aprint(
                        "    Falling back to the newest complete pair in the "
                        f"cache: {rel_name} / {org_name}"
                    )
            else:
                aprint(f"  AS relationships: {rel_name}")
                aprint(f"  AS organizations: {org_name}")
                _save_snapshot_memo(cache_dir, rel_name, org_name, now)

    rel_path = cache_dir / rel_name
    org_path = cache_dir / org_name

    with asection("Fetching CAIDA data"):
        download_file(AS_REL_INDEX_URL + rel_name, rel_path, f"{rel_name} (~2 MB)")
        download_file(AS_ORG_INDEX_URL + org_name, org_path, f"{org_name} (~4 MB)")

    return rel_path, org_path


def _prune_superseded_snapshots(
    cache_dir: Path, *, keep: int, current: tuple[str, str]
) -> None:
    """Delete all but the newest ``keep`` snapshot releases (plus ``current``).

    CAIDA publishes monthly, and nothing used to remove the superseded pairs, so
    the cache grew by ~6 MB of compressed snapshots plus their derived bundles
    every month, forever. Snapshot files are grouped by the 8-digit date in their
    name; the dates of the pair in use are always kept (the as-org2info release
    cadence is slower, so the current org file can be older than the newest one
    on disk), and each removed date takes its derived ``pipeline_*`` bundles —
    including a ``.pkl.corrupt`` / ``.pkl.tmp`` sibling ``cache_computed`` may
    have left — with it. Nothing else is touched: not the memo, not an
    unrecognised file, and deliberately not a ``layout3d_*`` cache. Those are
    small (~1 MB each) and keyed on the node SET rather than the release, so
    there is no date to prune them by, and an unchanged node set across releases
    reuses the same one.

    Deletions are best-effort: a file that vanishes underfoot is skipped rather
    than allowed to abort a run that has already paid for its download. There is
    a residual race left unaddressed (no locking): a run pinned to an older pair
    by the offline fallback can have its raw snapshot pruned by a concurrent run
    on the newest pair, in which case it simply re-downloads.
    """
    keep = max(1, keep)
    if not cache_dir.is_dir():
        return

    by_date: dict[str, list[Path]] = {}
    for path in sorted(cache_dir.iterdir()):
        if not path.is_file():
            continue
        if AS_REL_NAME_PATTERN.fullmatch(path.name) or AS_ORG_NAME_PATTERN.fullmatch(
            path.name
        ):
            by_date.setdefault(path.name[:8], []).append(path)
    if not by_date:
        return

    keep_dates = set(sorted(by_date, reverse=True)[:keep]) | {n[:8] for n in current}
    superseded = sorted(d for d in by_date if d not in keep_dates)

    collected: list[Path] = []
    for date in superseded:
        collected += by_date[date]
        # The derived bundle is keyed on both filenames, so a bundle naming a
        # removed date is unusable whatever its other date is.
        collected += sorted(
            p for p in cache_dir.glob(f"pipeline_*{date}*.pkl*") if p.is_file()
        )
    # A bundle whose two dates are BOTH superseded is collected once per date;
    # deleting it twice would report a spurious "Skipped … No such file".
    victims = list(dict.fromkeys(collected))
    if not victims:
        return

    with asection("Pruning superseded CAIDA snapshots"):
        freed = 0
        removed = 0
        for victim in victims:
            try:
                size = victim.stat().st_size
                victim.unlink()
            except OSError as exc:
                aprint(f"  Skipped {victim.name} ({exc})")
                continue
            freed += size
            removed += 1
            aprint(f"  Removed {victim.name} ({size / (1024 * 1024):.1f} MB)")
        aprint(
            f"  ✓ Reclaimed {freed / (1024 * 1024):.1f} MB from {removed} file(s) "
            f"({len(superseded)} superseded release(s))"
        )


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def parse_as_org(path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by ASN with ``org_name`` and ``country``.

    The CAIDA as-org2info file has two sections separated by format-comment
    headers, written with no space after the colon:
        # format:org_id|changed|org_name|country|source        (organizations)
        # format:aut|changed|aut_name|org_id|opaque_id|source  (ASes → org_id)
    We stream once, tracking which section we're in, and join. The markers are
    matched whitespace-tolerantly (:data:`ORG_SECTION_PATTERN` /
    :data:`AUT_SECTION_PATTERN`): a marker that fails to match is silent — every
    row is skipped and every AS comes out ``unknown``/``??`` — so it must not
    hinge on one spelling of the separator.
    """
    org_info: dict[str, tuple[str, str]] = {}
    asn_to_org: dict[str, str] = {}
    mode: str | None = None

    with asection("Parsing AS → organization mapping"):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("#"):
                    if ORG_SECTION_PATTERN.search(line):
                        mode = "org"
                    elif AUT_SECTION_PATTERN.search(line):
                        mode = "aut"
                    continue
                if not line.strip():
                    continue
                parts = line.split("|")
                if mode == "org" and len(parts) >= 4:
                    org_id, _changed, name, country = parts[:4]
                    org_info[org_id] = (name.strip(), country.strip() or "??")
                elif mode == "aut" and len(parts) >= 4:
                    asn, _changed, _aut_name, org_id = parts[:4]
                    asn_to_org[asn.strip()] = org_id.strip()

        rows = []
        for asn, org_id in asn_to_org.items():
            name, country = org_info.get(org_id, ("unknown", "??"))
            rows.append((asn, name, country))
        df = pd.DataFrame(rows, columns=["asn", "org_name", "country"])
        df = df.set_index("asn")
        aprint(
            f"  {len(org_info):,} orgs, {len(asn_to_org):,} ASes annotated "
            f"({df['country'].nunique():,} distinct countries)"
        )
    return df


def parse_as_rel(path: Path) -> pd.DataFrame:
    """Parse as-rel.txt.bz2 into a DataFrame [asn_a, asn_b, rel].

    ``rel`` is -1 for provider-customer (A is provider of B) or 0 for peers.
    Self-loops and malformed rows are silently dropped.
    """
    with asection("Parsing AS relationships"):
        opener = bz2.open if path.suffix == ".bz2" else open
        rows: list[tuple[str, str, int]] = []
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 3:
                    continue
                try:
                    a = parts[0].strip()
                    b = parts[1].strip()
                    rel = int(parts[2])
                except ValueError:
                    continue
                if not a or not b or a == b:
                    continue
                if rel not in (-1, 0):
                    continue
                rows.append((a, b, rel))
        df = pd.DataFrame(rows, columns=["asn_a", "asn_b", "rel"])
        # Deduplicate in case the file lists both orientations of a peer
        peer_mask = df["rel"] == 0
        peer_df = df[peer_mask].copy()
        peer_a = peer_df["asn_a"].to_numpy()
        peer_b = peer_df["asn_b"].to_numpy()
        lo = np.where(peer_a < peer_b, peer_a, peer_b)
        hi = np.where(peer_a < peer_b, peer_b, peer_a)
        peer_df["asn_a"] = lo
        peer_df["asn_b"] = hi
        peer_df = peer_df.drop_duplicates(subset=["asn_a", "asn_b"])
        pc_df = df[~peer_mask].drop_duplicates(subset=["asn_a", "asn_b"])
        df = pd.concat([pc_df, peer_df], ignore_index=True)

        n_pc = int((df["rel"] == -1).sum())
        n_peer = int((df["rel"] == 0).sum())
        aprint(
            f"  {len(df):,} unique relationships "
            f"({n_pc:,} provider-customer, {n_peer:,} peer)"
        )
    return df


def filter_to_lcc(
    edges: pd.DataFrame, org_df: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Restrict to the LCC. Attach org+country. Detect tier-1s.

    Returns:
        nodes: sorted ASN list
        node_df: [asn, org_name, country, tier1] indexed positionally
        edges: relationships within the LCC (unchanged direction for rel=-1)
        tier1: boolean array aligned with nodes
    """
    nx = require_module("networkx")

    with asection("Filtering to LCC + detecting tier-1s"):
        g = nx.Graph()
        g.add_edges_from(
            zip(edges["asn_a"].tolist(), edges["asn_b"].tolist(), strict=True)
        )
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        lcc = components[0]
        aprint(
            f"  {len(components):,} components; LCC has {len(lcc):,} ASes "
            f"(dropped {g.number_of_nodes() - len(lcc):,})"
        )

        lcc_set = set(lcc)
        mask = edges["asn_a"].isin(lcc_set) & edges["asn_b"].isin(lcc_set)
        edges = edges[mask].reset_index(drop=True)
        aprint(f"  {len(edges):,} relationships within the LCC")

        # Tier-1 = AS that is never a customer in any provider-customer edge
        pc = edges[edges["rel"] == -1]
        has_upstream = set(pc["asn_b"].tolist())
        tier1_set = lcc_set - has_upstream
        aprint(f"  {len(tier1_set):,} tier-1 ASes (no upstream providers)")

        # Build nodes in sorted ASN order (numeric-safe sort key)
        def _asn_sort_key(a: str) -> tuple[int, str]:
            try:
                return (0, f"{int(a):010d}")
            except ValueError:
                return (1, a)

        nodes = sorted(lcc_set, key=_asn_sort_key)
        asn_idx = {a: i for i, a in enumerate(nodes)}
        tier1 = np.zeros(len(nodes), dtype=bool)
        for a in tier1_set:
            tier1[asn_idx[a]] = True

        # Vectorized org/country lookup
        org_series = org_df["org_name"].reindex(nodes).fillna("unknown")
        country_series = org_df["country"].reindex(nodes).fillna("??")
        node_df = pd.DataFrame(
            {
                "asn": nodes,
                "org_name": org_series.to_numpy(),
                "country": country_series.to_numpy(),
                "tier1": tier1,
            }
        )
    return nodes, node_df, edges, tier1


# -----------------------------------------------------------------------------
# Derived pipeline bundle (parse → LCC/tier-1 → Louvain → degrees), cached
# -----------------------------------------------------------------------------


def _build_pipeline_bundle(rel_path: Path, org_path: Path) -> dict[str, np.ndarray]:
    """Run the whole deterministic derived chain into a bundle of arrays.

    Everything here is a pure function of the two snapshot files, and together it
    costs ~12 s on the real data (~3 s parse + LCC, ~9 s Louvain). Arrays rather
    than pickled DataFrames: the frames are rebuilt in :func:`load_pipeline`, so
    the cache holds only the columns the demo actually consumes and does not
    depend on the pandas version that wrote it. The string columns are
    ``dtype=object``, NOT a fixed-width unicode dtype: real org names run to
    ~200 characters, and ``<U200`` pads EVERY one of ~80k rows out to that width
    at 4 bytes a character — ~65 MB for a column that is a few MB as objects, in
    the very cache directory this bundle exists to keep small.
    """
    org_df = parse_as_org(org_path)
    edges_raw = parse_as_rel(rel_path)
    nodes, node_df, edges, tier1 = filter_to_lcc(edges_raw, org_df)
    communities = compute_communities(
        nodes,
        edges,
        columns=("asn_a", "asn_b"),
        unit_label="ASes",
        summary_suffix="",
    )
    degrees = _node_degrees(nodes, edges)

    idx = {a: i for i, a in enumerate(nodes)}
    n_edges = len(edges)
    return {
        # ``str(...)`` is defensive: whatever pandas hands back for these columns
        # (plain ``str``, or a numpy scalar subclass), the bundle stores plain
        # ``str``, which is what the demo's dict lookups and labels expect.
        "nodes": np.array([str(a) for a in nodes], dtype=object),
        "org_name": np.array(
            [str(x) for x in node_df["org_name"].tolist()], dtype=object
        ),
        "country": np.array(
            [str(x) for x in node_df["country"].tolist()], dtype=object
        ),
        "tier1": tier1.astype(bool),
        "edge_a": np.fromiter(
            (idx[a] for a in edges["asn_a"]), dtype=np.int32, count=n_edges
        ),
        "edge_b": np.fromiter(
            (idx[b] for b in edges["asn_b"]), dtype=np.int32, count=n_edges
        ),
        "rel": edges["rel"].to_numpy(dtype=np.int8),
        "communities": communities.astype(np.int32),
        "degrees": degrees.astype(np.int32),
    }


def load_pipeline(
    rel_path: Path,
    org_path: Path,
    cache_dir: Path,
    *,
    recompute: bool = False,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Cached ``(nodes, node_df, edges, tier1, communities, degrees)``.

    Keyed on the two snapshot FILENAMES, which is what makes invalidation
    correct: a new monthly release gets a new key (one recompute) while the
    previous release's bundle stays valid under its own key instead of being
    clobbered. ``version`` covers changes to the bundle layout itself.
    """
    with asection("AS graph (parse → LCC → communities)"):
        bundle = cache_computed(
            "caida",
            key=f"pipeline_{rel_path.name}_{org_path.name}",
            compute_fn=lambda: _build_pipeline_bundle(rel_path, org_path),
            cache_dir=cache_dir,
            recompute=recompute,
            version=PIPELINE_CACHE_VERSION,
        )

        # ``tolist()`` on the object arrays yields the plain Python ``str`` the
        # downstream ``idx[a]`` lookups, ``value_counts``, ``isin`` and hover
        # label formatting were written against.
        node_arr = bundle["nodes"]
        nodes: list[str] = node_arr.tolist()
        tier1 = bundle["tier1"].astype(bool)
        node_df = pd.DataFrame(
            {
                "asn": nodes,
                "org_name": bundle["org_name"].tolist(),
                "country": bundle["country"].tolist(),
                "tier1": tier1,
            }
        )
        edges = pd.DataFrame(
            {
                "asn_a": node_arr[bundle["edge_a"]].tolist(),
                "asn_b": node_arr[bundle["edge_b"]].tolist(),
                # int (not int8): the callers compare against -1 / 0 and build
                # masks from ``edges["rel"].to_numpy()``.
                "rel": bundle["rel"].astype(np.int64),
            }
        )
        communities = bundle["communities"].astype(np.int32)
        degrees = bundle["degrees"].astype(np.int32)
        # A fully-warm run does none of the work above, so print the shape of
        # what was loaded — otherwise the run says nothing about its own data.
        aprint(
            f"  {len(nodes):,} ASes · {len(edges):,} relationships · "
            f"{int(communities.max()) + 1 if len(communities) else 0} communities · "
            f"{int(tier1.sum())} tier-1"
        )
    return nodes, node_df, edges, tier1, communities, degrees


# -----------------------------------------------------------------------------
# Layout: spectral + UMAP (tuned) + PCA realign
# -----------------------------------------------------------------------------


def _pca_realign(coords: np.ndarray) -> np.ndarray:
    """Rotate coords so the largest-variance axis is X, then Y, then Z."""
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]  # descending
    rotated = centered @ eigvecs[:, order]
    # Enforce a deterministic sign (flip any axis whose skew is negative)
    for i in range(3):
        if rotated[:, i].mean() > 0 and np.median(rotated[:, i]) < 0:
            rotated[:, i] *= -1
    return rotated.astype(np.float32)


def compute_layout(
    nodes: list[str], edges: pd.DataFrame, cache_dir: Path, recompute: bool
) -> np.ndarray:
    """Spectral embedding + UMAP + PCA-realign → 3D coords, cached.

    The cache key IS the input identity — a hash of the node list plus the EDGE
    COUNT — so a changed graph lands in its own file instead of invalidating, and
    no separate hash guard is needed inside the loader. The edge count matters
    because the layout is computed from the adjacency, not from the nodes alone: a
    release whose LCC happens to have the same node set but different
    relationships must not silently reuse last month's geometry. It is a count
    rather than a full edge hash deliberately — hashing ~350k pairs on every warm
    run to catch a rewiring that preserves the edge count exactly is not worth it.
    """
    with asection("3D layout"):
        coords = cache_computed(
            "caida",
            key=f"layout3d_{nodes_hash(nodes)}_e{len(edges)}",
            compute_fn=lambda: _compute_layout_coords(nodes, edges),
            cache_dir=cache_dir,
            recompute=recompute,
            version=LAYOUT_CACHE_VERSION,
        )
        aprint(f"  {len(coords):,} coords")
    return coords


def _compute_layout_coords(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    """The expensive layout itself (spectral eigenvectors + UMAP + PCA realign)."""
    from scipy.sparse import eye as speye
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    with asection("Computing spectral + UMAP layout"):
        adj = build_adjacency(nodes, edges, columns=("asn_a", "asn_b"))
        aprint(f"  Adjacency: {adj.shape[0]:,} × {adj.shape[0]:,}, nnz={adj.nnz:,}")

        lap = laplacian(adj, normed=True).astype(np.float64)
        # Solve top-k LARGEST eigenvalues of (I - L), i.e. smallest of L
        shifted = speye(lap.shape[0], format="csr") - lap
        k = min(N_SPECTRAL_DIMS + 1, adj.shape[0] - 2)
        aprint(f"  Spectral: {k} eigenvectors (this is the slow step)")
        eigvals_shift, eigvecs = eigsh(shifted, k=k, which="LA")

        order = np.argsort(eigvals_shift)[::-1]
        eigvecs = eigvecs[:, order]
        features = eigvecs[:, 1:].astype(np.float32)
        # Row-normalize for cosine-distance UMAP
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / np.maximum(norms, 1e-10)
        aprint(f"  Spectral features: {features.shape}")

        UMAP = require_module("umap").UMAP

        aprint(
            f"  UMAP → 3D (n_neighbors={UMAP_N_NEIGHBORS}, "
            f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC}, "
            f"spread={UMAP_SPREAD})"
        )
        reducer = UMAP(
            n_components=3,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            spread=UMAP_SPREAD,
            metric=UMAP_METRIC,
            n_jobs=-1,
            verbose=False,
        )
        coords = reducer.fit_transform(features).astype(np.float32)

        coords = _pca_realign(coords)
        radius_95 = float(np.percentile(np.linalg.norm(coords, axis=1), 95))
        if radius_95 > 0:
            coords *= 10.0 / radius_95
        aprint(
            f"  Coords: x range {np.ptp(coords[:, 0]):.1f}, "
            f"y {np.ptp(coords[:, 1]):.1f}, z {np.ptp(coords[:, 2]):.1f}"
        )
    return coords


# -----------------------------------------------------------------------------
# Geometry
# -----------------------------------------------------------------------------


def _categorical_palette(n: int) -> np.ndarray:
    """(n, 3) float32 RGB in [0, 1]."""
    return (
        np.array(
            [get_categorical_color(i, n) for i in range(n)],
            dtype=np.float32,
        )
        / 255.0
    )


def build_node_points(
    nodes: list[str],
    node_df: pd.DataFrame,
    coords: np.ndarray,
    communities: np.ndarray,
    degrees: np.ndarray,
    tier1: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Assemble Points with two color views duplicated.

    Returns:
        positions (2N, 4), colors (2N, 3), radii (2N,), labels (2N,),
        country_cats, country_codes, community_palette, country_palette,
        per-node radii (N,) for reuse by edge geometry
    """
    n = len(nodes)

    # Community colors
    n_comms = int(communities.max()) + 1
    comm_palette = _categorical_palette(n_comms)
    comm_colors = comm_palette[communities]

    # Country colors — stable sort (top countries by count first, "??" last)
    country_vals = node_df["country"].tolist()
    country_counts = pd.Series(country_vals).value_counts()
    country_cats = [c for c in country_counts.index if c != "??"] + (
        ["??"] if "??" in country_counts.index else []
    )
    country_idx = {c: i for i, c in enumerate(country_cats)}
    country_codes = np.array([country_idx[c] for c in country_vals], dtype=np.int32)
    country_palette = _categorical_palette(len(country_cats))
    country_colors = country_palette[country_codes]

    # Radii: log(degree) with a tier-1 boost (1.7×) so the backbone pops
    log_deg = np.log1p(degrees.astype(np.float32))
    log_deg /= max(float(log_deg.max()), 1.0)
    base = 0.02 + 0.15 * log_deg
    base[tier1] *= 1.7
    radii_per_node = base.astype(np.float32)

    # Positions: two views stacked along attribute axis 0
    pos_view0 = np.empty((n, 4), dtype=np.float32)
    pos_view0[:, 0] = 0.0
    pos_view0[:, 1:] = coords
    pos_view1 = pos_view0.copy()
    pos_view1[:, 0] = 1.0
    positions = np.vstack([pos_view0, pos_view1])

    colors = np.vstack([comm_colors, country_colors])
    radii = np.concatenate([radii_per_node, radii_per_node])

    # Hover labels — one per node, replicated per view
    labels_per_node: list[str] = []
    asns = node_df["asn"].tolist()
    orgs = node_df["org_name"].tolist()
    countries = country_vals
    for i in range(n):
        tier_suffix = " · TIER-1" if tier1[i] else ""
        labels_per_node.append(
            f"AS{asns[i]} · {orgs[i]}\n"
            f"[{countries[i]} · deg {int(degrees[i])}{tier_suffix} · "
            f"community {int(communities[i])}]"
        )
    labels = labels_per_node * 2

    return (
        positions,
        colors,
        radii,
        labels,
        country_cats,
        country_codes,
        comm_palette,
        country_palette,
        radii_per_node,
    )


def build_edge_lines(
    nodes: list[str],
    edges: pd.DataFrame,
    coords: np.ndarray,
    communities: np.ndarray,
    tier1: np.ndarray,
    node_df: pd.DataFrame,
    max_edges: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Lines geometry duplicated across the two views.

    Provider→customer edges are width-tapered (thick at provider, thin at
    customer) and colored warm amber.  Peer edges are uniform width and
    cool cyan.  Both are rendered identically on both color views.

    Subsampling preference (if we exceed max_edges): keep *all* peer edges
    (there are fewer of them and they're the most interesting), then fill
    with a random sample of provider-customer edges.
    """
    idx = {s: i for i, s in enumerate(nodes)}
    a_idx = np.fromiter((idx[a] for a in edges["asn_a"]), dtype=np.int32)
    b_idx = np.fromiter((idx[b] for b in edges["asn_b"]), dtype=np.int32)
    rel = edges["rel"].to_numpy()

    if len(edges) > max_edges:
        with asection("Subsampling edges (60/40 PC/peer for visual balance)"):
            peer_mask = rel == 0
            pc_mask = rel == -1
            n_peer = int(peer_mask.sum())
            n_pc = int(pc_mask.sum())
            rng = np.random.default_rng(42)

            # Target a 60/40 provider-customer/peer split so the hierarchical
            # structure reads clearly. Fall back proportionally if one side
            # can't fill its quota.
            target_pc = int(max_edges * 0.6)
            target_peer = max_edges - target_pc
            keep_pc = min(n_pc, target_pc)
            keep_peer = min(n_peer, target_peer)
            slack = max_edges - keep_pc - keep_peer
            if slack > 0:
                keep_pc += min(slack, n_pc - keep_pc)
                slack = max_edges - keep_pc - keep_peer
                keep_peer += min(slack, n_peer - keep_peer)

            keep = np.zeros(len(edges), dtype=bool)
            pc_positions = np.nonzero(pc_mask)[0]
            peer_positions = np.nonzero(peer_mask)[0]
            if keep_pc > 0:
                chosen_pc = rng.choice(pc_positions, size=keep_pc, replace=False)
                keep[chosen_pc] = True
            if keep_peer > 0:
                chosen_peer = rng.choice(peer_positions, size=keep_peer, replace=False)
                keep[chosen_peer] = True

            edges = edges.loc[keep].reset_index(drop=True)
            a_idx = np.fromiter((idx[a] for a in edges["asn_a"]), dtype=np.int32)
            b_idx = np.fromiter((idx[b] for b in edges["asn_b"]), dtype=np.int32)
            rel = edges["rel"].to_numpy()
            aprint(
                f"  Kept {len(edges):,} edges: "
                f"{int((rel == -1).sum()):,} provider-customer + "
                f"{int((rel == 0).sum()):,} peer "
                f"(from {n_pc:,} PC + {n_peer:,} peer available)"
            )

    n_edges = len(edges)
    xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    # Vertices per view
    verts_xyz = np.empty((n_edges, 2, 3), dtype=np.float32)
    verts_xyz[:, 0, 0] = xs[a_idx]
    verts_xyz[:, 0, 1] = ys[a_idx]
    verts_xyz[:, 0, 2] = zs[a_idx]
    verts_xyz[:, 1, 0] = xs[b_idx]
    verts_xyz[:, 1, 1] = ys[b_idx]
    verts_xyz[:, 1, 2] = zs[b_idx]
    flat_xyz = verts_xyz.reshape(n_edges * 2, 3)
    view0 = np.zeros((n_edges * 2, 1), dtype=np.float32)
    view1 = np.ones((n_edges * 2, 1), dtype=np.float32)
    vertices = np.vstack([np.hstack([view0, flat_xyz]), np.hstack([view1, flat_xyz])])

    # Widths: tapered for provider→customer (A thick, B thin), uniform for peer
    #   Arrays are laid out as [a0, b0, a1, b1, ...] so index 0::2 is "A end"
    #   and 1::2 is "B end".
    widths_1view = np.empty(n_edges * 2, dtype=np.float32)
    is_pc = rel == -1
    widths_1view[0::2] = np.where(is_pc, 0.020, 0.012).astype(np.float32)
    widths_1view[1::2] = np.where(is_pc, 0.005, 0.012).astype(np.float32)
    widths = np.concatenate([widths_1view, widths_1view])

    # Colors: warm amber for PC, cool cyan for peer — equal on both endpoints
    color_per_edge = np.where(
        is_pc[:, None],
        np.array(COLOR_PROVIDER_CUSTOMER, dtype=np.float32)[None, :],
        np.array(COLOR_PEER, dtype=np.float32)[None, :],
    ).astype(np.float32)
    color_per_vertex_1view = np.repeat(color_per_edge, 2, axis=0)
    colors = np.vstack([color_per_vertex_1view, color_per_vertex_1view])

    # Labels with relationship narrative
    asns = node_df["asn"].tolist()
    orgs = node_df["org_name"].tolist()
    countries = node_df["country"].tolist()
    labels_per_edge: list[str] = []
    for i in range(n_edges):
        ia, ib = int(a_idx[i]), int(b_idx[i])
        sa = f"AS{asns[ia]} · {orgs[ia]} ({countries[ia]})"
        sb = f"AS{asns[ib]} · {orgs[ib]} ({countries[ib]})"
        if is_pc[i]:
            if tier1[ia] and tier1[ib]:
                # Shouldn't happen by definition of tier-1, but guard anyway
                narrative = "[Tier-1 transit — unusual]"
            elif tier1[ia]:
                narrative = "[Tier-1 transit]"
            else:
                narrative = "[Provider → Customer]"
            label = f"{sa}\n   ↓\n{sb}\n{narrative}"
        else:
            if tier1[ia] and tier1[ib]:
                narrative = "[Tier-1 peering]"
            elif tier1[ia] or tier1[ib]:
                narrative = "[Peering with tier-1]"
            else:
                narrative = "[Peering]"
            label = f"{sa}\n   ⇄\n{sb}\n{narrative}"
        labels_per_edge.append(label)

    labels_per_vertex_1view: list[str] = []
    for lab in labels_per_edge:
        labels_per_vertex_1view.extend([lab, lab])
    labels = labels_per_vertex_1view * 2

    return edges, vertices, widths, colors, labels


# -----------------------------------------------------------------------------
# Legends
# -----------------------------------------------------------------------------


def _build_edge_legend() -> str:
    pc_r, pc_g, pc_b = [int(x * 255) for x in COLOR_PROVIDER_CUSTOMER]
    p_r, p_g, p_b = [int(x * 255) for x in COLOR_PEER]
    return (
        '<div style="font-size:1.3vh;line-height:1.5;'
        "background:rgba(0,0,0,0.55);padding:0.6vh 0.8vh;"
        'border-radius:4px">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.4vh">'
        "Edges</div>"
        f'<div style="white-space:nowrap">'
        f'<span style="color:rgb({pc_r},{pc_g},{pc_b})">━▶</span> '
        "Provider → Customer</div>"
        f'<div style="white-space:nowrap">'
        f'<span style="color:rgb({p_r},{p_g},{p_b})">⇄</span> Peer</div>'
        "</div>"
    )


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


def build_scene(
    output_path: Path,
    nodes: list[str],
    node_df: pd.DataFrame,
    edges: pd.DataFrame,
    coords: np.ndarray,
    communities: np.ndarray,
    degrees: np.ndarray,
    tier1: np.ndarray,
    max_edges: int,
) -> tuple[int, int, int, int]:
    """Build and write the Points + Lines scene to ``output_path``.

    Returns the four integer counts ``(n_nodes, n_edges_kept, n_comms,
    n_tier1)``, where ``n_edges_kept`` is the edge count after the
    ``max_edges`` cap.
    """
    (
        node_positions,
        node_colors,
        node_radii,
        node_labels,
        country_cats,
        country_codes,
        _comm_palette,
        _country_palette,
        _radii_per_node,
    ) = build_node_points(nodes, node_df, coords, communities, degrees, tier1)

    edges_kept, verts, widths, edge_colors, edge_labels = build_edge_lines(
        nodes,
        edges,
        coords,
        communities,
        tier1,
        node_df,
        max_edges=max_edges,
    )

    n_nodes = len(nodes)
    n_edges_kept = len(edges_kept)
    n_comms = int(communities.max()) + 1
    n_tier1 = int(tier1.sum())

    with asection("Building Luxar scene"):
        dims = Dimensions(
            [
                Dimension(
                    "view",
                    unit="",
                    categories=["Community", "Country"],
                    display=False,
                    description="Color coding (press '1' then '[' / ']' to switch)",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "Autonomous Systems",
                positions=node_positions,
                colors=node_colors,
                radii=node_radii,
                sharpness=np.full(len(node_positions), 0.55, dtype=np.float32),
                opacity=0.95,
                intensity=0.2,
                labels=node_labels,
                layer=True,
            )

            if len(verts) > 0:
                scene.add_lines(
                    "Relationships",
                    vertices=verts,
                    widths=widths,
                    colors=edge_colors,
                    sharpness=np.full(len(verts), 0.85, dtype=np.float32),
                    line_type="segments",
                    blending_mode="luminous",
                    opacity=0.05,
                    intensity=0.55,
                    labels=edge_labels,
                    layer=True,
                )

            # Title
            scene.add_text(
                "CAIDA AS Topology — The Internet as a Graph",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )

            # Hover overlay (center-left)
            scene.add_text(
                "{hover_label}",
                position=(0.02, 0.5),
                anchor="center-left",
                font_size=0.022,
                color="white",
                background="rgba(0,0,0,0.72)",
                padding=0.01,
                text_align="left",
                opacity=1.0,
                transition="fade",
                transition_duration=0.15,
                hover=True,
            )

            # Edge legend (top-right, always visible — explains line colors)
            scene.add_html(
                _build_edge_legend(),
                position=(0.98, 0.02),
                anchor="top-right",
                opacity=0.92,
            )

            # Per-view captions + node legends
            community_legend = build_community_legend(communities)
            country_legend = build_legend_html("country", country_cats, country_codes)

            for view_id, caption, legend_html in [
                (0, "Colored by: Community (Louvain)", community_legend),
                (1, "Colored by: Country (CAIDA as2org)", country_legend),
            ]:
                scene.add_text(
                    caption,
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"view": view_id},
                    transition="fade",
                    transition_duration=0.2,
                )
                if legend_html:
                    scene.add_html(
                        legend_html,
                        position=(0.98, 0.5),
                        anchor="center-right",
                        opacity=0.92,
                        visible_range={"view": view_id},
                        transition="fade",
                        transition_duration=0.2,
                    )

            scene.add_text(
                (
                    f"{n_nodes:,} ASes · {n_edges_kept:,} relationships · "
                    f"{n_tier1} tier-1 · {n_comms} communities · CAIDA serial-2"
                ),
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )

        aprint(
            f"  ✓ Scene: {n_nodes:,} nodes, {n_edges_kept:,} edges, "
            f"{n_comms} communities, {n_tier1} tier-1"
        )

    return n_nodes, n_edges_kept, n_comms, n_tier1


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def _node_degrees(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    """Undirected degree per node (aligned with ``nodes``, missing → 0)."""
    counts = pd.concat([edges["asn_a"], edges["asn_b"]]).value_counts()
    return counts.reindex(nodes, fill_value=0).to_numpy(dtype=np.int32)


def main() -> None:
    """Fetch CAIDA data, compute the layout, and build/serve the scene."""
    argv = sys.argv[1:]
    max_edges = parse_int_arg("max-edges", DEFAULT_MAX_EDGES, argv)
    keep_snapshots = parse_int_arg("keep-snapshots", DEFAULT_KEEP_SNAPSHOTS, argv)
    cache_dir = parse_path_arg("cache-dir", argv) or CACHE_DIR
    recompute = "--recompute-layout" in argv
    recompute_pipeline = "--recompute-pipeline" in argv
    refresh_snapshots = "--refresh-snapshots" in argv

    aprint("=" * 70)
    aprint("CAIDA AS Topology — The Internet as a Graph")
    aprint("=" * 70)
    aprint("~80k ASes · ~350k BGP relationships · CAIDA serial-2")
    aprint(f"Edge cap: {max_edges:,}")
    aprint("")

    rel_path, org_path = ensure_data(cache_dir, refresh=refresh_snapshots)
    _prune_superseded_snapshots(
        cache_dir, keep=keep_snapshots, current=(rel_path.name, org_path.name)
    )
    nodes, node_df, edges, tier1, communities, degrees = load_pipeline(
        rel_path, org_path, cache_dir, recompute=recompute_pipeline
    )
    coords = compute_layout(
        nodes,
        edges,
        cache_dir=cache_dir,
        recompute=recompute,
    )

    if "--no-serve" in argv:
        output_path = get_demos_output_dir() / "caida_as_topology.luxar.zarr"
        build_scene(
            output_path,
            nodes,
            node_df,
            edges,
            coords,
            communities,
            degrees,
            tier1,
            max_edges,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_caida_") as tmpdir:
        output_path = Path(tmpdir) / "caida_as_topology.luxar.zarr"
        n_nodes, n_edges, n_comms, n_tier1 = build_scene(
            output_path,
            nodes,
            node_df,
            edges,
            coords,
            communities,
            degrees,
            tier1,
            max_edges,
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("  Press '1' then '[' / ']' to switch color view:")
        aprint("     0: Community (Louvain modules)")
        aprint("     1: Country   (from CAIDA as2org)")
        aprint("  Hover any node : ASN · org · country · tier · community")
        aprint("  Hover any edge : pair · relationship narrative")
        aprint("")
        aprint("  Line colors:  ▶ amber = Provider→Customer (tapered)")
        aprint("                ⇄ cyan  = Peer (uniform)")
        aprint("")
        aprint(
            f"  ASes: {n_nodes:,}   Relationships: {n_edges:,}   "
            f"Communities: {n_comms}   Tier-1: {n_tier1}"
        )
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
