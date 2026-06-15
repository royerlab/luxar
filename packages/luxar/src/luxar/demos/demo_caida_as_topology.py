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

The first run downloads ~60 MB total and computes the layout (~2-3 min
at this graph size). Subsequent runs reuse the cache at
``~/.cache/luxar/caida/`` — including the cached 3D coordinates.

Usage:
    python -m luxar.demos.demo_caida_as_topology
    python -m luxar.demos.demo_caida_as_topology --no-serve
    python -m luxar.demos.demo_caida_as_topology --max-edges 80000
    python -m luxar.demos.demo_caida_as_topology --recompute-layout
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import build_legend_html, get_categorical_color
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "luxar" / "caida"

AS_REL_INDEX_URL = "https://publicdata.caida.org/datasets/as-relationships/serial-2/"
AS_ORG_INDEX_URL = "https://publicdata.caida.org/datasets/as-organizations/"
AS_REL_FILE_PATTERN = re.compile(r'href="(\d{8}\.as-rel2\.txt\.bz2)"')
AS_ORG_FILE_PATTERN = re.compile(r'href="(\d{8}\.as-org2info\.txt\.gz)"')

LAYOUT_CACHE_FILENAME = "layout_3d.npz"

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


def _find_latest_file(index_url: str, pattern: re.Pattern) -> tuple[str, str]:
    """Hit the CAIDA directory listing and return (filename, absolute_url)."""
    r = requests.get(index_url, timeout=30)
    r.raise_for_status()
    matches = pattern.findall(r.text)
    if not matches:
        raise RuntimeError(f"No files matching {pattern.pattern} found at {index_url}")
    latest = sorted(matches)[-1]
    return latest, index_url + latest


def _download(url: str, dest: Path, description: str) -> None:
    """Stream a URL to ``dest``, with progress. No-op if file exists."""
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        aprint(f"  Using cached {dest.name} ({size_mb:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Downloading {description}")
    aprint(f"    URL: {url}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        last_pct = 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = 100.0 * written / total
                    if pct - last_pct >= 20.0:
                        aprint(
                            f"    {written / (1024 * 1024):,.0f} / "
                            f"{total / (1024 * 1024):,.0f} MB  ({pct:.0f}%)"
                        )
                        last_pct = pct
    tmp.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    aprint(f"  ✓ Saved {dest.name} ({size_mb:.1f} MB)")


def ensure_data(cache_dir: Path) -> tuple[Path, Path]:
    """Fetch latest AS-rel + AS-org2info files. Returns local paths."""
    with asection("Discovering latest CAIDA snapshots"):
        rel_name, rel_url = _find_latest_file(AS_REL_INDEX_URL, AS_REL_FILE_PATTERN)
        org_name, org_url = _find_latest_file(AS_ORG_INDEX_URL, AS_ORG_FILE_PATTERN)
        aprint(f"  AS relationships: {rel_name}")
        aprint(f"  AS organizations: {org_name}")

    rel_path = cache_dir / rel_name
    org_path = cache_dir / org_name

    with asection("Fetching CAIDA data"):
        _download(rel_url, rel_path, f"{rel_name} (~5-15 MB)")
        _download(org_url, org_path, f"{org_name} (~10-30 MB)")

    return rel_path, org_path


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def parse_as_org(path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by ASN with ``org_name`` and ``country``.

    The CAIDA as-org2info file has two sections separated by format-comment
    headers:
        # format: org_id|changed|name|country|source         (organizations)
        # format: aut|changed|aut_name|org_id|opaque_id|source  (ASes → org_id)
    We stream once, tracking which section we're in, and join.
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
                    if "format: org_id" in line:
                        mode = "org"
                    elif "format: aut" in line:
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
    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        aprint("❌ Missing dependency: networkx (>=3.0)")
        aprint("   Install with: pip install 'networkx>=3.0'")
        raise

    import networkx as nx

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
# Layout: spectral + UMAP (tuned) + PCA realign
# -----------------------------------------------------------------------------


def _build_adjacency(nodes: list[str], edges: pd.DataFrame):
    """Undirected, unweighted CSR adjacency (treats all edges as symmetric)."""
    from scipy.sparse import csr_matrix

    idx = {s: i for i, s in enumerate(nodes)}
    rows = np.fromiter((idx[a] for a in edges["asn_a"]), dtype=np.int32)
    cols = np.fromiter((idx[b] for b in edges["asn_b"]), dtype=np.int32)
    data = np.ones(len(edges), dtype=np.float32)
    n = len(nodes)
    mat = csr_matrix(
        (
            np.concatenate([data, data]),
            (np.concatenate([rows, cols]), np.concatenate([cols, rows])),
        ),
        shape=(n, n),
    )
    mat.sum_duplicates()
    mat.data = np.minimum(mat.data, 1.0)
    return mat


def _nodes_hash(nodes: list[str]) -> str:
    return hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()[:16]


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
    nodes: list[str], edges: pd.DataFrame, cache_path: Path, recompute: bool
) -> np.ndarray:
    """Spectral embedding + UMAP + PCA-realign → 3D coords, cached to npz."""
    node_hash = _nodes_hash(nodes)

    if cache_path.exists() and not recompute:
        with asection("Loading cached 3D layout"):
            data = np.load(cache_path, allow_pickle=False)
            cached_hash = str(data["node_hash"])
            if cached_hash == node_hash:
                coords = data["coords"].astype(np.float32)
                aprint(f"  Loaded {len(coords):,} coords from cache")
                return coords
            aprint("  Node set changed — recomputing layout")

    from scipy.sparse import eye as speye
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    with asection("Computing spectral + UMAP layout"):
        adj = _build_adjacency(nodes, edges)
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

        try:
            from umap import UMAP
        except ImportError:
            aprint("❌ Missing dependency: umap-learn")
            aprint("   Install with: pip install umap-learn")
            raise

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

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, coords=coords, node_hash=np.array(node_hash))
        aprint(f"  Cached to {cache_path.name}")
    return coords


# -----------------------------------------------------------------------------
# Communities
# -----------------------------------------------------------------------------


def compute_communities(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    """Louvain communities on the undirected projection of the AS graph."""
    import networkx as nx

    with asection("Detecting communities (Louvain)"):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(
            zip(edges["asn_a"].tolist(), edges["asn_b"].tolist(), strict=True)
        )
        partitions = nx.community.louvain_communities(g, seed=42)
        partitions = sorted(partitions, key=len, reverse=True)
        aprint(f"  {len(partitions)} communities")
        for i, p in enumerate(partitions[:5]):
            aprint(f"    #{i}: {len(p):,} ASes")
        if len(partitions) > 5:
            aprint(f"    ... + {len(partitions) - 5} smaller")

        idx = {s: i for i, s in enumerate(nodes)}
        comms = np.full(len(nodes), -1, dtype=np.int32)
        for comm_id, members in enumerate(partitions):
            for asn in members:
                comms[idx[asn]] = comm_id
    return comms


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


def _build_community_legend(communities: np.ndarray, top_n: int = 20) -> str:
    unique, counts = np.unique(communities, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    n_comms = len(unique)
    lines = [
        '<div style="font-size:1.3vh;line-height:1.5;'
        "background:rgba(0,0,0,0.55);padding:0.6vh 0.8vh;"
        'border-radius:4px;max-height:82vh;overflow:hidden">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.4vh">'
        f"Community ({n_comms})</div>"
    ]
    for comm_id, count in zip(unique[:top_n], counts[:top_n], strict=True):
        r, g, b = get_categorical_color(int(comm_id), n_comms)
        lines.append(
            f'<div style="white-space:nowrap">'
            f'<span style="color:rgb({r},{g},{b})">█</span> '
            f"#{int(comm_id)} ({int(count):,})</div>"
        )
    if n_comms > top_n:
        lines.append(f'<div style="color:#888">... +{n_comms - top_n} more</div>')
    lines.append("</div>")
    return "".join(lines)


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
            community_legend = _build_community_legend(communities)
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


def _int_arg(argv: list[str], flag: str, default: int) -> int:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return int(argv[i + 1])
        if arg.startswith(flag + "="):
            return int(arg.split("=", 1)[1])
    return default


def _path_arg(argv: list[str], flag: str) -> Path | None:
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if arg.startswith(flag + "="):
            return Path(arg.split("=", 1)[1]).expanduser()
    return None


def _node_degrees(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    counts = pd.concat([edges["asn_a"], edges["asn_b"]]).value_counts()
    return counts.reindex(nodes, fill_value=0).to_numpy(dtype=np.int32)


def main() -> None:
    argv = sys.argv[1:]
    max_edges = _int_arg(argv, "--max-edges", DEFAULT_MAX_EDGES)
    cache_dir = _path_arg(argv, "--cache-dir") or CACHE_DIR
    recompute = "--recompute-layout" in argv

    aprint("=" * 70)
    aprint("CAIDA AS Topology — The Internet as a Graph")
    aprint("=" * 70)
    aprint("~80k ASes · ~350k BGP relationships · CAIDA serial-2")
    aprint(f"Edge cap: {max_edges:,}")
    aprint("")

    rel_path, org_path = ensure_data(cache_dir)
    org_df = parse_as_org(org_path)
    edges_raw = parse_as_rel(rel_path)
    nodes, node_df, edges, tier1 = filter_to_lcc(edges_raw, org_df)
    coords = compute_layout(
        nodes,
        edges,
        cache_path=cache_dir / LAYOUT_CACHE_FILENAME,
        recompute=recompute,
    )
    communities = compute_communities(nodes, edges)
    degrees = _node_degrees(nodes, edges)

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
