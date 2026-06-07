#!/usr/bin/env python3
"""Self-Contained Demo: HuRI Protein-Protein Interaction Flow Field.

This demo turns the HuRI human protein-protein interaction network into a
continuous 3D flow landscape:

1. Download HuRI and HGNC metadata.
2. Compute PageRank centrality on the undirected PPI graph.
3. Orient every interaction from lower PageRank to higher PageRank.
4. Build a signed sparse adjacency / flow-profile matrix:
       X[tail, head] = +1, X[head, tail] = -1
   and feed that matrix to UMAP so proteins with similar upstream/downstream
   flow roles are colocated, while opposite flow roles separate.
5. Build a cubic vector field over the 3D UMAP bounding cube.  For every grid
   cell, nearby oriented edges are blended with a regularized 1/d^3 weight.
6. Smooth the vector field with a Gaussian blur (sigma=1 voxel).
7. Forward-advect all protein nodes through the field to make streamlines.

The result is a topology-derived "interactome wind map": proteins are points,
PPI edges are an optional faint directed layer, and luminous streamlines show
how the PageRank-oriented interaction flow converges through the volume.

Usage:
    hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py
    hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py --preset preview
    hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py --preset full
    hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py --no-serve
    hatch run python packages/luxar/src/luxar/demos/demo_ppi_flow_field.py --recompute-field

Requirements:
    pip install 'luxar[demos]' networkx umap-learn scipy pandas requests

Notes:
    - The default full preset uses the requested 256^3 grid and caches a large
      vector field under ~/.cache/luxar/huri/.
    - Use --preset preview for a faster 128^3 authoring/iteration run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, UIConfig, ViewerConfig
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import get_categorical_color
from luxar.utils.fields import (
    FlowField,
    add_reference_cube_to_scene,
    cubic_bounds,
    rk4_step,
)
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CACHE_DIR: Final = Path.home() / ".cache" / "luxar" / "huri"
HURI_URL: Final = "https://interactome-atlas.org/data/HuRI.tsv"
HURI_FILENAME: Final = "HuRI.tsv"
HGNC_URL: Final = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/"
    "hgnc_complete_set.txt"
)
HGNC_FILENAME: Final = "hgnc_complete_set.txt"

CACHE_VERSION: Final = "v1"
LAYOUT_CACHE_FILENAME: Final = f"ppi_flow_signed_umap_layout_{CACHE_VERSION}.npz"
PAGERANK_CACHE_FILENAME: Final = f"ppi_flow_pagerank_{CACHE_VERSION}.npz"
COMMUNITY_CACHE_FILENAME: Final = f"ppi_flow_communities_{CACHE_VERSION}.npz"

UMAP_N_NEIGHBORS: Final = 20
UMAP_MIN_DIST: Final = 0.22
UMAP_RANDOM_STATE: Final = 42

EDGE_TAIL_COLOR: Final = np.array([0.16, 0.42, 1.00], dtype=np.float32)
EDGE_HEAD_COLOR: Final = np.array([1.00, 0.62, 0.18], dtype=np.float32)
ATTRACTOR_COLOR: Final = np.array([1.35, 0.92, 0.28], dtype=np.float32)

# The viewer validates exposure to [-5, +5] EV.  The original scene used
# -3.6 EV; the requested -5 EV dimming would land at -8.6 EV, so we clamp the
# viewer exposure to -5 and apply the remaining -3.6 EV as a linear intensity
# multiplier on scene layers: 2^-3.6 ≈ 0.082.
VIEWER_EXPOSURE_EV: Final = -5.0
NODE_INTENSITY_SCALE: Final = float(2.0 ** (-8.6 - VIEWER_EXPOSURE_EV))


@dataclass(frozen=True)
class FlowPreset:
    """Numerical settings for vector-field construction and streamline tracing."""

    name: str
    grid_size: int
    nearest_samples: int
    edge_sample_step_voxels: float
    max_samples_per_edge: int
    chunk_cells: int
    gaussian_sigma: float
    step_voxels: float
    streamline_steps: int
    streamline_width: float
    edge_line_width_tail: float
    edge_line_width_head: float
    max_edge_lines: int
    max_streamline_seeds: int | None


PRESETS: Final[dict[str, FlowPreset]] = {
    "preview": FlowPreset(
        name="preview",
        grid_size=128,
        nearest_samples=32,
        edge_sample_step_voxels=1.25,
        max_samples_per_edge=48,
        chunk_cells=65_536,
        gaussian_sigma=1.0,
        step_voxels=0.75,
        streamline_steps=170,
        streamline_width=0.0105,
        edge_line_width_tail=0.001575,
        edge_line_width_head=0.0049,
        max_edge_lines=40_000,
        max_streamline_seeds=None,
    ),
    "full": FlowPreset(
        name="full",
        grid_size=256,
        nearest_samples=48,
        edge_sample_step_voxels=1.10,
        max_samples_per_edge=64,
        chunk_cells=32_768,
        gaussian_sigma=1.0,
        step_voxels=0.70,
        streamline_steps=260,
        streamline_width=0.0077,
        edge_line_width_tail=0.001225,
        edge_line_width_head=0.00385,
        max_edge_lines=60_000,
        max_streamline_seeds=None,
    ),
}


@dataclass(frozen=True)
class OrientedEdges:
    """Undirected PPI edges oriented from lower to higher PageRank."""

    tail_idx: np.ndarray
    head_idx: np.ndarray
    centrality_delta: np.ndarray
    tie_mask: np.ndarray


@dataclass(frozen=True)
class StreamlineData:
    """Indexed Luxar line geometry for advected protein streamlines."""

    vertices: np.ndarray
    segments: np.ndarray
    colors: np.ndarray
    streamline_count: int


@dataclass(frozen=True)
class SceneStats:
    """Summary of the generated scene."""

    n_nodes: int
    n_edges: int
    n_edge_lines: int
    n_streamlines: int
    n_communities: int
    grid_size: int


# -----------------------------------------------------------------------------
# Downloading and HuRI/HGNC loading
# -----------------------------------------------------------------------------


def _download(url: str, dest: Path, description: str) -> None:
    """Stream a URL to ``dest`` if it is not already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        aprint(f"  Using cached {dest.name} ({size_mb:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    aprint(f"  Downloading {description}")
    aprint(f"    URL: {url}")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        last_pct = 0.0
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = 100.0 * written / total
                    if pct - last_pct >= 20.0:
                        aprint(
                            f"    {written / (1024 * 1024):,.0f} / "
                            f"{total / (1024 * 1024):,.0f} MB ({pct:.0f}%)"
                        )
                        last_pct = pct
    tmp.replace(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    aprint(f"  ✓ Saved {dest.name} ({size_mb:.1f} MB)")


def ensure_data(cache_dir: Path) -> tuple[Path, Path]:
    """Fetch HuRI and HGNC metadata into the shared HuRI cache."""
    huri_path = cache_dir / HURI_FILENAME
    hgnc_path = cache_dir / HGNC_FILENAME
    with asection("Fetching HuRI / HGNC data"):
        _download(HURI_URL, huri_path, "HuRI network (~2 MB)")
        _download(HGNC_URL, hgnc_path, "HGNC complete set (~30 MB)")
    return huri_path, hgnc_path


def load_hgnc(tsv_path: Path) -> pd.DataFrame:
    """Return a frame indexed by Ensembl gene id with symbol/chromosome."""
    wanted = {"symbol", "ensembl_gene_id", "location", "status"}
    with asection("Loading HGNC metadata"):
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            usecols=lambda c: c in wanted,
            dtype=str,
            low_memory=False,
        )
        if "status" in df.columns:
            df = df[df["status"] == "Approved"]
        df = df.dropna(subset=["ensembl_gene_id", "symbol"])
        df["chromosome"] = (
            df.get("location", pd.Series([""] * len(df)))
            .fillna("?")
            .astype(str)
            .str.extract(r"^([0-9XYMT]+)", expand=False)
            .fillna("?")
        )
        df = df.set_index("ensembl_gene_id")[["symbol", "chromosome"]]
        df = df[~df.index.duplicated(keep="first")]
        aprint(f"  {len(df):,} Ensembl→symbol mappings")
    return df


def load_huri_edges(tsv_path: Path, hgnc: pd.DataFrame) -> pd.DataFrame:
    """Return the undirected, deduplicated HuRI edge frame in HGNC symbols."""
    with asection("Loading HuRI protein-protein interactions"):
        df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str, low_memory=False)
        if df.shape[1] < 2:
            raise RuntimeError(f"Unexpected HuRI format: {df.shape[1]} columns")
        df = df.iloc[:, :2].copy()
        df.columns = ["ensg_a", "ensg_b"]
        df = df[
            df["ensg_a"].str.startswith("ENSG", na=False)
            & df["ensg_b"].str.startswith("ENSG", na=False)
        ]
        df = df[df["ensg_a"] != df["ensg_b"]]
        aprint(f"  {len(df):,} ENSG-level pairs")

        symbol_map = hgnc["symbol"].to_dict()
        sym_a = df["ensg_a"].map(symbol_map)
        sym_b = df["ensg_b"].map(symbol_map)
        keep = sym_a.notna() & sym_b.notna()
        df = pd.DataFrame(
            {"sym_a": sym_a[keep].to_numpy(), "sym_b": sym_b[keep].to_numpy()}
        )
        aprint(f"  {len(df):,} pairs with both endpoints mapped to HGNC")

        a = df["sym_a"].to_numpy()
        b = df["sym_b"].to_numpy()
        lo = np.where(a < b, a, b)
        hi = np.where(a < b, b, a)
        df = pd.DataFrame({"sym_a": lo, "sym_b": hi}).drop_duplicates()
        aprint(f"  {len(df):,} unique undirected pairs")
    return df.reset_index(drop=True)


def filter_to_lcc(
    edges: pd.DataFrame, hgnc: pd.DataFrame
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Restrict the network to its largest connected component."""
    try:
        import networkx as nx  # noqa: F401
    except ImportError:
        aprint("❌ Missing dependency: networkx>=3.0")
        aprint("   Install with: pip install 'networkx>=3.0'")
        raise

    import networkx as nx

    with asection("Filtering to largest connected component"):
        graph = nx.Graph()
        graph.add_edges_from(
            zip(edges["sym_a"].tolist(), edges["sym_b"].tolist(), strict=True)
        )
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        lcc = components[0]
        aprint(
            f"  {len(components):,} components; LCC has {len(lcc):,} nodes "
            f"(dropped {graph.number_of_nodes() - len(lcc):,})"
        )

        lcc_set = set(lcc)
        mask = edges["sym_a"].isin(lcc_set) & edges["sym_b"].isin(lcc_set)
        edges = edges[mask].reset_index(drop=True)
        aprint(f"  {len(edges):,} edges within the LCC")

        nodes = sorted(lcc_set)
        sym_to_chrom = hgnc.groupby("symbol")["chromosome"].first().to_dict()
        node_df = pd.DataFrame(
            {
                "symbol": nodes,
                "chromosome": [sym_to_chrom.get(symbol, "?") for symbol in nodes],
            }
        )
    return nodes, node_df, edges


# -----------------------------------------------------------------------------
# Graph analysis: PageRank, communities, signed flow features
# -----------------------------------------------------------------------------


def network_hash(nodes: list[str], edges: pd.DataFrame) -> str:
    """Stable hash for cache invalidation when the graph changes."""
    digest = hashlib.sha256()
    digest.update(f"nodes:{len(nodes)}\n".encode())
    for node in nodes:
        digest.update(node.encode("utf-8"))
        digest.update(b"\n")
    digest.update(f"edges:{len(edges)}\n".encode())
    for sym_a, sym_b in edges[["sym_a", "sym_b"]].itertuples(index=False, name=None):
        digest.update(str(sym_a).encode("utf-8"))
        digest.update(b"\t")
        digest.update(str(sym_b).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def array_hash(array: np.ndarray) -> str:
    """Short content hash for numeric arrays used in derived caches."""
    arr = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _load_cache_hash(cache_path: Path, expected_hash: str) -> dict[str, Any] | None:
    """Load a small npz cache and validate its network hash."""
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=False)
    if "network_hash" not in data or str(data["network_hash"]) != expected_hash:
        return None
    return dict(data.items())


def compute_pagerank(
    nodes: list[str],
    edges: pd.DataFrame,
    cache_path: Path,
    graph_hash: str,
    recompute: bool,
) -> np.ndarray:
    """Compute PageRank centrality in node order, with cache."""
    if not recompute:
        cached = _load_cache_hash(cache_path, graph_hash)
        if cached is not None and "pagerank" in cached:
            with asection("Loading cached PageRank centrality"):
                pagerank = cached["pagerank"].astype(np.float64)
                aprint(f"  Loaded PageRank for {len(pagerank):,} proteins")
                return pagerank

    try:
        import networkx as nx
    except ImportError:
        aprint("❌ Missing dependency: networkx>=3.0")
        aprint("   Install with: pip install 'networkx>=3.0'")
        raise

    with asection("Computing PageRank centrality"):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(
            zip(edges["sym_a"].tolist(), edges["sym_b"].tolist(), strict=True)
        )
        rank_dict = nx.pagerank(graph, alpha=0.85, tol=1e-9, max_iter=200)
        pagerank = np.array([rank_dict[node] for node in nodes], dtype=np.float64)
        top = np.argsort(-pagerank)[:8]
        for rank, idx in enumerate(top, start=1):
            aprint(f"    #{rank}: {nodes[int(idx)]}  PageRank={pagerank[idx]:.3e}")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            network_hash=np.array(graph_hash),
            pagerank=pagerank.astype(np.float64),
        )
        aprint(f"  Cached to {cache_path.name}")
    return pagerank


def compute_degrees(nodes: list[str], edges: pd.DataFrame) -> np.ndarray:
    """Degree per node, in the same order as ``nodes``."""
    counts = pd.concat([edges["sym_a"], edges["sym_b"]]).value_counts()
    return counts.reindex(nodes, fill_value=0).to_numpy(dtype=np.int32)


def compute_communities(
    nodes: list[str],
    edges: pd.DataFrame,
    cache_path: Path,
    graph_hash: str,
    recompute: bool,
) -> np.ndarray:
    """Louvain community assignment per node, sorted by community size."""
    if not recompute:
        cached = _load_cache_hash(cache_path, graph_hash)
        if cached is not None and "communities" in cached:
            with asection("Loading cached Louvain communities"):
                communities = cached["communities"].astype(np.int32)
                aprint(f"  Loaded {int(communities.max()) + 1:,} communities")
                return communities

    try:
        import networkx as nx
    except ImportError:
        aprint("❌ Missing dependency: networkx>=3.0")
        aprint("   Install with: pip install 'networkx>=3.0'")
        raise

    with asection("Detecting communities (Louvain)"):
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(
            zip(edges["sym_a"].tolist(), edges["sym_b"].tolist(), strict=True)
        )
        partitions = nx.community.louvain_communities(graph, seed=42)
        partitions = sorted(partitions, key=len, reverse=True)
        aprint(f"  {len(partitions):,} communities detected")
        for i, part in enumerate(partitions[:5]):
            aprint(f"    #{i}: {len(part):,} nodes")

        node_index = {symbol: i for i, symbol in enumerate(nodes)}
        communities = np.full(len(nodes), -1, dtype=np.int32)
        for comm_id, members in enumerate(partitions):
            for symbol in members:
                communities[node_index[symbol]] = comm_id

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            network_hash=np.array(graph_hash),
            communities=communities,
        )
        aprint(f"  Cached to {cache_path.name}")
    return communities


def orient_edges_by_pagerank(
    nodes: list[str], edges: pd.DataFrame, pagerank: np.ndarray
) -> OrientedEdges:
    """Orient every undirected edge from low PageRank to high PageRank."""
    node_index = {symbol: i for i, symbol in enumerate(nodes)}
    a_idx = np.fromiter((node_index[s] for s in edges["sym_a"]), dtype=np.int32)
    b_idx = np.fromiter((node_index[s] for s in edges["sym_b"]), dtype=np.int32)

    pr_a = pagerank[a_idx]
    pr_b = pagerank[b_idx]
    tie_mask = pr_a == pr_b

    a_to_b = pr_a < pr_b
    b_to_a = pr_b < pr_a
    tie_a_to_b = tie_mask & (a_idx < b_idx)
    tail_idx = np.where(a_to_b | tie_a_to_b, a_idx, b_idx).astype(np.int32)
    head_idx = np.where(b_to_a | (tie_mask & (b_idx < a_idx)), a_idx, b_idx)
    head_idx = head_idx.astype(np.int32)
    centrality_delta = np.abs(pagerank[head_idx] - pagerank[tail_idx]).astype(
        np.float64
    )

    with asection("Orienting PPI edges by PageRank"):
        aprint(f"  Oriented {len(edges):,} interactions low→high PageRank")
        aprint(f"  Ties resolved deterministically: {int(tie_mask.sum()):,}")
        nonzero = centrality_delta[centrality_delta > 0]
        if len(nonzero):
            aprint(
                f"  PageRank Δ: median={np.median(nonzero):.2e}, "
                f"p95={np.percentile(nonzero, 95):.2e}, "
                f"max={nonzero.max():.2e}"
            )
    return OrientedEdges(tail_idx, head_idx, centrality_delta, tie_mask)


def build_signed_flow_adjacency(n_nodes: int, oriented: OrientedEdges):
    """Build sparse signed adjacency / flow-profile features for UMAP."""
    from scipy.sparse import csr_matrix

    n_edges = len(oriented.tail_idx)
    rows = np.concatenate([oriented.tail_idx, oriented.head_idx])
    cols = np.concatenate([oriented.head_idx, oriented.tail_idx])
    data = np.concatenate(
        [np.ones(n_edges, dtype=np.float32), -np.ones(n_edges, dtype=np.float32)]
    )
    matrix = csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    matrix.sum_duplicates()
    return matrix


def _stabilize_layout(coords: np.ndarray) -> np.ndarray:
    """Center, PCA-align, sign-stabilize, and scale a 3D layout."""
    coords = coords.astype(np.float32, copy=True)
    coords -= coords.mean(axis=0, keepdims=True)

    if len(coords) >= 3:
        _, _, vt = np.linalg.svd(coords.astype(np.float64), full_matrices=False)
        coords = (coords @ vt.T.astype(np.float32)).astype(np.float32)
        for axis in range(3):
            idx = int(np.argmax(np.abs(coords[:, axis])))
            if coords[idx, axis] < 0:
                coords[:, axis] *= -1.0

    radius_95 = float(np.percentile(np.linalg.norm(coords, axis=1), 95))
    if radius_95 > 0:
        coords *= np.float32(10.0 / radius_95)
    return coords.astype(np.float32)


def compute_signed_flow_layout(
    nodes: list[str],
    oriented: OrientedEdges,
    cache_path: Path,
    graph_hash: str,
    recompute: bool,
) -> np.ndarray:
    """Run UMAP on signed flow-profile adjacency features, with cache."""
    if cache_path.exists() and not recompute:
        data = np.load(cache_path, allow_pickle=False)
        cached_n_neighbors = int(data["n_neighbors"]) if "n_neighbors" in data else None
        cached_min_dist = float(data["min_dist"]) if "min_dist" in data else None
        if (
            "network_hash" in data
            and str(data["network_hash"]) == graph_hash
            and "coords" in data
            and cached_n_neighbors == UMAP_N_NEIGHBORS
            and cached_min_dist == UMAP_MIN_DIST
        ):
            with asection("Loading cached signed-flow UMAP layout"):
                coords = data["coords"].astype(np.float32)
                aprint(f"  Loaded {len(coords):,} coordinates from {cache_path.name}")
                return coords

    try:
        from umap import UMAP
    except ImportError:
        aprint("❌ Missing dependency: umap-learn")
        aprint("   Install with: pip install umap-learn")
        raise

    with asection("Computing signed-flow adjacency UMAP"):
        features = build_signed_flow_adjacency(len(nodes), oriented)
        aprint(
            f"  Signed feature matrix: {features.shape[0]:,} × "
            f"{features.shape[1]:,}, nnz={features.nnz:,}"
        )
        aprint("  Semantics: tail→head is +1 in the tail row, -1 in the head row")

        reducer = UMAP(
            n_components=3,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="cosine",
            random_state=UMAP_RANDOM_STATE,
            low_memory=True,
            verbose=False,
        )
        coords = reducer.fit_transform(features).astype(np.float32)
        coords = _stabilize_layout(coords)
        aprint(
            f"  Layout ranges: x={np.ptp(coords[:, 0]):.2f}, "
            f"y={np.ptp(coords[:, 1]):.2f}, z={np.ptp(coords[:, 2]):.2f}"
        )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            network_hash=np.array(graph_hash),
            coords=coords,
            n_neighbors=np.array(UMAP_N_NEIGHBORS),
            min_dist=np.array(UMAP_MIN_DIST),
        )
        aprint(f"  Cached to {cache_path.name}")
    return coords


# -----------------------------------------------------------------------------
# Vector-field construction
# -----------------------------------------------------------------------------


def _flow_cache_key(
    graph_hash: str,
    layout_hash: str,
    preset: FlowPreset,
    raw_edge_vectors: bool,
) -> str:
    mode = "raw" if raw_edge_vectors else "unit"
    return (
        f"{CACHE_VERSION}:{graph_hash}:{layout_hash}:{preset.grid_size}:"
        f"{preset.nearest_samples}:{preset.edge_sample_step_voxels}:"
        f"{preset.max_samples_per_edge}:{preset.gaussian_sigma}:{mode}"
    )


def _sample_oriented_edges(
    starts: np.ndarray,
    ends: np.ndarray,
    lengths: np.ndarray,
    spacing: float,
    preset: FlowPreset,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample each edge segment for KD-tree candidate lookup."""
    sample_step = max(spacing * preset.edge_sample_step_voxels, 1e-6)
    counts = np.ceil(lengths / sample_step).astype(np.int32) + 1
    counts = np.clip(counts, 2, preset.max_samples_per_edge)
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    total = int(offsets[-1])

    sample_points = np.empty((total, 3), dtype=np.float32)
    sample_edge_ids = np.empty(total, dtype=np.int32)
    for edge_id, count in enumerate(counts):
        start = int(offsets[edge_id])
        end = int(offsets[edge_id + 1])
        t = np.linspace(0.0, 1.0, int(count), dtype=np.float32)[:, None]
        sample_points[start:end] = starts[edge_id] * (1.0 - t) + ends[edge_id] * t
        sample_edge_ids[start:end] = edge_id

    aprint(
        f"  Edge KD samples: {total:,} points "
        f"({float(np.mean(counts)):.1f} per edge avg)"
    )
    return sample_points, sample_edge_ids


def _grid_points_for_flat_indices(
    flat_indices: np.ndarray, grid_min: np.ndarray, spacing: float, n: int
) -> np.ndarray:
    """Convert flattened C-order grid indices to xyz world coordinates."""
    ix = flat_indices // (n * n)
    rem = flat_indices - ix * n * n
    iy = rem // n
    iz = rem - iy * n
    coords = np.column_stack([ix, iy, iz]).astype(np.float32)
    coords *= np.float32(spacing)
    coords += grid_min[None, :]
    return coords


def _build_field_vectors(
    coords: np.ndarray,
    oriented: OrientedEdges,
    preset: FlowPreset,
    raw_edge_vectors: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute the unsmoothed vector field over the cubic grid."""
    from scipy.spatial import cKDTree

    grid_min, grid_max = cubic_bounds(coords, pad_fraction=0.08)
    n = preset.grid_size
    spacing = float((grid_max[0] - grid_min[0]) / max(n - 1, 1))

    starts_all = coords[oriented.tail_idx]
    ends_all = coords[oriented.head_idx]
    edge_vecs_all = ends_all - starts_all
    lengths_all = np.linalg.norm(edge_vecs_all, axis=1).astype(np.float32)
    valid_edges = lengths_all > np.float32(1e-6)
    starts = starts_all[valid_edges].astype(np.float32)
    ends = ends_all[valid_edges].astype(np.float32)
    edge_vecs = edge_vecs_all[valid_edges].astype(np.float32)
    lengths = lengths_all[valid_edges].astype(np.float32)
    edge_len2 = np.maximum(lengths * lengths, np.float32(1e-12))

    if raw_edge_vectors:
        median_len = max(float(np.median(lengths)), 1e-6)
        flow_vectors = (edge_vecs / np.float32(median_len)).astype(np.float32)
        aprint("  Field vectors: raw UMAP edge displacements / median edge length")
    else:
        flow_vectors = (edge_vecs / lengths[:, None]).astype(np.float32)
        aprint("  Field vectors: unit low→high PageRank edge directions")

    sample_points, sample_edge_ids = _sample_oriented_edges(
        starts, ends, lengths, spacing, preset
    )
    tree = cKDTree(sample_points)
    k = min(preset.nearest_samples, len(sample_points))

    n_cells = n**3
    field = np.zeros((n, n, n, 3), dtype=np.float32)
    flat_field = field.reshape(n_cells, 3)
    eps2 = np.float32(spacing * spacing)

    aprint(
        f"  Grid: {n}^3 = {n_cells:,} cells, spacing={spacing:.4f}, "
        f"raw field size≈{n_cells * 3 * 4 / (1024 * 1024):.1f} MB"
    )
    aprint(f"  Querying {k} nearest edge samples per cell")

    chunk = preset.chunk_cells
    for start in range(0, n_cells, chunk):
        stop = min(start + chunk, n_cells)
        flat = np.arange(start, stop, dtype=np.int64)
        points = _grid_points_for_flat_indices(flat, grid_min, spacing, n)

        _, nearest_sample_idx = tree.query(points, k=k, workers=-1)
        if k == 1:
            nearest_sample_idx = nearest_sample_idx[:, None]

        candidate_edges = sample_edge_ids[nearest_sample_idx]
        order = np.argsort(candidate_edges, axis=1)
        candidate_edges = np.take_along_axis(candidate_edges, order, axis=1)
        duplicate = np.zeros(candidate_edges.shape, dtype=bool)
        duplicate[:, 1:] = candidate_edges[:, 1:] == candidate_edges[:, :-1]

        p_minus_start = points[:, None, :] - starts[candidate_edges]
        cand_vecs = edge_vecs[candidate_edges]
        t = np.sum(p_minus_start * cand_vecs, axis=2) / edge_len2[candidate_edges]
        t = np.clip(t, 0.0, 1.0).astype(np.float32)
        closest_delta = p_minus_start - t[:, :, None] * cand_vecs
        d2 = np.sum(closest_delta * closest_delta, axis=2).astype(np.float32)

        weights = 1.0 / np.power(d2 + eps2, 1.5)
        weights = weights.astype(np.float32)
        weights[duplicate] = 0.0
        weight_sum = weights.sum(axis=1).astype(np.float32)
        weighted = weights[:, :, None] * flow_vectors[candidate_edges]
        vectors = weighted.sum(axis=1).astype(np.float32)
        nonzero = weight_sum > 0
        vectors[nonzero] /= weight_sum[nonzero, None]
        flat_field[start:stop] = vectors

        if start == 0 or stop == n_cells or (start // chunk) % 16 == 0:
            aprint(f"    field cells {stop:,}/{n_cells:,}")

    return field, grid_min, grid_max, spacing


def _smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth each vector component in voxel units."""
    from scipy.ndimage import gaussian_filter

    if sigma <= 0:
        return field

    smoothed = np.empty_like(field)
    for component in range(3):
        smoothed[..., component] = gaussian_filter(
            field[..., component], sigma=sigma, mode="nearest"
        )
    return smoothed.astype(np.float32)


def compute_vector_field(
    coords: np.ndarray,
    oriented: OrientedEdges,
    preset: FlowPreset,
    cache_path: Path,
    graph_hash: str,
    recompute: bool,
    raw_edge_vectors: bool,
) -> FlowField:
    """Build or load the regularized 1/d^3 edge-blended vector field."""
    layout_hash = array_hash(coords)
    cache_key = _flow_cache_key(graph_hash, layout_hash, preset, raw_edge_vectors)

    if cache_path.exists() and not recompute:
        data = np.load(cache_path, allow_pickle=False)
        if "cache_key" in data and str(data["cache_key"]) == cache_key:
            with asection("Loading cached PPI vector field"):
                vectors = data["vectors"].astype(np.float32)
                grid_min = data["grid_min"].astype(np.float32)
                grid_max = data["grid_max"].astype(np.float32)
                spacing = float(data["spacing"])
                aprint(f"  Loaded {vectors.shape[0]}^3 field from {cache_path.name}")
                return FlowField(vectors, grid_min, grid_max, spacing, cache_key)

    with asection("Building PPI vector field"):
        aprint(
            "  Each grid cell blends candidate oriented edges with "
            "1 / (d² + voxel²)^(3/2) weights"
        )
        field, grid_min, grid_max, spacing = _build_field_vectors(
            coords, oriented, preset, raw_edge_vectors
        )
        aprint(
            f"  Gaussian smoothing vector components (sigma={preset.gaussian_sigma})"
        )
        field = _smooth_field(field, preset.gaussian_sigma)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        aprint(f"  Caching vector field to {cache_path.name}")
        np.savez_compressed(
            cache_path,
            cache_key=np.array(cache_key),
            vectors=field,
            grid_min=grid_min,
            grid_max=grid_max,
            spacing=np.array(spacing, dtype=np.float32),
        )
        aprint("  ✓ Cached vector field")
    return FlowField(field, grid_min, grid_max, spacing, cache_key)


# -----------------------------------------------------------------------------
# Streamline integration
# -----------------------------------------------------------------------------


# Streamline integration uses rk4_step from luxar.utils.fields (imported above).


def _select_streamline_seeds(pagerank: np.ndarray, max_seeds: int | None) -> np.ndarray:
    """Select node indices to seed; HuRI defaults to all nodes."""
    n = len(pagerank)
    if max_seeds is None or max_seeds >= n:
        return np.arange(n, dtype=np.int32)

    rng = np.random.default_rng(42)
    top_count = min(max_seeds // 2, n)
    top = np.argsort(-pagerank)[:top_count]
    remaining = np.setdiff1d(np.arange(n, dtype=np.int32), top.astype(np.int32))
    random_count = max_seeds - top_count
    random = rng.choice(remaining, size=random_count, replace=False)
    seeds = np.unique(np.concatenate([top.astype(np.int32), random.astype(np.int32)]))
    if len(seeds) < max_seeds:
        extras = np.setdiff1d(np.arange(n, dtype=np.int32), seeds)
        seeds = np.concatenate([seeds, extras[: max_seeds - len(seeds)]])
    return np.sort(seeds.astype(np.int32))


def _streamline_cache_key(
    graph_hash: str,
    flow: FlowField,
    preset: FlowPreset,
    max_seeds: int | None,
) -> str:
    seed_text = "all" if max_seeds is None else str(max_seeds)
    return (
        f"{CACHE_VERSION}:{graph_hash}:{flow.cache_key}:"
        f"{preset.streamline_steps}:{preset.step_voxels}:{seed_text}"
    )


def integrate_streamlines(
    coords: np.ndarray,
    pagerank: np.ndarray,
    flow: FlowField,
    preset: FlowPreset,
    cache_path: Path,
    graph_hash: str,
    recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-advect protein nodes through the vector field."""
    cache_key = _streamline_cache_key(
        graph_hash, flow, preset, preset.max_streamline_seeds
    )
    if cache_path.exists() and not recompute:
        data = np.load(cache_path, allow_pickle=False)
        if "cache_key" in data and str(data["cache_key"]) == cache_key:
            with asection("Loading cached protein streamlines"):
                positions = data["positions"].astype(np.float32)
                valid = data["valid"].astype(bool)
                seed_indices = data["seed_indices"].astype(np.int32)
                aprint(
                    f"  Loaded {len(seed_indices):,} seeded streamlines "
                    f"from {cache_path.name}"
                )
                return positions, valid, seed_indices

    with asection("Integrating protein streamlines"):
        seed_indices = _select_streamline_seeds(pagerank, preset.max_streamline_seeds)
        seeds = coords[seed_indices].astype(np.float32)
        n_seeds = len(seeds)
        n_steps = preset.streamline_steps
        step_size = float(preset.step_voxels * flow.spacing)
        aprint(
            f"  Seeds: {n_seeds:,}; RK4 steps: {n_steps}; "
            f"step={step_size:.4f} ({preset.step_voxels} voxels)"
        )

        positions = np.full((n_seeds, n_steps + 1, 3), np.nan, dtype=np.float32)
        valid = np.zeros((n_seeds, n_steps + 1), dtype=bool)
        positions[:, 0] = seeds
        valid[:, 0] = True
        current = seeds.copy()
        active = np.ones(n_seeds, dtype=bool)

        for step in range(1, n_steps + 1):
            active_idx = np.flatnonzero(active)
            if len(active_idx) == 0:
                break

            next_points = rk4_step(current[active_idx], step_size, flow)
            inside = np.all(
                (next_points >= flow.grid_min[None, :])
                & (next_points <= flow.grid_max[None, :]),
                axis=1,
            )
            ok = np.isfinite(next_points).all(axis=1) & inside
            good_idx = active_idx[ok]
            bad_idx = active_idx[~ok]
            current[good_idx] = next_points[ok]
            positions[good_idx, step] = current[good_idx]
            valid[good_idx, step] = True
            active[bad_idx] = False

            if step == 1 or step % 40 == 0 or step == n_steps:
                aprint(
                    f"    RK4 step {step:>3}/{n_steps}: "
                    f"{int(active.sum()):,} active streamlines"
                )

        valid_counts = valid.sum(axis=1)
        n_valid = int(np.count_nonzero(valid_counts > 1))
        aprint(f"  ✓ Generated {n_valid:,} streamlines with ≥2 vertices")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            cache_key=np.array(cache_key),
            positions=positions,
            valid=valid,
            seed_indices=seed_indices,
        )
        aprint(f"  Cached to {cache_path.name}")
    return positions, valid, seed_indices


def build_streamline_geometry(
    positions: np.ndarray,
    valid: np.ndarray,
    seed_indices: np.ndarray,
    communities: np.ndarray,
    palette: np.ndarray,
) -> StreamlineData:
    """Convert line-major histories into indexed Luxar Lines geometry."""
    with asection("Building streamline line geometry"):
        valid_counts = valid.sum(axis=1)
        line_mask = valid_counts > 1
        streamline_count = int(np.count_nonzero(line_mask))
        if streamline_count == 0:
            return StreamlineData(
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
                0,
            )

        line_positions = positions[line_mask]
        line_valid = valid[line_mask]
        line_seed_indices = seed_indices[line_mask]
        n_vertices = int(np.count_nonzero(line_valid))

        index_map = np.full(line_valid.shape, -1, dtype=np.int32)
        index_map[line_valid] = np.arange(n_vertices, dtype=np.int32)
        start = index_map[:, :-1]
        end = index_map[:, 1:]
        segment_mask = (start >= 0) & (end >= 0)
        segments = np.column_stack([start[segment_mask], end[segment_mask]]).astype(
            np.uint32
        )
        vertices = line_positions[line_valid].astype(np.float32, copy=False)

        source_colors = palette[communities[line_seed_indices]].astype(np.float32)
        source_colors = np.clip(source_colors * 1.35, 0.0, 1.6)
        colors = np.repeat(source_colors[:, None, :], line_valid.shape[1], axis=1)[
            line_valid
        ].astype(np.float32)

        aprint(
            f"  {streamline_count:,} streamlines, {len(vertices):,} vertices, "
            f"{len(segments):,} segments"
        )
        return StreamlineData(vertices, segments, colors, streamline_count)


# -----------------------------------------------------------------------------
# Scene geometry helpers
# -----------------------------------------------------------------------------


def community_palette(n_communities: int) -> np.ndarray:
    """Return categorical RGB colors as float32 in [0, 1]."""
    return (
        np.array(
            [get_categorical_color(i, n_communities) for i in range(n_communities)],
            dtype=np.float32,
        )
        / 255.0
    )


def _normalized_log(values: np.ndarray) -> np.ndarray:
    """Robust [0, 1] normalized log scale."""
    values = np.asarray(values, dtype=np.float64)
    log_values = np.log1p(values / max(float(np.median(values[values > 0])), 1e-12))
    lo = float(np.percentile(log_values, 1))
    hi = float(np.percentile(log_values, 99))
    if hi <= lo:
        return np.zeros(len(values), dtype=np.float32)
    return np.clip((log_values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def build_node_geometry(
    nodes: list[str],
    node_df: pd.DataFrame,
    coords: np.ndarray,
    communities: np.ndarray,
    degrees: np.ndarray,
    pagerank: np.ndarray,
    palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build protein point positions/colors/radii/labels."""
    pr_norm = _normalized_log(pagerank)
    deg_norm = _normalized_log(degrees.astype(np.float64) + 1.0)
    colors = palette[communities].astype(np.float32)
    brightness = (0.62 + 0.72 * pr_norm)[:, None]
    colors = np.clip(colors * brightness, 0.0, 1.5).astype(np.float32)
    radii = (0.050 + 0.170 * pr_norm + 0.055 * deg_norm).astype(np.float32)

    chroms = node_df["chromosome"].tolist()
    labels = [
        f"{symbol}\n[chr{chrom} · PageRank {pr:.2e} · deg {int(deg)} · community {int(comm)}]"
        for symbol, chrom, pr, deg, comm in zip(
            nodes, chroms, pagerank, degrees, communities, strict=True
        )
    ]
    return coords.astype(np.float32), colors, radii, labels


def build_attractor_points(
    coords: np.ndarray,
    nodes: list[str],
    pagerank: np.ndarray,
    degrees: np.ndarray,
    top_n: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build a small highlight layer for the highest-PageRank proteins."""
    count = min(top_n, len(nodes))
    top = np.argsort(-pagerank)[:count]
    pr_norm = _normalized_log(pagerank)
    positions = coords[top].astype(np.float32)
    colors = np.tile(ATTRACTOR_COLOR[None, :], (count, 1)).astype(np.float32)
    radii = (0.18 + 0.20 * pr_norm[top]).astype(np.float32)
    labels = [
        f"Hub #{rank}: {nodes[int(idx)]}\n[PageRank {pagerank[idx]:.2e} · deg {int(degrees[idx])}]"
        for rank, idx in enumerate(top, start=1)
    ]
    return positions, colors, radii, labels


def build_oriented_edge_lines(
    nodes: list[str],
    coords: np.ndarray,
    oriented: OrientedEdges,
    pagerank: np.ndarray,
    preset: FlowPreset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """Build a capped, faint, tapered directed PPI edge layer."""
    n_edges_total = len(oriented.tail_idx)
    if n_edges_total > preset.max_edge_lines:
        order = np.argsort(-oriented.centrality_delta)
        chosen = np.sort(order[: preset.max_edge_lines])
    else:
        chosen = np.arange(n_edges_total, dtype=np.int64)

    tail = oriented.tail_idx[chosen]
    head = oriented.head_idx[chosen]
    n_edges = len(chosen)
    vertices = np.empty((n_edges * 2, 3), dtype=np.float32)
    vertices[0::2] = coords[tail]
    vertices[1::2] = coords[head]

    widths = np.empty(n_edges * 2, dtype=np.float32)
    widths[0::2] = preset.edge_line_width_tail
    widths[1::2] = preset.edge_line_width_head

    colors = np.empty((n_edges * 2, 3), dtype=np.float32)
    colors[0::2] = EDGE_TAIL_COLOR
    colors[1::2] = EDGE_HEAD_COLOR

    labels: list[str] = []
    for tail_idx, head_idx in zip(tail, head, strict=True):
        label = (
            f"{nodes[int(tail_idx)]} → {nodes[int(head_idx)]}\n"
            f"[PageRank {pagerank[tail_idx]:.2e} → {pagerank[head_idx]:.2e}]"
        )
        labels.extend([label, label])

    return vertices, widths, colors, labels, n_edges


def build_legend_html(communities: np.ndarray, top_n: int = 14) -> str:
    """Compact HTML legend and method summary."""
    unique, counts = np.unique(communities, return_counts=True)
    order = np.argsort(-counts)
    unique = unique[order]
    counts = counts[order]
    n_comms = len(unique)
    lines = [
        '<div style="font-size:1.25vh;line-height:1.45;'
        "background:rgba(0,0,0,0.58);padding:0.7vh 0.9vh;"
        'border-radius:5px;max-width:30vh">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.45vh">'
        "PPI flow field</div>"
        '<div style="color:#ddd;margin-bottom:0.6vh">'
        "HuRI edges are oriented low→high PageRank. UMAP sees a signed "
        "adjacency flow profile; streamlines follow the smoothed edge field."
        "</div>"
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.25vh">'
        f"Top communities ({n_comms})</div>"
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


# -----------------------------------------------------------------------------
# Scene writing and pipeline orchestration
# -----------------------------------------------------------------------------


def write_scene(
    output_path: Path,
    nodes: list[str],
    node_df: pd.DataFrame,
    edges: pd.DataFrame,
    coords: np.ndarray,
    oriented: OrientedEdges,
    pagerank: np.ndarray,
    degrees: np.ndarray,
    communities: np.ndarray,
    flow: FlowField,
    streamline_data: StreamlineData,
    preset: FlowPreset,
) -> SceneStats:
    """Write the final Luxar scene."""
    n_comms = int(communities.max()) + 1
    palette = community_palette(n_comms)
    node_positions, node_colors, node_radii, node_labels = build_node_geometry(
        nodes, node_df, coords, communities, degrees, pagerank, palette
    )
    hub_positions, hub_colors, hub_radii, hub_labels = build_attractor_points(
        coords, nodes, pagerank, degrees
    )
    (
        edge_vertices,
        edge_widths,
        edge_colors,
        edge_labels,
        n_edge_lines,
    ) = build_oriented_edge_lines(nodes, coords, oriented, pagerank, preset)

    center = (flow.grid_min + flow.grid_max) * 0.5
    side = float(flow.grid_max[0] - flow.grid_min[0])
    camera_position = (
        float(center[0] + 1.35 * side),
        float(center[1] - 1.65 * side),
        float(center[2] + 1.05 * side),
    )

    with asection("Writing Luxar PPI flow scene"):
        dims = Dimensions(
            [
                Dimension(
                    "x",
                    unit="flow-UMAP",
                    range=(flow.grid_min[0], flow.grid_max[0]),
                    display=True,
                ),
                Dimension(
                    "y",
                    unit="flow-UMAP",
                    range=(flow.grid_min[1], flow.grid_max[1]),
                    display=True,
                ),
                Dimension(
                    "z",
                    unit="flow-UMAP",
                    range=(flow.grid_min[2], flow.grid_max[2]),
                    display=True,
                ),
            ]
        )
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=camera_position,
                target=tuple(float(v) for v in center),
                up=(0.0, 0.0, 1.0),
                fov=42.0,
                near=0.01,
                far=side * 12.0,
            ),
            background_color="#10141c",
            tone_mapping="ACES",
            # Clamp to the viewer-supported floor; remaining dimming is applied
            # via per-layer intensities below.
            exposure=VIEWER_EXPOSURE_EV,
            bloom_enabled=True,
            bloom_strength=0.78,
            bloom_radius=0.55,
            bloom_threshold=0.04,
            auto_rotate=True,
            auto_rotate_speed=0.18,
            dynamic_clipping_enabled=True,
            ui=UIConfig(show_layers=True),
            theme="dark",
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            scene.add_points(
                "Proteins (community color, PageRank radius)",
                positions=node_positions,
                colors=node_colors,
                radii=node_radii,
                sharpness=np.full(len(node_positions), 0.6, dtype=np.float32),
                opacity=0.94,
                intensity=0.30 * NODE_INTENSITY_SCALE,
                labels=node_labels,
                layer=True,
            )

            scene.add_points(
                "Top PageRank attractors",
                positions=hub_positions,
                colors=hub_colors,
                radii=hub_radii,
                sharpness=np.full(len(hub_positions), 0.5, dtype=np.float32),
                opacity=0.82,
                intensity=0.85 * NODE_INTENSITY_SCALE,
                blending_mode="additive",
                labels=hub_labels,
                layer=True,
            )

            if len(edge_vertices) > 0:
                scene.add_lines(
                    "PPI edges low→high PageRank (tapered, optional)",
                    vertices=edge_vertices,
                    widths=edge_widths,
                    colors=edge_colors,
                    sharpness=np.full(len(edge_vertices), 0.85, dtype=np.float32),
                    line_type="segments",
                    blending_mode="additive",
                    opacity=0.18,
                    intensity=0.38 * NODE_INTENSITY_SCALE,
                    labels=edge_labels,
                    layer=True,
                    visible=False,
                )

            if len(streamline_data.vertices) > 0:
                scene.add_lines(
                    "Advected protein streamlines",
                    vertices=streamline_data.vertices,
                    widths=preset.streamline_width,
                    colors=streamline_data.colors,
                    sharpness=0.50,
                    indices=streamline_data.segments.ravel(),
                    line_type="indexed",
                    blending_mode="additive",
                    opacity=0.35,
                    intensity=0.92 * NODE_INTENSITY_SCALE,
                    layer=True,
                )

            add_reference_cube_to_scene(
                scene,
                flow.grid_min,
                flow.grid_max,
                name="vector field bounding cube",
                widths=0.0063,
                opacity=0.18,
                intensity=0.35 * NODE_INTENSITY_SCALE,
            )

            scene.add_text(
                "HuRI PPI Flow Field",
                position=(0.02, 0.02),
                font_size=0.052,
                anchor="top-left",
                color="rgba(255,255,255,0.68)",
                blend_mode="difference",
            )
            scene.add_text(
                "{hover_label}",
                position=(0.02, 0.50),
                anchor="center-left",
                font_size=0.020,
                color="white",
                background="rgba(0,0,0,0.74)",
                padding=0.010,
                text_align="left",
                opacity=1.0,
                transition="fade",
                transition_duration=0.15,
                hover=True,
            )
            scene.add_html(
                build_legend_html(communities),
                position=(0.98, 0.50),
                anchor="center-right",
                opacity=0.92,
            )
            scene.add_text(
                f"{len(nodes):,} proteins · {len(edges):,} HuRI interactions · "
                f"{streamline_data.streamline_count:,} streamlines · "
                f"{preset.grid_size}³ field · {preset.name} preset",
                position=(0.98, 0.97),
                font_size=0.0125,
                anchor="bottom-right",
                color="rgba(220,220,220,0.52)",
            )

        aprint(f"  ✓ Scene written to {output_path}")

    return SceneStats(
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_edge_lines=n_edge_lines,
        n_streamlines=streamline_data.streamline_count,
        n_communities=n_comms,
        grid_size=preset.grid_size,
    )


def generate_ppi_flow_field_scene(
    output_path: Path,
    cache_dir: Path,
    preset: FlowPreset,
    recompute_layout: bool,
    recompute_field: bool,
    recompute_streamlines: bool,
    recompute_graph: bool,
    raw_edge_vectors: bool,
) -> SceneStats:
    """Run the full HuRI → signed UMAP → vector field → Luxar pipeline."""
    huri_path, hgnc_path = ensure_data(cache_dir)
    hgnc = load_hgnc(hgnc_path)
    edges_raw = load_huri_edges(huri_path, hgnc)
    nodes, node_df, edges = filter_to_lcc(edges_raw, hgnc)
    graph_hash = network_hash(nodes, edges)
    aprint(f"Graph cache hash: {graph_hash}")

    pagerank = compute_pagerank(
        nodes,
        edges,
        cache_dir / PAGERANK_CACHE_FILENAME,
        graph_hash,
        recompute_graph,
    )
    degrees = compute_degrees(nodes, edges)
    communities = compute_communities(
        nodes,
        edges,
        cache_dir / COMMUNITY_CACHE_FILENAME,
        graph_hash,
        recompute_graph,
    )
    oriented = orient_edges_by_pagerank(nodes, edges, pagerank)
    coords = compute_signed_flow_layout(
        nodes,
        oriented,
        cache_dir / LAYOUT_CACHE_FILENAME,
        graph_hash,
        recompute_layout,
    )

    vector_mode = "raw" if raw_edge_vectors else "unit"
    field_cache = (
        cache_dir
        / f"ppi_flow_field_{preset.name}_{preset.grid_size}_{vector_mode}_{CACHE_VERSION}.npz"
    )
    flow = compute_vector_field(
        coords,
        oriented,
        preset,
        field_cache,
        graph_hash,
        recompute_field,
        raw_edge_vectors,
    )

    stream_cache = (
        cache_dir / f"ppi_flow_streamlines_{preset.name}_{preset.grid_size}_"
        f"{preset.streamline_steps}_{vector_mode}_{CACHE_VERSION}.npz"
    )
    positions, valid, seed_indices = integrate_streamlines(
        coords,
        pagerank,
        flow,
        preset,
        stream_cache,
        graph_hash,
        recompute_streamlines,
    )
    streamline_data = build_streamline_geometry(
        positions,
        valid,
        seed_indices,
        communities,
        community_palette(int(communities.max()) + 1),
    )

    return write_scene(
        output_path,
        nodes,
        node_df,
        edges,
        coords,
        oriented,
        pagerank,
        degrees,
        communities,
        flow,
        streamline_data,
        preset,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options for the demo."""
    parser = argparse.ArgumentParser(
        description="HuRI signed-adjacency UMAP PPI flow-field Luxar demo."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="full",
        help="Numerical preset. Default full uses 256^3; preview uses 128^3.",
    )
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="Write the .zarr dataset without launching the viewer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .zarr path. Defaults to datasets/demos for --no-serve.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=CACHE_DIR,
        help="Cache directory for downloads, UMAP layouts, fields, and streamlines.",
    )
    parser.add_argument("--recompute-layout", action="store_true")
    parser.add_argument("--recompute-field", action="store_true")
    parser.add_argument("--recompute-streamlines", action="store_true")
    parser.add_argument(
        "--recompute-graph",
        action="store_true",
        help="Recompute PageRank and Louvain communities.",
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute graph analysis, UMAP layout, vector field, and streamlines.",
    )
    parser.add_argument(
        "--raw-edge-vectors",
        action="store_true",
        help="Blend raw UMAP edge displacement vectors instead of unit directions.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="Override preset grid size. Useful for quick smoke tests.",
    )
    parser.add_argument(
        "--streamline-steps",
        type=int,
        default=None,
        help="Override the number of RK4 advection steps.",
    )
    parser.add_argument(
        "--step-voxels",
        type=float,
        default=None,
        help="Override RK4 step length in vector-field voxels.",
    )
    parser.add_argument(
        "--nearest-samples",
        type=int,
        default=None,
        help="Override KD-tree nearest edge-sample count per grid cell.",
    )
    parser.add_argument(
        "--max-edge-lines",
        type=int,
        default=None,
        help="Override displayed PPI edge cap.",
    )
    parser.add_argument(
        "--max-streamline-seeds",
        type=int,
        default=None,
        help="Seed at most this many proteins (default: all HuRI LCC nodes).",
    )
    return parser.parse_args(argv)


def apply_overrides(preset: FlowPreset, args: argparse.Namespace) -> FlowPreset:
    """Apply CLI overrides to a preset dataclass."""
    updates: dict[str, Any] = {}
    if args.grid_size is not None:
        if args.grid_size < 8:
            raise ValueError("--grid-size must be at least 8")
        updates["grid_size"] = int(args.grid_size)
    if args.streamline_steps is not None:
        if args.streamline_steps < 1:
            raise ValueError("--streamline-steps must be positive")
        updates["streamline_steps"] = int(args.streamline_steps)
    if args.step_voxels is not None:
        if args.step_voxels <= 0:
            raise ValueError("--step-voxels must be positive")
        updates["step_voxels"] = float(args.step_voxels)
    if args.nearest_samples is not None:
        if args.nearest_samples < 1:
            raise ValueError("--nearest-samples must be positive")
        updates["nearest_samples"] = int(args.nearest_samples)
    if args.max_edge_lines is not None:
        if args.max_edge_lines < 0:
            raise ValueError("--max-edge-lines must be non-negative")
        updates["max_edge_lines"] = int(args.max_edge_lines)
    if args.max_streamline_seeds is not None:
        if args.max_streamline_seeds < 1:
            raise ValueError("--max-streamline-seeds must be positive")
        updates["max_streamline_seeds"] = int(args.max_streamline_seeds)
    if not updates:
        return preset
    return replace(preset, **updates)


def main() -> None:
    args = parse_args(sys.argv[1:])
    try:
        preset = apply_overrides(PRESETS[args.preset], args)
    except ValueError as exc:
        aprint(f"❌ {exc}")
        raise SystemExit(2) from exc

    recompute_all = bool(args.recompute_all)
    recompute_graph = bool(args.recompute_graph or recompute_all)
    recompute_layout = bool(args.recompute_layout or recompute_all)
    recompute_field = bool(args.recompute_field or recompute_all)
    recompute_streamlines = bool(args.recompute_streamlines or recompute_all)

    aprint("=" * 72)
    aprint("HuRI PPI Flow Field — signed adjacency UMAP + vector streamlines")
    aprint("=" * 72)
    aprint(
        f"Preset: {preset.name} · grid {preset.grid_size}^3 · "
        f"nearest samples {preset.nearest_samples} · "
        f"RK4 steps {preset.streamline_steps}"
    )
    aprint(
        "Edge orientation: low→high PageRank; UMAP features: "
        "signed sparse adjacency flow profiles"
    )
    aprint("")

    if args.output is not None:
        output_path = args.output.expanduser()
        stats = generate_ppi_flow_field_scene(
            output_path,
            args.cache_dir.expanduser(),
            preset,
            recompute_layout,
            recompute_field,
            recompute_streamlines,
            recompute_graph,
            args.raw_edge_vectors,
        )
        aprint(
            f"Generated {stats.n_nodes:,} proteins, {stats.n_streamlines:,} "
            f"streamlines → {output_path}"
        )
        if not args.no_serve:
            launch_viewer(output_path)
        return

    if args.no_serve:
        output_path = get_demos_output_dir() / f"ppi_flow_field_{preset.name}.zarr"
        stats = generate_ppi_flow_field_scene(
            output_path,
            args.cache_dir.expanduser(),
            preset,
            recompute_layout,
            recompute_field,
            recompute_streamlines,
            recompute_graph,
            args.raw_edge_vectors,
        )
        aprint(
            f"Dataset generated at {output_path} "
            f"({stats.grid_size}^3 field, {stats.n_streamlines:,} streamlines)"
        )
        return

    with tempfile.TemporaryDirectory(prefix="luxar_ppi_flow_") as tmpdir:
        output_path = Path(tmpdir) / f"ppi_flow_field_{preset.name}.zarr"
        stats = generate_ppi_flow_field_scene(
            output_path,
            args.cache_dir.expanduser(),
            preset,
            recompute_layout,
            recompute_field,
            recompute_streamlines,
            recompute_graph,
            args.raw_edge_vectors,
        )

        aprint("")
        aprint("=" * 72)
        aprint("NAVIGATION")
        aprint("=" * 72)
        aprint("  Proteins: colored by Louvain community, sized by PageRank")
        aprint("  Streamlines: forward advection through low→high PageRank edge field")
        aprint("  Toggle the optional tapered PPI edge layer in the Layers panel")
        aprint("  Hover proteins or optional edges for labels")
        aprint(
            f"  {stats.n_nodes:,} proteins · {stats.n_edges:,} interactions · "
            f"{stats.n_streamlines:,} streamlines · {stats.n_communities} communities"
        )
        aprint("")
        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
