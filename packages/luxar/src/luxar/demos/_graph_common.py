"""Shared graph-network pipeline for the three large-graph demos.

``demo_caida_as_topology``, ``demo_huri_interactome`` and
``demo_ppi_flow_field`` all build a 3-D picture of a big graph the same way:
stream a public archive into ``~/.cache/luxar/``, restrict the network to its
largest connected component, build a sparse adjacency, and detect Louvain
communities. Each demo carried its own copy of that machinery and the copies
had drifted apart — three download helpers with two different timeouts and two
different temp-file promotions, two byte-identical community legends, two
``load_hgnc``/``load_huri_edges`` pairs differing in ``asection`` titles and in
one error message. This module holds the one copy, so each demo file stays a
readable manifest of what makes *its* network interesting.

Not every helper is shared by all three demos. Only :func:`download_file` and
:func:`compute_communities` are called by all of them. :func:`build_adjacency`,
:func:`nodes_hash` and :func:`build_community_legend` are shared by
``demo_caida_as_topology`` and ``demo_huri_interactome`` alone —
``demo_ppi_flow_field`` builds its own signed matrix and keeps its own
``network_hash`` and ``build_legend_html``. The HuRI/HGNC loaders
(:func:`load_hgnc`, :func:`load_huri_edges`) and :func:`filter_to_lcc` are the
mirror image, used by the two protein demos only; ``demo_caida_as_topology``
calls none of them, because its own LCC filter also detects tier-1 ASes, sorts
on a numeric ASN key and reindexes an org/country frame, and returns a 4-tuple
— a deliberate non-dedup.

Where two copies disagreed on console wording the majority spelling won, so
``demo_ppi_flow_field``'s log lines shift slightly; where they disagreed on
behaviour the safer one won (see :func:`download_file`).

:func:`build_adjacency` and :func:`compute_communities` are the two helpers
that take their edge-column names as an explicit ``columns`` argument: CAIDA's
frame is keyed on ``asn_a``/``asn_b`` and the protein networks on
``sym_a``/``sym_b``, and there is no sensible default for both. The HuRI
loaders have no such knob — :func:`load_huri_edges` emits, and
:func:`filter_to_lcc` consumes, hardcoded ``sym_a``/``sym_b``, being
protein-specific by construction.

Not a demo itself (no ``demo_`` prefix), so the demo import smoke test skips it.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from arbol import aprint, asection

from luxar.utils._umap_utils import get_categorical_color

from ._dependencies import require_module

# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------


def download_file(
    url: str, dest: Path, description: str, *, timeout: float = 180.0
) -> None:
    """Stream a URL to ``dest``, with progress. No-op if the file exists.

    The default ``timeout`` is sized for the slowest archive any of these demos
    fetches (the CAIDA snapshots); it is a per-read socket timeout, not a
    deadline for the whole transfer, so a generous value costs nothing on a
    healthy connection. The two protein demos previously used 120 s and now
    inherit this one.

    A failed transfer leaves nothing behind, and never disturbs a concurrent
    one: the body streams into a private per-invocation directory created
    beside ``dest``, promoted over ``dest`` only once the stream completes and
    removed whole otherwise. A fixed ``<dest>.part`` sibling would not do,
    because these demos share one cache directory — the two protein demos both
    default to ``~/.cache/luxar/huri/`` with the same filenames, so two
    overlapping runs would compute the same staging path, truncate each other's
    in-flight file, and then unlink it out from under the other's promotion.
    Staging inside ``dest.parent`` keeps that promotion a same-filesystem
    rename.

    Not :func:`luxar.utils.demos.cached_download`, which is the richer
    downloader (retry, resume, checksum, quarantine) but owns its destination:
    it always writes ``~/.cache/luxar/<name>/<filename>``. All three demos here
    accept a ``--cache-dir`` override and pass an explicit ``dest``, so they
    need a downloader that takes the path rather than deriving it.
    """
    if dest.exists() and dest.stat().st_size > 0:
        size_mb = dest.stat().st_size / (1024 * 1024)
        aprint(f"  Using cached {dest.name} ({size_mb:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Downloading {description}")
    aprint(f"    URL: {url}")

    staging = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
    tmp = staging / "part"
    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
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
                                f"{total / (1024 * 1024):,.0f} MB  ({pct:.0f}%)"
                            )
                            last_pct = pct
        # ``replace``, not ``rename``: on Windows, rename raises FileExistsError
        # when the destination is already there, so a re-fetch over a truncated
        # cache file would fail. On POSIX the two are the same call.
        tmp.replace(dest)
    finally:
        # A truncated body left in the cache directory outlives the run that
        # made it and is easy to mistake for a real artifact, so drop the
        # staging directory whatever happened — and only *this* call's
        # directory, so a concurrent run's in-flight file is never touched.
        # ``finally`` rather than an ``except BaseException``, so a Ctrl-C
        # cleans up too.
        shutil.rmtree(staging, ignore_errors=True)
    size_mb = dest.stat().st_size / (1024 * 1024)
    aprint(f"  ✓ Saved {dest.name} ({size_mb:.1f} MB)")


# -----------------------------------------------------------------------------
# HuRI / HGNC loading
# -----------------------------------------------------------------------------


def load_hgnc(tsv_path: Path) -> pd.DataFrame:
    """Return a frame indexed by Ensembl gene id with (symbol, chromosome)."""
    wanted = {"symbol", "ensembl_gene_id", "location", "status"}
    with asection("Loading HGNC (Ensembl → symbol)"):
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
        # location like "17q21.31" or "Xp22.2" → chromosome prefix. The
        # missing-column fallback must carry ``df.index``: the frame has been
        # filtered by now, so a fresh RangeIndex would align by label and inject
        # NaN for every surviving row whose label falls outside it.
        df["chromosome"] = (
            df.get("location", pd.Series("", index=df.index))
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
    """Return the undirected, deduplicated (sym_a, sym_b) edge frame."""
    with asection("Loading HuRI interactions"):
        df = pd.read_csv(tsv_path, sep="\t", header=None, dtype=str, low_memory=False)
        if df.shape[1] < 2:
            raise RuntimeError(f"Unexpected HuRI format (got {df.shape[1]} columns)")
        df = df.iloc[:, :2].copy()
        df.columns = ["ensg_a", "ensg_b"]
        # Filter to ENSG-looking rows (handles presence or absence of header)
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

        # Canonicalize so (A, B) == (B, A) then drop duplicates
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
    """Restrict to the largest connected component and attach chromosome info.

    Returns ``(nodes, node_df, edges)``: the LCC's gene symbols in sorted
    (deterministic → reproducible layout) order, a ``[symbol, chromosome]``
    frame aligned with them, and the subset of ``edges`` whose endpoints both
    survive. A symbol HGNC does not place gets chromosome ``"?"``.
    """
    nx = require_module("networkx")

    with asection("Filtering to largest connected component"):
        g = nx.Graph()
        g.add_edges_from(
            zip(edges["sym_a"].tolist(), edges["sym_b"].tolist(), strict=True)
        )
        components = sorted(nx.connected_components(g), key=len, reverse=True)
        lcc = components[0]
        aprint(
            f"  {len(components):,} components; LCC has "
            f"{len(lcc):,} nodes (dropped {g.number_of_nodes() - len(lcc):,})"
        )

        lcc_set = set(lcc)
        mask = edges["sym_a"].isin(lcc_set) & edges["sym_b"].isin(lcc_set)
        edges = edges[mask].reset_index(drop=True)
        aprint(f"  {len(edges):,} edges within the LCC")

        nodes = sorted(lcc_set)  # deterministic order → reproducible layout
        sym_to_chrom = hgnc.groupby("symbol")["chromosome"].first().to_dict()
        node_df = pd.DataFrame(
            {
                "symbol": nodes,
                "chromosome": [sym_to_chrom.get(s, "?") for s in nodes],
            }
        )
    return nodes, node_df, edges


# -----------------------------------------------------------------------------
# Graph structure
# -----------------------------------------------------------------------------


def build_adjacency(
    nodes: list[str], edges: pd.DataFrame, *, columns: tuple[str, str]
) -> Any:
    """Return a symmetric CSR adjacency matrix, one entry per undirected edge.

    ``columns`` names the two endpoint columns of ``edges`` and is deliberately
    required: the callers disagree (``("sym_a", "sym_b")`` for the protein
    networks, ``("asn_a", "asn_b")`` for the AS graph), and a default would make
    one call site's column naming look like the universal one.
    """
    from scipy.sparse import csr_matrix

    col_a, col_b = columns
    idx = {s: i for i, s in enumerate(nodes)}
    rows = np.fromiter((idx[s] for s in edges[col_a]), dtype=np.int32)
    cols = np.fromiter((idx[s] for s in edges[col_b]), dtype=np.int32)
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


def nodes_hash(nodes: list[str]) -> str:
    """Deterministic hash of the node list for layout cache invalidation."""
    return hashlib.sha256("\n".join(nodes).encode("utf-8")).hexdigest()[:16]


def compute_communities(
    nodes: list[str],
    edges: pd.DataFrame,
    *,
    columns: tuple[str, str],
    unit_label: str = "nodes",
    summary_suffix: str = " detected",
) -> np.ndarray:
    """Louvain community assignment per node. Communities sorted by size desc.

    ``columns`` names the two endpoint columns of ``edges`` (required, for the
    reason given on :func:`build_adjacency`). ``unit_label`` and
    ``summary_suffix`` only shape the console lines, so each demo keeps the
    wording it had: the AS-graph demo counts "ASes" and prints a bare
    "N communities".

    The Louvain seed is pinned so the demo's colors are reproducible.
    """
    nx = require_module("networkx")
    col_a, col_b = columns

    with asection("Detecting communities (Louvain)"):
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(zip(edges[col_a].tolist(), edges[col_b].tolist(), strict=True))
        partitions = nx.community.louvain_communities(g, seed=42)
        partitions = sorted(partitions, key=len, reverse=True)
        aprint(f"  {len(partitions):,} communities{summary_suffix}")
        for i, p in enumerate(partitions[:5]):
            aprint(f"    #{i}: {len(p):,} {unit_label}")
        if len(partitions) > 5:
            aprint(f"    ... + {len(partitions) - 5} smaller")

        idx = {s: i for i, s in enumerate(nodes)}
        comms = np.full(len(nodes), -1, dtype=np.int32)
        for comm_id, members in enumerate(partitions):
            for node in members:
                comms[idx[node]] = comm_id
    return comms


# -----------------------------------------------------------------------------
# Legend
# -----------------------------------------------------------------------------


def build_community_legend(communities: np.ndarray, top_n: int = 20) -> str:
    """Top-N communities by size, sorted descending."""
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
