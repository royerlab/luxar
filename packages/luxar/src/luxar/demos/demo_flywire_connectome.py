#!/usr/bin/env python3
"""Self-Contained Demo: FlyWire — the adult Drosophila brain connectome.

Visualize the first complete wiring diagram of an adult fruit fly brain:
~139k proofread neurons rendered as Points at their soma positions, and
millions of neuron-to-neuron connections rendered as Lines, all inside the
physical bounding box of the actual brain (coordinates in µm).

================================================================================
WHAT IS A CONNECTOME?
================================================================================

A *connectome* is a complete map of every neuron in a nervous system together
with every synaptic connection between them. FlyWire (Dorkenwald et al. 2024,
Nature) is the first such map for an adult animal brain: a female Drosophila
melanogaster, reconstructed from serial-section electron microscopy by a
consortium of 127 institutions and thousands of citizen-scientist proofreaders.

DATASET AT A GLANCE
-------------------
- ~139,000 proofread neurons (soma positions, cell types, neurotransmitter
  predictions, left/right/center laterality)
- ~54.5 million chemical synapses
- ~2-3 million neuron-to-neuron connections (after aggregating synapses per
  (pre, post) pair)
- Coordinate space: 4 × 4 × 40 nm voxels → physical brain is roughly
  700 × 350 × 250 µm

COLOR / STRUCTURE
-----------------
Neurons are colored by **super_class** — the top level of the FlyWire
taxonomy:

    optic                : ~70k neurons, the two optic lobes (vision)
    visual_projection_*  : optic → central brain relays
    central_brain        : the protocerebrum, mushroom body, central complex …
    sensory              : olfactory, mechanosensory, gustatory inputs
    descending           : brain → ventral nerve cord (motor command)
    ascending            : ventral nerve cord → brain
    motor                : direct output to muscles
    endocrine            : neurosecretory cells

Edges are drawn as straight line segments between the two somas, filtered
by minimum synapse count (default 20) and capped at the top N strongest
connections. Edges are split by **presynaptic neurotransmitter** into one
Lines layer per NT (acetylcholine, GABA, glutamate, dopamine, serotonin,
octopamine) so you can toggle excitatory, inhibitory, or neuromodulatory
circuits independently from the viewer's Layers panel.

NAVIGATION
----------
- Rotate to see left vs. right hemispheres and the large optic lobes on both
  flanks. The dense "ball" in the middle is the central brain.
- Open the Layers panel to toggle each NT individually — watch the ACh
  (excitatory) layer light up the full brain, then isolate the GABA
  (inhibitory) or dopamine subsets to see modulatory fiber systems.
- Hover any neuron to see its cell type, super-class, neurotransmitter, and
  side.

DATA SOURCES
------------
- Annotations (nodes): Schlegel et al. 2024 Nature, supplemental files:
    https://github.com/flyconnectome/flywire_annotations
- Connectivity (edges): FlyWire public release 783, Dorkenwald et al.:
    https://zenodo.org/records/10676866
- Portal:
    https://codex.flywire.ai/

The first run downloads ~850 MB of edge data to
``~/.cache/luxar/flywire/``. Subsequent runs reuse the cache.

Usage:
    python -m luxar.demos.demo_flywire_connectome
    python -m luxar.demos.demo_flywire_connectome --no-serve
    python -m luxar.demos.demo_flywire_connectome --min-synapses 50
    python -m luxar.demos.demo_flywire_connectome --max-edges 100000
"""

from __future__ import annotations

DEMO_META = {
    "key": "flywire_connectome",
    "title": "FlyWire Connectome",
    "description": "Adult Drosophila brain connectome: ~139k neurons as Points, ~2M connections as Lines.",
    "category": "connectome",
    "geometry": "points+lines",
    "requirements": {
        "download_mb": 850,
        "compute": "heavy",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["flywire"],
    "outputs": ["flywire_connectome"],
    "citation": {
        "short": "FlyWire Consortium 2024; Schlegel et al. 2024 annotations",
        "ref": "FlyWire Consortium 2024",
        "doi": "10.5281/zenodo.10676866",
        "license": "CC BY 4.0",
    },
}

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import ViewerConfig
from luxar.demos import (
    add_demo_caption,
    cached_download,
    launch_viewer,
    parse_int_arg,
    parse_path_arg,
)
from luxar.demos._support._umap_utils import format_label, get_categorical_color
from luxar.utils.paths import get_demos_output_dir

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "luxar" / "flywire"

ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations"
    "/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
ANNOTATIONS_FILE = CACHE_DIR / "Supplemental_file1_neuron_annotations.tsv"

CONNECTIONS_URL = (
    "https://zenodo.org/records/10676866/files/proofread_connections_783.feather"
    "?download=1"
)
CONNECTIONS_FILE = CACHE_DIR / "proofread_connections_783.feather"

# Voxel resolution from the FAFB / FlyWire segmentation (XYZ in nm)
VOXEL_NM = np.array([4.0, 4.0, 40.0], dtype=np.float32)
NM_PER_UM = 1000.0

# Default filters — balance visual impact vs. browser GPU limits
DEFAULT_MIN_SYNAPSES = 20
DEFAULT_MAX_EDGES = 300_000

# Neurotransmitter palette: distinctive, roughly matches community conventions
NT_COLORS: dict[str, tuple[float, float, float]] = {
    "acetylcholine": (1.00, 0.75, 0.25),  # warm amber (excitatory)
    "gaba": (0.35, 0.75, 1.00),  # blue (inhibitory)
    "glutamate": (1.00, 0.35, 0.85),  # magenta
    "dopamine": (0.40, 0.95, 0.55),  # green
    "serotonin": (1.00, 0.35, 0.35),  # red
    "octopamine": (0.75, 0.45, 1.00),  # purple
}
NT_UNKNOWN = (0.65, 0.65, 0.65)

EDGE_UNKNOWN_NT = (0.55, 0.55, 0.55)


# -----------------------------------------------------------------------------
# Download / cache
# -----------------------------------------------------------------------------


def _download(url: str, dest: Path, description: str) -> None:
    """Stream a URL to ``dest``, printing progress. Skips if file exists."""
    if dest.exists():
        size_mb = dest.stat().st_size / (1024 * 1024)
        aprint(f"  Using cached {dest.name} ({size_mb:.1f} MB)")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    aprint(f"  Downloading {description}")
    aprint(f"    URL: {url}")

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        written = 0
        last_reported = 0.0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MiB
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if total > 0:
                    pct = 100.0 * written / total
                    if pct - last_reported >= 10.0:
                        aprint(
                            f"    {written / (1024 * 1024):,.0f} / "
                            f"{total / (1024 * 1024):,.0f} MB  "
                            f"({min(pct, 100):.0f}%)"
                        )
                        last_reported = pct
                else:
                    mb = written / (1024 * 1024)
                    if mb - last_reported >= 20.0:
                        aprint(f"    {mb:,.0f} MB")
                        last_reported = mb
    tmp.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    aprint(f"  ✓ Saved {dest.name} ({size_mb:.1f} MB)")


def ensure_data(cache_dir: Path) -> tuple[Path, Path]:
    """Download FlyWire annotations + connections if not already cached."""
    global ANNOTATIONS_FILE, CONNECTIONS_FILE
    ANNOTATIONS_FILE = cache_dir / ANNOTATIONS_FILE.name
    connections_name = CONNECTIONS_FILE.name

    with asection("Fetching FlyWire data"):
        _download(
            ANNOTATIONS_URL,
            ANNOTATIONS_FILE,
            "neuron annotations (Schlegel et al. 2024, ~a few MB)",
        )
        # Route the large Zenodo connectivity download through the shared cache
        # helper (retry / resume / skip-if-present) under ~/.cache/luxar/flywire/.
        CONNECTIONS_FILE = cached_download(
            CONNECTIONS_URL,
            "flywire",
            connections_name,
        )
    return ANNOTATIONS_FILE, CONNECTIONS_FILE


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_neurons(tsv_path: Path) -> pd.DataFrame:
    """Read the annotation TSV and return a cleaned, position-enriched frame.

    Prefers soma coordinates; falls back to the backbone anchor for neurons
    without a localized soma (e.g. ascending / sensory fibers).
    """
    with asection("Loading neuron annotations"):
        # Low-memory dtype hints for the big id column; everything else left as
        # object so we can tolerate missing values cleanly.
        df = pd.read_csv(
            tsv_path,
            sep="\t",
            dtype={"root_id": "Int64"},
            low_memory=False,
        )
        aprint(f"  {len(df):,} rows loaded")

        # Prefer soma; fall back to pos_* for neurons whose soma wasn't localized
        soma = df[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float32)
        pos = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=np.float32)
        has_soma = np.isfinite(soma).all(axis=1) & (soma != 0).any(axis=1)
        combined = np.where(has_soma[:, None], soma, pos)
        finite = np.isfinite(combined).all(axis=1)

        df = df.loc[finite].copy()
        combined = combined[finite]

        # 4×4×40 nm voxels → micrometers, centered on the brain's bounding box
        xyz_um = combined * VOXEL_NM / NM_PER_UM
        center = 0.5 * (xyz_um.min(axis=0) + xyz_um.max(axis=0))
        xyz_um -= center

        df["x_um"] = xyz_um[:, 0]
        df["y_um"] = xyz_um[:, 1]
        df["z_um"] = xyz_um[:, 2]

        # Fill categorical columns
        for col in ("super_class", "cell_class", "cell_type", "side", "top_nt"):
            if col not in df.columns:
                df[col] = "unknown"
            df[col] = df[col].fillna("unknown").astype(str).str.lower()

        aprint(f"  {has_soma.sum():,} with localized soma; rest use backbone anchor")
        extent = np.ptp(xyz_um, axis=0)
        aprint(
            f"  Extent (µm): x={extent[0]:.0f}, y={extent[1]:.0f}, z={extent[2]:.0f}"
        )
        aprint(
            f"  super_class counts (top 8): "
            f"{dict(df['super_class'].value_counts().head(8))}"
        )
    return df


# Map from 783-release NT-average column names → our canonical NT labels.
NT_AVG_COLS: dict[str, str] = {
    "gaba_avg": "gaba",
    "ach_avg": "acetylcholine",
    "glut_avg": "glutamate",
    "oct_avg": "octopamine",
    "ser_avg": "serotonin",
    "da_avg": "dopamine",
}


def _find_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    """Return the first matching column name; raise if none found."""
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"None of {candidates} present in feather columns: {list(df.columns)}"
    )


def load_edges(
    feather_path: Path,
    neurons: pd.DataFrame,
    min_synapses: int,
    max_edges: int,
) -> pd.DataFrame:
    """Load the connectivity table and return a filtered per-pair edge frame.

    The Zenodo file stores one row per (pre, post, neuropil) triplet with a
    ``syn_count`` and six per-neurotransmitter probability columns
    (``gaba_avg``, ``ach_avg``, ``glut_avg``, ``oct_avg``, ``ser_avg``,
    ``da_avg``). We:

      1. Sum ``syn_count`` across neuropils.
      2. Take the synapse-count-weighted average of each NT probability and
         label the pair by the argmax across the six NTs.
      3. Keep only pairs with both endpoints in ``neurons`` and ``syn_count >=
         min_synapses``, capped at ``max_edges`` strongest pairs.
    """
    with asection("Loading + filtering connectivity"):
        df = pd.read_feather(feather_path)
        aprint(f"  {len(df):,} (pre, post, neuropil) rows loaded")
        aprint(f"  columns: {list(df.columns)}")

        pre_col = _find_col(df, ("pre_pt_root_id", "pre_root_id", "pre_root"))
        post_col = _find_col(df, ("post_pt_root_id", "post_root_id", "post_root"))
        syn_col = _find_col(df, ("syn_count", "n_syn", "synapse_count"))
        nt_avg_present = [c for c in NT_AVG_COLS if c in df.columns]

        # Weight each NT average by syn_count so per-pair means are synapse-
        # weighted across neuropils, not neuropil-weighted.
        for c in nt_avg_present:
            df[c] = df[c] * df[syn_col]

        group_cols = [pre_col, post_col]
        agg: dict[str, str] = {syn_col: "sum"}
        for c in nt_avg_present:
            agg[c] = "sum"
        pair = df.groupby(group_cols, sort=False, as_index=False).agg(agg)
        aprint(f"  {len(pair):,} unique (pre, post) pairs")

        # Divide through to recover synapse-weighted NT probabilities
        for c in nt_avg_present:
            pair[c] = pair[c] / pair[syn_col].clip(lower=1)

        # Argmax across NT columns → dominant presynaptic NT
        if nt_avg_present:
            nt_matrix = pair[nt_avg_present].to_numpy(dtype=np.float32)
            nt_names = np.array([NT_AVG_COLS[c] for c in nt_avg_present], dtype=object)
            pair["nt"] = nt_names[nt_matrix.argmax(axis=1)]
            pair = pair.drop(columns=nt_avg_present)
        else:
            pair["nt"] = "unknown"

        before = len(pair)
        pair = pair[pair[syn_col] >= min_synapses]
        aprint(
            f"  {len(pair):,} pairs at ≥{min_synapses} synapses "
            f"(culled {before - len(pair):,})"
        )

        # Keep only edges where both endpoints have positions in our frame
        have_pos = pd.Index(neurons["root_id"].astype("int64").to_numpy())
        pair = pair[
            pair[pre_col].astype("int64").isin(have_pos)
            & pair[post_col].astype("int64").isin(have_pos)
        ]
        aprint(f"  {len(pair):,} pairs with both endpoints located")

        if len(pair) > max_edges:
            pair = pair.nlargest(max_edges, syn_col)
            aprint(f"  ↳ capped at top {max_edges:,} by synapse count")

        pair = pair.rename(
            columns={pre_col: "pre_root", post_col: "post_root", syn_col: "syn_count"}
        )
        pair = pair.reset_index(drop=True)
        pair["nt"] = pair["nt"].fillna("unknown").astype(str)

        nt_summary = pair["nt"].value_counts().to_dict()
        aprint(f"  Dominant NT per edge: {nt_summary}")
    return pair[["pre_root", "post_root", "syn_count", "nt"]]


# -----------------------------------------------------------------------------
# Geometry builders
# -----------------------------------------------------------------------------


# Real Drosophila soma diameters are ~3-8 µm. Optic somas are small and
# densely packed; descending / ascending / motor are larger projection cells.
_SOMA_RADIUS_UM: dict[str, float] = {
    "optic": 2.5,
    "visual_projection": 2.5,
    "visual_centrifugal": 2.5,
    "descending": 5.5,
    "ascending": 5.5,
    "motor": 5.5,
}
_SOMA_RADIUS_DEFAULT = 4.0


def build_neuron_layer(
    sub: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None]:
    """Assemble Points attributes for one super_class subset.

    Returns:
        positions (N, 3) as [x, y, z]
        radii (N,) in µm
        labels (list of N hover strings)
        keys (list of N FlyWire root ids for the Codex link, or None when the
            frame carries no ``root_id`` column)
    """
    positions = np.column_stack(
        [
            sub["x_um"].to_numpy(dtype=np.float32),
            sub["y_um"].to_numpy(dtype=np.float32),
            sub["z_um"].to_numpy(dtype=np.float32),
        ]
    )

    sc = sub["super_class"].iloc[0] if len(sub) else ""
    radius = _SOMA_RADIUS_UM.get(sc, _SOMA_RADIUS_DEFAULT)
    radii = np.full(len(sub), radius, dtype=np.float32)

    labels = [
        f"{format_label(ct) or '(unnamed)'}\n[{scn} · {nt} · {sd}]"
        for ct, scn, nt, sd in zip(
            sub["cell_type"].tolist(),
            sub["super_class"].tolist(),
            sub["top_nt"].tolist(),
            sub["side"].tolist(),
            strict=True,
        )
    ]

    # Click a soma to open its FlyWire Codex cell page, right-click to copy the
    # root id (#1917). The id is a 19-digit segment id that appears nowhere in
    # the label — the label is the cell type and its classifications — so the
    # URL has to come from `keys=`, which is what that channel is for.
    #
    # `root_id` is a nullable Int64 column; a missing one becomes an empty key
    # rather than the string "<NA>", and the viewer then suppresses the link for
    # that neuron instead of opening a Codex page for a nonexistent cell.
    # `None`, not `[]`, when the column is absent: an empty list would be a
    # length mismatch against N points, so the caller omits `keys=` entirely and
    # the layer ships without links rather than failing to build.
    keys = (
        ["" if pd.isna(rid) else str(int(rid)) for rid in sub["root_id"].tolist()]
        if "root_id" in sub.columns
        else None
    )

    return positions, radii, labels, keys


def build_edge_lines(
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
    log_syn_max: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble the Lines attributes for one edge subset (e.g. one NT).

    Each edge is one two-vertex segment (line_type='segments'). Width encodes
    log(synapse_count), normalized against ``log_syn_max`` (usually the max
    from the *full* edge set, so per-NT layers stay mutually comparable in
    thickness). Direction is conveyed by the width taper: thick at pre, thin
    at post.

    Returns:
        vertices (2E, 3) as [x, y, z]
        widths (2E,)
        labels (list of 2E strings; same label repeated on both endpoints)
    """
    id_to_idx = pd.Series(
        np.arange(len(neurons), dtype=np.int64),
        index=neurons["root_id"].astype("int64").to_numpy(),
    )
    pre_idx = id_to_idx.reindex(edges["pre_root"].astype("int64").to_numpy()).to_numpy()
    post_idx = id_to_idx.reindex(
        edges["post_root"].astype("int64").to_numpy()
    ).to_numpy()

    xs = neurons["x_um"].to_numpy(dtype=np.float32)
    ys = neurons["y_um"].to_numpy(dtype=np.float32)
    zs = neurons["z_um"].to_numpy(dtype=np.float32)

    n_edges = len(edges)
    verts = np.empty((n_edges * 2, 3), dtype=np.float32)
    verts[0::2, 0] = xs[pre_idx]
    verts[0::2, 1] = ys[pre_idx]
    verts[0::2, 2] = zs[pre_idx]
    verts[1::2, 0] = xs[post_idx]
    verts[1::2, 1] = ys[post_idx]
    verts[1::2, 2] = zs[post_idx]

    syn = edges["syn_count"].to_numpy(dtype=np.float32)
    log_syn = np.log1p(syn) / max(log_syn_max, 1.0)
    # Widths in µm — thick enough to be visible at full-brain zoom (~800 µm
    # extent) but still thinner than soma radii so neurons remain the visual
    # primary, not the connections.
    base_w = (0.4 + 1.8 * log_syn).astype(np.float32)

    widths = np.empty(n_edges * 2, dtype=np.float32)
    widths[0::2] = base_w  # thicker at presynaptic cell
    widths[1::2] = base_w * 0.35  # tapers toward postsynaptic cell

    pre_types = neurons["cell_type"].to_numpy()[pre_idx]
    post_types = neurons["cell_type"].to_numpy()[post_idx]
    nt_vals = edges["nt"].to_numpy()
    labels: list[str] = []
    for pt, pot, count, nt in zip(pre_types, post_types, syn, nt_vals, strict=True):
        label = (
            f"{format_label(str(pt)) or '(pre)'} → "
            f"{format_label(str(pot)) or '(post)'}\n"
            f"[{int(count)} synapses · {nt}]"
        )
        labels.extend([label, label])

    return verts, widths, labels


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------


def _build_super_class_legend(
    categories: list[str],
    counts: dict[str, int],
    palette: dict[str, tuple[float, ...]],
) -> str:
    """HTML legend — one row per super-class with its colored square."""
    rows = [
        '<div style="font-size:1.3vh;line-height:1.55;'
        "background:rgba(0,0,0,0.55);padding:0.6vh 0.8vh;"
        'border-radius:4px;max-height:88vh;overflow-y:auto">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.4vh">'
        "super_class</div>"
    ]
    for name in categories:
        r, g, b = (int(c * 255) for c in palette[name])
        n = counts.get(name, 0)
        rows.append(
            f'<div style="white-space:nowrap">'
            f'<span style="color:rgb({r},{g},{b})">█</span> '
            f"{format_label(name) or name} ({n:,})</div>"
        )
    rows.append("</div>")
    return "".join(rows)


def _build_nt_legend() -> str:
    """HTML legend for the edge neurotransmitter palette."""
    rows = [
        '<div style="font-size:1.3vh;line-height:1.55;'
        "background:rgba(0,0,0,0.55);padding:0.6vh 0.8vh;"
        'border-radius:4px">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.4vh">'
        "edge NT (presynaptic)</div>"
    ]
    for name, (r, g, b) in NT_COLORS.items():
        rr, gg, bb = int(r * 255), int(g * 255), int(b * 255)
        rows.append(
            f'<div style="white-space:nowrap">'
            f'<span style="color:rgb({rr},{gg},{bb})">━</span> '
            f"{name}</div>"
        )
    rows.append("</div>")
    return "".join(rows)


def _build_legends_row(
    categories: list[str],
    counts: dict[str, int],
    palette: dict[str, tuple[float, ...]],
) -> str:
    """Both legends in one bottom-left block, side by side.

    One overlay rather than two positioned ones: the super-class legend's width
    depends on its longest label and its count digits, so any x-offset chosen
    for a second overlay would be a guess that breaks the moment a class is
    renamed or the neuron counts gain a digit. A flex row makes adjacency the
    layout's job. ``flex-end`` bottom-aligns them, so the two boxes share a
    baseline even though the super-class list is much taller.
    """
    return (
        '<div style="display:flex;align-items:flex-end;gap:0.8vh">'
        f"{_build_super_class_legend(categories, counts, palette)}"
        f"{_build_nt_legend()}"
        "</div>"
    )


def build_scene(
    output_path: Path,
    neurons: pd.DataFrame,
    edges: pd.DataFrame,
) -> tuple[int, int]:
    """Build the connectome scene (Points per super-class, Lines per NT).

    Returns ``(n_neurons, n_edges)``.
    """
    # Normalize edge widths against the *global* max synapse count so per-NT
    # layers share a consistent thickness scale.
    log_syn_max = (
        float(np.log1p(edges["syn_count"].to_numpy()).max()) if len(edges) else 1.0
    )

    # Stable super_class ordering by neuron count (largest first) — used both
    # for the legend palette and for layer order in the viewer.
    sc_counts = neurons["super_class"].value_counts().to_dict()
    sc_order = sorted(sc_counts, key=lambda s: -sc_counts[s])
    sc_palette = {
        name: tuple(c / 255.0 for c in get_categorical_color(i, len(sc_order)))
        for i, name in enumerate(sc_order)
    }

    # Order NT layers by edge count (most common first) so the viewer's
    # Layers panel reads as a natural biological hierarchy.
    nt_counts = edges["nt"].value_counts().to_dict()
    nt_order = [nt for nt in NT_COLORS if nt in nt_counts] + [
        nt for nt in nt_counts if nt not in NT_COLORS
    ]
    nt_order.sort(key=lambda n: -nt_counts.get(n, 0))

    with asection("Building Luxar scene"):
        dims = Dimensions(
            [
                Dimension("x", unit="µm", display=True),
                Dimension("y", unit="µm", display=True),
                Dimension("z", unit="µm", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(
                dimensions=dims,
                # ACES (the house recommendation), now that the A/B render the
                # old Neutral exception asked for has been done (#1459): with
                # the connection glow left faint (see below), ACES costs about
                # half a stop — the gallery still goes from 1.9% of its lit
                # pixels blown to 5.4% at the same exposure, and back to 0.9%
                # at the manifest's re-tuned -1.5 stops.
                viewer_config=ViewerConfig(
                    # Thin lines lose detail at CSS resolution; see ViewerConfig.allow_high_dpr.
                    allow_high_dpr=True,
                    cinematic_mode=True,
                    tone_mapping="ACES",
                ),
                citation=DEMO_META["citation"],
            )

            # One toggleable Points layer per super_class — optic, central,
            # sensory, descending etc. each get their own checkbox in the
            # viewer's Layers panel.
            for sc_name in sc_order:
                sub = neurons[neurons["super_class"] == sc_name]
                if len(sub) == 0:
                    continue
                pos, radii, labels, keys = build_neuron_layer(sub)
                link_attrs = (
                    {
                        "keys": keys,
                        "link": (
                            "https://codex.flywire.ai/app/cell_details"
                            "?root_id={hover_key}"
                        ),
                        "copy": "{hover_key}",
                    }
                    if keys is not None
                    else {}
                )
                scene.add_points(
                    f"Neurons — {sc_name}",
                    positions=pos,
                    colors=sc_palette[sc_name],
                    radii=radii,
                    sharpness=np.full(len(pos), 0.55, dtype=np.float32),
                    opacity=0.79,
                    # Was 0.1, i.e. a display window opening at 1/0.1 = 10.
                    # 139K cell bodies summed into a wash that blew out the
                    # centre of the brain; the fix is exposure, not fewer
                    # neurons. The display-range slider IS this attr — the
                    # viewer recovers its window as `(1-offset)/intensity` and
                    # only uses the data range when intensity is exactly 1.0
                    # (`computeDisplayRange` / viewer `layer-state.ts`) — so a
                    # window top of 270.91 is authored as its reciprocal.
                    intensity=1.0 / 270.91,
                    # Additive rather than luminous: with the exposure this far
                    # down the per-point falloff luminous adds buys nothing
                    # visible, and additive is order-independent, so the
                    # super-class layers composite the same however they are
                    # toggled.
                    blending_mode="additive",
                    labels=labels,
                    **link_attrs,
                    layer=True,
                )

            # One toggleable Lines layer per presynaptic neurotransmitter.
            # GABA is inhibitory; ACh, glutamate are excitatory; DA / 5-HT /
            # OA are neuromodulatory — users can isolate any subset in the
            # viewer's Layers panel.
            for nt in nt_order:
                sub = edges[edges["nt"] == nt]
                if len(sub) == 0:
                    continue
                nt_verts, nt_widths, nt_labels = build_edge_lines(
                    neurons, sub, log_syn_max=log_syn_max
                )
                nt_rgb = NT_COLORS.get(nt, EDGE_UNKNOWN_NT)
                scene.add_lines(
                    f"Connections — {nt}",
                    vertices=nt_verts,
                    widths=nt_widths,
                    colors=tuple(nt_rgb),
                    sharpness=np.full(len(nt_verts), 0.85, dtype=np.float32),
                    line_type="segments",
                    blending_mode="luminous",
                    # Very faint, and measured: 300K luminous connection lines
                    # accumulate into a white wash that hides the (beautifully
                    # colored) neuron cell bodies. #1459 asked for 0.79 here to
                    # match the other layers; the A/B render refused it — at
                    # 0.79 the gallery still blows 78% of its lit pixels
                    # against 1.9% here, and even handing the harness 2.5 extra
                    # stops of headroom leaves 17% blown and the super-class
                    # palette gone. So the connections stay a subtle connective
                    # glow and the neurons dominate the view; a brighter one
                    # needs fewer edges or a per-layer intensity rebalance, not
                    # an exposure knob.
                    #
                    # Re-measured on the hosted demo 2026-09-10: at
                    # opacity 0.08 / intensity 0.08 the whole brain rendered as
                    # one blown-out white blob with the neuron layers hidden
                    # underneath (toggling the connection layers off brought
                    # the coloured somas straight back). Sweeping the lines'
                    # gain live: 0.008 still left a milky wash over every
                    # super-class colour, 0.004 began to pale the optic lobes,
                    # 0.002 was clean but nearly invisible. 0.003 keeps a faint
                    # warm glow through the central brain with the palette
                    # intact. The gain multiplies the line colour before the
                    # luminous accumulation, so this is the exposure knob for
                    # this layer alone.
                    opacity=0.08,
                    intensity=0.003,
                    labels=nt_labels,
                    layer=True,
                )

            # Title
            scene.add_text(
                "FlyWire — adult Drosophila brain connectome",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )
            scene.add_text(
                "{hover_label}",
                # Top-right, not mid-left. Mid-left put the readout in the
                # middle of the frame's empty side, where it read as a caption
                # for the brain rather than as a response to the cursor, and it
                # sat directly across from the NT legend. The top-right corner
                # is out of the way of the title (top-left), the legends
                # (bottom-left) and the footer (bottom-right).
                position=(0.98, 0.02),
                anchor="top-right",
                font_size=0.022,
                color="white",
                background="rgba(0,0,0,0.7)",
                padding=0.01,
                text_align="left",
                opacity=1.0,
                transition="fade",
                transition_duration=0.15,
                hover=True,
            )

            # Legends — one bottom-left block holding both, side by side.
            scene.add_html(
                _build_legends_row(sc_order, sc_counts, sc_palette),
                position=(0.02, 0.97),
                anchor="bottom-left",
                opacity=0.92,
            )

            # Footer
            add_demo_caption(
                scene,
                f"{len(neurons):,} neurons • {len(edges):,} connections • "
                f"FlyWire release 783",
                DEMO_META.get("citation"),
            )

        aprint(
            f"  ✓ Scene: {len(neurons):,} neurons, "
            f"{len(edges):,} edges across {len(nt_order)} NT layers"
        )

    return len(neurons), len(edges)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    """Fetch FlyWire data, load neurons + edges, and build/serve the scene."""
    argv = sys.argv[1:]
    min_syn = parse_int_arg("min-synapses", DEFAULT_MIN_SYNAPSES, argv)
    max_edges = parse_int_arg("max-edges", DEFAULT_MAX_EDGES, argv)
    cache_dir = parse_path_arg("cache-dir", argv) or CACHE_DIR

    aprint("=" * 70)
    aprint("FLYWIRE — adult Drosophila brain connectome")
    aprint("=" * 70)
    aprint("~139k neurons · ~2-3M connections · public release 783")
    aprint(f"Filters: ≥{min_syn} synapses, top {max_edges:,} by count")
    aprint("")

    tsv_path, feather_path = ensure_data(cache_dir)
    neurons = load_neurons(tsv_path)
    edges = load_edges(feather_path, neurons, min_syn, max_edges)

    if "--no-serve" in argv:
        output_path = get_demos_output_dir() / "flywire_connectome.luxar.zarr"
        build_scene(output_path, neurons, edges)
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_flywire_") as tmpdir:
        output_path = Path(tmpdir) / "flywire_connectome.luxar.zarr"
        n_neurons, n_edges = build_scene(output_path, neurons, edges)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("  Rotate: see the two optic lobes flanking the central brain")
        aprint("  Hover a neuron: cell type · super-class · NT · side")
        aprint("")
        aprint(f"  Neurons: {n_neurons:,}   Edges: {n_edges:,}")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
