#!/usr/bin/env python3
"""Self-Contained Demo: Chromatrace CHOIR 3D UMAP — Sequential Cell-Type Walk.

Variant of ``demo_chromatrace_choir_umap`` that exposes an 89-step slider
stepping through each fine-grained CHOIR bio-term one at a time.

Slot ordering:
    - Bio-groups are traversed in legend order (Neuroectoderm → Neural
      Crest → Craniofacial Mesenchyme → Cardiac/Vascular → Mesoderm →
      Endoderm → Epidermis → Other → Unannotated).
    - Within each bio-group, cell types are re-ordered so consecutive
      slots are spatially adjacent in UMAP space: a nearest-neighbor
      Hamiltonian path through cluster centroids, refined by 2-opt.
    - Each group starts at the type whose centroid is closest to the
      previous group's endpoint, so cross-group transitions are smooth too.
    - Pass ``--no-tsp`` to fall back to the raw legend order.

At each slider position:
    - The current cell type is shown in its signature CHOIR palette color.
    - All other cells are rendered dimmed gray so the overall UMAP outline
      stays visible for spatial context.
    - A caption overlay announces the current bio-term and its bio-group.

Scene structure (two toggleable layers):
    - **Backdrop** — gray UMAP outline, stored as a single 60k-point cloud
      with ``extend_to_all=["cell_type"]`` so it stays visible across every
      slider slot without duplication.
    - **Highlight** — 60k CHOIR-colored points, each pinned to its own slot
      on the ``cell_type`` dim, so only the active type is rendered at a time.

    Both are exposed in the viewer's Layers panel for quick toggling.

Navigation:
    Press '1' to select the cell-type slider, then '[' / ']' to step.

Usage:
    python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py
    python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py --no-serve
    python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py --no-tsp
    python packages/luxar/src/luxar/demos/demo_chromatrace_choir_umap_sequence.py --data /path/to/zip-or-folder
"""

from __future__ import annotations

DEMO_META = {
    "key": "chromatrace_choir_umap_sequence",
    "title": "Chromatrace CHOIR 3D UMAP — Sequential Cell-Type Walk.",
    "description": "A 3D UMAP of ~60k single cells, stepping through each CHOIR cell type one slot at a time.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 0,
        "compute": "medium",
        "gpu": "none",
        "local_data": "manual-file",
    },
    "caches": ["chromatrace"],
    "outputs": ["chromatrace_choir_umap_sequence"],
}

import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.core.viewer_config import CameraConfig, ViewerConfig
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import format_label
from luxar.utils.paths import get_demos_output_dir

CACHE_DIR = Path.home() / ".cache" / "luxar" / "chromatrace"
PARQUET_NAME = "X_umap_3d.parquet"
COLORMAP_NAME = "choir_bio_term_colormap.json"

UNANNOTATED_COLOR = (0.55, 0.55, 0.55)
BACKDROP_COLOR = (0.35, 0.35, 0.35)
UNANNOTATED_SLOT_LABEL = "(unannotated)"


# ----------------------------------------------------------------------
# Data loading (mirrors demo_chromatrace_choir_umap.py)
# ----------------------------------------------------------------------


def _hex_to_rgb_float(hex_str: str) -> tuple[float, float, float]:
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _find_data_files(explicit: Path | None) -> tuple[Path, Path]:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates += [
        CACHE_DIR,
        Path.home() / "Downloads" / "choir_umap_3d_viewer",
        Path.home() / "Downloads" / "choir_umap_3d_viewer (1)",
    ]
    for root in candidates:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".zip":
            return _extract_zip(root)
        parquet = root / "data" / PARQUET_NAME
        colormap = root / "data" / COLORMAP_NAME
        if parquet.exists() and colormap.exists():
            return parquet, colormap
        parquet = root / PARQUET_NAME
        colormap = root / COLORMAP_NAME
        if parquet.exists() and colormap.exists():
            return parquet, colormap

    downloads = Path.home() / "Downloads"
    if downloads.exists():
        zips = sorted(downloads.glob("choir_umap_3d_viewer*.zip"))
        if zips:
            return _extract_zip(zips[-1])

    raise FileNotFoundError(
        "Could not find the CHOIR UMAP data bundle. Expected one of:\n"
        f"  - {CACHE_DIR}/data/{PARQUET_NAME}\n"
        f"  - ~/Downloads/choir_umap_3d_viewer/data/{PARQUET_NAME}\n"
        "  - ~/Downloads/choir_umap_3d_viewer*.zip\n"
        "Pass --data <path> to point at the zip or extracted folder."
    )


def _extract_zip(zip_path: Path) -> tuple[Path, Path]:
    parquet = CACHE_DIR / "data" / PARQUET_NAME
    colormap = CACHE_DIR / "data" / COLORMAP_NAME
    if parquet.exists() and colormap.exists():
        return parquet, colormap
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aprint(f"Extracting {zip_path.name} to {CACHE_DIR} …")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(PARQUET_NAME) or member.endswith(COLORMAP_NAME):
                out = CACHE_DIR / "data" / Path(member).name
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    if not (parquet.exists() and colormap.exists()):
        raise RuntimeError(f"Zip {zip_path} did not contain the expected data files.")
    return parquet, colormap


def load_chromatrace_data(
    explicit: Path | None = None,
) -> tuple[np.ndarray, dict, dict, dict, list[dict]]:
    parquet_path, colormap_path = _find_data_files(explicit)

    with asection("Loading Chromatrace CHOIR 3D UMAP"):
        aprint(f"  parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        aprint(f"  {len(df):,} cells loaded")

        coords = df[["UMAP1", "UMAP2", "UMAP3"]].to_numpy(dtype=np.float32)
        center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        coords -= center

        term_series = df["choir_bio_term"].astype("object").fillna("unannotated")
        group_series = df["choir_bio_group"].astype("object").fillna("Unannotated")
        term_cat = pd.Categorical(term_series)
        group_cat = pd.Categorical(group_series)

        attributes = {
            "bio_term": term_cat.codes.astype(np.int32),
            "bio_group": group_cat.codes.astype(np.int32),
        }
        category_maps = {
            "bio_term": list(term_cat.categories),
            "bio_group": list(group_cat.categories),
        }

        with open(colormap_path) as fh:
            palette = json.load(fh)
        term_colors = palette.get("colors", {})
        groups = palette.get("groups", [])

    return coords, attributes, category_maps, term_colors, groups


def _parse_data_arg(argv: list[str]) -> Path | None:
    for i, arg in enumerate(argv):
        if arg == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        m = re.match(r"^--data=(.+)$", arg)
        if m:
            return Path(m.group(1)).expanduser()
    return None


def _term_to_group(groups: list[dict]) -> dict[str, str]:
    """Flatten the palette groups into a ``term -> group name`` lookup."""
    lookup: dict[str, str] = {}
    for grp in groups:
        for ct in grp.get("cell_types", []):
            lookup[ct] = grp["name"]
    return lookup


def _ordered_terms(
    palette_order: list[str],
    present_terms: set[str],
) -> list[str]:
    """Keep the curated palette order, dropping terms with no cells in the data."""
    return [t for t in palette_order if t in present_terms]


def _cluster_centroids(
    coords: np.ndarray,
    term_name_of_cell: np.ndarray,
    terms: list[str],
) -> dict[str, np.ndarray]:
    """UMAP-space centroid of every cell type present in the data."""
    return {
        term: coords[term_name_of_cell == term].mean(axis=0)
        for term in terms
        if np.any(term_name_of_cell == term)
    }


def _path_length(path: list[int], dist: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(np.sum(dist[path[:-1], path[1:]]))


def _nn_path(dist: np.ndarray, start: int) -> list[int]:
    """Greedy nearest-neighbor Hamiltonian path starting from ``start``."""
    n = dist.shape[0]
    visited = [start]
    mask = np.ones(n, dtype=bool)
    mask[start] = False
    while mask.any():
        current = visited[-1]
        remaining_idx = np.where(mask)[0]
        nxt = int(remaining_idx[np.argmin(dist[current, remaining_idx])])
        visited.append(nxt)
        mask[nxt] = False
    return visited


def _two_opt(path: list[int], dist: np.ndarray, max_passes: int = 40) -> list[int]:
    """2-opt refinement of an open Hamiltonian path (keeps endpoints free)."""
    n = len(path)
    if n < 4:
        return path
    p = list(path)
    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = p[i - 1], p[i]
                c = p[j]
                d = p[j + 1] if j + 1 < n else None
                before = dist[a, b] + (dist[c, d] if d is not None else 0.0)
                after = dist[a, c] + (dist[b, d] if d is not None else 0.0)
                if after + 1e-12 < before:
                    p[i : j + 1] = reversed(p[i : j + 1])
                    improved = True
        if not improved:
            break
    return p


def optimize_path(
    centroids: np.ndarray,
    start: int | None = None,
) -> list[int]:
    """Nearest-neighbor Hamiltonian path + 2-opt refinement.

    If ``start`` is given, the path begins at that index. Otherwise we try
    every possible start and keep the best NN-path before refinement.
    """
    n = len(centroids)
    if n <= 1:
        return list(range(n))
    # Pairwise Euclidean distances between centroids.
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))

    if start is None:
        candidates = range(n)
    else:
        candidates = [start]
    best_path: list[int] = []
    best_len = float("inf")
    for s in candidates:
        path = _nn_path(dist, s)
        length = _path_length(path, dist)
        if length < best_len:
            best_path, best_len = path, length
    return _two_opt(best_path, dist)


def grouped_tsp_order(
    coords: np.ndarray,
    term_name_of_cell: np.ndarray,
    groups: list[dict],
    present_terms: set[str],
) -> list[str]:
    """Walk bio-groups in legend order, but re-order types inside each by TSP.

    Each group starts at the type whose centroid is closest to the previous
    group's endpoint, giving smooth cross-group transitions as well.
    """
    result: list[str] = []
    prev_endpoint_centroid: np.ndarray | None = None

    for grp in groups:
        grp_terms = [t for t in grp.get("cell_types", []) if t in present_terms]
        if not grp_terms:
            continue
        centroids_map = _cluster_centroids(coords, term_name_of_cell, grp_terms)
        terms_list = list(centroids_map.keys())
        centroid_arr = np.stack([centroids_map[t] for t in terms_list])

        # Pick a starting type closest to previous group's endpoint
        if prev_endpoint_centroid is not None and len(terms_list) > 1:
            d_to_prev = np.linalg.norm(centroid_arr - prev_endpoint_centroid, axis=1)
            start = int(np.argmin(d_to_prev))
        else:
            start = None  # try all starts, pick best NN-path

        order = optimize_path(centroid_arr, start=start)
        ordered_group_terms = [terms_list[i] for i in order]
        result.extend(ordered_group_terms)
        prev_endpoint_centroid = centroids_map[ordered_group_terms[-1]]

    return result


def build_sequence_scene(
    output_path: Path,
    coords: np.ndarray,
    attributes: dict,
    category_maps: dict,
    term_colors: dict[str, str],
    palette_order: list[str],
    groups: list[dict],
    use_tsp: bool = True,
) -> int:
    """Construct a scene with an 89-step slider walking through cell types.

    At each slot, all cells are present but only the matching type is
    shown in color — the rest form a dim gray backdrop for context.

    Args:
        use_tsp: If True (default), re-order types within each bio-group
            via nearest-neighbor + 2-opt on UMAP centroids so that
            consecutive slots are spatially adjacent. Each group still
            plays in legend order. If False, use the raw palette order.
    """
    n_cells = len(coords)

    term_cats = list(category_maps["bio_term"])
    group_cats = list(category_maps["bio_group"])
    term_to_group = _term_to_group(groups)
    term_code_of_cell = np.asarray(attributes["bio_term"], dtype=np.int32)
    term_name_of_cell = np.asarray(
        [term_cats[c] for c in term_code_of_cell], dtype=object
    )

    present_terms = {t for t in term_cats if t != "unannotated"}
    if use_tsp:
        with asection("Optimizing intra-group order by UMAP centroid proximity"):
            ordered_real = grouped_tsp_order(
                coords, term_name_of_cell, groups, present_terms
            )
            aprint(f"  TSP-reordered {len(ordered_real)} cell types")
    else:
        ordered_real = _ordered_terms(palette_order, present_terms)
    has_unannotated = "unannotated" in term_cats
    ordered_terms = ordered_real + ([UNANNOTATED_SLOT_LABEL] if has_unannotated else [])
    n_slots = len(ordered_terms)

    with asection(f"Building Sequence Scene ({n_slots} slots × {n_cells:,} cells)"):
        # Per-cell highlight color (CHOIR palette, gray for unannotated).
        highlight_rgb = np.zeros((n_cells, 3), dtype=np.float32)
        for i, term in enumerate(term_cats):
            if term == "unannotated":
                highlight_rgb[term_code_of_cell == i] = UNANNOTATED_COLOR
            else:
                hex_str = term_colors.get(term)
                if hex_str:
                    highlight_rgb[term_code_of_cell == i] = _hex_to_rgb_float(hex_str)

        aprint(
            f"  palette: {len(ordered_real)} terms in legend order"
            f"{' + unannotated slot' if has_unannotated else ''}"
        )

        # Map each cell → its slot index (its position on the cell_type dim).
        # TSP order defines the slot index for each real term; "unannotated"
        # cells land on the final dedicated slot.
        BACKDROP_RADIUS = 0.025
        HIGHLIGHT_RADIUS = 0.06
        term_to_slot = {term: idx for idx, term in enumerate(ordered_terms)}
        if has_unannotated:
            term_to_slot["unannotated"] = n_slots - 1  # trailing slot
        cell_slot_idx = np.asarray(
            [term_to_slot[t] for t in term_name_of_cell], dtype=np.float32
        )

        # Highlight node: ONE copy of each cell, positioned at its own slot.
        # Only visible when the slider matches that cell's type.
        highlight_positions = np.column_stack([cell_slot_idx, coords])
        highlight_colors = highlight_rgb
        highlight_radii = np.full(n_cells, HIGHLIGHT_RADIUS, dtype=np.float32)

        # Backdrop node: ONE copy of each cell, extend_to_all=["cell_type"]
        # makes it visible at every slider position without duplication.
        backdrop_positions = np.column_stack(
            [np.zeros(n_cells, dtype=np.float32), coords]
        )
        backdrop_colors = np.broadcast_to(
            np.asarray(BACKDROP_COLOR, dtype=np.float32), (n_cells, 3)
        ).copy()
        backdrop_radii = np.full(n_cells, BACKDROP_RADIUS, dtype=np.float32)

        aprint(
            f"  2 nodes × {n_cells:,} cells = {2 * n_cells:,} total points"
            f" (vs {n_slots * n_cells:,} before split)"
        )

        # Hover labels: 1 per cell, shared for both Highlight and Backdrop.
        per_cell_hover = [
            f"{format_label(term_name_of_cell[i])}\n"
            f"[{group_cats[attributes['bio_group'][i]]}]"
            for i in range(n_cells)
        ]

        dims = Dimensions(
            [
                Dimension(
                    "cell_type",
                    unit="",
                    categories=[
                        format_label(t) if t != UNANNOTATED_SLOT_LABEL else t
                        for t in ordered_terms
                    ],
                    display=False,
                    description=(
                        "Step through each CHOIR bio-term one at a time "
                        "(curated legend order)."
                    ),
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        # Explicit camera framing: auto-fit would include the 89-slot
        # cell_type dim in its bbox diagonal and place the camera far
        # outside the spatial volume. Frame the UMAP (x, y, z) directly.
        spatial_max = float(np.abs(coords).max())
        cam_dist = spatial_max * 2.2  # 2.2× half-extent → comfortable framing
        viewer_config = ViewerConfig(
            camera=CameraConfig(
                position=(cam_dist, cam_dist * 0.55, cam_dist),
                target=(0.0, 0.0, 0.0),
            ),
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims, viewer_config=viewer_config)

            # Backdrop layer: always visible across every slider slot.
            scene.add_points(
                "Backdrop",
                backdrop_positions,
                colors=backdrop_colors,
                radii=backdrop_radii,
                sharpness=np.full(n_cells, 0.6, dtype=np.float32),
                opacity=0.8,
                intensity=0.15,
                extend_to_all=["cell_type"],
                layer=True,
            )

            # Highlight layer: only the current slot's cells visible.
            scene.add_points(
                "Highlight",
                highlight_positions,
                colors=highlight_colors,
                radii=highlight_radii,
                sharpness=np.full(n_cells, 0.6, dtype=np.float32),
                opacity=0.95,
                intensity=0.3,
                labels=per_cell_hover,
                layer=True,
            )

            # Title
            scene.add_text(
                "Chromatrace 3D UMAP — Cell-type walkthrough",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )

            # Hover overlay (center-left, 2-line, matches base demo layout)
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

            # Per-slot captions (center-right).
            # Two text overlays per slot — using add_text (NOT add_html) so the
            # turntable recording pipeline scales them with the canvas height
            # (add_html's inline `vh` resolves against the browser window
            # during recording, producing tiny text).
            for slot_idx, slot_term in enumerate(ordered_terms):
                if slot_term == UNANNOTATED_SLOT_LABEL:
                    primary_color = "#a0a0a0"
                    primary = "Unannotated cells"
                    secondary = "[no CHOIR bio-term]"
                else:
                    primary_color = term_colors.get(slot_term, "#cccccc")
                    primary = format_label(slot_term)
                    group_name = term_to_group.get(slot_term, "—")
                    secondary = f"[{group_name}]"

                # Large colored type name
                scene.add_text(
                    primary,
                    position=(0.98, 0.47),
                    anchor="center-right",
                    font_size=0.035,
                    color=primary_color,
                    background="rgba(0,0,0,0.6)",
                    padding=0.008,
                    opacity=0.95,
                    text_align="right",
                    visible_range={"cell_type": slot_idx},
                    transition="fade",
                    transition_duration=0.15,
                )
                # Smaller group + counter on one line below (overlays use
                # white-space:nowrap, so \n would render as a space).
                scene.add_text(
                    f"{secondary}  ·  {slot_idx + 1}/{n_slots}",
                    position=(0.98, 0.53),
                    anchor="center-right",
                    font_size=0.018,
                    color="#c8c8c8",
                    background="rgba(0,0,0,0.6)",
                    padding=0.008,
                    opacity=0.9,
                    text_align="right",
                    visible_range={"cell_type": slot_idx},
                    transition="fade",
                    transition_duration=0.15,
                )

            # Bottom-left status / navigation hint
            scene.add_text(
                "← [  •  ] →   step through cell types",
                position=(0.02, 0.97),
                font_size=0.014,
                anchor="bottom-left",
                color="#ffcc44",
            )

            n_term_real = len(ordered_real)
            order_note = (
                "TSP-optimized within each CHOIR bio-group"
                if use_tsp
                else "CHOIR legend order"
            )
            scene.add_text(
                f"{n_cells:,} cells • {n_term_real} cell types • {order_note}",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )

    return n_cells


def main() -> None:
    aprint("=" * 70)
    aprint("CHROMATRACE 3D UMAP — Sequential Cell-Type Walkthrough")
    aprint("=" * 70)

    explicit_data = _parse_data_arg(sys.argv[1:])
    use_tsp = "--no-tsp" not in sys.argv
    coords, attributes, category_maps, term_colors, groups = load_chromatrace_data(
        explicit_data
    )
    # JSON 'order' list (legend order across bio-groups), used when --no-tsp.
    palette_order = []
    for grp in groups:
        palette_order.extend(grp.get("cell_types", []))

    if "--no-serve" in sys.argv:
        output_path = (
            get_demos_output_dir() / "chromatrace_choir_umap_sequence.luxar.zarr"
        )
        n = build_sequence_scene(
            output_path,
            coords,
            attributes,
            category_maps,
            term_colors,
            palette_order,
            groups,
            use_tsp=use_tsp,
        )
        aprint(f"Dataset generated at {output_path} ({n:,} cells)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_chromatrace_seq_") as tmpdir:
        output_path = Path(tmpdir) / "chromatrace_choir_umap_sequence.luxar.zarr"
        n = build_sequence_scene(
            output_path,
            coords,
            attributes,
            category_maps,
            term_colors,
            palette_order,
            groups,
            use_tsp=use_tsp,
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("  Press '1' to select CELL TYPE slider, then '[' / ']' to step")
        aprint("  through each CHOIR bio-term in legend order")
        aprint(f"  Total cells: {n:,}")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
