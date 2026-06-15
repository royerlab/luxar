#!/usr/bin/env python3
"""Self-Contained Demo: Chromatrace CHOIR 3D UMAP Visualization.

Visualizes a 3D UMAP of ~60k single cells annotated with CHOIR bio-terms
(fine-grained cell types) and bio-groups (8 broad tissue lineages).

The UMAP is colored two ways, switchable at runtime:

    - "Cell Type" (88 bio-terms) — using the curated palette shipped with
      the data (matches upstream plotly/matplotlib renderings exactly).
    - "Bio Group" (8 lineages) — Neuroectoderm, Neural Crest, Craniofacial
      Mesenchyme, Cardiac/Vascular, Mesoderm, Endoderm, Epidermis, Other.

Hover on any cell to see its bio-term and bio-group.

Data source:
    User-provided bundle ``choir_umap_3d_viewer.zip`` containing
    ``data/X_umap_3d.parquet`` + ``data/choir_bio_term_colormap.json``.
    The zip is auto-extracted to ``~/.cache/luxar/chromatrace/`` on first run.

Usage:
    python -m luxar.demos.demo_chromatrace_choir_umap
    python -m luxar.demos.demo_chromatrace_choir_umap --no-serve
    python -m luxar.demos.demo_chromatrace_choir_umap --data /path/to/zip-or-folder
"""

from __future__ import annotations

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
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import format_label, get_categorical_color
from luxar.utils.paths import get_demos_output_dir

CACHE_DIR = Path.home() / ".cache" / "luxar" / "chromatrace"
PARQUET_NAME = "X_umap_3d.parquet"
COLORMAP_NAME = "choir_bio_term_colormap.json"


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------


def _hex_to_rgb_float(hex_str: str) -> tuple[float, float, float]:
    """Parse '#rrggbb' into an (r, g, b) tuple in [0, 1]."""
    s = hex_str.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)


def _find_data_files(explicit: Path | None) -> tuple[Path, Path]:
    """Locate the parquet + colormap JSON, extracting the zip if needed."""
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
    """Extract the bundle zip into the cache and return the parquet/json paths."""
    parquet = CACHE_DIR / "data" / PARQUET_NAME
    colormap = CACHE_DIR / "data" / COLORMAP_NAME
    if parquet.exists() and colormap.exists():
        return parquet, colormap

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    aprint(f"Extracting {zip_path.name} to {CACHE_DIR} …")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith(PARQUET_NAME) or member.endswith(COLORMAP_NAME):
                target_name = Path(member).name
                out = CACHE_DIR / "data" / target_name
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out, "wb") as dst:
                    dst.write(src.read())
    if not (parquet.exists() and colormap.exists()):
        raise RuntimeError(f"Zip {zip_path} did not contain the expected data files.")
    return parquet, colormap


def load_chromatrace_data(
    explicit: Path | None = None,
) -> tuple[np.ndarray, dict, dict, dict, list[dict]]:
    """Load the Chromatrace CHOIR UMAP + annotations.

    Returns:
        coordinates: (N, 3) float32 UMAP positions
        attributes: dict with keys 'bio_term' and 'bio_group' (int codes)
        category_maps: dict mapping each key to its list of category names
        term_colors: {term_name: '#rrggbb'} from the shipped palette
        groups: list of {'name': str, 'cell_types': [...]} from the palette
    """
    parquet_path, colormap_path = _find_data_files(explicit)

    with asection("Loading Chromatrace CHOIR 3D UMAP"):
        aprint(f"  parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        aprint(f"  {len(df):,} cells loaded")

        coords = df[["UMAP1", "UMAP2", "UMAP3"]].to_numpy(dtype=np.float32)
        # Recenter so the viewer orbits around the cluster centroid
        # (raw UMAP coords can be asymmetric around the origin).
        center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        coords -= center
        aprint(
            f"  X range: [{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]  "
            f"Y: [{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]  "
            f"Z: [{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]"
        )

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
        n_unannotated = int((term_series == "unannotated").sum())
        aprint(
            f"  bio_term: {len(category_maps['bio_term'])} unique  "
            f"bio_group: {len(category_maps['bio_group'])} unique  "
            f"(incl. {n_unannotated:,} unannotated)"
        )

        with open(colormap_path) as fh:
            palette = json.load(fh)
        term_colors = palette.get("colors", {})
        groups = palette.get("groups", [])
        aprint(f"  palette: {len(term_colors)} term colors, {len(groups)} groups")

    return coords, attributes, category_maps, term_colors, groups


# ----------------------------------------------------------------------
# Coloring
# ----------------------------------------------------------------------


UNANNOTATED_COLOR = (0.55, 0.55, 0.55)


def _bio_term_colors(
    codes: np.ndarray, categories: list[str], term_colors: dict[str, str]
) -> np.ndarray:
    """Per-cell RGB colors using the shipped CHOIR palette, gray for missing."""
    lut = np.zeros((len(categories), 3), dtype=np.float32)
    for i, term in enumerate(categories):
        hex_str = term_colors.get(term)
        lut[i] = _hex_to_rgb_float(hex_str) if hex_str else UNANNOTATED_COLOR
    return lut[codes]


def _bio_group_colors(codes: np.ndarray, categories: list[str]) -> np.ndarray:
    """Per-cell RGB float32 colors, one distinct hue per bio_group.

    The 'Unannotated' group is rendered gray regardless of palette index.
    """
    n_real = sum(1 for c in categories if c != "Unannotated")
    lut = np.zeros((len(categories), 3), dtype=np.float32)
    palette_idx = 0
    for i, name in enumerate(categories):
        if name == "Unannotated":
            lut[i] = UNANNOTATED_COLOR
        else:
            r, g, b = get_categorical_color(palette_idx, n_real)
            lut[i] = (r / 255.0, g / 255.0, b / 255.0)
            palette_idx += 1
    return lut[codes]


# ----------------------------------------------------------------------
# Legend HTML
# ----------------------------------------------------------------------


def _build_bio_term_legend_html(
    groups: list[dict],
    term_colors: dict[str, str],
    present_terms: set[str],
    n_unannotated: int = 0,
) -> str:
    """Grouped HTML legend: cell types listed under each bio-group header."""
    lines = [
        '<div style="font-size:1.15vh;line-height:1.35;'
        "background:rgba(0,0,0,0.6);padding:0.6vh 0.8vh;"
        "border-radius:4px;max-height:92vh;overflow-y:auto;"
        'column-count:2;column-gap:1.2vh;width:34vh">'
    ]
    for grp in groups:
        grp_cts = [ct for ct in grp["cell_types"] if ct in present_terms]
        if not grp_cts:
            continue
        lines.append(
            f'<div style="color:#ffcc44;font-weight:bold;margin-top:0.4vh;'
            f'break-inside:avoid">[{grp["name"]}]</div>'
        )
        for ct in grp_cts:
            color = term_colors.get(ct, "#cccccc")
            lines.append(
                f'<div style="white-space:nowrap;break-inside:avoid">'
                f'<span style="color:{color}">█</span> '
                f"{format_label(ct)}</div>"
            )
    if n_unannotated > 0:
        lines.append(
            '<div style="color:#ffcc44;font-weight:bold;margin-top:0.4vh;'
            'break-inside:avoid">[Unannotated]</div>'
            '<div style="white-space:nowrap;break-inside:avoid">'
            '<span style="color:#8c8c8c">█</span> '
            f"no bio-term ({n_unannotated:,})</div>"
        )
    lines.append("</div>")
    return "".join(lines)


def _build_bio_group_legend_html(
    categories: list[str],
    counts: dict[str, int] | None = None,
) -> str:
    """Simple one-column HTML legend for the bio-groups."""
    lines = [
        '<div style="font-size:1.6vh;line-height:1.7;'
        "background:rgba(0,0,0,0.6);padding:0.8vh 1.0vh;"
        'border-radius:4px">'
        '<div style="color:#ffcc44;font-weight:bold;margin-bottom:0.5vh">'
        "Bio Group</div>"
    ]
    n_real = sum(1 for c in categories if c != "Unannotated")
    palette_idx = 0
    for name in categories:
        if name == "Unannotated":
            color_css = "rgb(140,140,140)"
        else:
            r, g, b = get_categorical_color(palette_idx, n_real)
            color_css = f"rgb({r},{g},{b})"
            palette_idx += 1
        suffix = f" ({counts[name]:,})" if counts and name in counts else ""
        lines.append(
            f'<div style="white-space:nowrap">'
            f'<span style="color:{color_css}">█</span> '
            f"{name}{suffix}</div>"
        )
    lines.append("</div>")
    return "".join(lines)


# ----------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------


def build_scene(
    output_path: Path,
    coords: np.ndarray,
    attributes: dict,
    category_maps: dict,
    term_colors: dict[str, str],
    groups: list[dict],
) -> int:
    n_cells = len(coords)

    term_cats = category_maps["bio_term"]
    group_cats = category_maps["bio_group"]

    with asection("Building Multi-Attribute Scene"):
        colors_term = _bio_term_colors(attributes["bio_term"], term_cats, term_colors)
        colors_group = _bio_group_colors(attributes["bio_group"], group_cats)

        pos_term = np.column_stack([np.zeros(n_cells, dtype=np.float32), coords])
        pos_group = np.column_stack([np.ones(n_cells, dtype=np.float32), coords])
        positions = np.vstack([pos_term, pos_group])
        colors = np.vstack([colors_term, colors_group])

        # Hover labels — two lines: cell type on line 1, bio-group on line 2
        per_cell_labels = [
            f"{format_label(term_cats[attributes['bio_term'][i]])}\n"
            f"[{group_cats[attributes['bio_group'][i]]}]"
            for i in range(n_cells)
        ]
        labels = per_cell_labels * 2

        group_counts = {
            name: int((attributes["bio_group"] == i).sum())
            for i, name in enumerate(group_cats)
        }
        present_terms = set(term_cats)
        n_unannotated = group_counts.get("Unannotated", 0)

        aprint(f"  2 views × {n_cells:,} cells = {len(positions):,} total points")

        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=["Cell Type (88)", "Bio Group (8)"],
                    display=False,
                    description="Color cells by fine-grained CHOIR bio-term or broad bio-group",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            scene.add_points(
                "Cells",
                positions,
                colors=colors,
                radii=np.full(len(positions), 0.04, dtype=np.float32),
                sharpness=np.full(len(positions), 0.6, dtype=np.float32),
                opacity=0.85,
                intensity=0.2,
                labels=labels,
            )

            # Title (top-left)
            scene.add_text(
                "Chromatrace 3D UMAP — CHOIR bio-term annotations",
                position=(0.02, 0.02),
                font_size=0.05,
                anchor="top-left",
                color="rgba(255,255,255,0.65)",
                blend_mode="difference",
            )

            # Custom hover overlay (center-left, two-line) — avoids the
            # right-side legend. Defining our own hover=True overlay suppresses
            # the default auto-injected top-right overlay.
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

            # Per-view captions + legends
            term_legend = _build_bio_term_legend_html(
                groups, term_colors, present_terms, n_unannotated=n_unannotated
            )
            group_legend = _build_bio_group_legend_html(group_cats, group_counts)

            for attr_id, caption, legend_html in [
                (0, "Colored by: CHOIR bio-term (88 cell types)", term_legend),
                (1, "Colored by: CHOIR bio-group (8 lineages)", group_legend),
            ]:
                scene.add_text(
                    caption,
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"attribute": attr_id},
                    transition="fade",
                    transition_duration=0.2,
                )
                scene.add_html(
                    legend_html,
                    position=(0.98, 0.5),
                    anchor="center-right",
                    opacity=0.92,
                    visible_range={"attribute": attr_id},
                    transition="fade",
                    transition_duration=0.2,
                )

            n_term_real = sum(1 for t in term_cats if t != "unannotated")
            n_group_real = sum(1 for g in group_cats if g != "Unannotated")
            scene.add_text(
                f"{n_cells:,} cells • {n_term_real} cell types • "
                f"{n_group_real} bio-groups • {n_unannotated:,} unannotated",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.5)",
            )

    return n_cells


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def _parse_data_arg(argv: list[str]) -> Path | None:
    for i, arg in enumerate(argv):
        if arg == "--data" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        m = re.match(r"^--data=(.+)$", arg)
        if m:
            return Path(m.group(1)).expanduser()
    return None


def main() -> None:
    aprint("=" * 70)
    aprint("CHROMATRACE 3D UMAP — CHOIR bio-term annotations")
    aprint("=" * 70)
    aprint("~60k cells • 88 bio-terms • 8 bio-groups")
    aprint("")

    explicit_data = _parse_data_arg(sys.argv[1:])
    coords, attributes, category_maps, term_colors, groups = load_chromatrace_data(
        explicit_data
    )

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "chromatrace_choir_umap.luxar.zarr"
        n = build_scene(
            output_path, coords, attributes, category_maps, term_colors, groups
        )
        aprint(f"Dataset generated at {output_path} ({n:,} cells)")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_chromatrace_") as tmpdir:
        output_path = Path(tmpdir) / "chromatrace_choir_umap.luxar.zarr"
        n = build_scene(
            output_path, coords, attributes, category_maps, term_colors, groups
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("  Press '1' then '[' / ']' to switch color views:")
        aprint("     0: Cell Type (88 CHOIR bio-terms, curated palette)")
        aprint("     1: Bio Group (8 broad lineages)")
        aprint("  Hover any cell for its term + group")
        aprint("")
        aprint(f"  Total cells: {n:,}")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
