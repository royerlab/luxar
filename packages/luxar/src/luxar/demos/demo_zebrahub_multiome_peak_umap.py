#!/usr/bin/env python3
"""Self-Contained Demo: Zebrahub Multiome Peak 3D UMAP Visualization

This demo demonstrates:
- Loading biological data from remote public zarr store
- 3D UMAP embedding of 640k single-cell ATAC-seq peaks
- Color-coded by cell type (30 cell types)
- Navigation through timepoints (6 developmental stages)
- Real scientific dataset from Zebrahub project
- Complete workflow: download → build scene → serve → cleanup

Data Source:
    CZ Biohub - Zebrahub Multiome Project
    3D UMAP of single-cell chromatin accessibility peaks
    URL: https://public.czbiohub.org/royerlab/zebrahub/...

    References:
    - Zebrahub: https://zebrahub.org
    - Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1

Dataset Structure:
    - 640,830 points in 3D UMAP space
    - 30 distinct cell types
    - 6 developmental timepoints
    - 6 lineages
    - 4 peak types

Usage:
    python demo_zebrahub_multiome_peak_umap.py

Controls:
    - Rotate to explore UMAP structure
    - Use dropdown to switch between 7 biological attributes (Cell Type, Chromosome, etc.)
    - Press '1' to select attribute dimension, then '['/']' to navigate
    - Different values shown in different colors
    - Ctrl+C to stop
"""

import sys
import tempfile
from pathlib import Path

import fsspec
import numpy as np
import zarr
from arbol import aprint, asection
from PIL import Image, ImageDraw, ImageFont

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir


def get_legends_dir() -> Path:
    """Get the legends output directory.

    Creates the directory if it doesn't exist.

    Returns:
        Path to legends/ directory under demos
    """
    legends_dir = Path(__file__).parent / "legends"
    legends_dir.mkdir(exist_ok=True)
    return legends_dir


# Curated color palettes for categorical data visualization
# Based on ColorBrewer and other well-designed palettes
# All colors are designed to be distinguishable and aesthetically pleasing

# Tab10-inspired palette for small categories (up to 10)
TAB10_COLORS = [
    (31, 119, 180),  # Blue
    (255, 127, 14),  # Orange
    (44, 160, 44),  # Green
    (214, 39, 40),  # Red
    (148, 103, 189),  # Purple
    (140, 86, 75),  # Brown
    (227, 119, 194),  # Pink
    (127, 127, 127),  # Gray
    (188, 189, 34),  # Olive
    (23, 190, 207),  # Teal
]

# Extended palette for medium categories (up to 20)
TAB20_COLORS = [
    (31, 119, 180),  # Blue
    (174, 199, 232),  # Light blue
    (255, 127, 14),  # Orange
    (255, 187, 120),  # Light orange
    (44, 160, 44),  # Green
    (152, 223, 138),  # Light green
    (214, 39, 40),  # Red
    (255, 152, 150),  # Light red
    (148, 103, 189),  # Purple
    (197, 176, 213),  # Light purple
    (140, 86, 75),  # Brown
    (196, 156, 148),  # Light brown
    (227, 119, 194),  # Pink
    (247, 182, 210),  # Light pink
    (127, 127, 127),  # Gray
    (199, 199, 199),  # Light gray
    (188, 189, 34),  # Olive
    (219, 219, 141),  # Light olive
    (23, 190, 207),  # Teal
    (158, 218, 229),  # Light teal
]


def get_sequential_color(t: float) -> tuple[int, int, int]:
    """Get a color from a sequential colormap for continuous data like time.

    Uses a plasma-inspired colormap that works well on black backgrounds:
    Deep purple -> magenta -> orange -> yellow

    Args:
        t: Normalized value between 0 and 1

    Returns:
        RGB tuple (0-255 range)
    """
    t = max(0.0, min(1.0, t))

    # Plasma-inspired colormap control points
    # Format: (t, r, g, b) - all in 0-1 range
    control_points = [
        (0.0, 0.05, 0.03, 0.53),  # Deep blue-purple
        (0.25, 0.42, 0.12, 0.66),  # Purple
        (0.5, 0.80, 0.24, 0.46),  # Magenta-pink
        (0.75, 0.97, 0.55, 0.20),  # Orange
        (1.0, 0.94, 0.98, 0.13),  # Bright yellow
    ]

    # Find the two control points to interpolate between
    for i in range(len(control_points) - 1):
        t0, r0, g0, b0 = control_points[i]
        t1, r1, g1, b1 = control_points[i + 1]
        if t0 <= t <= t1:
            # Linear interpolation
            s = (t - t0) / (t1 - t0) if t1 > t0 else 0
            r = r0 + s * (r1 - r0)
            g = g0 + s * (g1 - g0)
            b = b0 + s * (b1 - b0)
            return (int(r * 255), int(g * 255), int(b * 255))

    # Fallback to last color
    return (
        int(control_points[-1][1] * 255),
        int(control_points[-1][2] * 255),
        int(control_points[-1][3] * 255),
    )


def get_categorical_color(index: int, n_categories: int) -> tuple[int, int, int]:
    """Get a visually pleasing color for categorical data.

    Uses curated palettes for small category counts, and generates
    a well-distributed color scheme for larger counts using a
    golden-ratio-based hue distribution with varying saturation
    and lightness for visual distinction.

    Args:
        index: Category index (0-based)
        n_categories: Total number of categories

    Returns:
        RGB tuple (0-255 range)
    """
    if n_categories <= 10:
        return TAB10_COLORS[index % len(TAB10_COLORS)]
    elif n_categories <= 20:
        return TAB20_COLORS[index % len(TAB20_COLORS)]
    else:
        # For many categories, use golden-ratio-based hue distribution
        # This ensures colors are well-distributed around the color wheel
        golden_ratio = 0.618033988749895

        # Start with a nice blue-ish hue and distribute using golden ratio
        hue = (0.6 + index * golden_ratio) % 1.0

        # Vary saturation and value slightly to create more distinction
        sat_variation = 0.15
        val_variation = 0.15

        # Use index to create subtle variations in saturation and brightness
        sat = 0.75 + sat_variation * ((index * 7) % 3 - 1) / 2
        val = 0.90 + val_variation * ((index * 11) % 3 - 1) / 2

        sat = max(0.5, min(1.0, sat))
        val = max(0.7, min(1.0, val))

        # HSV to RGB conversion
        h = hue * 6.0
        c = val * sat
        x = c * (1 - abs(h % 2 - 1))
        m = val - c

        if h < 1:
            r, g, b = c, x, 0.0
        elif h < 2:
            r, g, b = x, c, 0.0
        elif h < 3:
            r, g, b = 0.0, c, x
        elif h < 4:
            r, g, b = 0.0, x, c
        elif h < 5:
            r, g, b = x, 0.0, c
        else:
            r, g, b = c, 0.0, x

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


def generate_legend_image(
    labels: list[str],
    title: str,
    output_path: Path,
    swatch_size: int = 48,
    spacing: int = 16,
    font_size: int = 36,
    title_font_size: int = 48,
    padding: int = 40,
    use_sequential: bool = False,
) -> None:
    """Generate a transparent PNG legend image.

    Creates a vertically arranged color legend with swatches and labels.
    Colors are generated using curated categorical palettes or sequential
    colormap for continuous data like time.

    Args:
        labels: List of category labels
        title: Title for the legend
        output_path: Where to save the PNG
        swatch_size: Size of color swatches in pixels
        spacing: Vertical spacing between items
        font_size: Font size for labels
        title_font_size: Font size for title
        padding: Padding around content
        use_sequential: If True, use sequential colormap (for time-like data)
    """
    n_items = len(labels)

    # Try to load a nice sans-serif font, fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        title_font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc", title_font_size
        )
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", title_font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    # Calculate image dimensions
    # First pass: measure text widths
    temp_img = Image.new("RGBA", (1, 1))
    temp_draw = ImageDraw.Draw(temp_img)

    max_text_width = 0
    for label in labels:
        bbox = temp_draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        max_text_width = max(max_text_width, text_width)

    title_bbox = temp_draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]

    # Calculate total dimensions
    content_width = swatch_size + spacing + max_text_width
    content_width = max(content_width, title_width)
    content_height = (
        title_height + spacing * 2 + n_items * (swatch_size + spacing) - spacing
    )

    img_width = content_width + padding * 2
    img_height = content_height + padding * 2

    # Create transparent image
    img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw title
    title_x = padding
    title_y = padding
    draw.text((title_x, title_y), title, fill=(255, 255, 255, 255), font=title_font)

    # Draw legend items
    y_offset = padding + title_height + spacing * 2

    for i, label in enumerate(labels):
        # Generate color using appropriate palette
        if use_sequential:
            t = i / max(n_items - 1, 1)  # Normalize to 0-1
            rgb = get_sequential_color(t)
        else:
            rgb = get_categorical_color(i, n_items)

        # Draw swatch
        x_swatch = padding
        y_swatch = y_offset + i * (swatch_size + spacing)
        draw.rectangle(
            [x_swatch, y_swatch, x_swatch + swatch_size, y_swatch + swatch_size],
            fill=(*rgb, 255),
        )

        # Draw label
        x_text = padding + swatch_size + spacing
        y_text = y_swatch + (swatch_size - font_size) // 2
        draw.text((x_text, y_text), label, fill=(255, 255, 255, 255), font=font)

    # Save as PNG with transparency
    img.save(output_path, "PNG")


def format_label(label: str) -> str:
    """Format a category label for display.

    Applies formatting rules:
    - Replace underscores with spaces
    - Add space between numbers and letters (e.g., "10somites" -> "10 somites")

    Args:
        label: Raw category label

    Returns:
        Formatted label for display
    """
    import re

    # Replace underscores with spaces
    label = label.replace("_", " ")

    # Add space between numbers and letters (e.g., "10somites" -> "10 somites")
    label = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", label)

    return label


def generate_all_legends(
    attributes: dict, category_maps: dict, prefix: str = "zebrafish"
) -> None:
    """Generate legend images for all attribute types.

    Args:
        attributes: Dict of attribute arrays from the dataset
        category_maps: Dict of attribute name -> list of category labels
        prefix: Prefix for legend filenames (e.g., "zebrafish", "human", "mouse")
    """
    legends_dir = get_legends_dir()

    # Define display names and attribute mappings
    attr_display_names = {
        "celltype": "Cell Type",
        "chromosome": "Chromosome",
        "leiden_coarse": "Leiden Coarse",
        "leiden_fine": "Leiden Fine",
        "lineage": "Lineage",
        "peak_type": "Peak Type",
        "timepoint": "Timepoint",
    }

    with asection("Generating Legend Images"):
        for attr_name, display_name in attr_display_names.items():
            if attr_name not in attributes:
                aprint(f"  Skipping {attr_name} (not in data)")
                continue

            # Get unique values (indices) sorted
            unique_indices = np.unique(attributes[attr_name])

            # Map indices to category names if available
            if attr_name in category_maps:
                category_names = category_maps[attr_name]
                labels = [
                    format_label(
                        category_names[idx] if idx < len(category_names) else str(idx)
                    )
                    for idx in unique_indices
                ]
            else:
                # Fall back to numeric labels
                labels = [str(idx) for idx in unique_indices]

            # Generate legend (use sequential colormap for timepoints)
            output_path = legends_dir / f"legend_{prefix}_{attr_name}.png"
            use_sequential = attr_name == "timepoint"
            generate_legend_image(
                labels, display_name, output_path, use_sequential=use_sequential
            )

            aprint(f"  {display_name}: {len(labels)} categories -> {output_path.name}")

        aprint(f"\n  Legends saved to: {legends_dir}")


def load_zebrahub_umap_data(
    base_url: str = "https://public.czbiohub.org/royerlab/zebrahub/sequencing/3d-umaps/peak_umap_3d_annotated_v6",
) -> tuple[np.ndarray, dict, dict]:
    """Load 3D UMAP data from Zebrahub public zarr store.

    Args:
        base_url: URL to zarr store

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    with asection("Downloading Zebrahub 3D UMAP Data"):
        # Load coordinates
        aprint("Loading 3D UMAP coordinates...")
        try:
            coords_store = fsspec.get_mapper(f"{base_url}/coords.zarr")
            coords_flat = zarr.open(coords_store, mode="r")[:]
        except Exception as e:
            aprint(f"❌ Failed to load data from {base_url}")
            aprint(f"   Error: {e}")
            aprint("")
            aprint("💡 Possible reasons:")
            aprint("   • Network connection issues")
            aprint("   • Remote server unavailable")
            aprint("   • Data URL has changed")
            aprint("")
            aprint("Please check your internet connection and try again.")
            raise

        # Reshape to (N, 3)
        n_points = len(coords_flat) // 3
        coordinates = coords_flat.reshape(n_points, 3)

        aprint(f"✓ Loaded {n_points:,} points")
        aprint(
            f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]"
        )
        aprint(
            f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]"
        )
        aprint(
            f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]"
        )

        # Load attributes and their category mappings
        aprint("\nLoading cell annotations...")
        attributes = {}
        category_maps = {}

        for attr_name in [
            "celltype",
            "chromosome",
            "leiden_coarse",
            "leiden_fine",
            "lineage",
            "peak_type",
            "timepoint",
        ]:
            try:
                attr_store = fsspec.get_mapper(f"{base_url}/attribute_{attr_name}.zarr")
                z = zarr.open(attr_store, mode="r")
                attr_data = z[:]
                attributes[attr_name] = attr_data

                # Extract category mapping from zarr attrs
                if "map" in z.attrs:
                    category_maps[attr_name] = list(z.attrs["map"])

                n_unique = len(np.unique(attr_data))
                aprint(f"  {attr_name}: {n_unique} unique values")
            except Exception as e:
                aprint(f"  {attr_name}: Failed to load - {e}")

    return coordinates, attributes, category_maps


def attribute_to_color(
    attribute_values: np.ndarray,
    attribute_name: str,
) -> np.ndarray:
    """Convert any attribute to distinct colors.

    Uses curated categorical color palettes for most attributes, and a
    sequential colormap for timepoints to convey temporal progression.

    Args:
        attribute_values: Attribute array
        attribute_name: Name of the attribute (used to determine colormap)

    Returns:
        RGB colors (float32, 0-1 range)
    """
    unique_values = np.unique(attribute_values)
    n_unique = len(unique_values)
    colors = np.zeros((len(attribute_values), 3), dtype=np.float32)

    # Use sequential colormap for timepoints
    use_sequential = attribute_name == "timepoint"

    for i, val in enumerate(unique_values):
        mask = attribute_values == val
        if use_sequential:
            t = i / max(n_unique - 1, 1)  # Normalize to 0-1
            rgb = get_sequential_color(t)
        else:
            rgb = get_categorical_color(i, n_unique)
        # Convert from 0-255 to 0-1 range for float32 colors
        colors[mask] = [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]

    return colors


def create_zebrahub_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
) -> int:
    """Create Luxar scene with categorical attribute visualization.

    Args:
        output_path: Where to write Luxar zarr
        coordinates: (N, 3) UMAP coordinates
        attributes: Dict of attribute arrays

    Returns:
        Number of points
    """
    n_points = len(coordinates)

    with asection("Building Multi-Attribute Scene"):
        # Define attribute types for categorical navigation
        attr_types = [
            "celltype",
            "chromosome",
            "leiden_coarse",
            "leiden_fine",
            "lineage",
            "peak_type",
            "timepoint",
        ]

        # Create one copy of points per attribute type
        all_positions = []
        all_colors = []

        for attr_idx, attr_name in enumerate(attr_types):
            if attr_name in attributes:
                # Generate colors for this attribute
                colors = attribute_to_color(attributes[attr_name], attr_name)

                # Create 4D positions: [attribute_view, x, y, z]
                positions_4d = np.column_stack(
                    [
                        np.full(n_points, attr_idx, dtype=np.float32),
                        coordinates[:, 0],
                        coordinates[:, 1],
                        coordinates[:, 2],
                    ]
                )

                all_positions.append(positions_4d)
                all_colors.append(colors)

                n_unique = len(np.unique(attributes[attr_name]))
                aprint(
                    f"  Attribute {attr_idx} ({attr_name}): {n_unique} unique values"
                )

        # Combine all attribute views
        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"✓ Created {len(attr_types)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Define dimensions with categorical attribute selector
        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=[
                        "Cell Type",
                        "Chromosome",
                        "Leiden Coarse",
                        "Leiden Fine",
                        "Lineage",
                        "Peak Type",
                        "Timepoint",
                    ],
                    display=False,
                    description="Biological attribute for color coding cells in UMAP space",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        # Create scene
        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            # Add points with small radii for dense point cloud
            total_points = len(positions_combined)
            radii = np.full(total_points, 0.02, dtype=np.float32)
            sharpnesses = np.full(total_points, 4.0, dtype=np.float32)

            scene.add_points(
                "Cells",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
            )

        aprint(f"✓ Scene created with {n_points:,} points")

    return n_points


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("ZEBRAHUB MULTIOME PEAK 3D UMAP DEMO")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing single-cell chromatin accessibility from zebrafish!")
    aprint("Data: 3D UMAP of 640k peaks across developmental stages")
    aprint("")
    aprint("📚 References:")
    aprint("   • Zebrahub: https://zebrahub.org")
    aprint("   • Paper: https://www.biorxiv.org/content/10.1101/2024.10.18.618987v1")
    aprint("")
    aprint("What you'll see:")
    aprint("  • 640k points representing chromatin accessibility peaks")
    aprint("  • Colors indicate 30 different cell types")
    aprint("  • Points cluster by cell type in UMAP space")
    aprint("  • Navigate timepoints to see developmental changes")
    aprint("")

    # Load data from remote zarr
    coordinates, attributes, category_maps = load_zebrahub_umap_data()

    # Generate legend images for all attributes
    generate_all_legends(attributes, category_maps)

    # If --no-serve, use persistent directory; otherwise temp for auto-cleanup
    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "zebrahub_multiome_peak_umap.zarr"
        _n_points = create_zebrahub_scene(output_path, coordinates, attributes)
        aprint(f"Dataset generated at {output_path}")
        return

    # Use temporary directory for serving (auto-cleanup on exit)
    with tempfile.TemporaryDirectory(prefix="luxar_demo_zebrahub_") as tmpdir:
        output_path = Path(tmpdir) / "zebrahub_umap.zarr"

        # Create scene
        _n_points = create_zebrahub_scene(output_path, coordinates, attributes)

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate to explore UMAP structure")
        aprint("  - Zoom in to see individual cells")
        aprint("")
        aprint("  Press '1' to select ATTRIBUTE VIEW, then use [/]:")
        aprint("     0: Cell Type (30 types)")
        aprint("     1: Chromosome (genomic location)")
        aprint("     2: Leiden Coarse (broad clusters)")
        aprint("     3: Leiden Fine (detailed clusters)")
        aprint("     4: Lineage (6 lineages)")
        aprint("     5: Peak Type (4 types)")
        aprint("     6: Timepoint (6 developmental stages)")
        aprint("")
        aprint("  Same structure, different colors reveal different biology!")
        aprint("")
        aprint("=" * 70)
        aprint("LAUNCHING VIEWER")
        aprint("=" * 70)
        aprint("Browser will open automatically. Press Ctrl+C when done.")
        aprint("")

        launch_viewer(output_path)

    aprint("Cleanup complete")


if __name__ == "__main__":
    main()
