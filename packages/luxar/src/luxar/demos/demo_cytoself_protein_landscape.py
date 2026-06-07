#!/usr/bin/env python3
"""Self-Contained Demo: CytoSelf Protein Localization 3D UMAP

Visualizes ~114K per-image CytoSelf embeddings from the OpenCell dataset as a
3D UMAP point cloud. Each point is a single fluorescence microscopy crop of an
endogenously tagged protein, colored by subcellular localization or protein
identity.

Run behavior:
  First run: Download embeddings from Google Drive, compute 3D UMAP (~10-30 min)
  Subsequent runs: Load cached results instantly

Data source: OpenCell / CytoSelf (CC BY 4.0)
  - Embeddings: Global VQ-VAE-2 representations (9,216-dim per image)
  - Labels: Protein name, subcellular localization
  - 114,806 images across ~1,311 proteins

References:
  - CytoSelf: Kobayashi et al., Nature Methods 2022
    https://doi.org/10.1038/s41592-022-01541-z
  - OpenCell: Cho et al., Science 2022
    https://doi.org/10.1126/science.abi6983
  - GitHub: https://github.com/royerlab/cytoself

Usage:
    python -m luxar.demos.demo_cytoself_protein_landscape
    python -m luxar.demos.demo_cytoself_protein_landscape --no-serve
    python -m luxar.demos.demo_cytoself_protein_landscape --recompute

Dependencies:
    pip install umap-learn pandas requests
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils._umap_utils import (
    attribute_to_color,
    build_legend_html,
    generate_all_legends,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

GDRIVE_EMBEDDINGS_ID = "1s9TL72912HH947SFWcO6tClF91xDOEEi"  # Global_representation.npy
GDRIVE_LABELS_ID = "1fl0lcrZCOkdN2vwXQSHe7i1MC04RXCiE"  # label.csv

# Image data: 10 numpy files with shape (batch, 100, 100, 4)
# Channels: [target_protein_GFP, nucleus_Hoechst, nuclear_distance, nuclear_segmentation]
# These contain ALL crops (train+val+test, ~1.1M total). The embeddings and
# label.csv are the TEST split only (~114K). We use the paired Label_data CSVs
# to identify which image rows correspond to the test split.
GDRIVE_IMAGE_IDS = {
    "Image_data00.npy": "15_CHBPT-p5JG44acP6D2hKd8jAacZatp",
    "Image_data01.npy": "1m7Cj2OALiZTIiHpvb9zFPG_I3j1wRnzK",
    "Image_data02.npy": "17nknzqlcYO3n9bAe4FwGVPkU-mJAhQ4j",
    "Image_data03.npy": "1vEsddF68dyOda-hwI-ptAL4vShBGl98Y",
    "Image_data04.npy": "1aB7WaRuhobG_IDl0l_PPeSJAxCYy-Pye",
    "Image_data05.npy": "1qb0waKcLprDtuFAdCec3WegWkmd-U45A",
    "Image_data06.npy": "1y-1vlfZ4eNhvTvpuqTZVL8DvSwYX3CH_",
    "Image_data07.npy": "1ejcPdh-d5lB1OcZ6x8SJx61pEUioZvB2",
    "Image_data08.npy": "1DOicAkruNsU5F4DWLzO2QrV6xU4kuVxs",
    "Image_data09.npy": "1a5YyHeRSRdJStG3KnFe2vsNjrsit9zbf",
}

# Label data: 10 CSV files paired with Image_data (same row ordering).
# Used to match test-split embeddings back to their image crops.
GDRIVE_LABEL_DATA_IDS = {
    "Label_data00.csv": "1CVwvXW2KhVBbTBixwRXIIiMhrlGDXz-4",
    "Label_data01.csv": "1mTYe5icvWXNfY5wEsuQUhSwgtefBJpjg",
    "Label_data02.csv": "1HckmktklyPo6qbakrwtERsCT34mRdn7l",
    "Label_data03.csv": "1GBxDmWcl_o49i4lGujA8EgIn5G4htkBr",
    "Label_data04.csv": "1G4FpJnlqB3ejmdw3SF2w3DFYt8Wnq0fT",
    "Label_data05.csv": "1Vo1J09qP2TAoXwltCF84socz2TPV92JU",
    "Label_data06.csv": "1d7gJjLTQhOw-e9KZJY9pr6KOCIN8NBvp",
    "Label_data07.csv": "1kr5EF0RA3ZwSXmoaBFwFDVnrokh2EaOE",
    "Label_data08.csv": "1mXyedmLezzty2LSSH3asw0LQeu-ie9mz",
    "Label_data09.csv": "1Vdv1cD75VhvC3FdKTen-5rqLJnWpHvmb",
}

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "luxar" / "cytoself"


# =============================================================================
# Google Drive Download
# =============================================================================


def _download_from_google_drive(
    file_id: str, output_path: Path, expected_min_size: int = 0
) -> Path:
    """Download a file from Google Drive, handling the virus-scan confirmation.

    Args:
        file_id: Google Drive file ID.
        output_path: Where to save the downloaded file.
        expected_min_size: Minimum expected file size in bytes (for cache check).

    Returns:
        Path to downloaded file.
    """
    import requests

    # Check if already downloaded
    if output_path.exists() and output_path.stat().st_size > expected_min_size * 0.9:
        aprint(f"File already downloaded: {output_path.name}")
        aprint(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={file_id}"

    session = requests.Session()

    with asection(f"Downloading {output_path.name} from Google Drive"):
        aprint(f"File ID: {file_id}")

        # Strategy 1: Direct download with confirm=t
        response = session.get(url, params={"confirm": "t"}, stream=True, timeout=60)

        # Strategy 2: Check cookies for download_warning token
        if response.headers.get("content-type", "").startswith("text/html"):
            aprint("Trying cookie-based confirmation...")
            confirm_token = None
            for key, value in response.cookies.items():
                if key.startswith("download_warning"):
                    confirm_token = value
                    break
            if confirm_token:
                response = session.get(
                    url,
                    params={"confirm": confirm_token},
                    stream=True,
                    timeout=60,
                )

        # Strategy 3: Parse the HTML confirmation page
        if response.headers.get("content-type", "").startswith("text/html"):
            import html as html_mod
            import re

            aprint("Parsing confirmation page for download form...")
            page_html = response.text

            action_match = re.search(r'action="([^"]*)"', page_html)
            form_inputs = dict(
                re.findall(
                    r'<input[^>]*name="([^"]*)"[^>]*value="([^"]*)"',
                    page_html,
                )
            )

            if action_match and form_inputs:
                action_url = html_mod.unescape(action_match.group(1))
                aprint(f"Found download form with {len(form_inputs)} params")
                response = session.get(
                    action_url,
                    params=form_inputs,
                    stream=True,
                    timeout=60,
                )
            else:
                # Strategy 4: Try the usercontent endpoint
                aprint("Trying usercontent endpoint...")
                uc_url = (
                    f"https://drive.usercontent.google.com/download"
                    f"?id={file_id}&export=download&confirm=t"
                )
                response = session.get(uc_url, stream=True, timeout=60)

        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        if total_size > 0:
            aprint(f"Download size: {total_size / (1024**2):.1f} MB")

        downloaded = 0
        last_report_mb = 0
        start_time = time.time()
        chunk_size = 1024 * 1024  # 1 MB

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    progress_mb = downloaded / (1024 * 1024)
                    if progress_mb - last_report_mb >= 100:
                        elapsed = time.time() - start_time
                        rate = (
                            downloaded / (1024 * 1024) / elapsed if elapsed > 0 else 0
                        )
                        if total_size > 0:
                            pct = downloaded / total_size * 100
                            aprint(
                                f"  {downloaded / (1024**3):.2f} / "
                                f"{total_size / (1024**3):.2f} GB "
                                f"({pct:.0f}%) - {rate:.1f} MB/s"
                            )
                        else:
                            aprint(
                                f"  {downloaded / (1024**3):.2f} GB - {rate:.1f} MB/s"
                            )
                        last_report_mb = progress_mb

        final_size = output_path.stat().st_size
        aprint(f"Download complete: {final_size / (1024**2):.1f} MB")

        # Sanity check — Google Drive HTML error pages are small
        if final_size < 1_000:
            output_path.unlink()
            raise RuntimeError(
                "Downloaded file is too small — likely a Google Drive error page. "
                "Try downloading manually from: "
                f"https://drive.google.com/file/d/{file_id}/view?usp=sharing "
                f"and place it at {output_path}"
            )

    return output_path


# =============================================================================
# Data Loading & UMAP
# =============================================================================


def load_cytoself_data(
    cache_dir: Path | None = None,
    recompute: bool = False,
) -> tuple[np.ndarray, dict, dict]:
    """Load CytoSelf embeddings and compute 3D UMAP.

    Downloads Global_representation.npy (~3.9 GB) and label.csv from Google Drive,
    then runs UMAP dimensionality reduction (cached after first run).

    Args:
        cache_dir: Directory for caching downloads and UMAP results.
        recompute: If True, recompute UMAP even if cache exists.

    Returns:
        Tuple of (coordinates, attributes, category_maps) where:
        - coordinates: (N, 3) array of 3D UMAP positions
        - attributes: dict of attribute arrays (numeric indices)
        - category_maps: dict of attribute name -> list of category labels
    """
    import pandas as pd

    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Download raw data ---
    embeddings_path = cache_dir / "Global_representation.npy"
    labels_path = cache_dir / "label.csv"

    with asection("Loading CytoSelf Data"):
        _download_from_google_drive(
            GDRIVE_EMBEDDINGS_ID,
            embeddings_path,
            expected_min_size=4_000_000_000,  # ~3.9 GB
        )
        _download_from_google_drive(
            GDRIVE_LABELS_ID,
            labels_path,
            expected_min_size=5_000_000,  # ~6 MB
        )

    # --- Extract attributes from labels ---
    with asection("Processing Labels"):
        df = pd.read_csv(labels_path)
        aprint(f"Loaded {len(df):,} rows with columns: {list(df.columns)}")

        attributes: dict[str, np.ndarray] = {}
        category_maps: dict[str, list] = {}

        # Columns: ensg, name, loc_grade1, loc_grade2, loc_grade3, protein_id, FOV_id
        # loc_grade1 = primary localization (semicolon-separated multi-labels)
        # name = protein name (~1,311 unique)

        # Extract primary localization (first label before semicolon, NaN -> "unknown")
        if "loc_grade1" in df.columns:
            primary_loc = df["loc_grade1"].fillna("unknown").str.split(";").str[0]
            cat = pd.Categorical(primary_loc)
            attributes["localization"] = cat.codes.astype(np.int32)
            category_maps["localization"] = list(cat.categories)
            aprint(f"  localization: {len(cat.categories)} unique values")

        # Protein name
        if "name" in df.columns:
            cat = pd.Categorical(df["name"])
            attributes["protein_name"] = cat.codes.astype(np.int32)
            category_maps["protein_name"] = list(cat.categories)
            aprint(f"  protein_name: {len(cat.categories)} unique values")

        if not attributes:
            raise RuntimeError(
                f"No usable columns found in label.csv. "
                f"Available columns: {list(df.columns)}"
            )

    # --- Compute or load cached 3D UMAP ---
    umap_cache = cache_dir / "umap_3d.npz"

    if umap_cache.exists() and not recompute:
        with asection("Loading cached 3D UMAP"):
            coordinates = np.load(umap_cache)["positions"]
            aprint(f"Loaded {len(coordinates):,} points from cache")
    else:
        from umap import UMAP

        with asection("Computing 3D UMAP (this may take 10-30 minutes)"):
            aprint("Loading embeddings into memory...")
            embeddings = np.load(embeddings_path)
            aprint(f"Embeddings shape: {embeddings.shape}")
            aprint("Parameters: n_neighbors=15, min_dist=0.1, metric=cosine")

            reducer = UMAP(
                n_components=3,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                n_jobs=-1,
                low_memory=True,
                verbose=True,
            )
            coordinates = reducer.fit_transform(embeddings).astype(np.float32)
            coordinates -= coordinates.mean(axis=0)
            aprint(f"UMAP complete: {coordinates.shape}")

            np.savez(umap_cache, positions=coordinates)
            aprint(f"Cached to {umap_cache}")

    aprint(f"Coordinates: {len(coordinates):,} points")
    aprint(f"  X range: [{coordinates[:, 0].min():.1f}, {coordinates[:, 0].max():.1f}]")
    aprint(f"  Y range: [{coordinates[:, 1].min():.1f}, {coordinates[:, 1].max():.1f}]")
    aprint(f"  Z range: [{coordinates[:, 2].min():.1f}, {coordinates[:, 2].max():.1f}]")

    return coordinates, attributes, category_maps


def _encode_crops_to_webp(
    images: np.ndarray,
) -> list[bytes]:
    """Encode (N, 100, 100, 4) image crops to WebP thumbnails.

    Takes channels 0 (protein GFP) and 1 (nucleus Hoechst), normalizes
    per-image to uint8, composites as green+blue RGB, and encodes to WebP.
    """
    import io

    from PIL import Image as PILImage

    n_crops = images.shape[0]

    # Extract channels: 0 = protein (GFP), 1 = nucleus (Hoechst)
    protein = images[:, :, :, 0].astype(np.float32)
    nucleus = images[:, :, :, 1].astype(np.float32)

    # Per-image min-max normalization to uint8
    def normalize(ch: np.ndarray) -> np.ndarray:
        flat = ch.reshape(ch.shape[0], -1)
        ch_min = flat.min(axis=1)[:, np.newaxis, np.newaxis]
        ch_max = flat.max(axis=1)[:, np.newaxis, np.newaxis]
        return ((ch - ch_min) / np.maximum(ch_max - ch_min, 1e-8) * 255).astype(
            np.uint8
        )

    protein_u8 = normalize(protein)
    nucleus_u8 = normalize(nucleus)

    # RGB composite: green = protein, blue = nucleus
    rgb = np.zeros((n_crops, 100, 100, 3), dtype=np.uint8)
    rgb[:, :, :, 1] = protein_u8
    rgb[:, :, :, 2] = nucleus_u8

    blobs: list[bytes] = []
    for i in range(n_crops):
        buf = io.BytesIO()
        PILImage.fromarray(rgb[i], mode="RGB").save(buf, format="webp", quality=85)
        blobs.append(buf.getvalue())

    return blobs


def _build_test_index_mapping(
    cache_dir: Path,
) -> tuple[dict[int, int], int]:
    """Build a mapping from test-split row index to global image row index.

    Downloads the 10 Label_data CSVs (small files), concatenates them to get
    the full ~1.1M-row label table, then matches rows from label.csv (the
    test split) to find their positions in the full dataset.

    The Image_data .npy files are row-aligned with the Label_data CSVs, so
    the resulting mapping tells us which image rows to encode.

    Args:
        cache_dir: Directory for caching downloads.

    Returns:
        Tuple of (mapping, n_test) where mapping is
        test_row_index -> global_image_row_index and n_test is the total
        number of test rows.
    """
    import pandas as pd

    with asection("Building test-to-image index mapping"):
        # Download the 10 Label_data CSVs (small: ~5-10 MB each)
        label_dfs = []
        for filename, file_id in GDRIVE_LABEL_DATA_IDS.items():
            csv_path = cache_dir / filename
            _download_from_google_drive(file_id, csv_path, expected_min_size=1_000_000)
            df = pd.read_csv(csv_path, header=None)
            label_dfs.append(df)

        full_labels = pd.concat(label_dfs, ignore_index=True)
        aprint(
            f"Full label table: {len(full_labels):,} rows x {full_labels.shape[1]} cols"
        )

        # Load test-split labels
        test_labels = pd.read_csv(cache_dir / "label.csv")
        aprint(f"Test labels: {len(test_labels):,} rows")

        # The Label_data CSVs have no header; label.csv has headers.
        # Assign column names from label.csv to the full table.
        if full_labels.shape[1] == len(test_labels.columns):
            full_labels.columns = test_labels.columns
        else:
            aprint(
                f"  ⚠ Column count mismatch: full={full_labels.shape[1]}, "
                f"test={len(test_labels.columns)}"
            )
            # Try using the first N columns that match
            full_labels.columns = [
                test_labels.columns[i] if i < len(test_labels.columns) else f"col_{i}"
                for i in range(full_labels.shape[1])
            ]

        # Build composite key for matching. Use all available columns for
        # an exact row match. Fill NaN consistently so keys match.
        key_cols = list(test_labels.columns)
        for col in key_cols:
            full_labels[col] = full_labels[col].fillna("__NA__").astype(str)
            test_labels[col] = test_labels[col].fillna("__NA__").astype(str)

        full_labels["_key"] = full_labels[key_cols].agg("|".join, axis=1)
        test_labels["_key"] = test_labels[key_cols].agg("|".join, axis=1)

        # Build lookup: key -> list of global indices (duplicates possible)
        from collections import defaultdict

        key_to_global: dict[str, list[int]] = defaultdict(list)
        for global_idx, key in enumerate(full_labels["_key"]):
            key_to_global[key].append(global_idx)

        # Match each test row. For duplicate keys, consume indices in order.
        key_usage: dict[str, int] = defaultdict(int)
        mapping: dict[int, int] = {}
        unmatched = 0

        for test_idx, key in enumerate(test_labels["_key"]):
            candidates = key_to_global.get(key, [])
            usage = key_usage[key]
            if usage < len(candidates):
                mapping[test_idx] = candidates[usage]
                key_usage[key] = usage + 1
            else:
                unmatched += 1

        aprint(
            f"Matched {len(mapping):,}/{len(test_labels):,} test rows "
            f"to image indices ({unmatched:,} unmatched)"
        )

    return mapping, len(test_labels)


def load_cytoself_images(
    cache_dir: Path | None = None,
) -> list[bytes]:
    """Load and encode CytoSelf image crops as WebP thumbnails.

    Downloads the 10 Label_data CSVs to identify which of the ~1.1M image
    crops correspond to the 114K test-split embeddings, then downloads and
    encodes only the matched crops from the Image_data .npy files.

    Args:
        cache_dir: Directory for caching downloads and encoded thumbnails.

    Returns:
        List of WebP-encoded bytes aligned with label.csv / embeddings.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for cached encoded thumbnails (test-aligned)
    thumbnails_cache = cache_dir / "image_labels_test_webp.npz"
    if thumbnails_cache.exists():
        with asection("Loading cached image thumbnails"):
            data = np.load(thumbnails_cache, allow_pickle=True)
            blobs = list(data["blobs"])
            aprint(f"Loaded {len(blobs):,} cached thumbnails (test-aligned)")
            return [bytes(b) for b in blobs]

    # Step 1: Build index mapping (test row -> global image row)
    mapping, n_test = _build_test_index_mapping(cache_dir)

    # Invert: global_index -> list of test_indices (for per-file processing)
    global_to_test: dict[int, list[int]] = {}
    for test_idx, global_idx in mapping.items():
        global_to_test.setdefault(global_idx, []).append(test_idx)

    # Step 2: Process Image_data files, encoding only matched crops
    result_blobs: list[bytes | None] = [None] * n_test
    global_offset = 0
    matched_count = 0

    with asection("Downloading and encoding matched image crops"):
        for i, (filename, file_id) in enumerate(GDRIVE_IMAGE_IDS.items()):
            img_path = cache_dir / filename

            _download_from_google_drive(
                file_id, img_path, expected_min_size=400_000_000
            )

            with asection(f"Processing {filename} ({i + 1}/{len(GDRIVE_IMAGE_IDS)})"):
                try:
                    arr = np.load(img_path)
                except Exception:
                    aprint("  ⚠ Corrupt file detected, re-downloading...")
                    img_path.unlink(missing_ok=True)
                    _download_from_google_drive(
                        file_id, img_path, expected_min_size=400_000_000
                    )
                    arr = np.load(img_path)

                n_crops = arr.shape[0]
                aprint(f"Shape: {arr.shape}, dtype: {arr.dtype}")

                # Find which local indices in this file are needed
                local_indices = []
                local_to_test_map: list[tuple[int, int]] = []
                for local_idx in range(n_crops):
                    global_idx = global_offset + local_idx
                    if global_idx in global_to_test:
                        local_indices.append(local_idx)
                        for test_idx in global_to_test[global_idx]:
                            local_to_test_map.append((local_idx, test_idx))

                if local_indices:
                    # Extract and encode only the needed crops
                    unique_local = sorted(set(local_indices))
                    subset = arr[unique_local]
                    encoded = _encode_crops_to_webp(subset)

                    # Map encoded blobs back to test indices
                    local_to_encoded = {li: ei for ei, li in enumerate(unique_local)}
                    for local_idx, test_idx in local_to_test_map:
                        encoded_idx = local_to_encoded[local_idx]
                        result_blobs[test_idx] = encoded[encoded_idx]
                        matched_count += 1

                    aprint(
                        f"Encoded {len(unique_local):,} matched crops "
                        f"(of {n_crops:,} total)"
                    )
                else:
                    aprint(f"No matched crops in this file ({n_crops:,} total)")

                global_offset += n_crops
                del arr

        aprint(f"Total matched: {matched_count:,}/{n_test:,}")

    # Fill any unmatched slots with a 1x1 transparent placeholder
    import io

    from PIL import Image as PILImage

    placeholder = io.BytesIO()
    PILImage.new("RGB", (1, 1), (0, 0, 0)).save(placeholder, format="webp")
    placeholder_bytes = placeholder.getvalue()
    n_filled = 0
    for idx in range(n_test):
        if result_blobs[idx] is None:
            result_blobs[idx] = placeholder_bytes
            n_filled += 1
    if n_filled:
        aprint(f"  Filled {n_filled:,} unmatched slots with placeholder")

    final_blobs: list[bytes] = [b for b in result_blobs if b is not None]

    # Cache the test-aligned thumbnails
    with asection("Caching test-aligned thumbnails"):
        np.savez(thumbnails_cache, blobs=np.array(final_blobs, dtype=object))
        aprint(f"Cached {len(final_blobs):,} thumbnails to {thumbnails_cache}")

    return final_blobs


# =============================================================================
# Scene Construction
# =============================================================================


def create_cytoself_scene(
    output_path: Path,
    coordinates: np.ndarray,
    attributes: dict,
    category_maps: dict | None = None,
    image_labels: list[bytes] | None = None,
) -> int:
    """Create Luxar scene with categorical attribute visualization.

    Args:
        output_path: Where to write Luxar zarr
        coordinates: (N, 3) UMAP coordinates
        attributes: Dict of attribute arrays
        category_maps: Dict of attribute name -> list of category labels
        image_labels: Optional list of WebP-encoded image blobs (one per point)

    Returns:
        Number of points
    """
    n_points = len(coordinates)

    with asection("Building Multi-Attribute Scene"):
        attr_types = ["localization", "protein_name"]
        category_labels = ["Localization", "Protein"]

        # Create one copy of points per attribute type
        all_positions = []
        all_colors = []
        available_attrs = [name for name in attr_types if name in attributes]

        for attr_idx, attr_name in enumerate(available_attrs):
            colors = attribute_to_color(attributes[attr_name], attr_name)

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
            aprint(f"  Attribute {attr_idx} ({attr_name}): {n_unique} unique values")

        positions_combined = np.vstack(all_positions)
        colors_combined = np.vstack(all_colors)

        aprint(f"Created {len(available_attrs)} attribute views")
        aprint(f"  Total points: {len(positions_combined):,} ({n_points:,} per view)")

        # Filter category_labels to match available attributes
        available_labels = [
            label
            for label, name in zip(category_labels, attr_types)
            if name in attributes
        ]

        dims = Dimensions(
            [
                Dimension(
                    "attribute",
                    unit="",
                    categories=available_labels,
                    display=False,
                    description="Color coding attribute for CytoSelf embeddings",
                ),
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            total_points = len(positions_combined)
            radii = np.full(total_points, 0.02, dtype=np.float32)
            sharpnesses = np.full(total_points, 0.6, dtype=np.float32)

            # Hover labels: resolve codes to category names, repeated per view
            per_cell_labels = []
            if category_maps:
                for i in range(n_points):
                    parts = []
                    for attr_name in available_attrs:
                        code = int(attributes[attr_name][i])
                        cats = category_maps.get(attr_name, [])
                        name = str(cats[code]) if code < len(cats) else str(code)
                        parts.append(name)
                    per_cell_labels.append("\n".join(parts))
            labels = per_cell_labels * len(available_attrs) if per_cell_labels else None

            # Image labels: replicate per attribute view (same as text labels).
            # Only use if count matches embeddings — the image .npy files may
            # contain more crops than the embedding/label rows.
            all_image_labels = None
            if image_labels is not None:
                if len(image_labels) == n_points:
                    all_image_labels = image_labels * len(available_attrs)
                else:
                    aprint(
                        f"  ⚠ Skipping image labels: count mismatch "
                        f"({len(image_labels):,} images vs {n_points:,} embeddings)"
                    )

            scene.add_points(
                "Images",
                positions_combined,
                colors=colors_combined,
                radii=radii,
                sharpness=sharpnesses,
                opacity=0.8,
                intensity=0.18,
                labels=labels,
                image_labels=all_image_labels,
            )

            # --- Overlays ---
            scene.add_text(
                "CytoSelf Protein Localization UMAP",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            attr_labels = available_labels
            attr_keys = available_attrs
            for attr_id, (label, attr_key) in enumerate(zip(attr_labels, attr_keys)):
                scene.add_text(
                    f"Colored by: {label}",
                    position=(0.02, 0.97),
                    font_size=0.015,
                    anchor="bottom-left",
                    color="#ffcc44",
                    visible_range={"attribute": attr_id},
                    transition="fade",
                    transition_duration=0.2,
                )

                if category_maps and attr_key in category_maps:
                    legend_html = build_legend_html(
                        attr_key, category_maps[attr_key], attributes.get(attr_key)
                    )
                    if legend_html:
                        scene.add_html(
                            legend_html,
                            position=(0.98, 0.5),
                            anchor="center-right",
                            opacity=0.9,
                            visible_range={"attribute": attr_id},
                            transition="fade",
                            transition_duration=0.2,
                        )

            # Custom hover overlays: image top-right, text to its left
            # (two lines via \n separator in labels). Defining these
            # suppresses the auto-injected default hover overlay.
            if all_image_labels is not None:
                scene.add_html(
                    "{hover_image_label}",
                    position=(0.98, 0.02),
                    anchor="top-right",
                    opacity=1.0,
                    transition="fade",
                    transition_duration=0.15,
                    hover=True,
                    hover_image_size=(0.15, 0.20),
                )
            if labels is not None:
                scene.add_text(
                    "{hover_label}",
                    position=(0.82, 0.02),
                    anchor="top-right",
                    font_size=0.018,
                    color="white",
                    width=0.12,
                    text_align="right",
                    background="rgba(0,0,0,0.7)",
                    padding=0.008,
                    opacity=1.0,
                    transition="fade",
                    transition_duration=0.15,
                    hover=True,
                )

            scene.add_text(
                f"{n_points:,} images \u2022 OpenCell \u2022 3D UMAP \u2022 Kobayashi et al., Nat Methods 2022",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

        aprint(f"Scene created with {n_points:,} points")

    return n_points


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("CYTOSELF PROTEIN LOCALIZATION LANDSCAPE")
    aprint("=" * 70)
    aprint("")
    aprint("Visualizing ~114K fluorescence microscopy image embeddings")
    aprint("from the OpenCell proteome-scale dataset.")
    aprint("")
    aprint("Each point is a single cell crop, positioned by CytoSelf's")
    aprint("self-supervised VQ-VAE-2 embedding, reduced to 3D via UMAP.")
    aprint("Points cluster by subcellular localization.")
    aprint("")
    aprint("References:")
    aprint("  CytoSelf: Kobayashi et al., Nature Methods 2022")
    aprint("  OpenCell: Cho et al., Science 2022")
    aprint("  GitHub:   https://github.com/royerlab/cytoself")
    aprint("")
    aprint("NOTE: First run downloads ~4 GB embeddings + ~4-17 GB images")
    aprint("      and computes UMAP (~10-30 min). Requires ~16 GB RAM.")
    aprint("      Subsequent runs load from cache.")
    aprint("      Use --without-images to skip image download.")
    aprint("")

    # Check runtime dependencies
    for module_name, pip_name in [
        ("umap", "umap-learn"),
        ("pandas", "pandas"),
        ("requests", "requests"),
    ]:
        try:
            __import__(module_name)
        except ImportError:
            aprint(f"Missing dependency: {pip_name}")
            aprint(f"Install with: pip install {pip_name}")
            sys.exit(1)

    recompute = "--recompute" in sys.argv
    without_images = "--without-images" in sys.argv

    # Load data (downloads + UMAP on first run, cached thereafter)
    coordinates, attributes, category_maps = load_cytoself_data(
        recompute=recompute,
    )

    # Load image labels (unless opted out)
    image_labels: list[bytes] | None = None
    if not without_images:
        try:
            from PIL import Image as _PILImage  # noqa: F401

            image_labels = load_cytoself_images()
        except ImportError:
            aprint("WARNING: Pillow not installed — skipping image labels.")
            aprint("  Install with: pip install Pillow")
        except Exception as e:
            aprint(f"WARNING: Failed to load images — skipping: {e}")
    else:
        aprint("Skipping image labels (--without-images)")

    # Generate legend images
    generate_all_legends(
        attributes,
        category_maps,
        prefix="cytoself",
        attr_display_names={
            "localization": "Localization",
            "protein_name": "Protein",
        },
    )

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "cytoself_protein_landscape.zarr"
        _n_points = create_cytoself_scene(
            output_path,
            coordinates,
            attributes,
            category_maps,
            image_labels=image_labels,
        )
        aprint(f"Dataset generated at {output_path}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_cytoself_") as tmpdir:
        output_path = Path(tmpdir) / "cytoself_landscape.zarr"

        _n_points = create_cytoself_scene(
            output_path,
            coordinates,
            attributes,
            category_maps,
            image_labels=image_labels,
        )

        aprint("")
        aprint("=" * 70)
        aprint("NAVIGATION")
        aprint("=" * 70)
        aprint("Once viewer opens:")
        aprint("")
        aprint("  - Rotate to explore UMAP structure")
        aprint("  - Zoom in to see individual images")
        aprint("  - Hover over a point to see its fluorescence image")
        aprint("")
        aprint("  Press '1' to select ATTRIBUTE VIEW, then use [/]:")
        aprint("     0: Localization (subcellular compartment)")
        aprint("     1: Protein (~1,311 unique proteins)")
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
