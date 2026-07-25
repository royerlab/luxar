"""
Spotify Tracks — 3D UMAP Embedding of Audio Features
=====================================================

Embeds ~114K Spotify tracks into 3D space using UMAP on 9 audio features
(danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo). Colored by genre, sized by popularity.

Hover over any point to see the track name, artist, and genre.

Data source: the `maharshipandya/spotify-tracks-dataset` on Hugging Face,
derived from the Spotify Web API (per-track audio features).

Usage:
    python -m luxar.demos.demo_spotify_tracks
    python -m luxar.demos.demo_spotify_tracks --no-serve
    python -m luxar.demos.demo_spotify_tracks --sample=50000

Dependencies:
    pip install luxar[demos]   # includes pandas, umap-learn
"""

DEMO_META = {
    "key": "spotify_tracks",
    "title": "Spotify Tracks",
    "description": "~114K Spotify tracks embedded in 3D by audio features (UMAP), colored by genre.",
    "category": "embeddings",
    "geometry": "points",
    "requirements": {
        "download_mb": 20,
        "compute": "medium",
        "gpu": "none",
        "local_data": None,
    },
    "caches": ["spotify"],
    "outputs": ["spotify_tracks"],
}

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import (
    cache_computed,
    cached_download,
    launch_viewer,
    require_module,
)
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 114000  # All tracks

# Per-track radius ramp (unpopular -> popular), in scene units. Sized against the
# measured local spacing: the track cloud has a median nearest-neighbour distance
# of ~0.051, so the median radius here (~0.020) is ~0.4x that. The previous ramp
# (0.03 + 0.08*p, median 0.058) exceeded the spacing outright, and because this
# UMAP is one dense ball rather than separated clusters it fused into a uniform
# pale blur — 97% of lit pixels lost their hue, so no genre was distinguishable.
RADIUS_BASE = 0.010
RADIUS_POPULARITY_GAIN = 0.028

DATASET_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"

# Audio features used for embedding (9 dimensions)
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Genre supercategory colors (top genres get distinct colors)
GENRE_COLORS: dict[str, tuple[float, float, float]] = {
    "pop": (1.0, 0.4, 0.7),
    "rock": (0.8, 0.2, 0.2),
    "hip-hop": (0.9, 0.6, 0.1),
    "r-n-b": (0.6, 0.3, 0.8),
    "electronic": (0.2, 0.8, 0.9),
    "jazz": (0.3, 0.6, 0.3),
    "classical": (0.9, 0.85, 0.5),
    "country": (0.7, 0.5, 0.2),
    "metal": (0.5, 0.5, 0.5),
    "latin": (1.0, 0.7, 0.2),
    "folk": (0.4, 0.7, 0.4),
    "blues": (0.2, 0.3, 0.8),
    "reggae": (0.2, 0.8, 0.3),
    "punk": (0.9, 0.3, 0.5),
    "soul": (0.7, 0.4, 0.6),
    "indie": (0.5, 0.7, 0.9),
    "dance": (0.1, 0.9, 0.7),
    "ambient": (0.4, 0.5, 0.7),
    "other": (0.5, 0.5, 0.5),
}


def _classify_genre(genre: str) -> str:
    """Map a detailed genre to a supercategory for coloring."""
    g = genre.lower().strip()
    for key in GENRE_COLORS:
        if key == "other":
            continue
        if key in g or g in key:
            return key
    # Common patterns
    if any(x in g for x in ["rap", "trap", "drill"]):
        return "hip-hop"
    if any(x in g for x in ["edm", "house", "techno", "trance", "dubstep"]):
        return "electronic"
    if any(x in g for x in ["r&b", "rnb"]):
        return "r-n-b"
    if any(x in g for x in ["opera", "piano", "symphony"]):
        return "classical"
    if any(x in g for x in ["alt-rock", "hard-rock", "grunge", "garage"]):
        return "rock"
    if any(x in g for x in ["k-pop", "j-pop", "synth-pop"]):
        return "pop"
    if any(x in g for x in ["salsa", "samba", "reggaeton", "bossa"]):
        return "latin"
    if any(x in g for x in ["acoustic", "singer-songwriter"]):
        return "folk"
    if any(x in g for x in ["death-metal", "black-metal", "heavy-metal", "metalcore"]):
        return "metal"
    return "other"


# =============================================================================
# Data Loading
# =============================================================================


def load_spotify_data(
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> tuple[np.ndarray, list[str], list[str], list[str], np.ndarray]:
    """Load Spotify tracks dataset.

    Returns:
        Tuple of (features, track_names, artists, genres, popularity)
    """
    # Gated here, not in main(): the CSV is parsed with pandas on every run.
    pd = require_module("pandas")

    with asection("Downloading Spotify dataset (~20 MB)"):
        csv_path = cached_download(DATASET_URL, "spotify", "dataset.csv")

    with asection("Loading dataset"):
        df = pd.read_csv(csv_path)
        aprint(f"✓ {len(df):,} tracks loaded")
        aprint(f"  Columns: {list(df.columns)}")

        # Drop rows with missing audio features
        df = df.dropna(subset=AUDIO_FEATURES + ["track_name", "artists", "track_genre"])
        aprint(f"  After cleaning: {len(df):,} tracks")

        # Subsample if requested
        if sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            aprint(f"  Sampled: {len(df):,} tracks")

        # Extract audio features
        features = df[AUDIO_FEATURES].values.astype(np.float32)

        # Normalize features to [0, 1]
        feat_min = features.min(axis=0)
        feat_max = features.max(axis=0)
        feat_range = feat_max - feat_min
        feat_range[feat_range == 0] = 1.0  # Avoid division by zero
        features = (features - feat_min) / feat_range

        # Extract metadata
        track_names = df["track_name"].astype(str).tolist()
        # Truncate long artist lists to first artist
        artists = [
            a.split(";")[0].strip() if ";" in a else a
            for a in df["artists"].astype(str)
        ]
        genres = df["track_genre"].astype(str).tolist()
        popularity = df["popularity"].values.astype(np.float32)

        aprint(f"  Unique genres: {len(set(genres))}")
        aprint(f"  Feature shape: {features.shape}")

    return features, track_names, artists, genres, popularity


# =============================================================================
# UMAP Reduction
# =============================================================================


def reduce_to_3d(features: np.ndarray) -> np.ndarray:
    """Reduce audio features to 3D with UMAP (pure compute; caching is external).

    Returns centered 3D positions.
    """
    # Gated here, not in main(): a warm spotify UMAP cache never calls this.
    UMAP = require_module("umap").UMAP

    with asection(
        f"UMAP reduction ({features.shape[0]:,} × {features.shape[1]}D → 3D)"
    ):
        reducer = UMAP(
            n_components=3,
            n_neighbors=30,
            min_dist=0.3,
            metric="euclidean",
            n_jobs=-1,
            verbose=True,
        )
        positions = reducer.fit_transform(features).astype(np.float32)

        # Center
        positions -= positions.mean(axis=0)

        aprint(f"✓ UMAP complete: {positions.shape}")

    return positions


# =============================================================================
# Scene Generation
# =============================================================================


def generate_spotify_landscape(
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> int:
    """Generate 3D landscape of Spotify tracks."""
    # Load data
    features, track_names, artists, genres, popularity = load_spotify_data(sample_size)
    n_tracks = len(track_names)

    if n_tracks == 0:
        aprint("❌ No tracks loaded")
        return 0

    # UMAP — cached under ~/.cache/luxar/spotify, keyed on the track count AND the
    # feature set (bump the key/version if AUDIO_FEATURES or normalization change).
    umap_key = f"umap3d_n{n_tracks}_f{len(AUDIO_FEATURES)}"
    positions = cache_computed("spotify", umap_key, lambda: reduce_to_3d(features))

    # Generate visualization
    with asection("Generating visualization"):
        # Colors by genre supercategory
        colors = np.zeros((n_tracks, 3), dtype=np.float32)
        genre_supercats = []
        for i, genre in enumerate(genres):
            supercat = _classify_genre(genre)
            genre_supercats.append(supercat)
            colors[i] = GENRE_COLORS.get(supercat, GENRE_COLORS["other"])

        # Count genres
        cat_counts: dict[str, int] = {}
        for sc in genre_supercats:
            cat_counts[sc] = cat_counts.get(sc, 0) + 1
        aprint("✓ Tracks by genre:")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
            aprint(f"  {cat}: {count:,}")

        # Size by popularity (more popular = larger)
        pop_norm = popularity / max(popularity.max(), 1.0)
        radii = (RADIUS_BASE + RADIUS_POPULARITY_GAIN * pop_norm).astype(np.float32)

        # Hover labels: track — artist (genre)
        labels = [
            f"{track_names[i][:50]}{'…' if len(track_names[i]) > 50 else ''} — {artists[i]} ({genres[i]})"
            for i in range(n_tracks)
        ]

    # Write to Zarr
    with asection("Writing to Zarr"):
        dims = Dimensions(
            [
                Dimension("x", unit="UMAP", display=True),
                Dimension("y", unit="UMAP", display=True),
                Dimension("z", unit="UMAP", display=True),
            ]
        )

        with LuxarZarrCompiler(output_path) as compiler:
            scene = compiler.create_scene(dimensions=dims)

            sharpness = np.full(n_tracks, 0.6, dtype=np.float32)

            scene.add_points(
                "tracks",
                positions=positions,
                colors=colors,
                radii=radii,
                sharpness=sharpness,
                opacity=0.9,
                intensity=0.12,
                labels=labels,
            )

            # Title overlay
            scene.add_text(
                "Spotify Tracks — Audio Feature Landscape",
                position=(0.02, 0.02),
                font_size=0.055,
                anchor="top-left",
                color="rgba(255,255,255,0.6)",
                blend_mode="difference",
            )

            # Info + source
            scene.add_text(
                f"{n_tracks:,} tracks • UMAP on {len(AUDIO_FEATURES)} audio features • Spotify dataset",
                position=(0.98, 0.97),
                font_size=0.012,
                anchor="bottom-right",
                color="rgba(200,200,200,0.45)",
            )

            # Genre color legend — built from the supercategories ACTUALLY present
            # (most common first) so the baked genre colors are decodable in-viewer.
            legend_cats = [
                c for c, _ in sorted(cat_counts.items(), key=lambda x: -x[1])
            ][:12]
            legend_html = (
                '<div style="font-size:1.3vh;line-height:1.6;background:rgba(0,0,0,0.5);'
                'padding:0.5vh;border-radius:3px">'
                '<div style="font-weight:bold;color:#ccc;margin-bottom:0.3vh">Genre</div>'
            )
            for cat in legend_cats:
                r, g, b = (
                    int(round(v * 255))
                    for v in GENRE_COLORS.get(cat, GENRE_COLORS["other"])
                )
                legend_html += f'<div><span style="color:#{r:02x}{g:02x}{b:02x}">█</span> {cat}</div>'
            legend_html += "</div>"
            scene.add_html(legend_html, position=(0.02, 0.97), anchor="bottom-left")

    aprint(f"✓ Wrote {n_tracks:,} tracks to {output_path}")
    return n_tracks


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """Main demo entry point."""
    aprint("=" * 70)
    aprint("SPOTIFY TRACKS — Audio Feature Landscape")
    aprint("=" * 70)
    aprint("")
    aprint("114K tracks embedded in 3D via UMAP on audio features.")
    aprint("Hover to see track name, artist, and genre.")
    aprint("")

    # Parse sample size
    sample_size = DEFAULT_SAMPLE_SIZE
    for arg in sys.argv[1:]:
        if arg.startswith("--sample="):
            sample_size = int(arg.split("=")[1])
            aprint(f"Sample size: {sample_size:,}")

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "spotify_tracks.luxar.zarr"
        try:
            n_tracks = generate_spotify_landscape(output_path, sample_size=sample_size)
            if n_tracks == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        aprint(f"Dataset generated at {output_path}")
        aprint(f"Total tracks: {n_tracks:,}")
        return

    with tempfile.TemporaryDirectory(prefix="luxar_demo_spotify_") as tmpdir:
        output_path = Path(tmpdir) / "spotify_tracks.luxar.zarr"

        try:
            n_tracks = generate_spotify_landscape(output_path, sample_size=sample_size)
            if n_tracks == 0:
                return
        except Exception as e:
            aprint(f"\nError: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        aprint("")
        aprint("=" * 70)
        aprint("VIEWING TIPS")
        aprint("=" * 70)
        aprint("")
        aprint("Explore the music landscape:")
        aprint("  - Clusters = genres with similar audio profiles")
        aprint("  - Bigger points = more popular tracks")
        aprint("  - Hover over any point to see track info")
        aprint("")
        aprint(f"Total tracks: {n_tracks:,}")
        aprint("")
        aprint("Press Ctrl+C when done.")

        launch_viewer(output_path)

    aprint("✓ Cleanup complete")


if __name__ == "__main__":
    main()
