"""
Spotify Tracks — 3D UMAP Embedding of Audio Features
=====================================================

Embeds ~114K Spotify tracks into 3D space using UMAP on 9 audio features
(danceability, energy, loudness, speechiness, acousticness, instrumentalness,
liveness, valence, tempo). Colored by genre, sized by popularity.

Hover over any point to see the track name, artist, and genre.

Data source: maharshipandya/spotify-tracks-dataset (Hugging Face, open access)

Usage:
    python -m luxar.demos.demo_spotify_tracks
    python -m luxar.demos.demo_spotify_tracks --no-serve
    python -m luxar.demos.demo_spotify_tracks --sample=50000

Dependencies:
    pip install luxar[demos]   # includes pandas, umap-learn
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from arbol import aprint, asection

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.demos import launch_viewer
from luxar.utils.paths import get_demos_output_dir

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SAMPLE_SIZE = 114000  # All tracks

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

# Map detailed genres to supercategories
GENRE_MAP: dict[str, str] = {}


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
    cache_dir: Path, sample_size: int = DEFAULT_SAMPLE_SIZE
) -> tuple[np.ndarray, list[str], list[str], list[str], np.ndarray]:
    """Load Spotify tracks dataset.

    Returns:
        Tuple of (features, track_names, artists, genres, popularity)
    """
    import pandas as pd

    csv_path = cache_dir / "dataset.csv"

    if not csv_path.exists():
        from luxar.utils.download import robust_download

        with asection("Downloading Spotify dataset (~20 MB)"):
            robust_download(DATASET_URL, csv_path)

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


def reduce_to_3d(
    features: np.ndarray,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Reduce audio features to 3D with UMAP.

    Returns centered 3D positions.
    """
    if cache_path and cache_path.exists():
        with asection("Loading cached UMAP"):
            positions = np.load(cache_path)["positions"]
            aprint(f"✓ Loaded {len(positions):,} positions from cache")
            return positions

    from umap import UMAP

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

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, positions=positions)
        aprint(f"✓ Cached to {cache_path}")

    return positions


# =============================================================================
# Scene Generation
# =============================================================================


def generate_spotify_landscape(
    output_path: Path,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    cache_dir: Path | None = None,
) -> int:
    """Generate 3D landscape of Spotify tracks."""
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "luxar" / "spotify"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    features, track_names, artists, genres, popularity = load_spotify_data(
        cache_dir, sample_size
    )
    n_tracks = len(track_names)

    if n_tracks == 0:
        aprint("❌ No tracks loaded")
        return 0

    # UMAP
    umap_cache = cache_dir / f"umap_{n_tracks}.npz"
    positions = reduce_to_3d(features, cache_path=umap_cache)

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
        radii = (0.03 + 0.08 * pop_norm).astype(np.float32)

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

    # Check dependencies
    try:
        import pandas  # noqa: F401
    except ImportError:
        aprint("Missing dependency: pandas")
        aprint("Install with: pip install luxar[demos]")
        sys.exit(1)

    try:
        import umap  # noqa: F401
    except ImportError:
        aprint("Missing dependency: umap-learn")
        aprint("Install with: pip install luxar[demos]")
        sys.exit(1)

    cache_dir = Path.home() / ".cache" / "luxar" / "spotify"

    if "--no-serve" in sys.argv:
        output_path = get_demos_output_dir() / "spotify_tracks.luxar.zarr"
        try:
            n_tracks = generate_spotify_landscape(
                output_path, sample_size=sample_size, cache_dir=cache_dir
            )
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
            n_tracks = generate_spotify_landscape(
                output_path, sample_size=sample_size, cache_dir=cache_dir
            )
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
