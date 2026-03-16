"""Post-batch merge orchestration: tiles -> per-(T,C) -> per-C -> final."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from arbol import aprint, asection

from luxar.gsplats.batch.manifest import BatchManifest, output_filename


def merge_batch_results(
    manifest: BatchManifest,
    output_dir: Path,
    channel_colors: Optional[List[Tuple[float, float, float]]] = None,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Run the 3-level fan-in merge for a completed batch job.

    Level 1: Per (T,C) — concatenate tiles.
    Level 2: Per C — stack timepoints via ``combine_as_new_dimension``.
    Level 3: Across channels — merge with channel colors (if provided).

    Args:
        manifest: Loaded batch manifest.
        output_dir: Batch output directory.
        channel_colors: Optional list of RGB tuples for channel coloring.
        force: Re-merge even if output already exists.
        verbose: Print progress.

    Returns:
        Path to the final merged output.
    """
    from luxar.gsplats.gsplat_data import GSplatData

    tiles_dir = output_dir / "tiles"
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    n_t = manifest.n_timepoints
    n_c = manifest.n_channels
    n_k = manifest.n_tiles

    # ================================================================
    # Level 1: Merge tiles per (T, C)
    # ================================================================
    tc_paths: dict[tuple[int, int], Path] = {}

    with asection("Level 1: Merging tiles per (timepoint, channel)"):
        for t in range(n_t):
            for c in range(n_c):
                out_path = merged_dir / f"t{t:02d}_c{c:02d}.gsplats.zarr"
                tc_paths[(t, c)] = out_path

                if out_path.exists() and not force:
                    if verbose:
                        aprint(f"  t={t} c={c}: exists, skipping")
                    continue

                tile_files = []
                for k in range(n_k):
                    fname = output_filename(t, c, k)
                    tile_path = tiles_dir / fname
                    if not tile_path.exists():
                        raise FileNotFoundError(
                            f"Missing tile output: {tile_path}\n"
                            f"Run `luxar gsplat batch status {output_dir}` "
                            f"to check job status."
                        )
                    tile_files.append(tile_path)

                if n_k == 1:
                    # Single tile — just copy/symlink
                    import shutil

                    if out_path.exists():
                        shutil.rmtree(out_path)
                    shutil.copytree(tile_files[0], out_path)
                else:
                    datasets = [GSplatData.load(p) for p in tile_files]
                    merged = GSplatData.concatenate(datasets)
                    merged.save(out_path)

                if verbose:
                    aprint(f"  t={t} c={c}: merged {n_k} tiles")

    # ================================================================
    # Level 2: Stack timepoints per channel (if T > 1)
    # ================================================================
    channel_paths: dict[int, Path] = {}

    if n_t > 1:
        with asection("Level 2: Stacking timepoints per channel"):
            for c in range(n_c):
                out_path = merged_dir / f"c{c:02d}_4d.gsplats.zarr"
                channel_paths[c] = out_path

                if out_path.exists() and not force:
                    if verbose:
                        aprint(f"  c={c}: exists, skipping")
                    continue

                tc_files = [tc_paths[(t, c)] for t in range(n_t)]
                datasets = [GSplatData.load(p) for p in tc_files]
                stacked = GSplatData.combine_as_new_dimension(
                    datasets,
                    values=[float(t) for t in range(n_t)],
                    sigma=0.0,
                )
                stacked.save(out_path)

                if verbose:
                    aprint(
                        f"  c={c}: stacked {n_t} timepoints "
                        f"-> {stacked.ndim}D ({stacked.n_splats:,} splats)"
                    )
    else:
        # Single timepoint — use Level 1 outputs directly
        for c in range(n_c):
            channel_paths[c] = tc_paths[(0, c)]

    # ================================================================
    # Level 3: Merge channels (if C > 1 and colors provided)
    # ================================================================
    if n_c > 1 and channel_colors:
        with asection("Level 3: Merging channels with colors"):
            final_path = merged_dir / "final.gsplats.zarr"

            if final_path.exists() and not force:
                if verbose:
                    aprint("  Final output exists, skipping")
                return final_path

            ch_files = [channel_paths[c] for c in range(n_c)]
            datasets = [GSplatData.load(p) for p in ch_files]
            final = GSplatData.merge_with_channel_colors(datasets, channel_colors)
            final.save(final_path)

            if verbose:
                aprint(f"  Merged {n_c} channels -> {final.n_splats:,} splats")
            return final_path

    # No channel merge needed — pick the single-channel output or
    # the most "final" thing we have
    if n_c == 1:
        final_path = channel_paths[0]
    else:
        # Multiple channels but no colors — concatenate
        with asection("Level 3: Concatenating channels"):
            final_path = merged_dir / "final.gsplats.zarr"
            if not final_path.exists() or force:
                ch_files = [channel_paths[c] for c in range(n_c)]
                datasets = [GSplatData.load(p) for p in ch_files]
                final = GSplatData.concatenate(datasets)
                final.save(final_path)
                if verbose:
                    aprint(f"  Concatenated {n_c} channels")

    return final_path
