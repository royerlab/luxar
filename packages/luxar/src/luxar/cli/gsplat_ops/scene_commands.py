"""``luxar gsplat scene`` commands (extracted from gsplat_commands.py).

Each command is a plain function; ``register_scene_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer (package-refactor-plan P3/P4/P6).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import typer
from arbol import aprint, asection

from .encoding import _resolve_encoding_mode

if TYPE_CHECKING:
    pass


def convert_to_scene(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene path"),
    center: bool = typer.Option(
        True, "--center/--no-center", help="Center at amplitude-weighted centroid"
    ),
    scale_intensity: Optional[float] = typer.Option(
        None, "--scale-intensity", help="Scale amplitudes by factor (e.g., 0.1)"
    ),
    opacity: float = typer.Option(1.0, "--opacity", help="Opacity (0.0-1.0)"),
    blending_mode: str = typer.Option(
        "additive", "--blending-mode", help="Blending: additive/normal/max/opaque"
    ),
    colormap: Optional[str] = typer.Option(
        None,
        "--colormap",
        help="Colormap for scalar amplitudes (e.g. plasma, viridis, gray). "
        "Builtin or any matplotlib/colorcet name. Default: gray.",
    ),
    tone_mapping: Optional[str] = typer.Option(
        None,
        "--tone-mapping",
        help="Scene HDR tone-mapping (None/Linear/Reinhard/Cineon/ACES/AgX/"
        "Neutral). Default: viewer default (ACES). Use 'Neutral' for faithful "
        "colormap colors (ACES shifts hues).",
    ),
    gamma: Optional[float] = typer.Option(
        None, "--gamma", help="Display gamma (default 1.0)"
    ),
    intensity: Optional[float] = typer.Option(
        None, "--intensity", help="Display intensity multiplier (default 1.0)"
    ),
    layer: bool = typer.Option(
        True,
        "--layer/--no-layer",
        help="List the gsplats node in the viewer Layers panel",
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode"
    ),
) -> None:
    """Convert a .gsplats.zarr dataset to a Luxar scene for the web viewer.

    Creates a persistent Luxar scene zarr that can be served with
    ``luxar serve``. By default, centers the data at the centroid.

    Appearance (colormap / tone-mapping / gamma / intensity) is baked into the
    scene here. For faithful scientific colors pair a colormap with Neutral
    tone-mapping — the viewer default (ACES) intentionally shifts hues.

    Examples:
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --no-center
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --scale-intensity 0.1
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr \\
            --colormap plasma --tone-mapping Neutral
    """
    try:
        import numpy as np

        from luxar import LuxarZarrCompiler
        from luxar.cli.gsplat_config import build_dimensions_from_data
        from luxar.core.viewer_config import VALID_TONE_MAPPINGS, ViewerConfig
        from luxar.gsplats.gsplat_data import GSplatData
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.tree import center_bounds, is_matrix_shaped

        # Validate + assemble appearance attrs (only forward what was set).
        appearance: dict = {"layer": layer}
        if colormap is not None:
            from luxar.colormaps import BUILTIN_COLORMAP_NAMES
            from luxar.colormaps.registry import resolve_colormap

            if colormap not in BUILTIN_COLORMAP_NAMES:
                try:
                    resolve_colormap(colormap)
                except Exception as e:
                    raise typer.BadParameter(
                        f"--colormap '{colormap}' is not a known builtin / "
                        f"matplotlib / colorcet colormap"
                    ) from e
            appearance["colormap"] = colormap
        if gamma is not None:
            appearance["gamma"] = gamma
        if intensity is not None:
            appearance["intensity"] = intensity

        viewer_config = None
        if tone_mapping is not None:
            if tone_mapping not in VALID_TONE_MAPPINGS:
                raise typer.BadParameter(
                    f"--tone-mapping must be one of {VALID_TONE_MAPPINGS}, "
                    f"got '{tone_mapping}'"
                )
            viewer_config = ViewerConfig(tone_mapping=tone_mapping)

        with asection(f"Converting: {input_path.name} -> {output_path.name}"):
            with asection("Loading gsplat dataset"):
                # Peek at the on-disk shape. A matrix-shaped tree (leaf / additive
                # ladder / kind=lod of leaves) round-trips through GSplatData and
                # supports --center / --scale-intensity. A partition / nested tree
                # has no flat GSplatData equivalent: it is grafted node-for-node.
                node, _ = load_gsplat_node(input_path)
                matrix = is_matrix_shaped(node)

            if matrix:
                data = GSplatData.from_tree(node)
                aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")

                if center:
                    aprint("Centering at amplitude-weighted centroid")
                    data = data.center_at_centroid()
                if scale_intensity is not None:
                    aprint(f"Scaling intensity by {scale_intensity}")
                    data = data.scale_intensity(scale_intensity)

                with asection("Creating Luxar scene"):
                    dims = build_dimensions_from_data(data.centers)
                    with LuxarZarrCompiler(
                        output_path, encoding_mode=_resolve_encoding_mode(encoding)
                    ) as compiler:
                        scene = compiler.create_scene(
                            dimensions=dims, viewer_config=viewer_config
                        )
                        scene.add_gsplats_from_data(
                            name="gsplats",
                            result=data,
                            opacity=opacity,
                            blending_mode=blending_mode,
                            **appearance,
                        )
            else:
                kind = (
                    "partition"
                    if node.__class__.__name__ == "GSplatPartition"
                    else "nested LOD"
                )
                aprint(f"Grafting a {kind} node tree (no flat-data transforms apply)")
                if scale_intensity is not None:
                    aprint(
                        "⚠️  --scale-intensity is ignored for a partition/nested "
                        "file (re-author intensity upstream with `gsplat transform`)."
                    )
                # --center defaults True; it does not apply to a graft (the file's
                # own coordinates are preserved), so note it rather than fail.
                if center:
                    aprint(
                        "ℹ️  --center is ignored for a partition/nested file; the "
                        "node tree keeps its authored coordinates."
                    )
                bounds = center_bounds(node)
                if bounds is None:
                    raise ValueError("Could not derive bounds from the node tree")
                bmin, bmax = bounds
                box = np.array([bmin, bmax], dtype=np.float32)

                with asection("Creating Luxar scene"):
                    dims = build_dimensions_from_data(box)
                    with LuxarZarrCompiler(
                        output_path, encoding_mode=_resolve_encoding_mode(encoding)
                    ) as compiler:
                        scene = compiler.create_scene(
                            dimensions=dims, viewer_config=viewer_config
                        )
                        scene.add_gsplats_from_file(
                            name="gsplats",
                            path=input_path,
                            opacity=opacity,
                            blending_mode=blending_mode,
                            **appearance,
                        )

            aprint(f"\nScene saved: {output_path}")
            aprint(f"Serve with: luxar serve {output_path} --viewer")

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:
        aprint(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def migrate_format_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Legacy .gsplats.zarr (v1.0 / v1.1 / v2.0), .gsplats.zarr.zip/.tar.gz, "
        "a substitutive directory (with manifest.json + level_<i>.gsplats.zarr), "
        "or a v3.0/v3.1 store whose kind=lod groups still carry the pre-v3.2 "
        "'pixel_size' selector attrs.",
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr (current node-tree format)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite output if it exists."
    ),
    lossless: bool = typer.Option(
        False,
        "--lossless",
        help="Preserve float32 Cholesky factors exactly (PRECISION encoding) "
        "for archival migration. Default re-encodes them with the AUTO policy "
        "(uint8 per-column with a covariance certificate; escalates to uint16 "
        "only when the measured error demands it).",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress the trailing 'wrote …' summary."
    ),
) -> None:
    """Convert a legacy .gsplats.zarr layout to the current node-tree format.

    Five input shapes are auto-detected:

    \b
    * v1.0  .gsplats.zarr (single flat splat set)
    * v1.1  .gsplats.zarr (multi-LOD additive, /splats/lod_<i>/ subgroups)
    * v2.0  .gsplats.zarr (2-D substitutive_<s>/additive_<a> matrix)
    * substitutive directory (manifest.json + level_<i>.gsplats.zarr files)
    * v3.0/v3.1 .gsplats.zarr with pre-v3.2 lod selector attrs
      (selector='pixel_size' / per-child min_pixel_size — rewritten as
      selector='coverage' + derived coverage_fraction thresholds)

    All migrate to a single current-format ``.gsplats.zarr`` node subtree. By
    default the output adopts the AUTO encoding policy, so legacy float32 Cholesky
    factors are re-encoded with certified uint8 per-column quantization (the
    encode-time covariance certificate escalates to uint16 when needed);
    pass ``--lossless`` to keep them float32 for archival fidelity.
    """
    try:
        from luxar.encoding import EncodingMode
        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.migrate import migrate_format
        from luxar.gsplats.io.save_gsplats import FORMAT_VERSION
        from luxar.gsplats.tree import total_splats

        with asection(f"Migrating {input_path.name} → v{FORMAT_VERSION}"):
            detected = migrate_format(
                input_path,
                output_path,
                overwrite=overwrite,
                encoding_mode=(
                    EncodingMode.PRECISION if lossless else EncodingMode.AUTO
                ),
            )
            aprint(f"Detected legacy format: {detected}")

            # Post-write read-back: confirm the output is a loadable current-format
            # file rather than reporting success blind. load_gsplat_node (not
            # GSplatData.load) so nested trees — a migrated v3.x partition /
            # multiscale / mosaic, which has no flat matrix equivalent — verify
            # too, not just leaf/lod-matrix shapes.
            import zarr

            verify_node, _ = load_gsplat_node(output_path, include_stats=False)
            out_attrs = dict(zarr.open_group(str(output_path), mode="r").attrs)
            fmt = out_attrs.get("format_version")
            if fmt != FORMAT_VERSION:
                aprint(
                    f"❌ Migration produced format_version={fmt!r}, "
                    f"expected {FORMAT_VERSION!r}"
                )
                raise typer.Exit(1)
            if not quiet:
                aprint(
                    f"✓ Verified v{FORMAT_VERSION} output: "
                    f"{total_splats(verify_node):,} splats → {output_path}"
                )
    except typer.Exit:
        raise
    except FileNotFoundError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except ValueError as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def reencode_command(
    input_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Input .gsplats.zarr (current node-tree format), or .zip/.tar.gz.",
    ),
    output_path: Path = typer.Argument(
        ..., help="Output .gsplats.zarr with the re-quantized Cholesky factors."
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "memory",
        "--encoding",
        "-e",
        help="Target Cholesky encoding: memory=uint8 (smallest, ~93 dB), "
        "auto=adaptive u8→u16→f32 ladder (near-lossless by certificate), "
        "precision=float32 (exact).",
    ),
    ordering: Literal["hilbert", "morton", "none"] = typer.Option(
        "hilbert", "--ordering", help="Spatial chunk ordering for the output."
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress the trailing 'wrote …' summary."
    ),
) -> None:
    """Re-quantize a fitted .gsplats.zarr's Cholesky factors to another encoding.

    A structure-preserving round-trip: the whole node tree (leaf / additive
    ladder / kind=lod / partition / nested) and its ``fitting`` / ``provenance``
    / ``pipeline`` groups are carried over verbatim — only the on-disk
    Cholesky encoding changes. Splat *count* and geometry are unchanged; decode
    is always to float32, so viewer/GPU/WASM paths are unaffected.

    Unlike ``migrate-format`` (legacy layout → current, which only exposes
    float32 vs the AUTO uint16 default via ``--lossless``), this exposes the
    full ladder including ``memory`` (uint8) and works on already-current files.

    Examples:
        luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory
        luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision
    """
    try:
        import zarr

        from luxar.gsplats.io.load_gsplats import load_gsplat_node
        from luxar.gsplats.io.save_gsplats import (
            FORMAT_VERSION,
            write_gsplats_tree,
        )
        from luxar.gsplats.tree import total_splats

        with asection(f"Re-encoding {input_path.name} → {encoding}"):
            node, _ = load_gsplat_node(input_path)

            # Preserve the aux provenance groups verbatim (write_gsplats_tree
            # drops them unless re-supplied). Read from the on-disk root; for
            # a compressed input the loader already handled extraction, so
            # re-open via the same path is safe for directory stores. Use the
            # in-memory node for a compressed source (no directory to reopen).
            pipeline_info = fitting_info = fitting_config = provenance_info = None
            if input_path.is_dir():
                root = zarr.open_group(str(input_path), mode="r")
                if "pipeline" in root:
                    pipeline_info = dict(root["pipeline"].attrs)
                if "fitting" in root:
                    fitting_info = dict(root["fitting"].attrs)
                    if "config" in root["fitting"]:
                        fitting_config = dict(root["fitting"]["config"].attrs)
                if "provenance" in root:
                    provenance_info = dict(root["provenance"].attrs)

            write_gsplats_tree(
                output_path,
                node,
                ordering=ordering,
                encoding_mode=_resolve_encoding_mode(encoding),
                pipeline_info=pipeline_info,
                fitting_info=fitting_info,
                fitting_config=fitting_config,
                provenance_info=provenance_info,
            )

            # Read-back verify — a loadable current-format file, not blind success.
            verify_node, _ = load_gsplat_node(output_path, include_stats=False)
            out_attrs = dict(zarr.open_group(str(output_path), mode="r").attrs)
            fmt = out_attrs.get("format_version")
            if fmt != FORMAT_VERSION:
                aprint(f"❌ Re-encode produced format_version={fmt!r}")
                raise typer.Exit(1)
            if not quiet:
                aprint(
                    f"✓ Re-encoded ({encoding}) v{FORMAT_VERSION}: "
                    f"{total_splats(verify_node):,} splats → {output_path}"
                )
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        aprint(f"Error: {exc}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


def register_scene_commands(app: typer.Typer) -> None:
    """Register the scene commands onto ``app_gsplat``."""
    app.command("convert")(convert_to_scene)
    app.command("migrate-format")(migrate_format_command)
    app.command("reencode")(reencode_command)
