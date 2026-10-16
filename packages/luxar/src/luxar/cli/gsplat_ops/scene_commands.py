"""``luxar gsplat scene`` commands.

Each command is a plain function; ``register_scene_commands(app)`` wires them onto
the shared ``app_gsplat`` Typer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import typer
from arbol import aprint, asection

from ...typing_utils.constants import DEFAULT_BLENDING_MODE
from .._traceback import exit_with_error
from .encoding import _resolve_encoding_mode


def _positive_scale(value: Optional[float]) -> Optional[float]:
    """``--scale-intensity`` must be a positive finite factor (or absent)."""
    import math

    if value is not None and not (math.isfinite(value) and value > 0.0):
        raise typer.BadParameter(f"must be a positive number, got {value!r}")
    return value


def convert_to_scene(
    input_path: Path = typer.Argument(
        ..., exists=True, help="Input .gsplats.zarr dataset (or .zip/.tar.gz)"
    ),
    output_path: Path = typer.Argument(..., help="Output .luxar.zarr scene path"),
    center: bool = typer.Option(
        True,
        "--center/--no-center",
        help="Center spatial axes at the amplitude-weighted centroid "
        "(a categorical time/channel axis is left in place)",
    ),
    scale_intensity: Optional[float] = typer.Option(
        None,
        "--scale-intensity",
        callback=_positive_scale,
        help="Display gain (brightness) of exactly this factor relative to the "
        "fitted amplitudes, which stay in data units (any tree shape); "
        "replaces automatic normalization and auto exposure",
    ),
    auto_exposure: bool = typer.Option(
        True,
        "--auto-exposure/--no-auto-exposure",
        help="Dim splat brightness so the opening view is not saturated: predict "
        "the per-pixel additive sum down the default view axis and set the "
        "node's display gain so its p99.9 is --target-brightness (never "
        "brightens; the amplitudes stay in data units), "
        "dimming further if a side view or another timepoint would exceed 4x "
        "that. Hidden axes are predicted where the viewer opens them (a "
        "timelapse at its first timepoint, up to 4 more sampled for the cap). "
        "--no-auto-exposure = gain from the legacy per-splat p99.9 "
        "normalization only.",
    ),
    target_brightness: Optional[float] = typer.Option(
        None,
        "--target-brightness",
        min=1e-6,
        help="Auto-exposure target for the predicted projected p99.9 (HDR, "
        "before tone mapping). Default: 2.0 under ACES/AgX/Neutral/Reinhard/"
        "Cineon (and when --tone-mapping is unset), 0.9 under None/Linear, "
        "which clip at 1.0. Lower is dimmer.",
    ),
    opacity: float = typer.Option(1.0, "--opacity", help="Opacity (0.0-1.0)"),
    absorption: float = typer.Option(
        1.0,
        "--absorption",
        help="Absorption kappa (>=0) for --blending-mode volumetric; 0 = additive",
    ),
    blending_mode: str = typer.Option(
        DEFAULT_BLENDING_MODE,
        "--blending-mode",
        help="Blending: additive/normal/max/opaque/luminous/volumetric",
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
        "Neutral). 'ACES' suits almost every scene and is also the viewer "
        "default when unset; pass it explicitly to record the choice. Use "
        "'None' — an exact passthrough — when a colormap carries an exact "
        "color encoding and the scene stays inside [0, 1].",
    ),
    gamma: Optional[float] = typer.Option(
        None, "--gamma", help="Display gamma (default 1.0)"
    ),
    intensity: Optional[float] = typer.Option(
        None,
        "--intensity",
        help="Display-window intensity relative to the display-gained "
        "amplitudes (after auto exposure / normalization, or "
        "--scale-intensity); a colormap window is restated in data units",
    ),
    layer: bool = typer.Option(
        True,
        "--layer/--no-layer",
        help="List the gsplats node in the viewer Layers panel",
    ),
    encoding: Literal["auto", "precision", "memory"] = typer.Option(
        "auto", "--encoding", "-e", help="Encoding mode"
    ),
    dim_order: Optional[str] = typer.Option(
        None,
        "--dim-order",
        help="Name each center column, comma-separated (e.g. 'z,y,x' or "
        "'t,z,y,x'). Default: the store's recorded axis names, else array order "
        "(z,y,x / y,x) for a store with fitting provenance, else positional. "
        "Columns are mapped onto x,y,z-ordered scene dims; a y column BEFORE the "
        "x column (image order) opens y-down like Fiji/napari. It NAMES columns, "
        "it does not move data: 'x,y,z' on a z,y,x fit is the legacy positional "
        "mode (the z column is labelled x, no flip).",
    ),
    view: Optional[str] = typer.Option(
        None,
        "--view",
        help="Opening view: xy (look down z; default for image fits), xz, or yz. "
        "Stored as viewer_config.camera.view_axis; ?viewAxis= overrides it.",
    ),
) -> None:
    """Convert a .gsplats.zarr dataset to a Luxar scene for the web viewer.

    Creates a persistent Luxar scene zarr that can be served with
    ``luxar serve``. By default, centers the data at the centroid.

    Appearance (colormap / tone-mapping / gamma / intensity) is baked into the
    scene here. Prefer ``--tone-mapping ACES`` (its filmic rolloff keeps bright
    structure from clipping flat, and passing it explicitly records the choice).
    When a colormap carries an exact scientific color encoding that ACES's hue
    shift would distort, pick by range: inside [0, 1], ``None`` is the only
    exact passthrough (the mega-shader applies exposure/offset/gamma *before*
    the tone-mapping switch, so those still work). ``Neutral`` is not a
    passthrough — even below its knee it subtracts an offset taken from the
    channel minimum, so anything but a fully saturated colour moves. Over range
    nothing is faithful: ``Neutral`` keeps the HSV hue angle exactly but sheds
    chroma (essentially white by peak 100), while a ``None`` clamp flattens
    everything above 1.0 and shifts hue when channels clip unequally, losing
    saturation too wherever the darkest channel is not already 0. So bring the
    scene back into [0, 1] with ``--intensity`` and use ``None``, or accept
    ACES's filmic rolloff (#1459).

    Orientation follows the IMAGE CONVENTION for fitted volumes: the center
    columns are array order (z, y, x), so they are mapped onto the scene's
    x, y, z dimensions with ``dim_order`` and the opening camera faces the XY
    plane with y pointing down (``viewer_config.camera`` ``view_axis="xy"``,
    ``up=(0, -1, 0)``) — the default Fiji/napari view. The flip is a camera
    choice, so coordinates, units, picking and bounds stay as fitted. A
    partition/nested store cannot be remapped (it is grafted node-for-node), so
    its scene dims keep column order and the same camera preset faces its XY
    plane by dimension name. ``--view xz|yz`` picks another face.
    ``--dim-order`` names the columns without moving data, so ``--dim-order
    x,y,z`` on a (z, y, x) fit is the legacy positional mode (z labelled x, no
    flip). A store without axis names or fitting provenance (geometric x, y, z
    data) keeps the positional mapping.

    Examples:
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --no-center
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --scale-intensity 0.1
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr \\
            --colormap plasma --tone-mapping ACES
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr \\
            --colormap plasma --tone-mapping None
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --view xz
        luxar gsplat convert fitted.gsplats.zarr scene.luxar.zarr --dim-order x,y,z
    """
    try:
        from .scene_build import SceneBuildOptions, build_gsplat_scene

        opts = SceneBuildOptions(
            center=center,
            scale_intensity=scale_intensity,
            auto_exposure=auto_exposure,
            target_brightness=target_brightness,
            opacity=opacity,
            absorption=absorption,
            blending_mode=blending_mode,
            colormap=colormap,
            tone_mapping=tone_mapping,
            gamma=gamma,
            intensity=intensity,
            layer=layer,
            encoding=encoding,
            dim_order=dim_order,
            view=view,
        )
        with asection(f"Converting: {input_path.name} -> {output_path.name}"):
            build_gsplat_scene(input_path, output_path, opts)
            aprint(f"\nScene saved: {output_path}")
            aprint(f"Serve with: luxar serve {output_path} --viewer")

    except (typer.Exit, typer.BadParameter):
        raise
    except Exception as e:
        exit_with_error(f"Error: {e}", e)


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
            # overview / adaptive, which has no flat matrix equivalent — verify
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
        exit_with_error(f"Error: {exc}", exc)


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
    / ``pipeline`` groups are carried over verbatim. Structure and splat *count*
    are unchanged, and decode is always to float32, so viewer/GPU/WASM paths are
    unaffected — but the Cholesky factors are not the only array re-encoded:
    ``auto`` and ``memory`` also re-quantize the CENTERS to per-axis uint16
    fixed-point — float32 once an axis spans 2**16, or once a NON-gridded
    axis's grid would displace MORE THAN 0.1% of the splats past their own
    sigma (a smaller degenerate population is quantized away silently), while a
    GRIDDED axis such as a stacked sigma=0 time axis instead keeps uint16 with
    its grid snapped onto the data's own spacing, which is also exact — so on
    ordinary spatial data center values are bit-exact only under ``precision``.

    Unlike ``migrate-format`` (legacy layout → current, which only exposes
    float32 vs the AUTO uint8-first policy via ``--lossless``), this exposes the
    full ladder including ``memory`` (uint8) and works on already-current files.

    Examples:
        luxar gsplat reencode fit.gsplats.zarr fit_u8.gsplats.zarr -e memory
        luxar gsplat reencode fit.gsplats.zarr fit_f32.gsplats.zarr -e precision
    """
    try:
        import shutil

        from luxar._zarr_compat import open_group as zc_open_group
        from luxar.gsplats.io._archive import resolve_store_path
        from luxar.gsplats.io.load_gsplats import (
            load_gsplat_node,
            read_rebuild_root_attrs,
        )
        from luxar.gsplats.io.save_gsplats import FORMAT_VERSION, write_gsplats_tree
        from luxar.gsplats.tree import total_splats

        with asection(f"Re-encoding {input_path.name} → {encoding}"):
            zarr_path, temp_dir = resolve_store_path(input_path)
            try:
                node, stats = load_gsplat_node(zarr_path, include_stats=True)
                root = zc_open_group(str(zarr_path), mode="r")

                pipeline_info = fitting_info = fitting_config = provenance_info = None
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
                    amplitude_bits="auto",
                    source_dtype=stats.get("source_dtype"),
                    pipeline_info=pipeline_info,
                    fitting_info=fitting_info,
                    fitting_config=fitting_config,
                    provenance_info=provenance_info,
                    root_attrs=read_rebuild_root_attrs(zarr_path),
                )

                # Read-back verify — a loadable current-format file, not blind success.
                verify_node, _ = load_gsplat_node(output_path, include_stats=False)
                out_attrs = dict(zc_open_group(str(output_path), mode="r").attrs)
                fmt = out_attrs.get("format_version")
                if fmt != FORMAT_VERSION:
                    aprint(f"❌ Re-encode produced format_version={fmt!r}")
                    raise typer.Exit(1)
                if not quiet:
                    aprint(
                        f"✓ Re-encoded ({encoding}) v{FORMAT_VERSION}: "
                        f"{total_splats(verify_node):,} splats → {output_path}"
                    )
            finally:
                if temp_dir is not None and temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
    except typer.Exit:
        raise
    except (FileNotFoundError, ValueError) as exc:
        aprint(f"Error: {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        exit_with_error(f"Error: {exc}", exc)


def register_scene_commands(app: typer.Typer) -> None:
    """Register the scene commands onto ``app_gsplat``."""
    app.command("convert")(convert_to_scene)
    app.command("migrate-format")(migrate_format_command)
    app.command("reencode")(reencode_command)
