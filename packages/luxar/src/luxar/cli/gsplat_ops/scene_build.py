"""Build a Luxar scene from a ``.gsplats.zarr`` — shared by ``convert`` and ``view``.

``luxar gsplat convert`` writes the scene a user keeps; ``luxar gsplat view``
writes the same scene into a temporary directory and serves it. One code path,
so the first look a user gets from ``view`` is exactly what ``convert`` would
publish: the image-convention orientation and opening camera
(:mod:`luxar.core.scene_orientation`), auto exposure
(:mod:`luxar.core.scene_exposure`) and the same appearance defaults, resolved
by :func:`luxar.core.scene_defaults.resolve_fitted_scene`, which Python callers
reach through :func:`~luxar.core.scene_defaults.fitted_scene_defaults`.

A matrix-shaped store round-trips through ``GSplatData`` (and supports
``center``); a partition / nested LOD store is grafted node for node by
``add_gsplats_from_file``. Either way the splats are re-written into the scene
(a graft rebuilds every node through the scene writers — there is no
zero-copy link), so the cost scales with the store.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import typer
from arbol import aprint, asection

from ...typing_utils.constants import DEFAULT_BLENDING_MODE
from .encoding import _resolve_encoding_mode

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ...core.scene_defaults import FittedScene
    from ...core.scene_orientation import ConvertOrientation

__all__ = ["SceneBuildOptions", "build_gsplat_scene"]


@dataclass(frozen=True)
class SceneBuildOptions:
    """Every ``convert`` knob that shapes the scene (CLI defaults)."""

    center: bool = True
    scale_intensity: Optional[float] = None
    auto_exposure: bool = True
    target_brightness: Optional[float] = None
    opacity: float = 1.0
    absorption: float = 1.0
    blending_mode: str = DEFAULT_BLENDING_MODE
    colormap: Optional[str] = None
    tone_mapping: Optional[str] = None
    gamma: Optional[float] = None
    intensity: Optional[float] = None
    layer: bool = True
    encoding: str = "auto"
    dim_order: Optional[str] = None
    view: Optional[str] = None


def _validate_colormap(colormap: str) -> None:
    """Refuse a colormap no LUT source knows, as a usage error."""
    from luxar.colormaps import BUILTIN_COLORMAP_NAMES
    from luxar.colormaps.registry import resolve_colormap

    if colormap in BUILTIN_COLORMAP_NAMES:
        return
    try:
        resolve_colormap(colormap)
    except Exception as e:
        raise typer.BadParameter(f"--colormap: {e}") from e


def scene_appearance(opts: SceneBuildOptions) -> Dict[str, Any]:
    """Validate the appearance flags; return the attrs to forward (set ones only)."""
    from luxar.core.viewer_config import VALID_TONE_MAPPINGS
    from luxar.validation.types import validate_absorption, validate_blending_mode

    try:
        validate_blending_mode(opts.blending_mode)
        validate_absorption(opts.absorption)
    except (ValueError, TypeError) as e:
        raise typer.BadParameter(str(e)) from e
    if opts.tone_mapping is not None and opts.tone_mapping not in VALID_TONE_MAPPINGS:
        raise typer.BadParameter(
            f"--tone-mapping must be one of {VALID_TONE_MAPPINGS}, "
            f"got '{opts.tone_mapping}'"
        )
    appearance: Dict[str, Any] = {"layer": opts.layer}
    if opts.colormap is not None:
        _validate_colormap(opts.colormap)
        appearance["colormap"] = opts.colormap
    if opts.gamma is not None:
        appearance["gamma"] = opts.gamma
    if opts.intensity is not None:
        appearance["intensity"] = opts.intensity
    return appearance


def _resolve_scene(
    node: Any,
    tree: Any,
    extent: Any,
    stats: dict,
    opts: SceneBuildOptions,
    *,
    remap: bool,
) -> "FittedScene":
    """Resolve convert's scene (shared with the Python helper), narrated.

    A bad ``--dim-order`` / ``--view`` becomes a usage error.
    """
    from luxar.core.scene_defaults import resolve_fitted_scene

    try:
        scene = resolve_fitted_scene(
            tree,
            extent,
            stats=stats,
            dimension_metadata=node.meta.get("dimension_metadata"),
            remap=remap,
            dim_order=opts.dim_order,
            view=opts.view,
            colormap=opts.colormap,
            tone_mapping=opts.tone_mapping,
            blending_mode=opts.blending_mode,
            opacity=opts.opacity,
            absorption=opts.absorption,
            layer=opts.layer,
            auto_exposure=opts.auto_exposure,
            target_brightness=opts.target_brightness,
            display_gain=opts.scale_intensity,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    aprint(scene.orientation.summary)
    if scene.exposure.line:
        aprint(scene.exposure.line)
    if opts.blending_mode == "volumetric":
        _narrate_volumetric_depth(tree, scene.orientation, opts, scene.exposure.gain)
    return scene


def _additive_gain(
    node: Any, orientation: "ConvertOrientation", opts: SceneBuildOptions
) -> Optional[float]:
    """The display gain additive auto exposure would pick, or ``None``.

    ``None`` when the gain was set by hand (``--display-gain``), auto exposure
    is off, or the opening view could not be predicted.
    """
    from luxar.core.scene_exposure import resolve_convert_exposure

    if opts.scale_intensity is not None or not opts.auto_exposure:
        return None
    additive = resolve_convert_exposure(
        node,
        orientation,
        auto_exposure=True,
        target=opts.target_brightness,
        scale_intensity=None,
        colormap=opts.colormap,
        opacity=opts.opacity,
        blending_mode="additive",
        tone_mapping=opts.tone_mapping,
    )
    if additive.attrs.get("amplitude_normalization_mode") != "projected":
        return None
    return additive.gain


def _narrate_volumetric_depth(
    node: Any, orientation: "ConvertOrientation", opts: SceneBuildOptions, gain: float
) -> None:
    """Print the brightest rays' optical depth under ``--absorption``.

    Auto exposure does not run for ``volumetric`` (it only tames SUMMING modes),
    so the per-ray density the absorption multiplies is whatever the fit's
    overlap makes it — often tens of times the additive target. One predicted
    number turns choosing kappa from guesswork into a division.

    Emission is that same density, so a small kappa leaves the opening view as
    saturated as additive. Unless the gain was set by hand, a second line names
    the ``--display-gain`` additive auto exposure would use and the kappa that
    gives ~1 optical depth at that gain (the density scales with the gain).
    """
    from luxar.core.scene_exposure import (
        predict_volumetric_ray_mass,
        volumetric_depth_line,
        volumetric_gain_line,
    )

    ray_mass = predict_volumetric_ray_mass(
        node, orientation, gain=gain, opacity=opts.opacity
    )
    if ray_mass is None:
        return
    aprint(volumetric_depth_line(ray_mass, opts.absorption))
    suggested = _additive_gain(node, orientation, opts)
    if suggested is not None and suggested < gain:
        aprint(volumetric_gain_line(suggested, ray_mass * suggested / gain))


def _windows_amplitude(node: Any, colormap: Optional[str]) -> bool:
    """Whether the node renders through a colormap, so ``intensity`` is a WINDOW.

    An explicit ``--colormap`` always does; otherwise a colorless store gets the
    writer's ``gray`` default, while a store carrying per-splat colours renders
    them directly (there ``intensity`` is a colour gain, unit-free).
    """
    from luxar.gsplats.tree import iter_default_leaves

    if colormap is not None:
        return True
    return not any(
        sub.colors is not None
        for leaf in iter_default_leaves(node)
        for sub in leaf.additive_sublods
    )


def _data_unit_intensity(intensity: float, gain: float) -> float:
    """Restate a ``--intensity`` WINDOW (``[0, 1/I]``) in data units.

    ``--intensity`` is authored against the display-gained amplitudes (what the
    stored values used to be), so the window ``[0, 1/I]`` covers data values up
    to ``1/(I·gain)``: the data-unit intensity is ``I·gain``.
    """
    from luxar.typing_utils.constants import INTENSITY_MAX, INTENSITY_MIN

    scaled = float(intensity) * float(gain)
    if not INTENSITY_MIN <= scaled <= INTENSITY_MAX:
        raise typer.BadParameter(
            f"--intensity {intensity:g} windows the colormap over [0, "
            f"{1.0 / max(scaled, 1e-300):.3g}] data units, which needs intensity "
            f"{scaled:.3g} (outside [{INTENSITY_MIN:g}, {INTENSITY_MAX:g}]). Leave "
            "--intensity unset and set the window in the viewer's Layers panel."
        )
    return scaled


def _matrix_source(node: Any, opts: SceneBuildOptions) -> Tuple[Any, Any]:
    """A matrix-shaped tree as ``GSplatData`` (centered) plus its extent."""
    from luxar.gsplats.gsplat_data import GSplatData

    data = GSplatData.from_tree(node)
    aprint(f"Loaded {data.n_splats:,} splats ({data.ndim}D)")
    if opts.center:
        aprint("Centering at amplitude-weighted centroid")
        data = data.center_at_centroid()
    return data, data.centers


def _graft_extent(node: Any, opts: SceneBuildOptions) -> Any:
    """The bounds box of a partition / nested tree, which is grafted as is."""
    import numpy as np

    from luxar.gsplats.tree import center_bounds

    kind = "partition" if node.__class__.__name__ == "GSplatPartition" else "nested LOD"
    aprint(f"Grafting a {kind} node tree (no flat-data transforms apply)")
    if opts.center:
        # --center defaults True; it does not apply to a graft (the file's own
        # coordinates are preserved), so note it rather than fail.
        aprint(
            "ℹ️  --center is ignored for a partition/nested file; the "
            "node tree keeps its authored coordinates."
        )
    bounds = center_bounds(node)
    if bounds is None:
        raise ValueError("Could not derive bounds from the node tree")
    return np.array([bounds[0], bounds[1]], dtype=np.float32)


def max_blending_seam_warning(
    blending_mode: str, stats: Dict[str, Any], *, partition: bool
) -> Optional[str]:
    """The warning for ``max`` blending over a fit whose tiles were blended.

    ``max`` keeps the brightest SINGLE splat on each ray; it is not a
    maximum-intensity projection of the fitted volume. A uniform tiled fit
    multiplies every tile by a Hann window, so where two tiles overlap each
    one's splats carry a window-weighted share of the signal — about half at
    the middle of the overlap. Additive blending sums the shares back to the
    whole; ``max`` keeps only the larger one, so the tile overlaps render as a
    dark seam cross. A content-tiled fit (``planned_fit``) keeps whole splats by
    core box and has no shares to lose, and a partition cut from ONE fit
    (``gsplat partition`` / ``lod --recipe tiles``) carries that fit's own
    ``iterations`` stamp. A partition with no record of any of these is warned
    about conditionally (a ``batch-fit`` merge of uniform tiles carries no root
    marker).
    """
    if blending_mode != "max" or stats.get("planned_fit"):
        return None
    if stats.get("tiled_fitting"):
        lead = "this is a uniform tiled fit"
    elif partition and "iterations" not in stats:
        lead = "if this partition came from uniform tiles (fit/batch-fit)"
    else:
        return None
    return (
        f"⚠ --blending-mode max: {lead}, expect dark seams where tiles "
        "overlap. `max` shows the brightest single splat per ray, not a "
        "maximum-intensity projection: the Hann-blended tiles split each "
        "overlap's signal between their splats, which additive blending sums "
        "back but max does not. For max-blend viewing refit with --tiling none "
        "(whole volume), or use additive/volumetric blending."
    )


def build_gsplat_scene(
    input_path: Path, output_path: Path, opts: SceneBuildOptions
) -> None:
    """Write the ``convert`` scene for ``input_path`` to ``output_path``.

    Raises:
        typer.BadParameter: On an invalid appearance/orientation flag.
        ValueError: When a node tree has no derivable bounds.
    """
    from luxar import LuxarZarrCompiler
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import is_matrix_shaped

    appearance = scene_appearance(opts)
    with asection("Loading gsplat dataset"):
        node, stats = load_gsplat_node(input_path, include_stats=True)
        matrix = is_matrix_shaped(node)
    seam_warning = max_blending_seam_warning(
        opts.blending_mode, stats, partition=not matrix
    )
    if seam_warning:
        aprint(seam_warning)
    if matrix:
        data, extent = _matrix_source(node, opts)
    else:
        data, extent = None, _graft_extent(node, opts)

    with asection("Creating Luxar scene"):
        # A grafted tree cannot be remapped (`add_gsplats_from_file` refuses
        # dim_order there): the scene dims keep column order and the camera
        # preset finds the XY plane by name.
        tree = data.tree if data is not None else node
        resolved = _resolve_scene(node, tree, extent, stats, opts, remap=matrix)
        attrs = dict(resolved.attrs)
        dim_order = attrs.pop("dim_order", None)
        # `--intensity 1` is the identity: no window was authored, the viewer
        # windows on the data range, and restating it would invent one.
        if appearance.get("intensity", 1.0) != 1.0 and _windows_amplitude(
            tree, opts.colormap
        ):
            appearance["intensity"] = _data_unit_intensity(
                appearance["intensity"], resolved.exposure.gain
            )
        attrs.update(appearance)
        with LuxarZarrCompiler(
            output_path,
            encoding_mode=_resolve_encoding_mode(opts.encoding),
            gsplat_amplitude_bits=resolved.amplitude_bits,
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=resolved.dimensions,
                viewer_config=resolved.viewer_config,
            )
            if data is not None:
                scene.add_gsplats_from_data(
                    name="gsplats",
                    result=data,
                    dim_order=dim_order,
                    **attrs,
                )
            else:
                # Already resolved above: the adder's own fitted default would
                # only re-read the store.
                scene.add_gsplats_from_file(
                    name="gsplats", path=input_path, fitted_defaults=False, **attrs
                )
