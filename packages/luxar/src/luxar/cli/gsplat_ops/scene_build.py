"""Build a Luxar scene from a ``.gsplats.zarr`` — shared by ``convert`` and ``view``.

``luxar gsplat convert`` writes the scene a user keeps; ``luxar gsplat view``
writes the same scene into a temporary directory and serves it. One code path,
so the first look a user gets from ``view`` is exactly what ``convert`` would
publish: the image-convention orientation and opening camera
(:mod:`.scene_orientation`), auto exposure (:mod:`.scene_exposure`) and the same
appearance defaults.

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
    from ...core.viewer_config import ViewerConfig
    from .scene_exposure import ConvertExposure
    from .scene_orientation import ConvertOrientation

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


def _orientation(
    extent: Any, node: Any, stats: dict, opts: SceneBuildOptions, *, remap: bool
) -> "ConvertOrientation":
    """Resolve the scene orientation, turning a bad flag into a usage error."""
    from .scene_orientation import resolve_convert_orientation

    try:
        return resolve_convert_orientation(
            extent,
            node.meta.get("dimension_metadata"),
            dim_order=opts.dim_order,
            view=opts.view,
            fitting_info=stats,
            remap=remap,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e


def _viewer_config(
    orientation: "ConvertOrientation", tone_mapping: Optional[str]
) -> "Optional[ViewerConfig]":
    """The scene's ViewerConfig: tone mapping and/or the orientation's camera."""
    from luxar.core.viewer_config import ViewerConfig

    aprint(orientation.summary)
    if tone_mapping is None and orientation.camera is None:
        return None
    return ViewerConfig(tone_mapping=tone_mapping, camera=orientation.camera)


def _exposure(
    node: Any,
    orientation: "ConvertOrientation",
    opts: SceneBuildOptions,
    stats: Optional[dict] = None,
) -> "ConvertExposure":
    """Auto exposure / manual gain for the scene, narrated in one line."""
    from .scene_exposure import resolve_convert_exposure

    exposure = resolve_convert_exposure(
        node,
        orientation,
        auto_exposure=opts.auto_exposure,
        target=opts.target_brightness,
        scale_intensity=opts.scale_intensity,
        colormap=opts.colormap,
        opacity=opts.opacity,
        blending_mode=opts.blending_mode,
        tone_mapping=opts.tone_mapping,
        stats=stats,
    )
    if exposure.line:
        aprint(exposure.line)
    return exposure


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
    from luxar.gsplats.io.save_gsplats import resolve_amplitude_bits
    from luxar.gsplats.tree import is_matrix_shaped

    appearance = scene_appearance(opts)
    with asection("Loading gsplat dataset"):
        node, stats = load_gsplat_node(input_path, include_stats=True)
        matrix = is_matrix_shaped(node)
        amplitude_bits = resolve_amplitude_bits(
            "auto", source_dtype=stats.get("source_dtype")
        )
    if matrix:
        data, extent = _matrix_source(node, opts)
    else:
        data, extent = None, _graft_extent(node, opts)

    with asection("Creating Luxar scene"):
        # A grafted tree cannot be remapped (`add_gsplats_from_file` refuses
        # dim_order there): the scene dims keep column order and the camera
        # preset finds the XY plane by name.
        orientation = _orientation(extent, node, stats, opts, remap=matrix)
        viewer_config = _viewer_config(orientation, opts.tone_mapping)
        tree = data.tree if data is not None else node
        exposure = _exposure(tree, orientation, opts, stats)
        # `--intensity 1` is the identity: no window was authored, the viewer
        # windows on the data range, and restating it would invent one.
        if appearance.get("intensity", 1.0) != 1.0 and _windows_amplitude(
            tree, opts.colormap
        ):
            appearance["intensity"] = _data_unit_intensity(
                appearance["intensity"], exposure.gain
            )
        attrs: Dict[str, Any] = {
            "normalize_amplitudes": exposure.normalize,
            "opacity": opts.opacity,
            "absorption": opts.absorption,
            "blending_mode": opts.blending_mode,
            **appearance,
            **exposure.attrs,
        }
        with LuxarZarrCompiler(
            output_path,
            encoding_mode=_resolve_encoding_mode(opts.encoding),
            gsplat_amplitude_bits=amplitude_bits,
        ) as compiler:
            scene = compiler.create_scene(
                dimensions=orientation.dimensions, viewer_config=viewer_config
            )
            if data is not None:
                scene.add_gsplats_from_data(
                    name="gsplats",
                    result=data,
                    dim_order=orientation.dim_order,
                    **attrs,
                )
            else:
                scene.add_gsplats_from_file(name="gsplats", path=input_path, **attrs)
