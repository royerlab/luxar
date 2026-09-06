"""add_sound body.

Pure function taking a ``group: Group`` parameter as the first arg. Called by
``Group.add_sound`` (a thin signature + docstring + delegate) in
``core/group/group.py``.

A sound node has none of the structural branches the geometry adders have (no
partition, no LOD — a clip is a clip), so this is the FLAT-LEAF write of
``add_points`` with the element arrays replaced by an opaque payload, plus the
one piece of authoring sugar the spec adds: ``hidden={"story": 3}`` builds the
single ``(1, ndim)`` row that binds a non-spatial clip to a hidden-dimension
value, and extends it over every other hidden dimension so the slab rule
answers "live at story 3, whatever the other sliders say".
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

import numpy as np
from arbol import aprint

from ....validation.sound import (
    validate_audio_input,
    validate_distance_model,
    validate_non_negative_finite,
    validate_sound_bus,
    validate_sound_licence,
    validate_sound_passthrough_attrs,
    validate_sound_trigger,
    validate_spatial_params,
)
from ...sound import Sound
from ..compositing import (
    funnel_add_error,
    reject_layer_order_inside_specialized_group,
    unnest_add_error,
)
from ..partition import reject_mismatched_partition_parent

if TYPE_CHECKING:
    from ...node import Node
    from ..group import Group


def _hidden_row(
    scene: Any, hidden: Mapping[str, float], name: str
) -> tuple[np.ndarray, List[str]]:
    """Build the one-row positions array ``hidden=`` stands for.

    Named hidden dimensions take their value, every other column is 0, and the
    returned ``extend_to_all`` list covers each NON-displayed dimension that was
    not named, so the row is live there regardless of the slice. Displayed
    dimensions are ignored by the slab rule, so naming one is a mistake worth
    refusing rather than a coordinate worth storing.
    """
    dims = scene._dimensions
    names: List[str] = list(dims.names)
    if not hidden:
        raise ValueError(
            f"hidden= for sound '{name}' must name at least one dimension; "
            "pass positions=None and hidden=None for a clip live everywhere"
        )
    unknown = [k for k in hidden if k not in names]
    if unknown:
        raise ValueError(
            f"hidden= for sound '{name}' names unknown dimension(s) {unknown}. "
            f"Scene dimensions: {names}"
        )
    displayed = [k for k in hidden if dims.dimensions[names.index(k)].display]
    if displayed:
        raise ValueError(
            f"hidden= for sound '{name}' names displayed dimension(s) {displayed}; "
            "the slab rule only reads non-displayed dimensions. Use positions= "
            "for a spatial source."
        )
    row = np.zeros((1, dims.ndim), dtype=np.float32)
    for key, value in hidden.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"hidden[{key!r}] must be a number, got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"hidden[{key!r}] must be finite, got {value}")
        row[0, names.index(key)] = float(value)
    extend = [d.name for d in dims.dimensions if not d.display and d.name not in hidden]
    return row, extend


def add_sound_impl(
    group: "Group",
    *,
    name: str,
    clip: Union[bytes, bytearray, str, Path],
    positions: Optional[Any],
    hidden: Optional[Mapping[str, float]],
    spatial: Optional[bool],
    trigger: str,
    delay_ms: float,
    gain: float,
    bus: str,
    fade_in_ms: float,
    fade_out_ms: float,
    distance_model: str,
    ref_distance: Optional[float],
    max_distance: Optional[float],
    rolloff: Optional[float],
    cone_inner_deg: Optional[float],
    cone_outer_deg: Optional[float],
    cone_outer_gain: Optional[float],
    orientation: Optional[Sequence[float]],
    license: str,
    attribution: str,
    source_url: str,
    parent: Optional["Node"],
    extend_to_all: Optional[Union[List[str], str]],
    attrs: Dict[str, Any],
) -> Sound:
    """Body of :meth:`Group.add_sound`; see it for the parameter contract."""
    try:
        # Fail-fast pre-write gate: everything below runs BEFORE any zarr write.
        from ....validation.base import validate_node_name

        validate_node_name(name)
        (parent or group)._ensure_no_duplicate_child(name)
        reject_mismatched_partition_parent(parent or group, "sound", name)
        reject_layer_order_inside_specialized_group(
            "sound", name, attrs, parent or group
        )
        validate_sound_passthrough_attrs(attrs)

        payload, fmt = validate_audio_input(clip)
        trigger = validate_sound_trigger(trigger)
        bus = validate_sound_bus(bus)
        distance_model = validate_distance_model(distance_model)
        gain_v = validate_non_negative_finite("gain", gain)
        delay_v = validate_non_negative_finite("delay_ms", delay_ms)
        fade_in_v = validate_non_negative_finite("fade_in_ms", fade_in_ms)
        fade_out_v = validate_non_negative_finite("fade_out_ms", fade_out_ms)
        license, attribution, source_url = validate_sound_licence(
            license, attribution, source_url
        )
        spatial_params = validate_spatial_params(
            ref_distance=ref_distance,
            max_distance=max_distance,
            rolloff=rolloff,
            cone_inner_deg=cone_inner_deg,
            cone_outer_deg=cone_outer_deg,
            cone_outer_gain=cone_outer_gain,
            orientation=orientation,
        )

        scene = group._find_scene()

        # Where the source lives: explicit rows, the hidden= sugar, or nowhere.
        pos_arr: Optional[np.ndarray] = None
        if positions is not None and hidden is not None:
            raise ValueError(
                "positions= and hidden= are mutually exclusive: hidden= builds "
                "the one row a non-spatial clip needs, positions= places a source"
            )
        if hidden is not None:
            if spatial:
                raise ValueError(
                    "spatial=True needs positions=; hidden= binds a NON-spatial "
                    "clip to a hidden-dimension value"
                )
            pos_arr, hidden_extend = _hidden_row(scene, hidden, name)
            if extend_to_all is None:
                extend_to_all = hidden_extend
            spatial = False
        elif positions is not None:
            pos_arr = np.asarray(positions, dtype=np.float32)
            if pos_arr.ndim != 2 or pos_arr.shape[0] == 0:
                raise ValueError(
                    f"positions for sound '{name}' must have shape (K, D) with "
                    f"K >= 1, got {pos_arr.shape}"
                )
            if spatial is None:
                spatial = True
        else:
            if spatial:
                raise ValueError(
                    "spatial=True needs positions= (one nD row per place the "
                    "source exists)"
                )
            spatial = False
            if extend_to_all not in (None, [], "all"):
                raise ValueError(
                    "extend_to_all only applies with positions= or hidden=; a "
                    "clip without either is audible everywhere already"
                )

        if not spatial and spatial_params:
            warnings.warn(
                f"Sound '{name}' is non-spatial; distance/cone/orientation knobs "
                f"{sorted(spatial_params)} have no effect and are not stored.",
                UserWarning,
                stacklevel=4,
            )
            spatial_params = {}

        final_extend: List[str] = []
        if pos_arr is not None:
            scene._validate_data_dimensions(pos_arr, name, data_type="positions")
            final_extend = scene._resolve_extend_to_all(extend_to_all, pos_arr, "sound")

        sound_attrs: Dict[str, Any] = {
            "spatial": bool(spatial),
            "trigger": trigger,
            "delay_ms": delay_v,
            "gain": gain_v,
            "bus": bus,
            "loop": trigger == "continuous",
            "fade_in_ms": fade_in_v,
            "fade_out_ms": fade_out_v,
            "license": license,
            "attribution": attribution,
            "source_url": source_url,
        }
        if spatial:
            sound_attrs["distance_model"] = distance_model
            sound_attrs.update(spatial_params)
        if final_extend:
            sound_attrs["extend_to_all"] = final_extend

        parent_node = parent or group
        writer = group._require_scene_writer(scene)
        path = f"{parent_node.path}/{name}" if parent_node.path else name
        metadata = writer.write_sound(
            path,
            payload,
            fmt,
            pos_arr,
            sound_attrs=sound_attrs,
            **attrs,
        )
        return Sound(
            name,
            metadata=metadata,
            parent=cast(Any, parent_node),
            writer=writer,
            **attrs,
        )
    except (ValueError, TypeError) as e:
        inner = unnest_add_error("sound", name, e)
        aprint(f"Failed to add sound node '{name}': {inner}")
        raise ValueError(funnel_add_error("sound", name, e)) from e
