"""Custom-colormap LUT resolution + zarr write for the compiler."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import zarr
from arbol import aprint

from ...encoding.compression import WIDTH_AWARE_DEFAULT, resolve_compressor


def write_colormap_lut_if_needed(
    group: zarr.Group,
    attrs: Dict[str, Any],
    scene_tone_mapping: Optional[str],
    lut_tone_mapping_warned: bool,
) -> bool:
    """Write custom colormap LUT to zarr if colormap is an array.

    If ``attrs["colormap"]`` is a numpy array, resolve it to a (256, 3) uint8
    LUT, write it as a dataset, and replace the attr value with ``"custom"``.
    String colormaps are left as-is.

    The at-most-once tone-mapping warning is threaded by value: the caller passes
    the current ``lut_tone_mapping_warned`` flag in, and this function returns the
    (possibly flipped) flag for the caller to store back. This preserves the
    one-shot warning behaviour without a back-pointer to the orchestrator.

    Args:
        group: Zarr group to write to
        attrs: Node attributes dict (modified in-place)
        scene_tone_mapping: The scene's configured tone mapping, or None.
        lut_tone_mapping_warned: Whether the LUT tone-mapping warning has already
            been emitted this session.

    Returns:
        The updated ``lut_tone_mapping_warned`` flag.
    """
    colormap = attrs.get("colormap")
    if colormap is None:
        return lut_tone_mapping_warned

    # The viewer defaults to ACES filmic tone-mapping, which intentionally
    # shifts hues for a pleasing HDR look. ACES is the right default for
    # almost every scene, but that hue shift does distort the exact colors of
    # a colormap LUT, so flag it for authors who have not considered the
    # choice at all — they may want to pin tone_mapping="Neutral" when the LUT
    # carries a scientific color encoding.
    #
    # Fires ONLY when the author set no tone_mapping. An explicit value —
    # including "ACES" — is a deliberate decision and must not be second-
    # guessed; the message itself speaks of "the viewer's *default*", which is
    # only what the author gets when they said nothing. Also skipped for the
    # implicit grayscale default ("gray"), which has no hue for ACES to
    # distort and is not a deliberate LUT choice.
    is_grayscale_default = isinstance(colormap, str) and colormap == "gray"
    if (
        not lut_tone_mapping_warned
        and scene_tone_mapping is None
        and not is_grayscale_default
    ):
        warnings.warn(
            "This scene uses a colormap LUT and sets no tone_mapping, so the "
            "viewer's default 'ACES' applies. ACES intentionally shifts hues, "
            "which can distort LUT colors. That is usually the look you want; "
            "if exact colormap fidelity matters (e.g. for scientific color "
            "encoding), set tone_mapping='Neutral' in the scene's "
            "viewer_config — or set 'ACES' explicitly to silence this.",
            UserWarning,
            stacklevel=3,
        )
        lut_tone_mapping_warned = True

    from ...colormaps import resolve_colormap
    from ...colormaps.builtins import BUILTIN_COLORMAP_NAMES

    if isinstance(colormap, str):
        if colormap in BUILTIN_COLORMAP_NAMES:
            # Built-in name — viewer resolves it directly, no LUT needed
            return lut_tone_mapping_warned

        # Non-built-in name (matplotlib/colorcet) — resolve to LUT and
        # store as "custom" so the viewer can render it without needing
        # matplotlib/colorcet at display time.
        lut = resolve_colormap(colormap)  # Raises ValueError if unknown
        group.create_dataset(
            "colormap_lut",
            data=lut,
            chunks=(256, 3),
            dtype=np.uint8,
            compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, np.uint8),
        )
        attrs["colormap"] = "custom"
        aprint(f"  ✓ Resolved '{colormap}' to LUT and wrote as custom (256x3 uint8)")
        return lut_tone_mapping_warned

    # Array colormap — resolve and write as dataset
    lut = resolve_colormap(colormap)  # (256, 3) uint8
    group.create_dataset(
        "colormap_lut",
        data=lut,
        chunks=(256, 3),
        dtype=np.uint8,
        compressor=resolve_compressor(WIDTH_AWARE_DEFAULT, np.uint8),
    )
    attrs["colormap"] = "custom"
    aprint("  ✓ Wrote custom colormap LUT (256x3 uint8)")
    return lut_tone_mapping_warned
