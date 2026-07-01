"""Interoperability adapters between Luxar Gaussian splats and external tools.

Currently provides a bridge to `tracksdata <https://github.com/royerlab/tracksdata>`_
(the Royer-lab multi-object-tracking data structure). The adapters import their
external dependency lazily, so this package imports cleanly without the optional
extras installed.
"""

from luxar.gsplats.interop.tracksdata import (
    gsplats_to_tracksdata_graph,
    splat_mask_and_bbox,
)

__all__ = ["gsplats_to_tracksdata_graph", "splat_mask_and_bbox"]
