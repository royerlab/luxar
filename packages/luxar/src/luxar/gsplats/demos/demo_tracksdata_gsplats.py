"""Demo: bridge fitted Gaussian splats into a `tracksdata` graph.

Builds a small set of 2D Gaussian splats (standing in for fitted cells), converts
each into a `tracksdata` node carrying a binary mask + bbox via
:func:`luxar.gsplats.interop.gsplats_to_tracksdata_graph`, prints the resulting
graph stats, and — if napari is available — renders the nodes as an image with
tracksdata's ``GraphArrayView``.

Requires the optional extra::

    pip install 'luxar[tracksdata]'

In a real pipeline you would ``GSplatData.load(...)`` a fitted ``.gsplats.zarr``
(2D/3D, one frame) instead of the synthetic splats here, and call the adapter
once per timepoint into the same graph to assemble a time-lapse for tracking.

Originally prototyped by Jordão Bragantini (@JoOkuma) in PR #20.
"""

from __future__ import annotations

import numpy as np
from arbol import aprint, asection

from luxar.gsplats import GSplatData
from luxar.gsplats.interop import gsplats_to_tracksdata_graph

FRAME_SHAPE = (256, 256)


def _synthetic_cells(n: int = 12, seed: int = 0) -> GSplatData:
    """A handful of well-separated, variably-shaped 2D Gaussian 'cells'."""
    rng = np.random.RandomState(seed)
    centers = (rng.rand(n, 2) * (np.array(FRAME_SHAPE) - 40) + 20).astype(np.float32)
    amplitudes = (0.3 + 0.7 * rng.rand(n)).astype(np.float32)
    # packed lower-triangular [L00, L10, L11] per splat: random-ish ellipses
    scale = 4.0 + 6.0 * rng.rand(n)
    skew = (rng.rand(n) - 0.5) * 3.0
    cholesky = np.stack([scale, skew, scale * (0.6 + 0.6 * rng.rand(n))], axis=1)
    return GSplatData(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky.astype(np.float32),
    )


def main() -> None:
    with asection("tracksdata <- gsplats demo"):
        gsplats = _synthetic_cells()
        aprint(f"Built {gsplats.n_splats} synthetic {gsplats.ndim}D splats")

        try:
            graph = gsplats_to_tracksdata_graph(gsplats, FRAME_SHAPE, t=0)
        except ImportError as exc:
            aprint(f"⚠️  {exc}")
            aprint(
                "Install the extra to run the full demo:  pip install 'luxar[tracksdata]'"
            )
            return

        aprint(f"tracksdata graph: {graph.num_nodes()} nodes")
        aprint(f"node attributes : {sorted(graph.node_attr_keys())}")

        try:
            import napari
            import tracksdata as td
        except ImportError:
            aprint(
                "napari not installed — skipping the viewer (graph built successfully)."
            )
            return

        view = td.array.GraphArrayView(
            graph, attr_key="amplitude", shape=(1, *FRAME_SHAPE)
        )
        viewer = napari.Viewer()
        viewer.add_image(
            np.asarray(view), colormap="magma", name="splats→graph (amplitude)"
        )
        napari.run()


if __name__ == "__main__":
    main()
