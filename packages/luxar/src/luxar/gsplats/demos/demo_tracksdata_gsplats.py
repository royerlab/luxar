from typing import Any

import numpy as np
import napari
import tracksdata as td

from luxar.gsplats.fit_result import GSplatData
from luxar.gsplats.utils import unpack_tril


def _to_bbox_and_mask(
    grid: np.ndarray,
    center: np.ndarray,
    cholesky_factor: np.ndarray,
    threshold: float = 2.0,
) -> tuple[np.ndarray, td.nodes.Mask]:

    centered = grid - center
    shape = centered.shape[:-1]

    z = np.linalg.solve(cholesky_factor, centered.reshape(-1, centered.shape[-1]).T).T
    sq_dist = np.square(z).sum(axis=-1)

    array_mask = sq_dist <= threshold**2
    array_mask = array_mask.reshape(shape)

    start = np.min(grid[array_mask], axis=0)
    end = np.max(grid[array_mask], axis=0)

    bbox = np.concatenate([start, end])

    mask = td.nodes.Mask(
        mask=array_mask[tuple(slice(s, e) for s, e in zip(start, end, strict=True))],
        bbox=bbox,
    )

    return bbox, mask


def _add_gsplats_node(
    graph: td.graph.BaseGraph,
    gsplats: GSplatData,
    frame_shape: tuple[int, ...],
) -> dict[str, Any]:

    n_nodes = gsplats.centers.shape[0]
    nodes_data = []

    grid = np.stack(
        np.meshgrid(*[np.arange(s) for s in frame_shape], indexing="ij"), axis=-1
    )
    dim = len(frame_shape)
    cholesky_factors = unpack_tril(gsplats.cholesky_factors, len(frame_shape))

    # adding keys
    graph.add_node_attr_key("center", default_value=np.zeros(dim))
    graph.add_node_attr_key("amplitude", default_value=0.0)
    graph.add_node_attr_key("cholesky_factor", default_value=np.zeros((dim, dim)))
    graph.add_node_attr_key("sharpness", default_value=0.0)
    graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.BBOX, default_value=np.zeros(2 * dim, dtype=int))
    graph.add_node_attr_key(td.DEFAULT_ATTR_KEYS.MASK, default_value=None)
    for c in ["z", "y", "x"][-dim:]:
        graph.add_node_attr_key(c, default_value=0.0)

    for i in range(n_nodes):
        center = gsplats.centers[i]
        amplitude = gsplats.amplitudes[i]
        cholesky_factor = cholesky_factors[i]
        sharpness = gsplats.sharpnesses[i]

        node = {
            td.DEFAULT_ATTR_KEYS.T: 0,
            "center": center,
            "amplitude": amplitude,
            "cholesky_factor": cholesky_factor,
            "sharpness": sharpness,
        }
        for c, v in zip(["z", "y", "x"][-dim:], center, strict=True):
            node[c] = v

        bbox, mask = _to_bbox_and_mask(grid, center, cholesky_factor)
        node[td.DEFAULT_ATTR_KEYS.BBOX] = bbox
        node[td.DEFAULT_ATTR_KEYS.MASK] = mask
        
        nodes_data.append(node)
    
    # for visualization purposes, sort by amplitude
    nodes_data = sorted(nodes_data, key=lambda x: x["amplitude"])
    
    graph.bulk_add_nodes(nodes_data)


def main() -> None:
    gsplats = GSplatData.load("mitosis_splats.gsplats.zarr")
    graph = td.graph.InMemoryGraph()

    _add_gsplats_node(graph, gsplats, frame_shape=(256, 256))

    viewer = napari.Viewer()

    arr_view = td.array.GraphArrayView(
        graph,
        shape=(1, 256, 256),
        # attr_key=td.DEFAULT_ATTR_KEYS.NODE_ID,
        attr_key="amplitude",
    )

    viewer = napari.Viewer()
    # viewer.add_labels(arr_view)
    viewer.add_image(arr_view, colormap="magma", name="amplitude")

    napari.run()


if __name__ == "__main__":
    main()
