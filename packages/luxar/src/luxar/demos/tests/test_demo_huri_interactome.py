"""Scene metadata regressions for the HuRI interactome demo."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

pd = pytest.importorskip("pandas")

from luxar.demos import demo_huri_interactome as demo  # noqa: E402


def test_protein_layer_uses_canonical_genecards_url(tmp_path) -> None:
    nodes = ["TP53", "MDM2"]
    node_df = pd.DataFrame({"chromosome": ["17", "12"]}, index=nodes)
    edges = pd.DataFrame({"sym_a": ["TP53"], "sym_b": ["MDM2"]})
    output_path = tmp_path / "huri.luxar.zarr"

    assert demo.build_scene(
        output_path,
        nodes,
        node_df,
        edges,
        coords=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32),
        communities=np.array([0, 0], dtype=np.int32),
        degrees=np.array([1, 1], dtype=np.int32),
        corum_lookup={},
        max_edges=1,
    ) == (2, 1, 1)

    attrs = dict(zarr.open_group(str(output_path), mode="r")["Proteins"].attrs)
    assert attrs["link"] == "https://www.genecards.org/card/{hover_key}"
    assert attrs["copy"] == "{hover_key}"
    assert attrs["has_keys"] is True
