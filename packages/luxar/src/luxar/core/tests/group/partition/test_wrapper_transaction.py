from __future__ import annotations

import numpy as np
import pytest

from ..conftest import open_scene


def test_partition_late_child_failure_rolls_back_store_graph_and_bounds(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, scene, _ = open_scene(tmp_path, "transaction.zarr")
    positions = np.arange(36, dtype=np.float32).reshape(12, 3)
    write_points = compiler.write_points
    calls = 0

    def fail_second_child(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("late child failure")
        return write_points(*args, **kwargs)

    monkeypatch.setattr(compiler, "write_points", fail_second_child)

    with pytest.raises(RuntimeError, match="late child failure"):
        scene.add_points("broken", positions, partition={"max_elements": 4})

    assert calls == 2
    assert "broken" not in compiler.store
    assert [child.name for child in scene.children] == []
    assert compiler._scene_bounds is None
