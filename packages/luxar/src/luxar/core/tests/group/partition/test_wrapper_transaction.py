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


def test_partition_rollback_restores_labels_and_array_dedup_registry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, scene, _ = open_scene(tmp_path, "transaction-labels.zarr")
    rng = np.random.default_rng(7)
    positions = rng.random((12, 3), dtype=np.float32)
    colors = rng.random((12, 3), dtype=np.float32)
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
        scene.add_points(
            "broken",
            positions,
            colors=colors,
            labels=[f"point {index}" for index in range(12)],
            partition={"max_elements": 4},
        )

    assert not scene._has_labels
    monkeypatch.setattr(compiler, "write_points", write_points)
    scene.add_points(
        "good", positions, colors=colors, partition={"max_elements": 4}
    )
    compiler.finalize()

    def encodings(group: object) -> list[str]:
        names: list[str] = []
        for array_name in group.array_keys():
            encoding = group[array_name].attrs.get("encoding", {})
            names.append(encoding.get("name", ""))
        for group_name in group.group_keys():
            names.extend(encodings(group[group_name]))
        return names

    assert "array_ref" not in encodings(compiler.store)
    assert "overlays" not in compiler.store


def test_lod_late_child_failure_rolls_back_wrapper(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    compiler, scene, _ = open_scene(tmp_path, "transaction-lod.zarr")
    positions = np.random.default_rng(8).random((24, 3), dtype=np.float32)
    write_gsplats = compiler.write_gsplats
    calls = 0

    def fail_second_level(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("late LOD failure")
        return write_gsplats(*args, **kwargs)

    monkeypatch.setattr(compiler, "write_gsplats", fail_second_level)
    with pytest.raises(RuntimeError, match="late LOD failure"):
        scene.add_points(
            "broken",
            positions,
            substitutive_lod={"levels": 2, "device": "cpu", "seed": 0},
        )

    assert calls == 2
    assert "broken" not in compiler.store
    assert not scene.children
