"""Tests for the budding-organoid demo: source decoding, lineage, time axis, scene."""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from luxar.demos import demo_gsplats_4d_organoid_budding as demo
from luxar.demos._scene_size import SIZE_PASSES
from luxar.demos.registry import dir_size_bytes
from luxar.gsplats.gsplat_data import GSplatData


def _write_vti(
    path: Path,
    voxels: np.ndarray,
    origin_xyz: tuple[float, float, float],
    *,
    spacing_xyz: tuple[float, float, float] = (0.26, 0.26, 2.0),
    block: int = 1000,
) -> None:
    """Write ``(Z, Y, X, 3)`` uint8 voxels the way the LSTree export does."""
    nz, ny, nx, _ = voxels.shape
    raw = np.ascontiguousarray(voxels, dtype=np.uint8).tobytes()
    chunks = [zlib.compress(raw[i : i + block]) for i in range(0, len(raw), block)]
    last = len(raw) - block * (len(chunks) - 1)
    header = struct.pack(
        f"<{3 + len(chunks)}I", len(chunks), block, last, *map(len, chunks)
    )
    payload = base64.b64encode(header) + base64.b64encode(b"".join(chunks))
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" '
        'header_type="UInt32" compressor="vtkZLibDataCompressor">\n'
        f'<ImageData WholeExtent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}" '
        f'Origin="{" ".join(map(str, origin_xyz))}" '
        f'Spacing="{" ".join(map(str, spacing_xyz))}" '
        'Direction="1 0 0 0 1 0 0 0 1">\n'
        f'<Piece Extent="0 {nx - 1} 0 {ny - 1} 0 {nz - 1}"><PointData Scalars="values">'
        '<DataArray type="UInt8" Name="values" NumberOfComponents="3" format="binary">\n'
        f"{payload.decode()}\n"
        "</DataArray></PointData></Piece></ImageData></VTKFile>\n"
    )


def _write_mamut(path: Path, spots: list[tuple[int, int, float]], edges) -> None:
    """Minimal MaMuT XML: ``spots`` are (id, frame, x); ``edges`` (source, target)."""
    spot_xml = "".join(
        f'<Spot ID="{i}" FRAME="{f}" POSITION_X="{x}" POSITION_Y="{2 * x}" '
        f'POSITION_Z="{3 * x}" />'
        for i, f, x in spots
    )
    edge_xml = "".join(
        f'<Edge SPOT_SOURCE_ID="{a}" SPOT_TARGET_ID="{b}" />' for a, b in edges
    )
    path.write_text(
        f"<TrackMate><Model><AllSpots><SpotsInFrame>{spot_xml}</SpotsInFrame>"
        f"</AllSpots><AllTracks><Track>{edge_xml}</Track></AllTracks></Model>"
        "</TrackMate>"
    )


def test_time_axis_is_binary_exact_and_spans_the_recording() -> None:
    assert len(demo.FRAMES) == 167
    assert demo.FRAMES[0] == 0 and demo.FRAMES[-1] == 664
    assert demo.TIME_STEP_MIN == 40.0
    # Every coordinate sits EXACTLY on k * step, so the last slider stop exists.
    for k, frame in enumerate(demo.FRAMES):
        assert frame * demo.FRAME_INTERVAL_MIN == k * demo.TIME_STEP_MIN
    assert demo.OPENING_FRAME in demo.FRAMES
    assert demo.frame_member(0).endswith("rgb_grid/grid-T0001.vti")


def test_the_empty_export_frame_is_drawn_from_its_predecessor() -> None:
    # T0245 is an empty 2 x 15 x 15 crop in the record; T0244 stands in.
    assert demo.source_frame(244) == 243
    assert demo.frame_member(demo.source_frame(244)).endswith("grid-T0244.vti")
    assert "_t0244." in demo._frame_cache_name("nuclei", 244)
    assert all(demo.source_frame(f) == f for f in demo.FRAMES if f != 244)
    # So the frames actually drawn step by 4 except 3 then 5 around that slot,
    # which is what the docstring and the scene description state.
    drawn = [demo.source_frame(f) for f in demo.FRAMES]
    steps = [b - a for a, b in zip(drawn, drawn[1:], strict=False)]
    assert sorted(set(steps)) == [3, 4, 5]
    assert steps.count(3) == steps.count(5) == 1
    assert steps.index(3) + 1 == steps.index(5)


def test_stand_in_slot_shows_the_stand_in_lineage(monkeypatch) -> None:
    monkeypatch.setattr(demo, "FRAMES", (4,))
    monkeypatch.setattr(demo, "FRAME_STAND_INS", {4: 3})
    spots = {i: (i, np.full(3, float(i), np.float32)) for i in range(5)}
    geo = demo.lineage_geometry(spots, {i: i - 1 for i in range(1, 5)})
    # The marker is the frame-3 cell, placed at the slot's nominal time.
    np.testing.assert_allclose(geo["points"], [[3.0, 3.0, 3.0, 40.0]])


def test_read_rgb_grid_round_trips_voxels_and_origin(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    voxels = rng.integers(0, 256, (3, 5, 7, 3), dtype=np.uint8)
    path = tmp_path / "grid-T0001.vti"
    _write_vti(path, voxels, (55.38, 52.26, 36.0))

    decoded, origin_zyx = demo.read_rgb_grid(path)

    np.testing.assert_array_equal(decoded, voxels)
    np.testing.assert_allclose(origin_zyx, (36.0, 52.26, 55.38))


def test_read_rgb_grid_refuses_an_unexpected_voxel_size(tmp_path: Path) -> None:
    path = tmp_path / "grid.vti"
    _write_vti(path, np.zeros((2, 2, 2, 3), np.uint8), (0, 0, 0), spacing_xyz=(1, 1, 1))
    with pytest.raises(ValueError, match="spacing"):
        demo.read_rgb_grid(path)


def test_lineage_orients_reversed_links_and_drops_same_frame_ones(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mamut.xml"
    # 0 -> 1 -> {2, 3}; the 1-2 link is stored later-to-earlier, 2-4 is same-frame.
    _write_mamut(
        path,
        [(0, 0, 1.0), (1, 1, 2.0), (2, 2, 3.0), (3, 2, 4.0), (4, 2, 5.0)],
        [(0, 1), (2, 1), (1, 3), (2, 4)],
    )

    spots, parent = demo.load_lineage(path)

    assert parent == {1: 0, 2: 1, 3: 1}
    frame, zyx = spots[2]
    assert frame == 2
    np.testing.assert_allclose(zyx, (9.0, 6.0, 3.0))


def test_lineage_hues_split_at_divisions_and_stop_at_the_generation_cap() -> None:
    # A chain that divides every frame: 0 -> {1, 2}, 1 -> {3, 4}, ...
    spots = {i: (i, np.zeros(3, np.float32)) for i in range(31)}
    parent = {child: (child - 1) // 2 for child in range(1, 31)}

    hues = demo.lineage_hues(spots, parent)

    assert hues[0] == 0.5
    assert {hues[1], hues[2]} == {0.25, 0.75}
    gen3 = sorted(hues[i] for i in range(7, 15))
    assert gen3 == [(k + 0.5) / 8 for k in range(8)]
    # Past COLOUR_GENERATIONS a daughter keeps her mother's hue.
    assert all(hues[c] == hues[(c - 1) // 2] for c in range(15, 31))


def test_lineage_geometry_trails_stop_at_the_history_and_share_joints(
    monkeypatch,
) -> None:
    monkeypatch.setattr(demo, "FRAMES", (0, 4))
    monkeypatch.setattr(demo, "TRAIL_HISTORY", 3)
    spots = {i: (i, np.full(3, float(i), np.float32)) for i in range(5)}
    parent = {i: i - 1 for i in range(1, 5)}

    geo = demo.lineage_geometry(spots, parent)

    # Frame 0: one marker, no trail. Frame 4: links 3-4, 2-3, 1-2 (age <= 3).
    assert geo["points"].shape == (2, 4)
    assert len(geo["trail_edges"]) == 3
    assert len(geo["trail_vertices"]) == 4  # each joint authored once
    assert set(geo["trail_vertices"][:, 3]) == {4 * demo.FRAME_INTERVAL_MIN}
    newest = geo["trail_vertices"][:, 0].argmax()
    assert geo["trail_colors"][newest].max() > geo["trail_colors"].min(axis=1).min()


def test_splat_budget_is_proportional_and_clipped_per_channel() -> None:
    volume = np.zeros((10, 100, 100), np.uint8)
    for ch in demo.CHANNELS:
        assert demo.splat_budget(volume, ch["name"]) == demo.SPLATS_MIN
    volume[:] = 255
    for ch in demo.CHANNELS:
        expected = round(volume.size / ch["voxels_per_seed"])
        assert demo.splat_budget(volume, ch["name"]) == min(
            max(expected, demo.SPLATS_MIN), ch["splats_max"]
        )
    big = np.full((40, 400, 400), 255, np.uint8)
    assert demo.splat_budget(big, "membranes") == 120_000
    assert demo.splat_budget(big, "nuclei") == 60_000


def test_membranes_are_unmixed_from_the_additive_composite() -> None:
    # The export stores (min(255, m + n), n, n): membranes m, nuclei n.
    m = np.array([0, 40, 40, 0, 200], np.int16)
    n = np.array([0, 0, 30, 90, 100], np.int16)
    rgb = np.stack([np.minimum(m + n, 255), n, n], axis=-1).astype(np.uint8)
    np.testing.assert_array_equal(
        demo.channel_volume(rgb, "membranes"), [0, 40, 40, 0, 155]
    )
    np.testing.assert_array_equal(demo.channel_volume(rgb, "nuclei"), n)


def test_cache_names_change_with_the_recipe(monkeypatch) -> None:
    before = demo._frame_cache_name("membranes", 0)
    assert "red-minus-green" in before and "_max120000_" in before
    assert "_green_" in demo._frame_cache_name("nuclei", 0)
    monkeypatch.setattr(demo, "SPLATS_MIN", 1000)
    assert demo._frame_cache_name("membranes", 0) != before


def test_fetch_source_refuses_bytes_that_miss_the_pinned_digest(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(demo, "FRAMES", (0,))

    def fake_download(url: str, members: dict) -> dict:
        for path in members.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"not the recording")
        return members

    monkeypatch.setattr(demo, "download_zip_members", fake_download)
    with pytest.raises(ValueError, match="Source digest mismatch"):
        demo.fetch_source()


def test_manifest_hosts_the_frame_bundle_on_the_cc_by_record() -> None:
    from luxar.demos import load_manifest

    spec = load_manifest()["datasets"][demo.DATASET_NAME]
    assert spec["bucket"] == "zenodo" and spec["record"] == "cc-by"
    assert [f["name"] for f in spec["files"]] == [demo.BUNDLE_NAME]
    assert spec["acquisition"]["comparable"] is False
    assert demo.DEMO_META["requirements"]["compute"] == "medium"


def test_hosted_fits_split_the_bundle_per_channel(monkeypatch) -> None:
    """Members are requested CHANNELS x FRAMES and split back per channel."""
    monkeypatch.setattr(demo, "FRAMES", (0, 4))
    calls = {}

    def fake_bundle(name, bundle, names, *, recompute, include_stats):
        calls.update(name=name, bundle=bundle, include_stats=include_stats)
        return list(names)

    monkeypatch.setattr(demo, "load_dataset_bundle", fake_bundle)
    fits = demo.hosted_fits()
    assert calls == {
        "name": demo.DATASET_NAME,
        "bundle": demo.BUNDLE_NAME,
        "include_stats": True,
    }
    for ch in demo.CHANNELS:
        assert fits[ch["name"]] == [
            demo._frame_cache_name(ch["name"], f) for f in (0, 4)
        ]


def _empty_cache(monkeypatch, tmp_path: Path) -> None:
    """Point the manifest resolver at an empty cache, so the bundle must download."""
    from luxar.demos._support.datasets import data_fetch

    monkeypatch.setattr(data_fetch, "_DEFAULT_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(data_fetch, "_DEMOS_DATA_DIR", tmp_path / "no-lfs")


def test_an_offline_bundle_download_falls_back_to_the_local_build(
    monkeypatch, tmp_path: Path
) -> None:
    """Offline, the download layer raises what robust_download really raises."""
    import requests

    from luxar.demos._support.downloads import download

    def offline(*_args, **_kwargs):
        raise requests.exceptions.ConnectionError("Failed to resolve 'zenodo.org'")

    _empty_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "robust_download", offline)
    assert demo.hosted_fits() is None


def test_a_bundle_that_fails_its_checksum_still_raises(
    monkeypatch, tmp_path: Path
) -> None:
    from luxar.demos._support.downloads import download

    def corrupt(_url, output_path, **_kwargs):
        Path(output_path).write_bytes(b"not the bundle")
        return Path(output_path)

    _empty_cache(monkeypatch, tmp_path)
    monkeypatch.setattr(download, "robust_download", corrupt)
    with pytest.raises(ValueError, match="checksum"):
        demo.hosted_fits()


def test_an_unavailable_bundle_falls_back_to_the_local_build(monkeypatch) -> None:
    from luxar.demos import DatasetUnavailable

    def unavailable(*_args, **_kwargs):
        raise DatasetUnavailable("pending upload")

    monkeypatch.setattr(demo, "load_dataset_bundle", unavailable)
    assert demo.hosted_fits() is None


def test_other_fetch_faults_are_not_swallowed(monkeypatch) -> None:
    def damaged(*_args, **_kwargs):
        raise RuntimeError("the cache write is damaged")

    monkeypatch.setattr(demo, "load_dataset_bundle", damaged)
    with pytest.raises(RuntimeError, match="damaged"):
        demo.hosted_fits()


def _no_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("a cached run must not touch the network")


def test_a_verified_source_cache_skips_the_network(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(demo, "FRAMES", (0,))
    members = [demo.frame_member(0), demo.LINEAGE_MEMBER]
    for member in members:
        demo._source_path(member).parent.mkdir(parents=True, exist_ok=True)
        demo._source_path(member).write_bytes(member.encode())
    monkeypatch.setattr(demo, "SOURCE_SET_SHA256", demo._source_set_digest(members))
    monkeypatch.setattr(demo, "SOURCE_BYTES", sum(len(m) for m in members))
    monkeypatch.setattr(demo, "download_zip_members", _no_network)
    frames, lineage = demo.fetch_source()
    assert frames == [demo._source_path(members[0])]
    assert lineage == demo._source_path(demo.LINEAGE_MEMBER)


def test_lineage_is_pinned_and_fetched_only_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    import hashlib

    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path)
    path = demo._source_path(demo.LINEAGE_MEMBER)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"<TrackMate/>")
    monkeypatch.setattr(
        demo, "LINEAGE_SHA256", hashlib.sha256(b"<TrackMate/>").hexdigest()
    )
    monkeypatch.setattr(demo, "download_zip_members", _no_network)
    assert demo.fetch_lineage() == path
    path.write_bytes(b"tampered")
    seen: list[bool] = []

    def _refetch(url: str, members: dict) -> dict:
        # A corrupt copy is removed first, so the download cannot skip it.
        seen.append(path.exists())
        path.write_bytes(b"still wrong")
        return members

    monkeypatch.setattr(demo, "download_zip_members", _refetch)
    with pytest.raises(ValueError, match="Lineage digest mismatch"):
        demo.fetch_lineage()
    assert seen == [False]


def test_cached_fits_need_every_frame(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(demo, "FRAMES", (0, 4))
    monkeypatch.setattr(demo, "local_fit_path", lambda name, f: tmp_path / f)
    assert demo.cached_fits() is None


def _tiny_frame(seed: int, t_offset: float) -> GSplatData:
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=(rng.uniform(0.0, 10.0, (50, 3)) + t_offset).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, 50).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (50, 1)).astype(np.float32),
    )


def test_scene_builds_with_both_layers_and_the_lineage(
    monkeypatch, tmp_path: Path
) -> None:
    import zarr

    frames = (0, 4, 8)
    monkeypatch.setattr(demo, "FRAMES", frames)
    monkeypatch.setattr(demo, "OPENING_FRAME", 8)
    stacks = {
        ch["name"]: demo.stack_channel([_tiny_frame(i, i) for i in range(len(frames))])
        for ch in demo.CHANNELS
    }
    np.testing.assert_array_equal(
        np.unique(stacks["nuclei"].centers[:, 3]), [0.0, 40.0, 80.0]
    )
    spots = {i: (i, np.full(3, 5.0 + i, np.float32)) for i in range(9)}
    lineage = demo.lineage_geometry(spots, {i: i - 1 for i in range(1, 9)})

    out = demo.create_luxar_scene(
        stacks, lineage, tmp_path / "organoid.luxar.zarr", 88_569_352
    )

    root = zarr.open_group(str(out), mode="r")
    names = set(root.group_keys())
    assert {"membranes", "nuclei", "lineage_trails", "tracked_cells"} <= names
    assert root.attrs["citation"]["doi"] == "10.1038/s41467-022-32465-z"


def test_source_size_pins_cover_every_fetched_member() -> None:
    # 167 VTI frames (T0244 standing in for T0245, so 167 distinct files) plus
    # the lineage XML, measured on the extracted, digest-verified source.
    members = {demo.frame_member(demo.source_frame(f)) for f in demo.FRAMES}
    assert len(members) == len(demo.FRAMES) == 167
    assert demo.frame_member(244) not in members
    assert demo.SOURCE_FRAME_BYTES == 455_390_806
    assert demo.LINEAGE_BYTES == 6_711_980
    assert demo.SOURCE_BYTES == demo.SOURCE_FRAME_BYTES + demo.LINEAGE_BYTES


def test_a_source_that_misses_the_size_pin_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(demo, "CACHE_DIR", tmp_path)
    members = [demo.frame_member(0), demo.LINEAGE_MEMBER]
    for member in members:
        demo._source_path(member).parent.mkdir(parents=True, exist_ok=True)
        demo._source_path(member).write_bytes(b"12345")
    monkeypatch.setattr(demo, "SOURCE_BYTES", 10)
    assert demo.check_source_bytes(members) == 10
    monkeypatch.setattr(demo, "SOURCE_BYTES", 11)
    with pytest.raises(ValueError, match="update the size pins"):
        demo.check_source_bytes(members)


def _tiny_scene_inputs(monkeypatch) -> tuple[dict, dict]:
    frames = (0, 4, 8)
    monkeypatch.setattr(demo, "FRAMES", frames)
    monkeypatch.setattr(demo, "OPENING_FRAME", 8)
    stacks = {
        ch["name"]: demo.stack_channel([_tiny_frame(i, i) for i in range(len(frames))])
        for ch in demo.CHANNELS
    }
    spots = {i: (i, np.full(3, 5.0 + i, np.float32)) for i in range(9)}
    return stacks, demo.lineage_geometry(spots, {i: i - 1 for i in range(1, 9)})


def _scene_texts(scene: Path) -> tuple[str, list[str]]:
    from luxar._zarr_compat import group_keys, open_group

    root = open_group(str(scene), mode="r")
    overlays = root["overlays"]
    return dict(root.attrs)["description"], [
        dict(overlays[name].attrs)["text"] for name in group_keys(overlays)
    ]


def test_scene_text_states_its_size_against_the_source(
    monkeypatch, tmp_path: Path
) -> None:
    stacks, lineage = _tiny_scene_inputs(monkeypatch)
    out = demo.create_luxar_scene(
        stacks, lineage, tmp_path / "organoid.luxar.zarr", 88_569_352
    )
    description, texts = _scene_texts(out)
    assert description.endswith(
        "Scene size: 88.6 MB on disk, from the 462 MB source frames and "
        "lineage (≈5.2x smaller)."
    )
    # CC BY 4.0 attribution travels with the store: data DOI, licence link and
    # the changes made, before the size sentence the restater rewrites.
    assert (
        "Data: doi:10.5281/zenodo.6828906, CC BY 4.0 "
        "(https://creativecommons.org/licenses/by/4.0/); changed by unmixing the "
        "membrane channel (red minus green), showing every fourth frame (T0244 "
        "standing in for the empty T0245), Gaussian-splat fitting and overlaying "
        "the published lineage as smoothed trails. Scene size: "
    ) in description
    assert (
        "Light-sheet • membranes + nuclei • lineage • 88.6 MB, ≈5.2x smaller "
        "than the source frames and lineage • de Medeiros et al. 2022"
    ) in texts


def test_build_scene_states_the_measured_size_and_restamps_the_hash(
    monkeypatch, tmp_path: Path
) -> None:
    import shutil

    from luxar._zarr_compat import open_group
    from luxar.io._compiler.finalize.hashing import compute_content_hashes

    stacks, lineage = _tiny_scene_inputs(monkeypatch)
    restated: list[int] = []
    restate = demo.restate_scene_size

    def counting_restate(scene: Path, stated: int, measured: int) -> None:
        restated.append(measured)
        restate(scene, stated, measured)

    monkeypatch.setattr(demo, "restate_scene_size", counting_restate)
    out = tmp_path / "out.luxar.zarr"
    measured = demo.build_scene(stacks, lineage, out)

    assert measured == dir_size_bytes(out)
    # The tiny scene is nowhere near the 88.6 MB estimate, so it was restated.
    assert 1 <= len(restated) < SIZE_PASSES
    description, texts = _scene_texts(out)
    note, caption = demo._size_texts(measured)
    assert note in description and "88.6 MB" not in description
    assert any(caption in t for t in texts)
    # The stamped hash is the one a fresh walk computes over the restated text.
    stamped = dict(open_group(str(out), mode="r").attrs)["content_hash"]
    copy = tmp_path / "copy.luxar.zarr"
    shutil.copytree(out, copy)
    assert compute_content_hashes(open_group(str(copy), mode="r+")) == stamped
    assert not (tmp_path / ".building.out.luxar.zarr").exists()


def test_an_interrupted_build_leaves_output_path_untouched(
    monkeypatch, tmp_path: Path
) -> None:
    stacks, lineage = _tiny_scene_inputs(monkeypatch)
    out = tmp_path / "out.luxar.zarr"
    out.mkdir()
    (out / "previous.txt").write_text("the scene served before this build")
    restate = demo.restate_scene_size

    def dying_restate(scene: Path, stated: int, measured: int) -> None:
        raise KeyboardInterrupt  # Ctrl-C, OOM, ...

    monkeypatch.setattr(demo, "restate_scene_size", dying_restate)
    with pytest.raises(KeyboardInterrupt):
        demo.build_scene(stacks, lineage, out)
    assert [p.name for p in out.iterdir()] == ["previous.txt"]

    # The next run clears the interrupted scratch and replaces the old scene.
    monkeypatch.setattr(demo, "restate_scene_size", restate)
    measured = demo.build_scene(stacks, lineage, out)
    assert not (out / "previous.txt").exists()
    assert not (tmp_path / ".building.out.luxar.zarr").exists()
    assert not (tmp_path / ".out.luxar.zarr.old").exists()
    assert demo._size_texts(measured)[0] in _scene_texts(out)[0]
