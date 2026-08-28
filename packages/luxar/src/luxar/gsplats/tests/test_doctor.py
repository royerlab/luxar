"""``gsplat doctor`` — diagnosing, and repairing, an existing store.

The conditions the doctor exists for are SILENT: the dataset loads, renders, and
says nothing, while the viewer quietly orders its parts wrongly. So these tests
assert on two things a "did it run?" test would miss — that a repair lands in
BOTH the per-node attrs and the consolidated metadata that shadows them, and
that what was written actually orders the parts correctly.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import create_array as zc_create_array
from luxar._zarr_compat import open_group as zc_open_group
from luxar._zarr_compat import read_consolidated_attrs, read_node_attrs
from luxar.conftest import confine_temp_dirs
from luxar.gsplats.doctor import diagnose_store
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree


def _partition_store(tmp: Path, n: int = 400, parts_cap: int = 80) -> Path:
    """A real BSP partition on disk, split planes and all."""
    rng = np.random.default_rng(4)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    node = GSplatData(
        centers=(rng.random((n, 3)) * 100).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    ).to_spatial_partition(max_elements=parts_cap)
    assert node.bsp_tree is not None
    path = tmp / "part.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path


def _legacy_gsplat_store(tmp: Path, version: object) -> Path:
    path = tmp / "legacy.gsplats.zarr"
    root = zc_open_group(path, mode="w")
    root.attrs["format_type"] = "gsplats_zarr"
    if version is not None:
        root.attrs["format_version"] = version
    return path


def _flat_store(tmp: Path, n: int = 40) -> Path:
    rng = np.random.default_rng(8)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    path = tmp / "flat.gsplats.zarr"
    data = GSplatData(
        centers=rng.random((n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )
    write_gsplats_tree(path, data.tree)
    return path


def _partition_scene(
    tmp: Path, geometry: str = "points", *, drop_tree: bool = True
) -> tuple[Path, str]:
    """A real scene containing one native points or mesh partition."""
    from luxar import Dimensions, LuxarZarrCompiler

    rng = np.random.default_rng(12)
    positions = (rng.random((120, 3)) * 100).astype(np.float32)
    path = tmp / "scene.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        if geometry == "points":
            scene.add_points(
                "points",
                positions,
                radii=1.0,
                partition={"max_elements": 40},
                additive_lod=False,
            )
        elif geometry == "mesh":
            offsets = np.arange(40, dtype=np.float32)[:, None] * 10.0
            vertices = np.stack(
                (
                    np.concatenate(
                        (offsets, np.zeros((40, 2), dtype=np.float32)), axis=1
                    ),
                    np.concatenate(
                        (offsets + 1.0, np.zeros((40, 2), dtype=np.float32)), axis=1
                    ),
                    np.concatenate(
                        (
                            offsets,
                            np.ones((40, 1), dtype=np.float32),
                            np.zeros((40, 1), dtype=np.float32),
                        ),
                        axis=1,
                    ),
                ),
                axis=1,
            )
            faces = np.arange(120, dtype=np.uint32).reshape(40, 3)
            scene.add_mesh(
                "mesh",
                vertices.reshape(120, 3),
                faces,
                partition={"max_elements": 12},
            )
        else:
            raise ValueError(f"unsupported geometry: {geometry}")
        scene.add_points("sibling", np.zeros((3, 3), dtype=np.float32))

    root = zc_open_group(str(path), mode="r+")
    group = root[geometry]
    if drop_tree and "bsp_tree" in group.attrs:
        del group.attrs["bsp_tree"]
    zc_consolidate(root)
    return path, geometry


def _hidden_first_partition_scene(tmp: Path) -> Path:
    """A native 4D partition whose widest displayed column is index 3."""
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    rng = np.random.default_rng(7)
    positions = np.empty((400, 4), dtype=np.float32)
    positions[:, 0] = rng.integers(0, 3, 400)
    positions[:, 1:3] = rng.uniform(-1, 1, (400, 2))
    positions[:, 3] = rng.uniform(-40, 40, 400)
    dimensions = Dimensions(
        [
            Dimension("state", display=False, spatial=True, range=(0, 2)),
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
        ]
    )
    path = tmp / "hidden-first.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)
        scene.add_points(
            "points",
            positions,
            partition={"max_elements": 100},
            extend_to_all=[],
        )
    return path


def _disjoint_centroid_split_lines_scene(tmp: Path) -> Path:
    """A native lines partition whose valid plane crosses one part's bounds."""
    from luxar import Dimensions, LuxarZarrCompiler

    first = np.column_stack(
        (
            np.arange(11, dtype=np.float32),
            np.zeros(11, dtype=np.float32),
            np.zeros(11, dtype=np.float32),
        )
    )
    second = np.column_stack(
        (
            np.arange(12, 15, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
        )
    )
    positions = np.concatenate((first, second))
    indices = np.array(
        [(i, i + 1) for i in range(10)] + [(i, i + 1) for i in range(11, 13)],
        dtype=np.uint32,
    )

    path = tmp / "disjoint-lines.luxar.zarr"
    with LuxarZarrCompiler(path) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(
            "lines",
            positions,
            widths=0.1,
            indices=indices,
            line_type="indexed",
            partition={"max_elements": 12, "rule": "median"},
        )
    return path


def _uniform_tiled_store(tmp: Path, *, stacked: bool = False) -> Path:
    """A uniform-tiled partition on disk: overlapping parts, approximate planes.

    The shape this PR's own uniform-tiling producer writes — apodized tiles keep
    their overlap band, so the parts genuinely intersect and the grid's cuts sit
    at each band's midplane. No tree separates these parts.
    """
    from luxar.gsplats.tiling import compute_tile_specs, grid_bsp_tree

    specs = compute_tile_specs((48, 48, 48), 32, 8)
    rng = np.random.default_rng(7)
    regions = []
    for spec in specs:
        lo = np.array(spec.origin, dtype=float)
        hi = lo + np.array(spec.shape, dtype=float)
        chol = np.zeros((40, 10 if stacked else 6), dtype=np.float32)
        chol[:, [0, 2, 5] + ([9] if stacked else [])] = 1.0
        centers = rng.uniform(lo, hi, size=(40, 3)).astype(np.float32)
        centers[0] = lo
        centers[1] = hi
        if stacked:
            centers = np.column_stack(
                (centers, np.full(40, spec.index, dtype=np.float32))
            )
        regions.append(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(40, dtype=np.float32),
                cholesky_factors=chol,
            )
        )
    node = GSplatData.partition_from_regions(
        regions,
        bsp_tree=grid_bsp_tree(specs),
        region_labels=[s.index for s in specs],
    )
    assert node.bsp_tree is not None
    path = tmp / "uniform.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path


def _sparse_uniform_tiled_store(tmp: Path) -> Path:
    """A producer-shaped grid whose sparse content under-fills one halo."""
    from luxar.gsplats.tiling import compute_tile_specs, grid_bsp_tree

    specs = compute_tile_specs((96, 16, 16), 32, 8)
    x_bounds = ((0.0, 32.0), (24.0, 56.0), (72.0, 74.0), (72.0, 74.0))
    assert len(specs) == len(x_bounds)
    regions = []
    for lo_x, hi_x in x_bounds:
        centers = np.array([[lo_x, 1.0, 1.0], [hi_x, 2.0, 2.0]], dtype=np.float32)
        chol = np.zeros((2, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        regions.append(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=chol,
            )
        )
    node = GSplatData.partition_from_regions(
        regions,
        bsp_tree=grid_bsp_tree(specs),
        region_labels=[spec.index for spec in specs],
    )
    path = tmp / "sparse-uniform.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path


def _mixed_overlap_store(tmp: Path) -> tuple[Path, dict]:
    """Three x-ordered parts: one overlapping cut and one clean cut."""
    bounds = ((0.0, 32.0), (24.0, 56.0), (80.0, 112.0))
    regions = []
    for lo_x, hi_x in bounds:
        lo = np.array([lo_x, 0.0, 0.0], dtype=np.float32)
        hi = np.array([hi_x, 10.0, 10.0], dtype=np.float32)
        centers = np.stack((lo, hi))
        chol = np.zeros((2, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        regions.append(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=chol,
            )
        )
    tree = {
        "axis": 0,
        "split": 68.0,
        "left": {
            "axis": 0,
            "split": 28.0,
            "left": {"part": 0},
            "right": {"part": 1},
        },
        "right": {"part": 2},
    }
    node = GSplatData.partition_from_regions(
        regions, bsp_tree=tree, region_labels=[0, 1, 2]
    )
    path = tmp / "mixed.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path, tree


def _centered_mixed_overlap_store(tmp: Path) -> tuple[Path, dict]:
    """Three x-ordered parts with an overlapping cut and a clean cut at zero."""
    bounds = ((-80.0, -48.0), (-56.0, -24.0), (24.0, 56.0))
    regions = []
    for lo_x, hi_x in bounds:
        lo = np.array([lo_x, 0.0, 0.0], dtype=np.float32)
        hi = np.array([hi_x, 10.0, 10.0], dtype=np.float32)
        centers = np.stack((lo, hi))
        chol = np.zeros((2, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        regions.append(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=chol,
            )
        )
    tree = {
        "axis": 0,
        "split": 0.0,
        "left": {
            "axis": 0,
            "split": -52.0,
            "left": {"part": 0},
            "right": {"part": 1},
        },
        "right": {"part": 2},
    }
    node = GSplatData.partition_from_regions(
        regions, bsp_tree=tree, region_labels=[0, 1, 2]
    )
    path = tmp / "centered-mixed.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path, tree


def _two_part_overlap_store(tmp: Path) -> tuple[Path, dict]:
    """Two overlapping x-ordered parts with one band-midpoint cut."""
    regions = []
    for lo_x, hi_x in ((0.0, 32.0), (24.0, 56.0)):
        lo = np.array([lo_x, 0.0, 0.0], dtype=np.float32)
        hi = np.array([hi_x, 10.0, 10.0], dtype=np.float32)
        centers = np.stack((lo, hi))
        chol = np.zeros((2, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        regions.append(
            GSplatData(
                centers=centers,
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=chol,
            )
        )
    tree = {
        "axis": 0,
        "split": 28.0,
        "left": {"part": 0},
        "right": {"part": 1},
    }
    node = GSplatData.partition_from_regions(
        regions, bsp_tree=tree, region_labels=[0, 1]
    )
    path = tmp / "two-part.gsplats.zarr"
    write_gsplats_tree(path, node)
    return path, tree


def _scale_tree_planes(tree: dict, factors: tuple[float, ...]) -> dict:
    if "part" in tree:
        return dict(tree)
    axis = int(tree["axis"])
    return {
        "axis": axis,
        "split": float(tree["split"]) * factors[axis],
        "left": _scale_tree_planes(tree["left"], factors),
        "right": _scale_tree_planes(tree["right"], factors),
    }


def _root_attrs(path: Path) -> dict:
    """The root node's attributes as they sit on disk, in either zarr format."""
    attrs = read_node_attrs(path)
    assert attrs is not None, f"no readable root metadata under {path}"
    return attrs


def _consolidated_attrs(path: Path) -> dict:
    """The root's attributes as recorded in the CONSOLIDATED index.

    Deliberately distinct from :func:`_root_attrs`: the doctor's job includes
    noticing when the two disagree, which is exactly what a stale consolidation
    looks like. Format 2 keys the index by metadata document and format 3 by
    node, so the distinction is preserved through the facade rather than by
    naming either layout.
    """
    return read_consolidated_attrs(path)["/"]


def _set_root_attr(path: Path, key: str, value) -> None:
    """Write (or delete) a root attr through the facade, refreshing consolidation.

    Through the facade, as Luxar's own in-place editors are: re-opening an
    already-consolidated store with plain ``zarr.open_group`` and
    re-consolidating leaves a NESTED index holding the pre-edit attributes,
    which later reads prefer over the correct per-node documents.
    """
    root = zc_open_group(str(path), mode="r+")
    if value is None:
        del root.attrs[key]
    else:
        root.attrs[key] = value
    zc_consolidate(root)


def _part_boxes(path: Path) -> list:
    boxes = {}
    for part in path.glob("part_*"):
        attrs = read_node_attrs(part) or {}
        boxes[int(attrs["child_index"])] = (
            np.array(attrs["position_bounds"]["min"][:3]),
            np.array(attrs["position_bounds"]["max"][:3]),
        )
    return [boxes[i] for i in range(len(boxes))]


def _required_last(box_a, box_b, eye):
    """Which of two disjoint boxes MUST be drawn last, or ``None`` if either may.

    A separating axis-aligned plane puts the eye's own side nearer, so that side
    is drawn last; an eye strictly inside a separating gap proves mutual
    non-occlusion and vetoes every other axis's verdict.
    """
    verdicts = set()
    for k in range(3):
        for name, lo, hi in (("a", box_a, box_b), ("b", box_b, box_a)):
            if lo[1][k] <= hi[0][k]:
                if lo[1][k] < eye[k] < hi[0][k]:
                    return None
                verdicts.add(
                    name if eye[k] <= lo[1][k] else ("b" if name == "a" else "a")
                )
    return verdicts.pop() if len(verdicts) == 1 else None


def _order_violations(tree: dict, boxes: list, poses: int = 60) -> int:
    """Pairwise ordering constraints the tree's traversal violates.

    Same model as ``test_bsp_tree_serialized``: a separating plane makes one box
    unambiguously nearer, and an eye inside a separating gap proves mutual
    non-occlusion and vetoes.
    """

    def traverse(node, eye, out):
        if "part" in node:
            out.append(node["part"])
            return
        if eye[node["axis"]] < node["split"]:
            traverse(node["right"], eye, out)
            traverse(node["left"], eye, out)
        else:
            traverse(node["left"], eye, out)
            traverse(node["right"], eye, out)

    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo))
    rng = np.random.default_rng(2)
    bad = 0
    for mult in (1.5, 0.35):  # outside the volume, then inside it
        for _ in range(poses):
            d = rng.normal(size=3)
            eye = centre + d / np.linalg.norm(d) * radius * mult
            order: list = []
            traverse(tree, eye, order)
            rank = {p: i for i, p in enumerate(order)}
            for a, b in itertools.combinations(range(len(boxes)), 2):
                need = _required_last(boxes[a], boxes[b], eye)
                if need is None:
                    continue
                if ("a" if rank[a] > rank[b] else "b") != need:
                    bad += 1
    return bad


class TestGsplatReadabilityCheck:
    @pytest.mark.parametrize("version", ["2.0", None])
    def test_unsupported_or_missing_format_version_is_diagnosed(
        self, version: object
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _legacy_gsplat_store(Path(tmp), version)

            report = diagnose_store(path)

            assert not report.healthy
            (finding,) = report.findings
            assert finding.check == "format-version"
            assert finding.severity == "error"
            assert finding.path == ""
            assert repr(version) in finding.summary
            assert not finding.fixable
            assert "migrate-format" in finding.remedy

    def test_supported_but_unreadable_store_is_diagnosed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _legacy_gsplat_store(Path(tmp), "3.4")

            report = diagnose_store(path)

            assert not report.healthy
            (finding,) = report.findings
            assert finding.check == "readability"
            assert finding.severity == "error"
            assert finding.path == ""
            assert "cannot be read" in finding.summary
            assert "missing required array 'centers'" in finding.detail
            assert not finding.fixable

    def test_all_tree_layouts_are_checked_without_decoding(self, monkeypatch) -> None:
        from luxar.encoding import ArrayDecoder
        from luxar.gsplats.doctor.checks import check_gsplat_readable
        from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition

        calls = 0
        original = ArrayDecoder.decode

        def counted_decode(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(ArrayDecoder, "decode", counted_decode)
        with tempfile.TemporaryDirectory() as tmp:
            flat_path = _flat_store(Path(tmp))
            flat = GSplatData.load(flat_path).tree
            assert isinstance(flat, GSplatLeaf)
            sublod = flat.additive_sublods[0]
            nodes = {
                "additive": GSplatLeaf([sublod, sublod]),
                "lod": GSplatLodGroup([GSplatLeaf([sublod]), GSplatLeaf([sublod])]),
                "partition": GSplatPartition(
                    [GSplatLeaf([sublod]), GSplatLeaf([sublod])]
                ),
            }
            paths = [flat_path]
            for name, node in nodes.items():
                path = Path(tmp) / f"{name}.gsplats.zarr"
                write_gsplats_tree(path, node, ordering="none")
                paths.append(path)

            calls = 0
            reports = [
                diagnose_store(path, checks=[check_gsplat_readable]) for path in paths
            ]

        assert all(report.findings == [] for report in reports)
        assert calls == 0

    def test_mismatched_leaf_array_lengths_are_diagnosed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _flat_store(Path(tmp))
            root = zc_open_group(path, mode="r+")
            del root["amplitudes"]
            zc_create_array(
                root,
                "amplitudes",
                data=np.ones(39, dtype=np.float32),
                compressor=None,
            )
            zc_consolidate(root)

            report = diagnose_store(path)

            assert not report.healthy
            (finding,) = report.findings
            assert finding.check == "readability"
            assert "array lengths disagree" in finding.detail
            assert "amplitudes=39" in finding.detail
            assert "centers=40" in finding.detail


class TestSplitPlanesCheck:
    def test_axis_three_partition_is_healthy_and_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _hidden_first_partition_scene(Path(tmp))
            attrs = read_node_attrs(path / "points") or {}
            before = attrs["bsp_tree"]
            assert before["axis"] == 3

            report = diagnose_store(path)
            assert report.healthy

            fixed = diagnose_store(path, fix=True)
            assert fixed.healthy
            attrs = read_node_attrs(path / "points") or {}
            assert attrs["bsp_tree"] == before

    def test_missing_axis_three_planes_are_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _hidden_first_partition_scene(Path(tmp))
            root = zc_open_group(str(path), mode="r+")
            del root["points"].attrs["bsp_tree"]
            zc_consolidate(root)

            report = diagnose_store(path)
            assert [finding.severity for finding in report.findings] == ["error"]
            assert report.findings[0].fixable
            assert "no split planes recorded for 4 parts" in report.findings[0].summary

            fixed = diagnose_store(path, fix=True)
            assert fixed.healthy
            attrs = read_node_attrs(path / "points") or {}
            assert attrs["bsp_tree"]["axis"] == 3
            assert diagnose_store(path).healthy

    def test_narrow_part_bounds_report_a_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _hidden_first_partition_scene(Path(tmp))
            root = zc_open_group(str(path), mode="r+")
            for name in root["points"].group_keys():
                child = root["points"][name]
                bounds = dict(child.attrs["position_bounds"])
                bounds["min"] = bounds["min"][:3]
                bounds["max"] = bounds["max"][:3]
                child.attrs["position_bounds"] = bounds
            zc_consolidate(root)

            report = diagnose_store(path)

            assert [finding.path for finding in report.findings] == ["points"]
            assert report.findings[0].summary == (
                "split planes disagree with the parts, and cannot be rebuilt"
            )
            assert report.findings[0].fixable
            assert not report.healthy

    def test_ragged_part_bounds_report_a_finding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _hidden_first_partition_scene(Path(tmp))
            root = zc_open_group(str(path), mode="r+")
            names = sorted(root["points"].group_keys())
            for name in names[1:]:
                child = root["points"][name]
                bounds = dict(child.attrs["position_bounds"])
                bounds["min"] = bounds["min"][:3]
                bounds["max"] = bounds["max"][:3]
                child.attrs["position_bounds"] = bounds
            zc_consolidate(root)

            report = diagnose_store(path)

            assert [finding.path for finding in report.findings] == ["points"]
            assert report.findings[0].summary == (
                "split planes present but not verifiable"
            )
            assert report.healthy

    def test_a_healthy_partition_reports_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            report = diagnose_store(path)
            assert report.findings == []
            assert report.healthy
            assert report.checks_run  # says what it looked at

    def test_missing_planes_are_diagnosed_and_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", None)  # an old, pre-#1555 store

            report = diagnose_store(path)
            assert not report.healthy
            (finding,) = report.findings
            assert finding.severity == "error"
            assert finding.fixable and not finding.fixed
            assert "no split planes" in finding.summary
            # A diagnosis must not write.
            assert "bsp_tree" not in _root_attrs(path)

            fixed = diagnose_store(path, fix=True)
            assert fixed.healthy
            assert all(f.fixed for f in fixed.findings)
            assert diagnose_store(path).findings == []

    def test_a_repair_reaches_the_consolidated_metadata_too(self) -> None:
        """Consolidated metadata SHADOWS per-node attrs — a repair that only
        wrote the latter would look applied on disk and be invisible to every
        reader, including the viewer."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            before = _root_attrs(path)["content_hash"]
            _set_root_attr(path, "bsp_tree", None)

            diagnose_store(path, fix=True)

            assert "bsp_tree" in _root_attrs(path)
            assert "bsp_tree" in _consolidated_attrs(path)
            assert (
                _root_attrs(path)["bsp_tree"] == _consolidated_attrs(path)["bsp_tree"]
            )
            # ...and the content hash moved, so the viewer's cache invalidates.
            assert _root_attrs(path)["content_hash"] != before

    def test_the_recovered_planes_actually_order_the_parts(self) -> None:
        """The point of the repair, not merely that an attr appeared."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", None)
            diagnose_store(path, fix=True)

            boxes = _part_boxes(path)
            assert len(boxes) > 3, "need a non-trivial partition to mean anything"
            assert _order_violations(_root_attrs(path)["bsp_tree"], boxes) == 0

    def test_stale_planes_are_caught_and_rebuilt(self) -> None:
        """A tree left in a pre-transform coordinate space is worse than none:
        it still traverses to a plausible permutation, so the ordering is
        confidently wrong rather than falling back."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            stale = _root_attrs(path)["bsp_tree"]

            def shift(node: dict) -> dict:
                if "part" in node:
                    return node
                return {
                    "axis": node["axis"],
                    "split": node["split"] + 1000.0,  # way off the parts
                    "left": shift(node["left"]),
                    "right": shift(node["right"]),
                }

            _set_root_attr(path, "bsp_tree", shift(stale))

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "error"
            assert "disagree" in finding.summary
            assert finding.fixable

            diagnose_store(path, fix=True)
            boxes = _part_boxes(path)
            assert _order_violations(_root_attrs(path)["bsp_tree"], boxes) == 0

    def test_small_stale_offset_on_points_remains_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, group_path = _partition_scene(Path(tmp), drop_tree=False)
            node_path = path / group_path
            attrs = read_node_attrs(node_path)
            assert attrs is not None

            def shift(node: dict) -> dict:
                if "part" in node:
                    return node
                return {
                    "axis": node["axis"],
                    "split": node["split"] + 2.0,
                    "left": shift(node["left"]),
                    "right": shift(node["right"]),
                }

            root = zc_open_group(str(path), mode="r+")
            root[group_path].attrs["bsp_tree"] = shift(attrs["bsp_tree"])
            zc_consolidate(root)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.path == group_path
            assert finding.severity == "error"
            assert finding.fixable
            assert not report.healthy

    def test_an_approximate_tree_over_overlapping_parts_is_left_alone(self) -> None:
        """A uniform-tiled fit's parts overlap, so NO tree separates them and
        failing the separation test says nothing about staleness. Condemning one
        would delete the producer's own (documented, band-bounded) planes and
        drop the viewer back to the centroid order those planes exist to avoid."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            before = _root_attrs(path)["bsp_tree"]

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "note"
            assert not finding.fixable
            assert "centroid-split lines or mesh" in finding.detail
            assert report.healthy  # a note does not fail the gate

            diagnose_store(path, fix=True)
            assert _root_attrs(path)["bsp_tree"] == before

    def test_a_stacked_axis_does_not_make_overlapping_parts_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp), stacked=True)
            before = _root_attrs(path)["bsp_tree"]

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "note"
            assert not finding.fixable
            assert report.healthy

            diagnose_store(path, fix=True)
            assert _root_attrs(path)["bsp_tree"] == before

    def test_disjoint_centroid_split_lines_are_approximate_and_rebuilt(self) -> None:
        from luxar.core.group.partition import serialized_bsp_tree_separates

        with tempfile.TemporaryDirectory() as tmp:
            path = _disjoint_centroid_split_lines_scene(Path(tmp))
            node_path = path / "lines"
            before_attrs = read_node_attrs(node_path)
            assert before_attrs is not None
            before = before_attrs["bsp_tree"]
            boxes = _part_boxes(node_path)
            assert not serialized_bsp_tree_separates(before, boxes)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.path == "lines"
            assert finding.severity == "note"
            assert finding.fixable
            assert report.healthy

            fixed = diagnose_store(path, fix=True)
            after_attrs = read_node_attrs(node_path)
            assert after_attrs is not None
            after = after_attrs["bsp_tree"]
            assert after != before
            assert serialized_bsp_tree_separates(after, boxes)
            assert fixed.healthy

    def test_sparse_uniform_content_keeps_the_producer_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _sparse_uniform_tiled_store(Path(tmp))
            before = _root_attrs(path)["bsp_tree"]

            report = diagnose_store(path)

            (finding,) = report.findings
            assert finding.severity == "note"
            assert not finding.fixable
            assert report.healthy

            diagnose_store(path, fix=True)
            assert _root_attrs(path)["bsp_tree"] == before

            _set_root_attr(path, "bsp_tree", _scale_tree_planes(before, (0.25,) * 3))
            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "error"
            assert finding.fixable

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == before

    def test_an_approximate_tree_is_recovered_without_overclaiming_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            healthy = _root_attrs(path)["bsp_tree"]
            _set_root_attr(
                path, "bsp_tree", _scale_tree_planes(healthy, (0.25, 0.5, 0.125))
            )

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "error"
            assert finding.fixable
            assert "coordinate frame" not in finding.summary
            assert "overlap band" in finding.detail

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == healthy

    def test_a_clean_cut_does_not_disable_overlap_scale_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, healthy = _mixed_overlap_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", _scale_tree_planes(healthy, (0.25,) * 3))

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.fixable
            assert "coordinate frame" in finding.summary

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == healthy

    def test_a_zero_clean_cut_does_not_disable_overlap_scale_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, healthy = _centered_mixed_overlap_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", _scale_tree_planes(healthy, (0.25,) * 3))

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.fixable

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == healthy

    def test_uniform_scale_across_axes_has_store_wide_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            healthy = _root_attrs(path)["bsp_tree"]
            _set_root_attr(path, "bsp_tree", _scale_tree_planes(healthy, (0.25,) * 3))

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.fixable
            assert "coordinate frame" in finding.summary
            assert "downscale" in finding.detail

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == healthy

    def test_overlap_recovery_does_not_claim_a_frame_scale_without_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, healthy = _mixed_overlap_store(Path(tmp))
            broken = _scale_tree_planes(healthy, (0.25,) * 3)
            broken["left"]["split"] *= 1.2
            _set_root_attr(path, "bsp_tree", broken)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.fixable
            assert "coordinate frame" not in finding.summary
            assert "downscale" not in finding.detail
            assert "overlap band" in finding.detail

    def test_one_plane_is_recovered_without_claiming_a_frame_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, healthy = _two_part_overlap_store(Path(tmp))
            broken = dict(healthy)
            broken["split"] = 1e-6
            _set_root_attr(path, "bsp_tree", broken)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.fixable
            assert "coordinate frame" not in finding.summary
            assert "downscale" not in finding.detail
            assert "overlap band" in finding.detail

            repaired = diagnose_store(path, fix=True)
            assert repaired.healthy
            assert _root_attrs(path)["bsp_tree"] == healthy

    def test_frame_scale_recovery_pads_unsplit_nd_axes(self) -> None:
        from luxar.gsplats.doctor.checks import _recover_frame_scale

        stored = {
            "axis": 0,
            "split": 1.0,
            "left": {"part": 0},
            "right": {"part": 1},
        }
        boxes = [
            (np.array([0.0, 0.0, 0.0, 0.0]), np.array([2.0, 1.0, 1.0, 1.0])),
            (np.array([2.0, 0.0, 0.0, 0.0]), np.array([4.0, 1.0, 1.0, 1.0])),
        ]

        recovered = _recover_frame_scale(stored, boxes)

        assert recovered is not None
        repaired, factors, frame_scale_supported = recovered
        assert repaired["split"] == 2.0
        assert factors == (2.0, 1.0, 1.0, 1.0)
        assert not frame_scale_supported

    def test_frame_scale_recovery_repairs_an_nd_split_axis(self) -> None:
        from luxar.gsplats.doctor.checks import _recover_frame_scale

        stored = {
            "axis": 3,
            "split": 1.0,
            "left": {"part": 0},
            "right": {"part": 1},
        }
        boxes = [
            (np.array([0.0, 0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 2.0])),
            (np.array([0.0, 0.0, 0.0, 2.0]), np.array([1.0, 1.0, 1.0, 4.0])),
        ]

        recovered = _recover_frame_scale(stored, boxes)

        assert recovered is not None
        repaired, factors, frame_scale_supported = recovered
        assert repaired["split"] == 2.0
        assert factors == (1.0, 1.0, 1.0, 2.0)
        assert not frame_scale_supported

    def test_unrecoverable_overlap_violation_names_the_actual_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            broken = _scale_tree_planes(_root_attrs(path)["bsp_tree"], (0.25,) * 3)
            broken["left"]["split"] *= 2.0
            _set_root_attr(path, "bsp_tree", broken)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert "centers do not straddle" in finding.detail
            assert "measured overlap band" in finding.detail

    def test_inconsistent_plane_scales_are_not_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            broken = _scale_tree_planes(_root_attrs(path)["bsp_tree"], (0.25,) * 3)
            broken["left"]["split"] *= 2.0
            _set_root_attr(path, "bsp_tree", broken)

            report = diagnose_store(path)
            (finding,) = report.findings
            assert finding.severity == "error"
            assert "cannot be rebuilt" in finding.summary

            diagnose_store(path, fix=True)
            assert "bsp_tree" not in _root_attrs(path)

    def test_a_repair_that_leaves_a_lesser_condition_does_not_report_healthy(
        self,
    ) -> None:
        """Removing a misleading tree from parts that cannot be ordered exactly
        still leaves them unorderable. A run that called every fix must not
        report a clean bill of health for a store the next run condemns."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _uniform_tiled_store(Path(tmp))
            # Names a different part set, so it is broken however the parts sit.
            _set_root_attr(
                path,
                "bsp_tree",
                {"axis": 0, "split": 20.0, "left": {"part": 0}, "right": {"part": 1}},
            )

            report = diagnose_store(path, fix=True)
            assert all(f.fixed for f in report.findings)
            assert "bsp_tree" not in _root_attrs(path)
            assert not report.healthy
            assert [f.severity for f in report.unresolved] == ["warning"]
            assert not diagnose_store(path).healthy  # and the next run agrees

    def test_a_repair_that_cures_the_store_reports_healthy(self) -> None:
        """The negative control: the re-check must not turn every fix run red."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", None)
            report = diagnose_store(path, fix=True)
            assert report.residual == []
            assert report.healthy

    def test_a_tree_naming_the_wrong_part_set_is_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(
                path,
                "bsp_tree",
                {"axis": 0, "split": 50.0, "left": {"part": 0}, "right": {"part": 1}},
            )
            report = diagnose_store(path)
            assert [f.severity for f in report.findings] == ["error"]


class TestStoreGuards:
    def test_a_scene_partition_is_diagnosed_and_repaired_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, group_path = _partition_scene(Path(tmp))
            root = zc_open_group(str(path), mode="r+")
            before = root.attrs["content_hash"]
            sibling_before = root["sibling"].attrs["content_hash"]

            report = diagnose_store(path)
            assert [finding.path for finding in report.findings] == [group_path]
            assert report.findings[0].severity == "error"
            assert "no split planes" in report.findings[0].summary

            fixed = diagnose_store(path, fix=True)
            assert fixed.healthy
            reopened = zc_open_group(str(path), mode="r")
            assert reopened.attrs["content_hash"] != before
            assert reopened["sibling"].attrs["content_hash"] == sibling_before
            node_attrs = read_node_attrs(path / group_path)
            assert node_attrs is not None
            assert (
                node_attrs["bsp_tree"]
                == read_consolidated_attrs(path)[group_path]["bsp_tree"]
            )
            assert diagnose_store(path).findings == []

    def test_a_native_mesh_partition_is_diagnosed_and_repaired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, group_path = _partition_scene(Path(tmp), geometry="mesh")
            report = diagnose_store(path)
            assert [finding.path for finding in report.findings] == [group_path]
            assert "no split planes" in report.findings[0].summary

            fixed = diagnose_store(path, fix=True)
            assert fixed.healthy
            assert diagnose_store(path).findings == []

    def test_a_non_gsplats_store_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.zarr"
            zarr.open_group(str(path), mode="w")
            with pytest.raises(ValueError, match="not a Luxar scene"):
                diagnose_store(path)

    @staticmethod
    def _archive(tmp: Path) -> Path:
        """A partition with no split planes, packed as a .gsplats.zarr.zip."""
        import shutil

        store = _partition_store(tmp)
        _set_root_attr(store, "bsp_tree", None)
        base = tmp / "packed"
        shutil.make_archive(str(base), "zip", root_dir=str(tmp), base_dir=store.name)
        archive = tmp / "packed.gsplats.zarr.zip"
        (tmp / "packed.zip").rename(archive)
        return archive

    def test_a_compressed_store_can_be_diagnosed(self) -> None:
        """Most bundled demo datasets ship as .zip, so refusing archives outright
        would put the common case out of reach of a read-only sweep."""
        with tempfile.TemporaryDirectory() as tmp:
            archive = self._archive(Path(tmp))
            report = diagnose_store(archive)
            assert not report.healthy
            (finding,) = report.findings
            assert "no split planes" in finding.summary
            # The report names what the user asked about, not the temp extraction.
            assert report.path == str(archive)

    def test_fixing_a_compressed_store_is_refused(self) -> None:
        """There is nothing to write back to in place."""
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(ValueError, match="unpack"):
                diagnose_store(self._archive(Path(tmp)), fix=True)

    def test_diagnosing_an_archive_leaves_no_temp_directory_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = self._archive(tmp_path)
        confine_temp_dirs(tmp_path, monkeypatch)
        assert Path(tempfile.gettempdir()) == tmp_path

        # Stand in for a concurrent compressed save in the confined root.
        with tempfile.TemporaryDirectory(prefix="luxar_gsplat_save_", dir=tmp_path):
            diagnose_store(archive)
            assert list(tmp_path.glob("luxar_gsplat_archive_*")) == []
