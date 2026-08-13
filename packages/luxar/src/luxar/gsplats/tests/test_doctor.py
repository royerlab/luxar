"""``gsplat doctor`` — diagnosing, and repairing, an existing store.

The conditions the doctor exists for are SILENT: the dataset loads, renders, and
says nothing, while the viewer quietly orders its parts wrongly. So these tests
assert on two things a "did it run?" test would miss — that a repair lands in
BOTH the per-node attrs and the consolidated metadata that shadows them, and
that what was written actually orders the parts correctly.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

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


def _root_attrs(path: Path) -> dict:
    return json.loads((path / ".zattrs").read_text())


def _consolidated_attrs(path: Path) -> dict:
    return json.loads((path / ".zmetadata").read_text())["metadata"][".zattrs"]


def _set_root_attr(path: Path, key: str, value) -> None:
    """Write (or delete) a root attr through zarr, refreshing consolidation."""
    root = zarr.open_group(str(path), mode="r+")
    if value is None:
        del root.attrs[key]
    else:
        root.attrs[key] = value
    zarr.consolidate_metadata(root.store)


def _part_boxes(path: Path) -> list:
    boxes = {}
    for part in path.glob("part_*"):
        attrs = json.loads((part / ".zattrs").read_text())
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


class TestSplitPlanesCheck:
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
    def test_a_non_gsplats_store_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plain.zarr"
            zarr.open_group(str(path), mode="w")
            with pytest.raises(ValueError, match="not a standalone"):
                diagnose_store(path)

    def test_fixing_a_compressed_store_is_refused_but_diagnosis_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            archive = Path(tmp) / "packed.gsplats.zarr.zip"
            import shutil

            shutil.make_archive(str(archive).replace(".zip", ""), "zip", str(path))
            with pytest.raises(ValueError, match="unpack"):
                diagnose_store(archive, fix=True)


class TestCli:
    def test_doctor_exits_nonzero_while_a_problem_stands(self) -> None:
        from typer.testing import CliRunner

        from luxar.cli import app

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", None)

            result = runner.invoke(app, ["gsplat", "doctor", str(path), "--no-info"])
            assert result.exit_code == 1, result.stdout
            assert "no split planes" in result.stdout

            fixed = runner.invoke(
                app, ["gsplat", "doctor", str(path), "--no-info", "--fix"]
            )
            assert fixed.exit_code == 0, fixed.stdout

            again = runner.invoke(app, ["gsplat", "doctor", str(path), "--no-info"])
            assert again.exit_code == 0, again.stdout
            assert "No problems found" in again.stdout

    def test_doctor_writes_a_json_report(self) -> None:
        from typer.testing import CliRunner

        from luxar.cli import app

        with tempfile.TemporaryDirectory() as tmp:
            path = _partition_store(Path(tmp))
            _set_root_attr(path, "bsp_tree", None)
            out = Path(tmp) / "report.json"
            CliRunner().invoke(
                app,
                ["gsplat", "doctor", str(path), "--no-info", "--json", str(out)],
            )
            payload = json.loads(out.read_text())
            assert payload["healthy"] is False
            assert payload["findings"][0]["check"] == "split-planes"
            assert payload["findings"][0]["fixable"] is True
