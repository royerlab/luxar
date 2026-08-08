"""Unit tests for demo_dmri_tractography.

Exercises the network-free helpers (arc-length resampling, direction colouring,
the RAS -> scene rotation, the indexed edge builder, the deterministic
subsampler, the sizing/reuse gates) plus the scene authoring itself, on
synthetic bundles — nothing here downloads the 588 MB atlas. The demo is loaded
by file path so importing it never triggers the module-level
``parse_demo_flags`` scan of pytest's argv.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_dmri_tractography.py"


def _load_demo_module():
    name = "_luxar_demo_tractography_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


class TestResamplePolyline:
    def test_output_shape_and_dtype(self) -> None:
        pts = np.cumsum(np.ones((17, 3)), axis=0)
        out = _demo.resample_polyline(pts, 28)
        assert out.shape == (28, 3)
        assert out.dtype == np.float32

    def test_endpoints_are_preserved(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 5.0, 0.0], [10.0, 2.0, -3.0]])
        out = _demo.resample_polyline(pts, 40)
        np.testing.assert_allclose(out[0], pts[0], atol=1e-5)
        np.testing.assert_allclose(out[-1], pts[-1], atol=1e-5)

    def test_points_are_evenly_spaced_in_arc_length(self) -> None:
        # A straight line sampled with deliberately UNEVEN input spacing: an
        # index-space resample would inherit the clustering, an arc-length one
        # must not. This is the property the whole helper exists for.
        t = np.array([0.0, 0.01, 0.02, 0.03, 1.0])
        pts = np.column_stack([t * 100.0, np.zeros(5), np.zeros(5)])
        out = _demo.resample_polyline(pts, 21)
        step = np.linalg.norm(np.diff(out, axis=0), axis=1)
        assert step.std() < 1e-3
        np.testing.assert_allclose(step.mean(), 5.0, rtol=1e-4)

    def test_curved_path_stays_on_the_curve(self) -> None:
        theta = np.linspace(0.0, np.pi, 200)
        pts = np.column_stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)])
        out = _demo.resample_polyline(pts, 32)
        radius = np.linalg.norm(out[:, :2], axis=1)
        np.testing.assert_allclose(radius, 1.0, atol=2e-3)

    def test_degenerate_zero_length_streamline(self) -> None:
        pts = np.zeros((6, 3))
        out = _demo.resample_polyline(pts, 12)
        assert out.shape == (12, 3)
        assert np.all(np.isfinite(out))
        assert np.all(out == 0.0)

    @pytest.mark.parametrize(
        "pts,n",
        [
            (np.zeros((1, 3)), 8),  # too few input points
            (np.zeros((5, 2)), 8),  # not 3D
            (np.zeros((5, 3)), 1),  # too few output points
        ],
    )
    def test_rejects_bad_input(self, pts: np.ndarray, n: int) -> None:
        with pytest.raises(ValueError):
            _demo.resample_polyline(pts, n)


class TestDirectionColors:
    def test_axis_aligned_paths_get_pure_channels(self) -> None:
        # R = left-right (x), G = anterior-posterior (y), B = inferior-superior (z).
        steps = np.linspace(0.0, 10.0, 12)
        zero = np.zeros_like(steps)
        for axis, channel in ((0, 0), (1, 1), (2, 2)):
            cols = [zero, zero, zero]
            cols[axis] = steps
            path = np.column_stack(cols)[None, ...]
            rgb = _demo.direction_colors(path)
            assert rgb.shape == (1, 12, 3)
            assert np.all(rgb[..., channel] == 255)
            for other in range(3):
                if other != channel:
                    assert np.all(rgb[..., other] == 0)

    def test_is_orientation_only_not_direction(self) -> None:
        # Reversing a streamline must not change its colour — that is the whole
        # point of taking |tangent|.
        path = np.cumsum(np.random.default_rng(0).normal(size=(1, 30, 3)), axis=1)
        forward = _demo.direction_colors(path)
        backward = _demo.direction_colors(path[:, ::-1])
        np.testing.assert_array_equal(forward, backward[:, ::-1])

    def test_diagonal_splits_channels(self) -> None:
        steps = np.linspace(0.0, 10.0, 9)
        path = np.column_stack([steps, steps, np.zeros_like(steps)])[None, ...]
        rgb = _demo.direction_colors(path)
        expected = round(255.0 / np.sqrt(2.0))
        assert np.all(np.abs(rgb[..., 0].astype(int) - expected) <= 1)
        assert np.all(np.abs(rgb[..., 1].astype(int) - expected) <= 1)
        assert np.all(rgb[..., 2] == 0)

    def test_coincident_vertices_do_not_produce_nan(self) -> None:
        path = np.zeros((1, 10, 3))
        rgb = _demo.direction_colors(path)
        assert rgb.dtype == np.uint8
        assert np.all(rgb == 0)

    @pytest.mark.parametrize(
        "bad", [np.zeros((4, 3)), np.zeros((2, 1, 3)), np.zeros((2, 5, 2))]
    )
    def test_rejects_bad_input(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            _demo.direction_colors(bad)


class TestRasToScene:
    def test_superior_maps_to_plus_y(self) -> None:
        # RAS +z (superior) must become scene +y so the brain is upright.
        out = _demo.ras_to_scene(np.array([[0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(out, [[0.0, 1.0, 0.0]], atol=1e-6)

    def test_right_stays_plus_x_and_anterior_goes_minus_z(self) -> None:
        out = _demo.ras_to_scene(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        np.testing.assert_allclose(out[0], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(out[1], [0.0, 0.0, -1.0], atol=1e-6)

    def test_is_a_rigid_rotation(self) -> None:
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(50, 3))
        out = _demo.ras_to_scene(pts)
        np.testing.assert_allclose(
            np.linalg.norm(out, axis=1), np.linalg.norm(pts, axis=1), rtol=1e-6
        )

    def test_preserves_leading_shape(self) -> None:
        assert _demo.ras_to_scene(np.zeros((4, 7, 3))).shape == (4, 7, 3)

    def test_rejects_non_3_last_axis(self) -> None:
        with pytest.raises(ValueError):
            _demo.ras_to_scene(np.zeros((5, 2)))


class TestPolylineSegmentIndices:
    def test_count_and_dtype(self) -> None:
        idx = _demo.polyline_segment_indices(7, 28)
        assert idx.dtype == np.uint32
        assert len(idx) == 2 * 7 * 27

    def test_no_edge_crosses_a_streamline_boundary(self) -> None:
        n_paths, n_vertices = 11, 9
        edges = _demo.polyline_segment_indices(n_paths, n_vertices).reshape(-1, 2)
        path_of_vertex = np.repeat(np.arange(n_paths), n_vertices)
        assert np.all(path_of_vertex[edges[:, 0]] == path_of_vertex[edges[:, 1]]), (
            "an edge joined two different streamlines"
        )

    def test_every_vertex_referenced_and_joints_shared(self) -> None:
        n_paths, n_vertices = 6, 12
        edges = _demo.polyline_segment_indices(n_paths, n_vertices).reshape(-1, 2)
        degree = np.bincount(edges.reshape(-1), minlength=n_paths * n_vertices)
        assert np.all(degree > 0), "emitted an unreferenced vertex"
        # Interior vertices have degree 2 (shared joint); the two ends have 1.
        assert int((degree == 1).sum()) == 2 * n_paths
        assert int((degree == 2).sum()) == n_paths * (n_vertices - 2)

    def test_indices_are_in_range(self) -> None:
        n_paths, n_vertices = 5, 20
        idx = _demo.polyline_segment_indices(n_paths, n_vertices)
        assert int(idx.min()) >= 0
        assert int(idx.max()) < n_paths * n_vertices

    def test_rejects_single_vertex_paths(self) -> None:
        with pytest.raises(ValueError):
            _demo.polyline_segment_indices(4, 1)


class TestSubsampleIndices:
    def test_returns_all_when_under_the_cap(self) -> None:
        out = _demo.subsample_indices(500, 6000, seed=0)
        np.testing.assert_array_equal(out, np.arange(500))

    def test_caps_and_stays_sorted_and_unique(self) -> None:
        out = _demo.subsample_indices(50_000, 6_000, seed=0)
        assert len(out) == 6_000
        assert np.all(np.diff(out) > 0)
        assert int(out.max()) < 50_000

    def test_is_deterministic_for_a_given_seed(self) -> None:
        a = _demo.subsample_indices(10_000, 100, seed=7)
        b = _demo.subsample_indices(10_000, 100, seed=7)
        c = _demo.subsample_indices(10_000, 100, seed=8)
        np.testing.assert_array_equal(a, b)
        assert not np.array_equal(a, c)


class TestCheckSizing:
    def test_defaults_pass_without_a_warning(self) -> None:
        assert (
            _demo.check_sizing(_demo.DEFAULT_POINTS, _demo.DEFAULT_PER_BUNDLE) is None
        )

    @pytest.mark.parametrize("points,per_bundle", [(1, 6000), (0, 6000), (-3, 6000)])
    def test_rejects_unusable_point_counts(self, points: int, per_bundle: int) -> None:
        with pytest.raises(ValueError, match="--points"):
            _demo.check_sizing(points, per_bundle)

    @pytest.mark.parametrize("per_bundle", [0, -1])
    def test_rejects_empty_bundles(self, per_bundle: int) -> None:
        # Without this the empty selection only blows up much later, in the
        # centroid of a zero-row array — after the 588 MB download.
        with pytest.raises(ValueError, match="--per-bundle"):
            _demo.check_sizing(_demo.DEFAULT_POINTS, per_bundle)

    def test_warns_past_the_un_laddered_leaf_gate(self) -> None:
        warning = _demo.check_sizing(28, 50_000)
        assert warning is not None
        assert "1,400,000" in warning


class TestSceneMarker:
    @staticmethod
    def _write(marker: Path, payload: str) -> Path:
        marker.write_text(payload)
        return marker

    def _current(self, tmp_path: Path) -> Path:
        return self._write(
            tmp_path / "scene_build.json",
            json.dumps(
                {
                    "version": _demo.SCENE_SCHEMA_VERSION,
                    "points": 28,
                    "per_bundle": 6000,
                }
            ),
        )

    def test_missing_marker_forces_a_rebuild(self, tmp_path: Path) -> None:
        assert not _demo.scene_marker_matches(
            tmp_path / "absent.json", points=28, per_bundle=6000
        )

    def test_matching_marker_allows_reuse(self, tmp_path: Path) -> None:
        marker = self._current(tmp_path)
        assert _demo.scene_marker_matches(marker, points=28, per_bundle=6000)

    @pytest.mark.parametrize("points,per_bundle", [(20, 6000), (28, 3000), (20, 3000)])
    def test_different_sizing_forces_a_rebuild(
        self,
        tmp_path: Path,
        points: int,
        per_bundle: int,
    ) -> None:
        marker = self._current(tmp_path)
        assert not _demo.scene_marker_matches(
            marker, points=points, per_bundle=per_bundle
        )

    def test_pre_label_marker_forces_a_rebuild(self, tmp_path: Path) -> None:
        # A scene built before hover labels existed: right sizing, no version
        # key. Reusing it would serve a label-less scene while the docs (and
        # the auto-injected hover overlay) promise tooltips. Anyone with a warm
        # datasets/demos/ or a gallery box is in exactly this state.
        marker = self._write(
            tmp_path / "scene_build.json", '{"points": 28, "per_bundle": 6000}'
        )
        assert not _demo.scene_marker_matches(marker, points=28, per_bundle=6000)

    def test_older_schema_version_forces_a_rebuild(self, tmp_path: Path) -> None:
        marker = self._write(
            tmp_path / "scene_build.json",
            json.dumps(
                {
                    "version": _demo.SCENE_SCHEMA_VERSION - 1,
                    "points": 28,
                    "per_bundle": 6000,
                }
            ),
        )
        assert not _demo.scene_marker_matches(marker, points=28, per_bundle=6000)

    def test_corrupt_marker_forces_a_rebuild(self, tmp_path: Path) -> None:
        marker = self._write(tmp_path / "scene_build.json", "{not json")
        assert not _demo.scene_marker_matches(marker, points=28, per_bundle=6000)


class TestNodeBudget:
    """The two viewer limits the sizing constants exist to respect."""

    #: scripts/check_demo_ladders.py fails an un-laddered lines leaf above this.
    LADDER_GATE_VERTICES = 200_000
    #: 682 * maxTextureSize segments per node, on a 4096-class GPU.
    TEXTURE_CAP_SEGMENTS = 682 * 4096

    def test_defaults_keep_a_node_under_both_limits(self) -> None:
        assert _demo.LADDER_GATE_VERTICES == self.LADDER_GATE_VERTICES
        vertices = _demo.DEFAULT_PER_BUNDLE * _demo.DEFAULT_POINTS
        segments = _demo.DEFAULT_PER_BUNDLE * (_demo.DEFAULT_POINTS - 1)
        assert vertices < self.LADDER_GATE_VERTICES
        assert segments < self.TEXTURE_CAP_SEGMENTS

    def test_divisions_cover_the_atlas_layout(self) -> None:
        assert set(_demo.DIVISIONS) == {
            "association",
            "projection",
            "commissural",
            "cerebellum",
            "cranial nerve",
        }


#: The HCP-1065 archive's member list, by division — enumerated once from the
#: release zip's central directory rather than inferred. Paired codes are listed
#: by their base and expanded to ``_L`` / ``_R`` below; the unpaired (midline)
#: ones are listed as-is. Note ``SCP``: ``CB`` and ``ICP`` pair, the superior
#: cerebellar peduncle does not.
#:
#: Being a literal, this is a *transcription* of the archive, and the tests
#: below compare it against ``BUNDLE_INFO`` — two in-repo tables, not the
#: archive itself. Nothing here can catch a wrong transcription; the runtime
#: signal for that is ``build_scene``'s "missing from BUNDLE_INFO" WARNING,
#: which fires on the real member names during a real build.
_PAIRED_CODES = {
    "association": [
        "AF",
        "C_FP",
        "C_FPH",
        "C_PH",
        "C_PHP",
        "C_PO",
        "EMC",
        "FAT",
        "IFOF",
        "ILF",
        "MdLF",
        "PAT",
        "SLF1",
        "SLF2",
        "SLF3",
        "UF",
        "VOF",
    ],
    "cerebellum": ["CB", "ICP"],
    "cranial nerve": ["CNII", "CNIII", "CNV", "CNVII", "CNVIII"],
    "projection": [
        "AR",
        "CBT",
        "CPT_F",
        "CPT_O",
        "CPT_P",
        "CST",
        "CS_A",
        "CS_P",
        "CS_S",
        "DRTT",
        "F",
        "ML",
        "OR",
        "RST",
        "TR_A",
        "TR_P",
        "TR_S",
    ],
}
_UNPAIRED_CODES = {
    "cerebellum": ["MCP", "SCP", "V"],
    "commissural": ["AC", "CC"],
}

_ALL_BUNDLES: list[tuple[str, str]] = [
    *(
        (f"{code}_{side}", division)
        for division, codes in _PAIRED_CODES.items()
        for code in codes
        for side in ("L", "R")
    ),
    *(
        (code, division)
        for division, codes in _UNPAIRED_CODES.items()
        for code in codes
    ),
]


class TestTractLabel:
    """The hover-label expansion: every atlas code must resolve."""

    def test_the_code_list_fixture_still_has_87_entries(self) -> None:
        # A fixture-integrity guard, NOT a check on the atlas: _ALL_BUNDLES is a
        # literal 30 lines up. It catches an accidental edit to the paired /
        # unpaired split — moving SCP back into _PAIRED_CODES, say — which would
        # otherwise quietly weaken every test below.
        assert len(_ALL_BUNDLES) == 87

    def test_the_table_and_the_code_list_agree(self) -> None:
        # Both directions: no atlas code without a table entry (a fallback
        # label in the viewer) and no table entry the atlas never ships (dead
        # data). The 87 members reduce to 46 base codes.
        assert {_demo.split_hemisphere(b)[0] for b, _ in _ALL_BUNDLES} == set(
            _demo.BUNDLE_INFO
        )
        assert len(_demo.BUNDLE_INFO) == 46

    @pytest.mark.parametrize("bundle,division", _ALL_BUNDLES)
    def test_every_atlas_code_resolves(self, bundle: str, division: str) -> None:
        label = _demo.tract_label(bundle, division)
        # Never the bare fallback.
        assert label != f"{bundle} — {division}"
        base, _ = _demo.split_hemisphere(bundle)
        name, gloss = _demo.BUNDLE_INFO[base]
        # Leads with the raw code, so the Layers panel row and the tooltip agree.
        assert label.startswith(f"{bundle} — ")
        assert name in label
        assert division in label
        assert gloss in label

    def test_underscore_base_codes_are_not_mis_split(self) -> None:
        # THE regression: several base codes themselves end in _F/_O/_P/_A/_S,
        # so only a BUNDLE_INFO-guarded _L/_R strip gets these right.
        assert _demo.split_hemisphere("CPT_F_L") == ("CPT_F", "left")
        assert "Frontal corticopontine tract (left)" in _demo.tract_label(
            "CPT_F_L", "projection"
        )
        assert "Superior corticostriatal tract (right)" in _demo.tract_label(
            "CS_S_R", "projection"
        )
        assert "Cingulum, frontal-parietal segment (left)" in _demo.tract_label(
            "C_FP_L", "association"
        )
        assert "Cingulum, parolfactory segment (right)" in _demo.tract_label(
            "C_PO_R", "association"
        )
        assert "Superior longitudinal fasciculus III (left)" in _demo.tract_label(
            "SLF3_L", "association"
        )

    @pytest.mark.parametrize(
        "bundle,division",
        [
            ("MCP", "cerebellum"),
            # SCP is the non-obvious one: CB and ICP pair, this does not.
            ("SCP", "cerebellum"),
            ("V", "cerebellum"),
            ("CC", "commissural"),
        ],
    )
    def test_unpaired_codes_get_no_hemisphere(self, bundle: str, division: str) -> None:
        assert _demo.split_hemisphere(bundle) == (bundle, "")
        label = _demo.tract_label(bundle, division)
        assert "(left)" not in label
        assert "(right)" not in label

    def test_matches_the_documented_shape(self) -> None:
        assert _demo.tract_label("AF_L", "association") == (
            "AF_L — Arcuate fasciculus (left) · association · "
            "frontal (Broca) and temporal (Wernicke) language areas"
        )

    def test_unknown_code_falls_back_without_raising(self) -> None:
        assert _demo.tract_label("ZZZ_L", "projection") == "ZZZ_L — projection"
        assert _demo.split_hemisphere("ZZZ_L") == ("ZZZ_L", "")


class TestBuildScene:
    """The authoring path, on synthetic bundles — no atlas download needed.

    Everything above tests the helpers in isolation; this is the only check
    that what they feed ``add_lines`` actually survives the compiler: a
    ``kind=lod`` group per tract whose finest child is the indexed Lines node
    (edges in range, one colour per vertex, no orphans), with the tuned
    compositing on the group where the Layers panel and attribute composition
    expect it.
    """

    POINTS = 6
    PER_BUNDLE = 4

    #: One REAL atlas code per division, in DIVISIONS order — synthetic names
    #: would silently take ``tract_label``'s unknown-code fallback.
    NAMES = ("AF_L", "CST_R", "CC", "MCP", "CNII_L")

    def _bundles(self) -> dict:
        rng = np.random.default_rng(0)
        bundles: dict = {"names": [], "divisions": [], "positions": [], "colors": []}
        for name, division in zip(self.NAMES, _demo.DIVISIONS, strict=True):
            paths = np.cumsum(
                rng.normal(size=(self.PER_BUNDLE, self.POINTS, 3)), axis=1
            )
            bundles["names"].append(name)
            bundles["divisions"].append(division)
            bundles["positions"].append(_demo.ras_to_scene(paths).reshape(-1, 3))
            bundles["colors"].append(_demo.direction_colors(paths).reshape(-1, 3))
        return bundles

    def test_every_bundle_becomes_an_indexed_node(self, tmp_path: Path) -> None:
        output = tmp_path / "tractography.luxar.zarr"
        _demo.build_scene(self._bundles(), output, points=self.POINTS)

        root = zarr.open_group(output, mode="r")
        for name, division in zip(self.NAMES, _demo.DIVISIONS, strict=True):
            # The group name is the division with its space replaced.
            tract = root[f"{division.replace(' ', '_')}/{name}"]

            # Compositing rides on the lod wrapper, not the levels: opacity and
            # intensity multiply root-to-leaf and blending takes the nearest
            # ancestor, so the authored look reaches every level from here — and
            # `layer` here is what gives the Layers panel one row per tract.
            assert tract.attrs["kind"] == "lod"
            assert tract.attrs["layer"] is True
            assert tract.attrs["blending_mode"] == "additive"
            assert tract.attrs["opacity"] == _demo.LINE_OPACITY
            assert tract.attrs["intensity"] == _demo.LINE_INTENSITY

            leaves = [
                child
                for _, child in tract.groups()
                if "original_line_type" in child.attrs
            ]
            assert len(leaves) == 1, "expected exactly one Lines level"
            node = leaves[0]
            n_vertices = int(node.attrs["n_vertices"])
            assert node.attrs["original_line_type"] == "indexed"
            assert n_vertices == self.PER_BUNDLE * self.POINTS
            assert node["vertices"].shape[0] == n_vertices
            assert node["colors"].shape[0] == n_vertices

            edges = np.asarray(node["segments"][:])
            assert edges.ndim == 2 and edges.shape[1] == 2
            assert len(edges) == self.PER_BUNDLE * (self.POINTS - 1)
            assert int(edges.min()) >= 0
            assert int(edges.max()) < n_vertices
            degree = np.bincount(edges.reshape(-1), minlength=n_vertices)
            assert np.all(degree > 0), "orphaned vertex"
            # Interior joints share an index; only the two ends have degree 1.
            assert int((degree == 1).sum()) == 2 * self.PER_BUNDLE

    def test_the_labelled_level_is_gated_at_full_viewport_coverage(
        self, tmp_path: Path
    ) -> None:
        # The premise the whole "get close before hover says anything" note
        # rests on. If a future default made the Lines level selectable below
        # full coverage, hover would start working at the opening pose and the
        # docstring's headline caveat would silently become wrong.
        output = tmp_path / "tractography.luxar.zarr"
        _demo.build_scene(self._bundles(), output, points=self.POINTS)

        root = zarr.open_group(output, mode="r")
        for name, division in zip(self.NAMES, _demo.DIVISIONS, strict=True):
            tract = root[f"{division.replace(' ', '_')}/{name}"]
            assert tract.attrs["selector"] == "coverage"

            children = sorted(
                (child for _, child in tract.groups()),
                key=lambda c: int(c.attrs["child_index"]),
            )
            fractions = [float(c.attrs["coverage_fraction"]) for c in children]
            # Coarsest to finest, ascending, with the finest anchored at 1.0 —
            # i.e. only a node filling the viewport selects it.
            assert fractions == sorted(fractions)
            assert fractions[-1] == 1.0
            assert "original_line_type" in children[-1].attrs, (
                "the level gated at 1.0 must be the labelled Lines level"
            )

    def test_hover_labels_reach_the_finest_lines_level(self, tmp_path: Path) -> None:
        output = tmp_path / "tractography.luxar.zarr"
        _demo.build_scene(self._bundles(), output, points=self.POINTS)

        root = zarr.open_group(output, mode="r")
        for name, division in zip(self.NAMES, _demo.DIVISIONS, strict=True):
            tract = root[f"{division.replace(' ', '_')}/{name}"]
            leaves = [
                child
                for _, child in tract.groups()
                if "original_line_type" in child.attrs
            ]
            # The ladder shape itself: two synthesised coarse gsplat levels
            # plus the one real Lines level.
            children = [child for _, child in tract.groups()]
            assert len(children) == 3
            assert len(leaves) == 1

            # THE headline caveat, pinned: the coarse levels carry NO labels.
            # If a core change ever propagated labels down the gsplat lift, the
            # docstring's "zoom in before hover says anything" would silently
            # become false with every other assertion still green.
            for child in children:
                if "original_line_type" not in child.attrs:
                    assert child.attrs.get("has_labels") is not True

            node = leaves[0]
            n_vertices = int(node.attrs["n_vertices"])

            assert node.attrs["has_labels"] is True
            offsets = np.asarray(node["label_offsets"][:])
            assert len(offsets) == n_vertices + 1

            # Labels are per-vertex CSR: label i is the byte slice
            # [offsets[i]:offsets[i + 1]] of label_bytes, UTF-8 decoded.
            raw = np.asarray(node["label_bytes"][:]).tobytes()
            expected = _demo.tract_label(name, division)
            decoded = {
                raw[int(offsets[i]) : int(offsets[i + 1])].decode("utf-8")
                for i in range(n_vertices)
            }
            assert decoded == {expected}, (
                "every vertex of a bundle must carry the same tract label"
            )

    def test_labels_trigger_the_auto_injected_hover_overlay(
        self, tmp_path: Path
    ) -> None:
        # The demo authors no hover overlay of its own: the compiler injects one
        # only because a node carried labels, so its presence is the end-to-end
        # signal that the labels reached the scene. Placement is hover_inject's
        # default, not the demo's choice, so it is not asserted here.
        output = tmp_path / "tractography.luxar.zarr"
        _demo.build_scene(self._bundles(), output, points=self.POINTS)

        root = zarr.open_group(output, mode="r")
        overlays = [child for _, child in root["overlays"].groups()]
        hover = [o for o in overlays if o.attrs.get("hover")]
        assert len(hover) == 1, "expected exactly one hover overlay"
        assert hover[0].attrs["type"] == "overlay_text"
        assert "{hover_label}" in hover[0].attrs["text"]

    def test_reports_bundles_missing_from_the_table(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The fallback path has to be visible: a future atlas revision adding a
        # code must not silently ship code-only tooltips.
        bundles = self._bundles()
        bundles["names"][0] = "ZZZ_L"

        _demo.build_scene(bundles, tmp_path / "t.luxar.zarr", points=self.POINTS)

        out = capsys.readouterr().out
        assert "Hover labels: 4/5 tracts named" in out
        assert "1 bundle(s) missing from BUNDLE_INFO" in out
        assert "ZZZ_L" in out

    def test_reports_full_coverage_when_every_code_is_known(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _demo.build_scene(
            self._bundles(), tmp_path / "t.luxar.zarr", points=self.POINTS
        )

        out = capsys.readouterr().out
        assert "Hover labels: 5/5 tracts named" in out
        assert "missing from BUNDLE_INFO" not in out
