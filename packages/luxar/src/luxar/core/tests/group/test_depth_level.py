"""Authoring tests for ``depth_level`` — the cross-layer draw order.

``depth_level`` states where a layer draws relative to the layers it overlaps
(higher = nearer the camera = drawn later, the CSS ``z-index`` convention). It
is a compositing attr composed nearest-setter-wins by the viewer, and it may be
authored ONLY on a node that is a layer — the scene root, a plain group, or a
top-level leaf.

The refusals below are the load-bearing half. A level authored strictly inside a
``kind=partition`` wrapper would split that wrapper across draw-order bands and
destroy the exact Fuchs-Kedem-Naylor part order its stored BSP planes guarantee;
one inside a ``kind=lod`` group would simply be inert, since only one level
renders. Both are refused rather than ignored, through BOTH doors — the adder
kwarg and the post-hoc ``node.attrs[...] = ...`` write-through — because an attr
that writes cleanly and silently does nothing is the most expensive kind of bug
this codebase produces.

Spec: ``docs/guides/specs/LAYER_DEPTH_LEVEL_SPEC.md``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from luxar.core.group.compositing import (
    AUTHORED_APPEARANCE_ATTRS,
    COMPOSITING_ATTRS,
    IDENTITY_COMPOSITING_ATTRS,
    WRITER_STAMPED_APPEARANCE_DEFAULTS,
)
from luxar.io._compiler.node_common import KNOWN_RENDER_ATTRS
from luxar.validation.types import validate_depth_level

from .conftest import cholesky_rows, open_scene, random_positions


class TestValidateDepthLevel:
    @pytest.mark.parametrize("value", [0, 1, 10, -5, 999999, np.int32(7), np.int64(-2)])
    def test_accepts_any_integer(self, value: Any) -> None:
        assert validate_depth_level(value) == int(value)
        assert isinstance(validate_depth_level(value), int)

    # `isinstance(True, int)` is true in Python, so a bool would sail through a
    # naive check and silently band the layer at level 1. Almost certainly the
    # author confused this with a flag.
    @pytest.mark.parametrize("value", [True, False, np.bool_(True)])
    def test_refuses_a_bool(self, value: Any) -> None:
        with pytest.raises(TypeError, match="depth_level must be an integer"):
            validate_depth_level(value)

    @pytest.mark.parametrize("value", [1.5, 2.0, "10", None, [1], {"a": 1}])
    def test_refuses_a_non_integer(self, value: Any) -> None:
        with pytest.raises(TypeError, match="depth_level must be an integer"):
            validate_depth_level(value)


class TestRegistryMembership:
    """The four set memberships that define the attr's behaviour."""

    def test_is_a_compositing_attr(self) -> None:
        # The wrapper IS the layer, so a partitioned node carries ONE level for
        # the whole block rather than copying it onto every internal child.
        assert "depth_level" in COMPOSITING_ATTRS

    def test_is_carried_through_structure_only_rebuilds(self) -> None:
        # Otherwise `gsplat lod` and friends would silently drop it (#1600).
        assert "depth_level" in AUTHORED_APPEARANCE_ATTRS

    def test_is_advertised_as_a_known_render_attr(self) -> None:
        # Without this the writer's unknown-key gate rejects the attr outright
        # before any validator runs.
        assert "depth_level" in KNOWN_RENDER_ATTRS

    # THE safety property (spec D2). An explicit level suppresses the
    # containment rule while an unset one must not, so the viewer has to be able
    # to tell them apart. A stamped default would put every store ever written
    # afterwards on band 0 EXPLICITLY, silently disabling containment for all of
    # them. Asserting ABSENCE is the only way to catch that.
    def test_has_no_writer_stamped_default(self) -> None:
        assert "depth_level" not in WRITER_STAMPED_APPEARANCE_DEFAULTS
        assert "depth_level" not in IDENTITY_COMPOSITING_ATTRS


class TestAuthoringOnALayer:
    def test_round_trips_on_a_leaf(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "leaf.luxar.zarr")
        with compiler:
            node = scene.add_points(
                "pts", random_positions(16, seed=0), depth_level=30, layer=True
            )
            assert node.attrs["depth_level"] == 30

    def test_round_trips_on_a_group(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "grp.luxar.zarr")
        with compiler:
            group = scene.add_group("layer", depth_level=10, layer=True)
            scene.add_points("pts", random_positions(16, seed=1), parent=group)
            assert group.attrs["depth_level"] == 10

    def test_accepts_a_negative_level(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "neg.luxar.zarr")
        with compiler:
            node = scene.add_points(
                "back", random_positions(8, seed=2), depth_level=-10
            )
            assert node.attrs["depth_level"] == -10

    def test_refuses_a_bad_value_before_writing(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "bad.luxar.zarr")
        with compiler:
            with pytest.raises((TypeError, ValueError)):
                scene.add_points(
                    "pts", random_positions(8, seed=3), depth_level="front"
                )

    # A `None` refuses loudly here rather than being read as "absent", which is
    # why `depth_level` is NOT in ABSENT_WHEN_NONE_RENDER_ATTRS: that set is for
    # attrs whose None would otherwise reach disk unchecked.
    def test_refuses_none_rather_than_treating_it_as_absent(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, "none.luxar.zarr")
        with compiler:
            with pytest.raises((TypeError, ValueError)):
                scene.add_points("pts", random_positions(8, seed=4), depth_level=None)


class TestRefusedInsideASpecializedGroup:
    """Both doors, both kinds — the rule the exact BSP part order depends on."""

    @staticmethod
    def _partition(scene: Any, seed: int) -> Any:
        return scene.add_gsplats(
            "tiles",
            random_positions(64, seed=seed),
            amplitudes=np.ones(64, dtype=np.float32),
            cholesky_factors=cholesky_rows(64),
            partition={"max_elements": 16},
        )

    def test_adder_refuses_inside_a_partition(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "part.luxar.zarr")
        with compiler:
            wrapper = self._partition(scene, seed=5)
            # A GSPLATS child, so the partition-homogeneity guard
            # (`reject_mismatched_partition_parent`, which refuses a foreign
            # display_type) passes and the depth_level refusal is the one that
            # fires. A points child here is rejected earlier for a different
            # reason and would make this test prove nothing.
            with pytest.raises(ValueError, match="depth_level"):
                scene.add_gsplats(
                    "sneak",
                    random_positions(8, seed=6),
                    amplitudes=np.ones(8, dtype=np.float32),
                    cholesky_factors=cholesky_rows(8),
                    parent=wrapper,
                    depth_level=5,
                )

    def test_write_through_door_refuses_on_a_part(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "part2.luxar.zarr")
        with compiler:
            wrapper = scene.add_gsplats(
                "tiles",
                random_positions(64, seed=7),
                amplitudes=np.ones(64, dtype=np.float32),
                cholesky_factors=cholesky_rows(64),
                partition={"max_elements": 16},
            )
            parts = [c for c in wrapper.children if c.name.startswith("part_")]
            assert parts, "expected the partition to have produced parts"
            with pytest.raises(ValueError, match="depth_level"):
                parts[0].attrs["depth_level"] = 5

    def test_the_wrapper_itself_is_allowed(self, tmp_path: Any) -> None:
        """The wrapper IS the layer — a level there moves the whole block."""
        compiler, scene, path = open_scene(tmp_path, "part3.luxar.zarr")
        with compiler:
            wrapper = scene.add_gsplats(
                "tiles",
                random_positions(64, seed=8),
                amplitudes=np.ones(64, dtype=np.float32),
                cholesky_factors=cholesky_rows(64),
                partition={"max_elements": 16},
                depth_level=40,
            )
            assert wrapper.attrs["depth_level"] == 40
            # And it must NOT have been copied onto the parts, or they would
            # each carry their own band.
            for child in wrapper.children:
                assert "depth_level" not in child.attrs

    def test_message_names_the_wrapper_and_the_fix(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "part4.luxar.zarr")
        with compiler:
            wrapper = self._partition(scene, seed=9)
            with pytest.raises(ValueError) as excinfo:
                scene.add_gsplats(
                    "sneak",
                    random_positions(8, seed=10),
                    amplitudes=np.ones(8, dtype=np.float32),
                    cholesky_factors=cholesky_rows(8),
                    parent=wrapper,
                    depth_level=1,
                )
        message = str(excinfo.value)
        assert "kind=partition" in message
        # A refusal has to say what to do instead, or it is just an obstacle.
        assert "WRAPPER" in message
        assert "BSP" in message

    def test_refuses_inside_a_lod_group_too(self, tmp_path: Any) -> None:
        """D6 covers `kind=lod`, where a level would simply be INERT.

        Only one level of a LOD group renders at a time, so banding one of them
        separately means nothing. Refused for the same reason the codebase
        refuses every other write-cleanly-do-nothing attr.
        """
        compiler, scene, path = open_scene(tmp_path, "lod.luxar.zarr")
        with compiler:
            wrapper = scene.add_points(
                "ladder",
                random_positions(64, seed=11),
                substitutive_lod={"levels": 2},
            )
            assert wrapper.attrs.get("kind") == "lod"
            levels = list(wrapper.children)
            assert levels, "expected the ladder to have produced levels"
            with pytest.raises(ValueError, match="depth_level") as excinfo:
                levels[0].attrs["depth_level"] = 3
        assert "kind=lod" in str(excinfo.value)

    def test_a_level_on_a_lod_WRAPPER_is_allowed(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lod2.luxar.zarr")
        with compiler:
            wrapper = scene.add_points(
                "ladder",
                random_positions(64, seed=12),
                substitutive_lod={"levels": 2},
                depth_level=25,
            )
            assert wrapper.attrs["depth_level"] == 25
