"""An explicit ``None`` render attr means ABSENT at the LEAF adders too (#1574).

#1496 settled the rule one level up, on the three ``**attrs``-forwarding gsplats
doors (``add_gsplats_from_data`` / ``add_gsplats_from_file`` / the graft entry —
see ``lod/test_source_validation.py``'s "#1496" section for the four-route
matrix), and deliberately left the leaf adders alone. Two present-but-``None``
render attrs were therefore still mishandled inside the leaves themselves, on
ALL FOUR geometry types, which is why this suite is parametrized over the adder
table rather than living next to the gsplats-pipeline cases.

Measured on the unfixed leaf adders, uncoloured data, a 3-dimension scene:

  * ``colormap=None`` wrote ``colormap='custom'`` — the silent half.
    ``validate_render_attrs`` guards its colormap check on ``is not None`` so the
    key passes, and ``compositing.sync_custom_colormap_attr`` then rewrites the
    None to ``'custom'`` (a None is not a str in ``BUILTIN_COLORMAP_NAMES``)
    WITHOUT writing any ``colormap_lut``. The viewer's ``build-scene-graph.ts``
    answers a ``'custom'`` with no LUT by warning and falling back to VIRIDIS,
    where omitting the key gives ``gray``. Points/Lines/Mesh all wrote
    ``'custom'`` against an omitted-key control of NO key at all; GSplats wrote
    ``'custom'`` against a control of ``'gray'`` (its writer stamps that default
    only when the key is absent).
  * ``coverage_fraction=None`` persisted a literal ``coverage_fraction: null``
    LOD selector threshold into the node's zarr attrs.

Both assertions are made against what LANDED IN THE STORE, not merely against
the returned node object: the complaint is about the file the viewer reads.

The last two classes are scope controls rather than regression cases. They pass
before and after the fix by construction — their job is to kill an OVER-broad
strip, which would swallow the Nones that are supposed to be loud, and to pin
that the leaf key set is the same one the gsplats pipeline derives its wider set
from rather than a hand-copied second list.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Set, Tuple

import numpy as np
import pytest
import zarr

from luxar.core.group.compositing import ABSENT_WHEN_NONE_RENDER_ATTRS
from luxar.core.group.gsplats_pipeline.from_data import ABSENT_WHEN_NONE_ATTRS

from .conftest import cholesky_rows, grid_mesh, open_scene, random_positions

#: Element count for every leaf built here. Small: none of these cases depends
#: on the data, only on which attr keys survive to disk.
_N = 8

#: Sentinel for "the key is not in the node's attrs at all", so an absent key and
#: a key valued ``None`` can never compare equal (``dict.get`` alone returns None
#: for both, which is precisely the distinction under test).
_ABSENT = "<absent>"


def _add_points(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_points(name, positions=random_positions(_N, seed=1574), **attrs)


def _add_lines(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_lines(
        name,
        vertices=random_positions(_N, seed=1575),
        widths=np.full(_N, 0.5, dtype=np.float32),
        **attrs,
    )


def _add_mesh(scene: Any, name: str, **attrs: Any) -> Any:
    vertices, faces = grid_mesh(4)
    return scene.add_mesh(name, vertices=vertices, faces=faces, **attrs)


def _add_gsplats(scene: Any, name: str, **attrs: Any) -> Any:
    return scene.add_gsplats(
        name,
        centers=random_positions(_N, seed=1576),
        amplitudes=np.ones(_N, dtype=np.float32),
        cholesky_factors=cholesky_rows(_N),
        **attrs,
    )


#: ``(geometry, adder)`` for all four leaf adders. Every case runs against the
#: whole table on purpose: the bug was measured on all four, and each adder owns
#: its own copy of the attrs handling the fix had to reach.
LEAF_ADDERS = [
    ("points", _add_points),
    ("lines", _add_lines),
    ("mesh", _add_mesh),
    ("gsplats", _add_gsplats),
]


def _write_leaf(
    tmp_path: Any, filename: str, adder: Callable[..., Any], **attrs: Any
) -> Tuple[Dict[str, Any], Set[str]]:
    """Write one leaf, finalize, and return its ON-DISK attrs and array names."""
    compiler, scene, path = open_scene(tmp_path, filename)
    adder(scene, "leaf", **attrs)
    compiler.finalize()
    node = zarr.open_group(path, mode="r")["leaf"]
    return dict(node.attrs), set(node.array_keys())


class TestAColormapNoneIsAbsentNotCustom:
    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_it_matches_the_omitted_key_control(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """``colormap=None`` writes whatever omitting the key writes.

        Stated as parity with the control rather than against a literal, because
        the right answer differs by geometry — no key for Points/Lines/Mesh,
        ``'gray'`` for GSplats — and the point of the fix is exactly that the two
        spellings agree, whatever that shared answer is.
        """
        explicit, _ = _write_leaf(
            tmp_path, f"{geometry}_none.luxar.zarr", adder, colormap=None
        )
        omitted, _ = _write_leaf(tmp_path, f"{geometry}_omit.luxar.zarr", adder)

        assert explicit.get("colormap", _ABSENT) == omitted.get("colormap", _ABSENT)

    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_no_lut_less_custom_node_is_shipped(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """The specific state the viewer renders as viridis is never written.

        The parity case above would also be satisfied if BOTH spellings started
        writing a LUT-less ``'custom'``, so the failure mode is pinned directly:
        ``colormap='custom'`` is only meaningful alongside the ``colormap_lut``
        dataset the writer resolves a real custom colormap into.
        """
        attrs, arrays = _write_leaf(
            tmp_path, f"{geometry}_custom.luxar.zarr", adder, colormap=None
        )

        assert attrs.get("colormap") != "custom"
        assert "colormap_lut" not in arrays


class TestACoverageFractionNoneIsNotPersisted:
    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_the_key_is_absent_from_the_written_attrs(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        """No ``coverage_fraction: null`` selector threshold reaches the store.

        ``in`` rather than ``get(...) is None``: a null VALUE is what was written,
        so only key membership tells the two states apart.
        """
        attrs, _ = _write_leaf(
            tmp_path, f"{geometry}_cf.luxar.zarr", adder, coverage_fraction=None
        )

        assert "coverage_fraction" not in attrs


class TestTheDataDoorAndTheLeafDoorAgree:
    def test_colormap_none_reads_the_same_through_both(self, tmp_path: Any) -> None:
        """``add_gsplats_from_data`` and the ``add_gsplats`` it delegates to.

        The asymmetry #1574 names: the outer door stripped the None (#1496) and
        wrote ``'gray'``, the leaf it delegates to rewrote it to ``'custom'``, so
        the same argument meant two different things one call apart.
        """
        from luxar.gsplats.gsplat_data import GSplatData

        data = GSplatData(
            centers=random_positions(_N, seed=1577),
            amplitudes=np.ones(_N, dtype=np.float32),
            cholesky_factors=cholesky_rows(_N),
        )

        compiler, scene, path = open_scene(tmp_path, "both_doors.luxar.zarr")
        scene.add_gsplats_from_data("from_data", data, colormap=None)
        _add_gsplats(scene, "leaf", colormap=None)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["leaf"].attrs.get("colormap", _ABSENT) == store[
            "from_data"
        ].attrs.get("colormap", _ABSENT)


class TestARenderAttrOutsideTheSetStillRefusesItsNone:
    """Scope control: the strip must not start swallowing the LOUD Nones.

    Every render attr other than the two in the set runs its value validator
    unconditionally, so a None there is already reported rather than persisted —
    and reading it as "absent" would only mask a typo. These cases pass before
    and after the fix; they fail against a strip widened past its set.
    """

    @pytest.mark.parametrize("geometry,adder", LEAF_ADDERS)
    def test_opacity_none_is_reported(
        self, tmp_path: Any, geometry: str, adder: Callable[..., Any]
    ) -> None:
        with pytest.raises(ValueError, match="convertible to float, got NoneType"):
            _write_leaf(tmp_path, f"{geometry}_opacity.luxar.zarr", adder, opacity=None)

    def test_truncation_radius_none_is_reported(self, tmp_path: Any) -> None:
        """The gsplat-only member of the same family, and the near miss.

        It IS in the pipeline's wider set (there an explicit None must decline to
        override the ``GSplatData``'s own radius), so a leaf strip reusing that
        set unchanged would have silently dropped a caller's None here instead of
        answering it.
        """
        with pytest.raises(ValueError, match="convertible to float, got NoneType"):
            _write_leaf(
                tmp_path, "trunc.luxar.zarr", _add_gsplats, truncation_radius=None
            )


class TestTheLeafSetIsTheOnePipelineDerivesFrom:
    """Scope control: one definition of the rule, not two lists free to drift."""

    def test_the_leaf_set_is_the_two_silently_wrong_render_attrs(self) -> None:
        assert set(ABSENT_WHEN_NONE_RENDER_ATTRS) == {"colormap", "coverage_fraction"}

    def test_the_pipeline_set_is_a_superset_of_it(self) -> None:
        """#1496's wider set ADDS to this one; it does not restate it."""
        assert set(ABSENT_WHEN_NONE_RENDER_ATTRS) <= set(ABSENT_WHEN_NONE_ATTRS)
