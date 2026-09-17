"""``luxar.mesh`` must expose the decimation API the docs already promise.

``core/README.md`` and ``MESH_NODE_SPEC.md`` name ``luxar.mesh.decimate`` as the
producer of substitutive mesh levels, and ``luxar.mesh`` mirrors ``luxar.gsplats``
as the package's public toolbox — so the names must be importable from the
package, not only from the submodule.
"""

from __future__ import annotations

import importlib

import pytest

import luxar.mesh
from luxar.mesh import decimate as decimate_module_or_function

DECIMATION_API = (
    "decimate",
    "decimate_ladder",
    "decimate_cluster",
    "DecimatedMesh",
    "DECIMATION_METHODS",
    "resolve_decimation_method",
)


@pytest.mark.parametrize("name", DECIMATION_API)
def test_decimation_api_is_re_exported_from_the_package(name: str) -> None:
    submodule = importlib.import_module("luxar.mesh.decimate")
    assert name in luxar.mesh.__all__
    assert getattr(luxar.mesh, name) is getattr(submodule, name)


def test_from_luxar_mesh_import_decimate_is_the_function() -> None:
    """``from luxar.mesh import decimate`` binds the FUNCTION.

    The submodule shares the name. Python resolves the ``from … import`` form
    against the package's attributes first, so once ``__init__`` re-exports the
    function, that is what a caller gets — exactly what the docs' prose
    ``luxar.mesh.decimate`` reads as. ``import luxar.mesh.decimate as m`` still
    reaches the module through ``sys.modules``.
    """
    assert callable(decimate_module_or_function)
    assert decimate_module_or_function is luxar.mesh.decimate


def test_mesh_still_does_not_re_export_the_node_class() -> None:
    """The package docstring's promise: no ``luxar.core`` edge via ``Mesh``."""
    assert "Mesh" not in luxar.mesh.__all__
    assert not hasattr(luxar.mesh, "Mesh")
