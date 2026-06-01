"""Group subpackage: the scene-graph container with `add_*` factories.

`Group` is re-exported here so `from luxar.core.group import Group` and
`from luxar import Group` continue to resolve through the same path as
before. Helpers (partition, lod, adders, compositing, dim_order) live in
sibling submodules and subpackages.
"""

from .group import Group

__all__ = ["Group"]
