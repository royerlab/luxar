"""Node subpackage: the scene-graph base class.

`Node` is re-exported here so `from luxar import Node` and
`from luxar.core.node import Node` continue to resolve through the same
path as before. Helpers (rendering_attrs, specialized_groups) live in
sibling submodules.
"""

from .node import Node

__all__ = ["Node"]
