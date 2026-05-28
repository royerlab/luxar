"""Scene subpackage: the scene-graph root node.

`Scene` is re-exported here so `from luxar import Scene`,
`from luxar.core import Scene`, and `from luxar.core.scene import Scene`
all resolve through the same path as before. Helpers (validation,
dim_order, overlays) live in sibling submodules.
"""

from .scene import Scene

__all__ = ["Scene"]
