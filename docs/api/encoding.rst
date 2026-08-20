Encoding Package
================

The encoding package provides semantic type-aware array encoding and quantization.

.. automodule:: luxar.encoding
   :no-members:

ArrayEncoder
------------

.. autoclass:: luxar.encoding.ArrayEncoder
   :members:
   :undoc-members:
   :show-inheritance:

Semantic Types
--------------

.. autoclass:: luxar.encoding.SemanticType
   :members:
   :undoc-members:

Encoding Modes
--------------

.. autoclass:: luxar.encoding.EncodingMode
   :members:
   :undoc-members:

Coordinate Grid Snap
--------------------

The shared predicate behind the COORDINATE grid snap, and the level count it is
asked about. Both are public because the gsplat writer's sigma rail must ask the
encoder the same question it asks itself.

.. autofunction:: luxar.encoding.gridded_axis_step

.. autodata:: luxar.encoding.COORDINATE_LEVELS
