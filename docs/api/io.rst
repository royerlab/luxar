I/O Package
===========

The io package handles reading and writing Luxar scenes to Zarr format with spatial indexing.

.. automodule:: luxar.io
   :no-members:

Overview
--------

The I/O package provides two main classes:

* **LuxarZarrCompiler**: Progressive writer for creating Zarr datasets
* **LuxarScene**: Reader for loading and querying Zarr datasets

Both classes work together to enable efficient streaming of massive n-dimensional scenes with points, lines, Gaussian splats, and other primitives.

LuxarZarrCompiler
-----------------

The compiler writes data progressively to Zarr format without caching in memory,
enabling creation of TB-scale datasets on GB-scale machines.

.. autoclass:: luxar.io.LuxarZarrCompiler
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
~~~~~~~~~~~~

* **Progressive Writing**: No memory caching - stream directly to disk
* **Spatial Indexing**: Automatic Morton/Hilbert ordering for fast queries
* **Chunk Bounds**: Precomputed bounding boxes for O(chunks) spatial queries
* **Context Manager**: Automatic finalization with ``with`` statement
* **Hierarchy Support**: Create nested groups with full path notation

Example::

   from luxar.io import LuxarZarrCompiler
   from luxar.core.dimensions import Dimensions
   import numpy as np

   dims = Dimensions.default_3d()
   with LuxarZarrCompiler('scene.luxar.zarr') as compiler:
       scene = compiler.create_scene(dimensions=dims)

       # Add root-level points
       compiler.write_points('cloud', positions, colors)

       # Create hierarchy with full paths
       compiler.write_points('GroupA/SubPoints', positions2, colors2)

LuxarScene
----------

The reader provides efficient access to Zarr datasets with lazy loading and spatial queries.

.. autoclass:: luxar.io.LuxarScene
   :members:
   :undoc-members:
   :show-inheritance:

Key Features
~~~~~~~~~~~~

* **Lazy Loading**: Only load metadata initially, fetch data on demand
* **Spatial Queries**: Efficient chunk-based queries using precomputed bounds
* **Node Discovery**: List and inspect all nodes in the scene graph
* **Type-Safe Access**: Type-specific getters (``get_points``, ``get_lines``, ``get_gsplats``)

Example::

   from luxar.io import LuxarScene

   # Load scene (metadata only)
   scene = LuxarScene.load('scene.luxar.zarr')

   # List available nodes
   print(scene.list_points())
   print(scene.list_lines())

   # Get node data (lazy loading)
   points_node = scene.get_points('cloud')
   positions = points_node.positions  # NumPy array attribute

Spatial Ordering
----------------

Spatial ordering functions for Morton/Hilbert curves and chunk bounds calculation.

.. automodule:: luxar.io.ordering
   :members:
   :undoc-members:

Morton and Hilbert Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~

Space-filling curves map n-dimensional coordinates to 1-dimensional ordering while
preserving spatial locality. This enables:

* **Better Compression**: Similar values group together (2-10x improvement)
* **Faster Queries**: Scan fewer chunks for spatial queries (10-100x speedup)
* **Cache Efficiency**: Adjacent points in space are adjacent in memory

See :doc:`../concepts/architecture` for detailed explanation of spatial indexing strategy.

Writer Protocol
---------------

.. autoclass:: luxar.io.writer.ZarrWriterProtocol
   :members:
   :undoc-members:

``ZarrWriterProtocol`` is a ``Protocol`` (abstract interface), not a concrete class.
It defines the writer contract that any Zarr writer implementation must satisfy. This
allows for alternative implementations (e.g., remote writers, streaming writers) while
maintaining compatibility with the compiler.
