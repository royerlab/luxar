I/O Package
===========

The io package handles reading and writing Luxar scenes to Zarr format with spatial indexing.

.. automodule:: luxar.io
   :no-members:

Overview
--------

The I/O package provides two main classes and a filterable capacity warning:

* **LuxarZarrCompiler**: Progressive writer for creating Zarr datasets
* **LuxarScene**: Reader for loading and querying Zarr datasets
* **ElementCapacityWarning**: An authored node may exceed the viewer's
  conservative per-node element-texture capacity

Both classes work together to enable efficient streaming of massive n-dimensional scenes with points, lines, Gaussian splats, and other primitives.

.. autoexception:: luxar.io.ElementCapacityWarning

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

The contract is enforced, not merely declared: ``LuxarZarrCompiler`` is checked
against it by mypy with no ``type: ignore[override]`` escapes, and
``test_writer_protocol_agreement.py`` additionally pins the parameter names,
order and defaults of every method it declares. Order matters to a caller because
these are positional-or-keyword parameters, and a parameter present in one
signature but not the other renumbers every argument after it.

Volume Loading
--------------

Load an nD image / volume from ``.zarr``, ``.zarr.zip``, ``.tiff``, ``.npy``, or
``.npz`` — with channel/timepoint selection and axis-order handling — for
Gaussian splat fitting and calibration.

.. automodule:: luxar.io.volume
   :no-members:

.. autofunction:: luxar.io.volume.load_volume

OME-Zarr Discovery
------------------

Discover the shape, axis labels, and voxel size of an OME-Zarr multiscale
dataset without loading the pixel data.

.. automodule:: luxar.io.ome_zarr
   :no-members:

.. autofunction:: luxar.io.ome_zarr.discover_ome_zarr_shape

.. autofunction:: luxar.io.ome_zarr.resolve_ngff_attrs

.. autofunction:: luxar.io.ome_zarr.ngff_scale_transform

.. autoclass:: luxar.io.ome_zarr.OMEZarrInfo
   :members:
   :undoc-members:
   :show-inheritance:

Chunk-Layout Optimization
-------------------------

Re-chunk an existing store for streaming in one structure-preserving pass —
values, codecs, the spatial-index grid, the zarr format version and every
attribute except the cache guard (the root's ``content_hash``, restamped, and
the ``chunk_layout`` summary beside it) all survive; only zarr chunk shapes
change. Backs the ``luxar optimize`` command and the ``luxar info --stats``
chunk diagnostic.

.. automodule:: luxar.io.optimize
   :no-members:

.. autofunction:: luxar.io.optimize.optimize_store

.. autofunction:: luxar.io.optimize.plan_optimization

.. autofunction:: luxar.io.optimize.summarize_chunk_layout

.. autofunction:: luxar.io.optimize.summarize_plan

.. autofunction:: luxar.io.optimize.resolve_target_bytes

.. autodata:: luxar.io.optimize.CHUNK_PROFILES

.. autoclass:: luxar.io.optimize.OptimizePlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: luxar.io.optimize.ArrayPlan
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: luxar.io.optimize.ChunkLayoutSummary
   :members:
   :undoc-members:
   :show-inheritance:
