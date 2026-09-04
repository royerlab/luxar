Mesh Package
============

The mesh package holds everything *about* triangle meshes that is not the
scene-graph object. The node class itself is :class:`luxar.core.Mesh` (see
:doc:`core`); this package is its toolbox, mirroring how :mod:`luxar.gsplats`
sits beside :class:`luxar.core.GSplats`.

Mesh is the only *shaded* geometry type — Points, Lines and GSplats are purely
emissive — and the only one with no per-element size, because a triangle's
extent comes from its own vertices.

.. automodule:: luxar.mesh
   :members:
   :undoc-members:
   :show-inheritance:

Importing Classical Mesh Files
------------------------------

Reads PLY, OBJ, STL, VTP and glTF/GLB into a common :class:`TriangleMesh`,
including indexed directories of per-timepoint files.

.. automodule:: luxar.mesh.interop
   :members:
   :undoc-members:
   :show-inheritance:

Decimation (substitutive LOD)
-----------------------------

Produces the genuinely coarser *surfaces* a ``kind=lod`` group needs. A surface
cannot be coarsened by dropping elements the way points, lines and gsplats can —
dropping triangles punches holes — so decimation is what makes a substitutive
mesh ladder possible at all.

Two methods are offered: ``cluster`` (vectorized vertex-grid collapse, O(V log V),
usable at the writer's 2\ :sup:`27`-vertex cap) and ``qem`` (quadric error
metrics, higher quality at small vertex counts). ``auto`` resolves to ``qem``
through 10,000 vertices and ``cluster`` above that.

.. automodule:: luxar.mesh.decimate
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: luxar.mesh.qem.decimate_qem

.. autofunction:: luxar.mesh.qem.decimate_qem_ladder

Splitting by Faces
------------------

The by-face re-indexing behind ``add_mesh(partition=…)``. Splitting a mesh
duplicates the vertices that straddle a part boundary, so
:func:`~luxar.mesh.split.duplication_factor` reports what that cost.

.. automodule:: luxar.mesh.split
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

* :doc:`shading` — bake ambient occlusion for the emissive geometry types
* :doc:`core` — the :class:`luxar.core.Mesh` node itself
* :doc:`cli` — the ``luxar mesh import`` / ``luxar mesh lod`` commands
