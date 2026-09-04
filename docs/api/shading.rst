Shading Package
===============

Appearance baked from geometry at scene-authoring time.

Luxar's emissive geometry types — Points, Lines and GSplats — have shaders that
know nothing about neighbouring geometry, so shape cues a shaded renderer would
get for free have to be computed by the author and written into the scene. This
package holds those bakes; :func:`~luxar.shading.bake_ambient_occlusion` is the
sanctioned way to make a dense emissive cloud read as three-dimensional rather
than as a flat haze.

The dividing line is deliberate: **bake what the geometry knows, leave to the
shader what the camera knows.** Ambient occlusion is a scalar function of the
geometry alone, so a value computed offline is correct from every camera and
belongs in the store. A key light depends on a direction relative to the viewer,
so baking one fixes it in world space and it stops reading the moment the camera
orbits — that belongs in a material, not in a store.

.. automodule:: luxar.shading
   :no-members:

Everything in ``luxar.shading``'s ``__all__`` is re-exported from
:mod:`luxar.shading.occlusion`, so the members are documented once, below,
rather than twice under two names.

Occlusion
---------

.. automodule:: luxar.shading.occlusion
   :members:
   :undoc-members:
   :show-inheritance:

See Also
--------

* :doc:`mesh` — the shaded geometry type, which needs no bake
* :doc:`core` — :class:`luxar.core.Points`, :class:`luxar.core.Lines` and
  :class:`luxar.core.GSplats`, the emissive types this package exists for
