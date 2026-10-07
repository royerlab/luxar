Core Package
============

The core package provides fundamental data structures for building Luxar scenes.

.. automodule:: luxar.core
   :no-members:

Scene
-----

.. autoclass:: luxar.core.Scene
   :members:
   :undoc-members:
   :show-inheritance:

Node
----

.. autoclass:: luxar.core.Node
   :members:
   :undoc-members:
   :show-inheritance:

Group
-----

.. autoclass:: luxar.core.Group
   :members:
   :undoc-members:
   :show-inheritance:

Points
------

.. autoclass:: luxar.core.Points
   :members:
   :undoc-members:
   :show-inheritance:

Lines
-----

.. autoclass:: luxar.core.Lines
   :members:
   :undoc-members:
   :show-inheritance:

GSplats
-------

.. autoclass:: luxar.core.GSplats
   :members:
   :undoc-members:
   :show-inheritance:

Mesh
----

.. autoclass:: luxar.core.Mesh
   :members:
   :undoc-members:
   :show-inheritance:

Sound
-----

.. autoclass:: luxar.core.Sound
   :members:
   :undoc-members:
   :show-inheritance:

Dimensions
----------

.. autoclass:: luxar.core.Dimensions
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: luxar.core.Dimension
   :members:
   :no-undoc-members:
   :show-inheritance:

Transforms
----------

.. automodule:: luxar.core.transforms
   :members:
   :undoc-members:

Waypoint trajectories
---------------------

How a :class:`~luxar.core.viewer_config.Waypoint` flies the camera to its pose.
Pass one of these as ``Waypoint(trajectory=...)``, or its name for its defaults.
The user guide's *Waypoint trajectories* section compares them side by side.

.. automodule:: luxar.core.trajectories
   :members: Orbit, ZoomPan, Arc, Straight, Swing, FlyThrough, Via
   :no-undoc-members:
   :show-inheritance:
