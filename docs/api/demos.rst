Demos Package
=============

The demos package is the supported public barrel for dataset acquisition,
download integrity, runtime flags, provenance, and viewer helpers used by
Luxar's executable demos. The concern packages below ``luxar.demos._support``
are implementation details and should not be imported directly by demos.

.. automodule:: luxar.demos
   :members:
   :exclude-members: FlowField
   :undoc-members:
   :show-inheritance:

.. autoclass:: luxar.demos.FlowField
   :members:
   :exclude-members: vectors, grid_min, grid_max, spacing, cache_key
