Utils Package
=============

The utils package provides cross-cutting utility functions used by Luxar's core
runtime. Demo-owned download, dataset, and runtime helpers are exposed through
the :mod:`luxar.demos` barrel instead.

.. automodule:: luxar.utils
   :members:
   :undoc-members:
   :show-inheritance:

Array Utilities
---------------

.. automodule:: luxar.utils.array
   :members:
   :undoc-members:

Reusable Demo Generators
------------------------

.. automodule:: luxar.utils.colors
   :members:
   :undoc-members:

.. automodule:: luxar.utils.scenes
   :members:
   :undoc-members:

Console Output
--------------

``luxar.verbosity`` turns Luxar's own narration down, or off. The library layers
below the CLI narrate through arbol — the right default for a long CLI run, and
the wrong one in a notebook cell or a napari plugin. Reachable from the package
root as ``luxar.set_verbosity()`` / ``luxar.verbosity()``.

Note that this writes process-global arbol state, so it is neither per-call nor
thread-safe; the module docstring states the constraints in full.

.. automodule:: luxar.verbosity
   :members:
   :undoc-members:
