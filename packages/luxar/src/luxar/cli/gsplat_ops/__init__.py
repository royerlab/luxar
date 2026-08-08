"""Command-group modules for the ``luxar gsplat`` CLI.

Each thematic group exposes a ``register_<group>(app)`` (or, for ``batch-fit``,
an ``app_batch`` sub-app) that wires its commands onto the shared ``app_gsplat``
Typer defined in ``cli/gsplat_commands.py``. The aggregator imports and calls
them — keeping ``gsplat_commands.py`` a thin registration surface rather than a
4.7k-line god-file.

Small groups stay as single root modules (``scene_commands``,
``inspect_commands``, ``interchange_commands``, ``benchmark``) alongside the
shared root helpers (``recipe_shared``, ``planner``, ``encoding``). The three
large groups are subpackages whose registration surface is ``commands.py``:

- ``fitting/`` — ``fit`` / ``cal`` / ``render`` / ``denoise``
- ``batch/`` — the ``batch-fit`` group (local multi-GPU + Slurm)
- ``transforms/`` — edit-style commands on a fitted ``.gsplats.zarr``

Their ``__init__.py`` files are docstring-only: importers name the owning module
directly, never a re-export root.
"""
