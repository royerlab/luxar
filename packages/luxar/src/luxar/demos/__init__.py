"""Luxar Demos - Self-contained demonstration scripts.

This package contains executable demo scripts that showcase Luxar's capabilities.
Each demo is completely self-contained with all generation code in a single file.

To run a demo:
    hatch run python packages/luxar/src/luxar/demos/demo_lorenz.py
    hatch run python packages/luxar/src/luxar/demos/demo_cubic_array.py

See demos/README.md for more information on creating new demos.

Note: The `launch_viewer` helper function is imported from `luxar.demos`
which is aliased to `luxar.utils.demos` in the main package.
"""

# Note: Demos are meant to be run as scripts, not imported as modules.
# The `launch_viewer` function is available via `from luxar.demos import launch_viewer`
# which resolves to `luxar.utils.demos.launch_viewer` due to the alias in luxar/__init__.py
