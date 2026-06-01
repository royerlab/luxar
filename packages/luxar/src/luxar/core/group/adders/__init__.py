"""Per-leaf adder implementations for Group.

Each geometry type (Points, Lines, GSplats) has its own module here
holding the body of `Group.add_<type>` along with its partition-wrapper and
multi-LOD-wrapper helpers. The orchestrator file `core/group/group.py`
keeps the public method signatures + docstrings and delegates to these
free functions.
"""
