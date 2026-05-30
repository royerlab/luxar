"""GSplats high-level pipeline: from_data / file / volume + LOD dispatch.

These free functions back `Group.add_gsplats_from_data`,
`add_gsplats_from_file`, and `add_gsplats_from_volume`. The orchestrator
in `core/group/group.py` keeps the public signatures + docstrings and
delegates to these.
"""
