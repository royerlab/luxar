#### Reuse mypy state across CI runs

The required Python 3.12 job now restores and saves the complete `.mypy_cache`
tree, including any nested per-pass cache directories. Each commit saves a fresh
entry keyed by runner, Python version, and `pyproject.toml`, while a prefix
fallback lets mypy safely reuse valid module state after configuration or
dependency changes.
