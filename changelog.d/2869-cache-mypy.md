#### Reuse mypy state across CI runs

The required Python 3.12 job now restores and saves the complete `.mypy_cache`
tree, including any nested platform- and version-pinned pass caches. The exact
cache key follows the runner, Python version, mypy pin, and Python dependency
set, so unchanged runs avoid rebuilding full-tree type information while
dependency or type-checker declaration changes still start clean.
