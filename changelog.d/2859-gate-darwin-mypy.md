#### Gate Darwin-specific Python typing

CI and `make type-check-python` now run a second mypy pass pinned to Darwin, with
an isolated cache, so Linux CI catches uses of platform-specific Python APIs
before they break the macOS pre-commit hook. The hook itself remains host-only
to avoid doubling the full-tree type-check cost on every commit.
