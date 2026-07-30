# luxar.tests

Top-level test suite for the `luxar` package.

## Scope

This directory contains package-level tests (e.g., import/export sanity, version
checks, and the custom Hatch build-hook contract). Subpackage-specific tests are
colocated with each subpackage in their own `tests/` directories.

## Running Tests

```bash
hatch run test                              # All tests
hatch run pytest packages/luxar/src/luxar/tests/  # Only this directory
hatch run test-cov                          # With coverage
```

## Test Layout

Tests are colocated throughout the package:

- `luxar/tests/` -- top-level package tests (this directory)
- `luxar/core/tests/` -- scene graph, transforms, dimensions
- `luxar/io/tests/` -- compiler, reader, writer
- `luxar/encoding/tests/` -- encoding strategies
- `luxar/colormaps/tests/` -- colormap utilities
- `luxar/validation/tests/` -- validation logic and nD transforms
- `luxar/gsplats/**/tests/` -- Gaussian splatting (fitting, models, seeds, etc.)
- `luxar/cli/tests/` -- CLI commands
- `luxar/utils/tests/` -- utility functions
