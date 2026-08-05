# luxar.tests

Top-level test suite for the `luxar` package.

## Quick Start

Run the suite with Hatch (see [Running Tests](#running-tests) for more targets):

```bash
hatch run test
```

Tests here are plain PyTest functions with a one-line docstring. A typical
package-level check looks like this:

```python
def test_luxar_advertises_a_version() -> None:
    """The package exposes a version string at its root."""
    import luxar

    assert isinstance(luxar.__version__, str)
```

Checks that must guarantee a *pristine* interpreter — e.g. that a bare
`import luxar` does not eagerly import torch — run the probe in a subprocess so
another already-collected test module can't have imported the dependency first;
see `test_lazy_imports.py`.

## Scope

This directory contains package-level tests (e.g., import/export sanity, version
checks, and the custom Hatch build-hook contract) plus tests for repository-level
maintenance scripts that exercise the installed package. Subpackage-specific tests
are colocated with each subpackage in their own `tests/` directories.

## Running Tests

```bash
hatch run test                              # All tests
hatch run pytest packages/luxar/src/luxar/tests/  # Only this directory
hatch run test-cov                          # With coverage
```

## Test Layout

Tests are colocated throughout the package:

- `luxar/tests/` -- top-level package and maintenance-script tests (this directory)
- `luxar/core/tests/` -- scene graph, transforms, dimensions
- `luxar/io/tests/` -- compiler, reader, writer
- `luxar/encoding/tests/` -- encoding strategies
- `luxar/colormaps/tests/` -- colormap utilities
- `luxar/validation/tests/` -- validation logic and nD transforms
- `luxar/gsplats/**/tests/` -- Gaussian splatting (fitting, models, seeds, etc.)
- `luxar/cli/tests/` -- CLI commands
- `luxar/utils/tests/` -- utility functions
