"""Importing the base package must stay lightweight (no eager torch/scipy).

Scene construction, dimensions, and basic zarr compilation should not pull in
the heavy gsplats subsystem. The GSplat re-exports resolve lazily (PEP 562) on
first attribute access.
"""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


def test_import_luxar_does_not_import_torch() -> None:
    """A bare `import luxar` must not import torch (the gsplats-only dep)."""
    result = _run(
        "import luxar, sys; "
        "assert 'torch' not in sys.modules, 'torch imported eagerly by import luxar'; "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_lazy_gsplat_export_is_accessible() -> None:
    """Accessing luxar.GSplatData resolves it lazily (and only then loads gsplats)."""
    result = _run(
        "import luxar; "
        "cls = luxar.GSplatData; "
        "from luxar import fit_gaussian_splats; "
        "assert cls.__name__ == 'GSplatData'; "
        "assert callable(fit_gaussian_splats); "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """The lazy __getattr__ must not swallow genuinely missing attributes."""
    import luxar

    try:
        luxar.does_not_exist  # noqa: B018
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError for unknown attribute")
