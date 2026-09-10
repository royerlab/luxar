"""Shared helpers for fitting-pipeline tests."""

from __future__ import annotations

from typing import Any

try:
    from luxar.gsplats.fitting.config import FitParameters
    from luxar.gsplats.fitting.validation import (
        prepare_fit_config as _prepare_fit_config,
    )
except ModuleNotFoundError as exc:
    if (exc.name or "").split(".", 1)[0] not in {"scipy", "torch"}:
        raise
else:

    def prepare_fit_config(fitter: Any, V: Any, **kwargs: Any) -> Any:
        """Build raw parameters while keeping direct validation tests concise."""
        return _prepare_fit_config(fitter, FitParameters(V=V, **kwargs))
