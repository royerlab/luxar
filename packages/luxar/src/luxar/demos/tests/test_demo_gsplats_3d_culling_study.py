"""Regression tests for the culling study's runtime metadata."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_culling_study.py"


def _load_demo_module(name: str = "_luxar_demo_culling_study_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def test_scene_description_uses_loaded_base_count() -> None:
    description = _demo.scene_description([{"splats": 200_023}, {"splats": 95_160}])

    assert "200,023 splats" in description
    assert "200,155 splats" not in description
