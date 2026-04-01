#!/usr/bin/env python3
"""Smoke test: verify sharpness removal from GSplats and preservation for Points/Lines.

Tests are maintained in the pytest suite at:
    packages/luxar/src/luxar/gsplats/tests/test_gsplat_data.py::TestGSplatsWithoutSharpness

This script imports and runs those tests standalone (e.g. on HPC without pytest).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

# Re-export test functions from the pytest suite
from luxar.gsplats.tests.test_gsplat_data import TestGSplatsWithoutSharpness


def _run_test(name, func, *args):
    """Run a test method, passing fixtures manually."""
    try:
        func(*args)
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name}\n  {e}")
        return False


if __name__ == "__main__":
    inst = TestGSplatsWithoutSharpness()
    passed = failed = 0

    tests_no_fixture = [
        ("gsplat_data_no_sharpnesses_field", inst.test_gsplat_data_no_sharpnesses_field),
        ("add_gsplats_signature_no_sharpness", inst.test_add_gsplats_signature_no_sharpness),
        ("fitting_returns_no_sharpness", inst.test_fitting_returns_no_sharpness),
    ]

    tests_with_tmp_path = [
        ("gsplats_roundtrip_no_sharpness", inst.test_gsplats_roundtrip_no_sharpness),
        ("points_still_have_sharpness", inst.test_points_still_have_sharpness),
        ("lines_still_have_sharpness", inst.test_lines_still_have_sharpness),
    ]

    for name, func in tests_no_fixture:
        if _run_test(name, func):
            passed += 1
        else:
            failed += 1

    for name, func in tests_with_tmp_path:
        with tempfile.TemporaryDirectory() as tmp:
            if _run_test(name, func, Path(tmp)):
                passed += 1
            else:
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
