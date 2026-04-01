#!/usr/bin/env python3
"""Smoke tests for batch plan HPC fixes.

Tests are maintained in the pytest suite at:
    packages/luxar/src/luxar/gsplats/tests/test_batch.py::TestBatchPlanRegression

This script imports and runs those tests standalone (e.g. on HPC without pytest).

Run with:
    hatch run python scripts/test_batch_plan_fixes.py
"""

import sys
import tempfile
from pathlib import Path

# Re-export test class from the pytest suite
from luxar.gsplats.tests.test_batch import TestBatchPlanRegression


def _run_test(name, func, *args):
    """Run a test method, passing fixtures manually."""
    try:
        func(*args)
        print(f"PASS: {name}")
        return True
    except AssertionError as e:
        print(f"FAIL: {name}\n  {e}")
        return False
    except Exception as e:
        print(f"ERROR: {name}\n  {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    inst = TestBatchPlanRegression()
    passed = failed = 0

    print(f"\n{'=' * 60}")
    print("Batch Plan HPC Fixes — Smoke Tests")
    print(f"{'=' * 60}\n")

    tests_no_fixture = [
        ("zarr_zip_suffix_detection", inst.test_zarr_zip_suffix_detection),
        ("custom_axes_parsing", inst.test_custom_axes_parsing),
        ("auto_tile_small_volume", inst.test_auto_tile_small_volume),
        ("cull_retention_defaults", inst.test_cull_retention_defaults),
        ("6d_channel_decoding", inst.test_6d_channel_decoding),
        ("tasks_per_job_manifest", inst.test_tasks_per_job_manifest),
        (
            "env_capture_ld_library_path_prepend",
            inst.test_env_capture_ld_library_path_prepend,
        ),
        (
            "sbatch_omits_channel_when_single",
            inst.test_sbatch_omits_channel_when_single,
        ),
    ]

    tests_with_tmp_path = [
        ("axes_override_validation", inst.test_axes_override_validation),
        ("array_selection_consistency", inst.test_array_selection_consistency),
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

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)
