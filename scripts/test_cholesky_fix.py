"""Test the Cholesky regularisation fix in embed_cholesky_packed.

Tests are maintained in the pytest suite at:
    packages/luxar/src/luxar/gsplats/utils/tests/test_trils.py::TestEmbedCholeskyPackedNoNan

This script imports and runs those tests standalone (e.g. on HPC without pytest).
"""

import sys

import numpy as np

# Re-export test class from the pytest suite
from luxar.gsplats.utils.tests.test_trils import TestEmbedCholeskyPackedNoNan
from luxar.gsplats.utils.trils import pack_tril


def main():
    # Build the fixture manually (mirrors the pytest fixture)
    rng = np.random.default_rng(42)
    N, d = 100, 3
    A = rng.standard_normal((N, d, d))
    Sigma = A @ A.transpose(0, 2, 1) + np.eye(d) * 0.1
    L = np.linalg.cholesky(Sigma)
    good_packed = pack_tril(L.astype(np.float32))

    inst = TestEmbedCholeskyPackedNoNan()

    tests = [
        ("good_splats", inst.test_good_splats, good_packed),
        ("some_degenerate_zeros", inst.test_some_degenerate_zeros, good_packed),
        ("all_degenerate_zeros", inst.test_all_degenerate_zeros, good_packed),
        ("near_singular", inst.test_near_singular, good_packed),
        ("empty", inst.test_empty),
        ("single_splat", inst.test_single_splat, good_packed),
    ]

    passed = total = 0
    for entry in tests:
        name = entry[0]
        func = entry[1]
        args = entry[2:]
        total += 1
        try:
            func(*args)
            print(f"  {name}: PASS")
            passed += 1
        except Exception as e:
            print(f"  {name}: FAIL ({e})")

    print(f"\nResults: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
