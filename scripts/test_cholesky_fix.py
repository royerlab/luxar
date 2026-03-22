"""Test the Cholesky regularisation fix in embed_cholesky_packed."""
import traceback
import numpy as np
from luxar.gsplats.utils.trils import embed_cholesky_packed, pack_tril


def test_case(name, packed, d_src, d_dst, dim_mapping, fill_sigma):
    try:
        result = embed_cholesky_packed(
            packed, d_src=d_src, d_dst=d_dst,
            dim_mapping=dim_mapping, fill_sigma=fill_sigma,
        )
        ok = bool(np.all(np.isfinite(result)))
        status = "PASS" if ok else "FAIL (non-finite)"
        print(f"  {name}: shape={result.shape}, {status}")
        return ok
    except Exception as e:
        print(f"  {name}: FAIL (exception: {e})")
        traceback.print_exc()
        return False


def main():
    rng = np.random.default_rng(42)
    N, d = 100, 3
    A = rng.standard_normal((N, d, d))
    Sigma = A @ A.transpose(0, 2, 1) + np.eye(d) * 0.1
    L = np.linalg.cholesky(Sigma)
    packed = pack_tril(L.astype(np.float32))

    passed = 0
    total = 0

    print("Test 1: All good splats")
    total += 1; passed += test_case("good", packed, 3, 4, [1, 2, 3], {0: 1e-7})

    print("Test 2: Some degenerate (zeros)")
    p2 = packed.copy(); p2[5] = 0; p2[50] = 0
    total += 1; passed += test_case("some_bad", p2, 3, 4, [1, 2, 3], {0: 1e-7})

    print("Test 3: ALL degenerate (zeros)")
    p3 = np.zeros_like(packed)
    total += 1; passed += test_case("all_bad", p3, 3, 4, [1, 2, 3], {0: 1e-7})

    print("Test 4: Near-singular (tiny values)")
    p4 = packed.copy(); p4 *= 1e-15
    total += 1; passed += test_case("near_singular", p4, 3, 4, [1, 2, 3], {0: 1e-7})

    print("Test 5: N=0 (empty)")
    p5 = np.empty((0, 6), dtype=np.float32)
    total += 1; passed += test_case("empty", p5, 3, 4, [1, 2, 3], {0: 1e-7})

    print("Test 6: N=1")
    total += 1; passed += test_case("single", packed[:1], 3, 4, [1, 2, 3], {0: 1e-7})

    print(f"\nResults: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
