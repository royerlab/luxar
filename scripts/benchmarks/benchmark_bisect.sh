#!/bin/bash
#
# CUDA Performance Bisection Script
#
# Benchmarks CUDA gaussian splatting across multiple commits to identify
# performance regressions. Creates a worktree for each commit, builds
# the CUDA extension, runs reduced benchmarks, and collects results.
#
# Usage: bash scripts/benchmarks/benchmark_bisect.sh
#

set -euo pipefail

# Configuration
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Interpreter for the benchmark runs. Resolved from hatch rather than hardcoded:
# this used to name one developer's virtualenv by absolute path, which no other
# machine has. Override when hatch cannot report it.
if [ -n "${LUXAR_BENCH_PYTHON:-}" ]; then
    HATCH_PYTHON="${LUXAR_BENCH_PYTHON}"
else
    HATCH_ENV="$(cd "${REPO_ROOT}" && hatch env find default 2>/dev/null || true)"
    # `hatch env find` emits ANSI colour even when its output is captured, so the
    # raw substitution is not a usable path — resolution would fail on every
    # machine where hatch colourises. Safely (the -f guard rejects it) but
    # unconditionally, which makes the script unusable rather than merely strict.
    #
    # Stripped with bash parameter expansion rather than `sed`/`head`: this runs
    # with a deliberately minimal PATH in the regression tests, and a pipeline
    # through absent tools fails into the same empty result it is meant to fix.
    shopt -s extglob
    HATCH_ENV="${HATCH_ENV//$'\033'\[*([0-9;])[a-zA-Z]/}"
    HATCH_ENV="${HATCH_ENV%%$'\n'*}"
    shopt -u extglob
    HATCH_PYTHON="${HATCH_ENV:-/nonexistent}/bin/python"
fi
BENCHMARK_SCRIPT="${REPO_ROOT}/scripts/benchmarks/benchmark_bisect_runner.py"
RESULTS_DIR="${REPO_ROOT}/docs/benchmarks/bisection"
WORKTREE_BASE="/tmp/luxar-bench"
BUILD_SCRIPT_REL="packages/luxar/src/luxar/gsplats/models/gsplats/cuda/build.py"
CUDA_DIR_REL="packages/luxar/src/luxar/gsplats/models/gsplats/cuda"

# Commits to benchmark (oldest to newest) - all commits touching CUDA code
# These SHAs predate the September 2026 git history rewrite and no longer
# resolve in this repository. Recover a rewritten commit by searching its
# subject, for example: git log --all --grep='feat: add CUDA backend for gsplats'
COMMITS=(
    "f8a1ebb:feat: add CUDA backend for gsplats"
    "922493e:fix(cuda): fix build system after make clean-cuda"
    "cc0882f:feat(cuda): add global splat handling for large splats"
    "5ec1367:feat(cuda): add FP16 precision support"
    "aba16bb:feat(cuda): add AMP support and expand FP16/test coverage"
    "18e4d09:feat(cuda): add grad_output caching and refactor kernels"
    "b985c6f:feat(cuda): add dynamic GPU capability detection for tile sizing"
    "e47c55d:feat(cuda): add dynamic SPLAT_BATCH_SIZE scaling"
    "daa7d5f:perf(cuda): architecture-aware batch size selection"
    "adcdb2d:fix(cuda): use architecture defaults for shared_memory_per_block"
    "ee57fea:feat: add gsplat constraint params, dynamic range quantization"
    "101fd4d:feat: add comprehensive gsplat CLI tools, improve dynamic ops"
    "0629c4f:feat: comprehensive codebase review improvements"
    "136b5d7:perf(cuda): revert register-spilling pixel precomputation"
    "cee7441:chore: misc fixes from concurrent agents"
)

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "========================================================================"
echo "CUDA PERFORMANCE BISECTION"
echo "========================================================================"
echo ""
echo "Commits to benchmark: ${#COMMITS[@]}"
echo "Results directory: ${RESULTS_DIR}"
echo "Python: ${HATCH_PYTHON}"
echo ""

# Check prerequisites
if [ ! -f "${HATCH_PYTHON}" ]; then
    echo "ERROR: Hatch Python not found at ${HATCH_PYTHON}"
    echo "       Run 'hatch env create' or set LUXAR_BENCH_PYTHON to the interpreter to use."
    exit 1
fi

if ! "${HATCH_PYTHON}" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch CUDA not available"
    exit 1
fi

echo "GPU: $("${HATCH_PYTHON}" -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo ""

# Track results
COMPLETED=0
FAILED=0
SKIPPED=0

for entry in "${COMMITS[@]}"; do
    SHA="${entry%%:*}"
    MSG="${entry#*:}"
    WORKTREE="${WORKTREE_BASE}-${SHA}"
    OUTPUT_JSON="${RESULTS_DIR}/bench_${SHA}.json"

    echo "---------------------------------------------------------------"
    echo "[$((COMPLETED + FAILED + SKIPPED + 1))/${#COMMITS[@]}] ${SHA} - ${MSG}"
    echo "---------------------------------------------------------------"

    # Skip if already benchmarked
    if [ -f "${OUTPUT_JSON}" ]; then
        echo "  Already benchmarked, skipping. Delete ${OUTPUT_JSON} to re-run."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Clean up any leftover worktree
    if [ -d "${WORKTREE}" ]; then
        echo "  Cleaning up stale worktree..."
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || rm -rf "${WORKTREE}"
    fi

    # Create worktree
    echo "  Creating worktree at ${WORKTREE}..."
    if ! git -C "${REPO_ROOT}" worktree add "${WORKTREE}" "${SHA}" --detach 2>/dev/null; then
        echo "  ERROR: Failed to create worktree for ${SHA}"
        FAILED=$((FAILED + 1))
        echo "{\"commit\": \"${SHA}\", \"error\": \"Failed to create worktree\"}" > "${OUTPUT_JSON}"
        continue
    fi

    # Check if CUDA code exists in this commit
    if [ ! -f "${WORKTREE}/${BUILD_SCRIPT_REL}" ]; then
        echo "  SKIP: No CUDA build script in this commit"
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || true
        SKIPPED=$((SKIPPED + 1))
        echo "{\"commit\": \"${SHA}\", \"error\": \"No CUDA code in this commit\"}" > "${OUTPUT_JSON}"
        continue
    fi

    # Build CUDA extension
    echo "  Building CUDA extension..."
    BUILD_START=$(date +%s)
    if ! "${HATCH_PYTHON}" "${WORKTREE}/${BUILD_SCRIPT_REL}" > "${WORKTREE}/build.log" 2>&1; then
        echo "  ERROR: CUDA build failed. See ${WORKTREE}/build.log"
        cat "${WORKTREE}/build.log" | tail -20
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || true
        FAILED=$((FAILED + 1))
        echo "{\"commit\": \"${SHA}\", \"error\": \"CUDA build failed\"}" > "${OUTPUT_JSON}"
        continue
    fi
    BUILD_END=$(date +%s)
    echo "  Build completed in $((BUILD_END - BUILD_START))s"

    # Verify .so file exists
    SO_FILE=$(ls "${WORKTREE}/${CUDA_DIR_REL}"/cuda_splatting_backend*.so 2>/dev/null | head -1)
    if [ -z "${SO_FILE}" ]; then
        # Check build directory too
        SO_FILE=$(ls "${WORKTREE}/${CUDA_DIR_REL}"/build/cuda_splatting_backend*.so 2>/dev/null | head -1)
    fi
    if [ -z "${SO_FILE}" ]; then
        echo "  ERROR: No .so file found after build"
        git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || true
        FAILED=$((FAILED + 1))
        echo "{\"commit\": \"${SHA}\", \"error\": \"No .so file after build\"}" > "${OUTPUT_JSON}"
        continue
    fi
    echo "  Built: $(basename ${SO_FILE})"

    # Run benchmark
    echo "  Running benchmarks..."
    BENCH_START=$(date +%s)
    if ! "${HATCH_PYTHON}" "${BENCHMARK_SCRIPT}" "${WORKTREE}" "${SHA}" "${OUTPUT_JSON}" 2>&1; then
        echo "  ERROR: Benchmark failed"
        FAILED=$((FAILED + 1))
        if [ ! -f "${OUTPUT_JSON}" ]; then
            echo "{\"commit\": \"${SHA}\", \"error\": \"Benchmark script failed\"}" > "${OUTPUT_JSON}"
        fi
    else
        COMPLETED=$((COMPLETED + 1))
    fi
    BENCH_END=$(date +%s)
    echo "  Benchmark completed in $((BENCH_END - BENCH_START))s"

    # Clean up worktree
    echo "  Cleaning up worktree..."
    git -C "${REPO_ROOT}" worktree remove --force "${WORKTREE}" 2>/dev/null || rm -rf "${WORKTREE}"

    echo ""
done

echo ""
echo "========================================================================"
echo "BISECTION COMPLETE"
echo "========================================================================"
echo "  Completed: ${COMPLETED}"
echo "  Failed:    ${FAILED}"
echo "  Skipped:   ${SKIPPED}"
echo "  Results:   ${RESULTS_DIR}/"
echo ""
echo "Run the analysis script to generate the comparison report:"
echo "  ${HATCH_PYTHON} scripts/benchmarks/benchmark_bisect_analyze.py"
echo ""
