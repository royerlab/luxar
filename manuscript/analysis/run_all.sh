#!/bin/bash
# Master script: runs all manuscript analyses sequentially.
#
# Ordered fastest-to-slowest, with PDF consolidation after each analysis:
#   1. Noise floor estimation (~seconds)
#   2. Splat count vs quality — rate-distortion + Noise2Self + plots (~2h)
#      → builds splat_count_vs_quality supplementary PDF
#   3. Progressive vs single-pass — 4 conditions per dataset (~1h)
#      → builds progressive supplementary PDF
#   4. Convergence — 5 counts x 11 checkpoints per dataset (~3h) [SLOWEST]
#      → builds convergence supplementary PDF
#
# All scripts are resumable — re-running skips completed entries.
#
# Usage:
#   bash manuscript/analysis/run_all.sh
#   bash manuscript/analysis/run_all.sh 2>&1 | tee manuscript/analysis/run_all.log

set -e
cd "$(git rev-parse --show-toplevel)"

DATASETS="opencell_map4_ch0 opencell_map4_ch1 opencell_lmnb1_ch0 opencell_lmnb1_ch1 kidney_dapi kidney_actin cells3d_nuclei cells3d_membrane organoid_ch0 celegans_t100 tribolium acto3d_heart_nuclei"

echo "================================================================"
echo "  MASTER ANALYSIS PIPELINE"
echo "  Datasets: $(echo $DATASETS | wc -w) total"
echo "  Started: $(date -Iseconds)"
echo "================================================================"
echo ""

# ──────────────────────────────────────────────────────────────────────
# 1. NOISE FLOOR ESTIMATION (fastest — seconds per dataset)
# ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  1. NOISE FLOOR ESTIMATION                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
hatch run python manuscript/analysis/splat_count_vs_quality/run_noise_floor.py --all

# ──────────────────────────────────────────────────────────────────────
# 2. SPLAT COUNT VS QUALITY (rate-distortion + Noise2Self + plots)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  2. SPLAT COUNT VS QUALITY                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for ds in $DATASETS; do
    echo ""
    echo "--- Rate-distortion: $ds ---"
    hatch run python manuscript/analysis/splat_count_vs_quality/run_analysis.py --dataset $ds

    echo "--- Noise2Self: $ds ---"
    hatch run python manuscript/analysis/splat_count_vs_quality/run_noise2self.py --dataset $ds

    echo "--- Plotting: $ds ---"
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_results.py --dataset $ds
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_noise2self.py --dataset $ds
done

echo ""
echo "--- Building splat_count_vs_quality supplementary PDF ---"
hatch run python manuscript/analysis/build_supp_docs.py --analysis splat_count_vs_quality

# ──────────────────────────────────────────────────────────────────────
# 3. PROGRESSIVE VS SINGLE-PASS (4 conditions per dataset)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3. PROGRESSIVE VS SINGLE-PASS                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for ds in $DATASETS; do
    echo "--- Progressive: $ds ---"
    hatch run python manuscript/analysis/progressive_vs_single/run_analysis.py --dataset $ds
    hatch run python manuscript/analysis/progressive_vs_single/plot_results.py --dataset $ds
done

echo ""
echo "--- Building progressive supplementary PDF ---"
hatch run python manuscript/analysis/build_supp_docs.py --analysis progressive

# ──────────────────────────────────────────────────────────────────────
# 4. CONVERGENCE ANALYSIS (5 counts x 11 checkpoints — SLOWEST)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  4. CONVERGENCE ANALYSIS (slowest — runs last)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for ds in $DATASETS; do
    echo "--- Convergence: $ds ---"
    hatch run python manuscript/analysis/convergence/run_convergence.py --dataset $ds
    hatch run python manuscript/analysis/convergence/plot_convergence.py --dataset $ds
done

echo ""
echo "--- Building convergence supplementary PDF ---"
hatch run python manuscript/analysis/build_supp_docs.py --analysis convergence

# ──────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  ALL ANALYSES COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "================================================================"
