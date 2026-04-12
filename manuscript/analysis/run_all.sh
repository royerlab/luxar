#!/bin/bash
# Master script: runs all manuscript analyses sequentially.
#
# Usage:
#   bash manuscript/analysis/run_all.sh
#   bash manuscript/analysis/run_all.sh 2>&1 | tee manuscript/analysis/run_all.log
#
# All scripts are resumable — re-running skips completed entries.

set -e
cd "$(git rev-parse --show-toplevel)"

DATASETS="opencell_map4_ch0 opencell_map4_ch1 kidney_dapi kidney_actin organoid_ch0 celegans_t100 tribolium"

echo "================================================================"
echo "  MASTER ANALYSIS PIPELINE"
echo "  Datasets: $DATASETS"
echo "  Started: $(date -Iseconds)"
echo "================================================================"
echo ""

# ──────────────────────────────────────────────────────────────────────
# 1. SPLAT COUNT VS QUALITY (rate-distortion + Noise2Self + plots)
# ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  1. SPLAT COUNT VS QUALITY                                  ║"
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

# ──────────────────────────────────────────────────────────────────────
# 2. NOISE FLOOR ESTIMATION
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  2. NOISE FLOOR ESTIMATION                                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
hatch run python manuscript/analysis/splat_count_vs_quality/run_noise_floor.py --all

# Re-plot with noise floor overlay
echo "--- Re-plotting with noise floor ---"
for ds in $DATASETS; do
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_results.py --dataset $ds
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_noise2self.py --dataset $ds
done

# ──────────────────────────────────────────────────────────────────────
# 3. CONVERGENCE ANALYSIS
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3. CONVERGENCE ANALYSIS                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for ds in $DATASETS; do
    echo "--- Convergence: $ds ---"
    hatch run python manuscript/analysis/convergence/run_convergence.py --dataset $ds
    hatch run python manuscript/analysis/convergence/plot_convergence.py --dataset $ds
done

# ──────────────────────────────────────────────────────────────────────
# 4. PROGRESSIVE VS SINGLE-PASS
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  4. PROGRESSIVE VS SINGLE-PASS                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
for ds in $DATASETS; do
    echo "--- Progressive: $ds ---"
    hatch run python manuscript/analysis/progressive_vs_single/run_analysis.py --dataset $ds
    hatch run python manuscript/analysis/progressive_vs_single/plot_results.py --dataset $ds
done

# ──────────────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  ALL ANALYSES COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo "================================================================"
