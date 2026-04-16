#!/bin/bash
# Generate all figure PDFs from pre-computed analysis results (TSVs / NPZs).
#
# NO GPU REQUIRED — this script only reads cached data and produces plots.
# Assumes analysis results already exist (run analysis/run_all.sh first).
# Does NOT run analyses, build supp docs, or build the preprint.
#
# Slice montages and visual comparisons are skipped if .npz slice files
# are not available (these require the GPU analysis phase).
#
# Usage:
#   bash manuscript/build_figures.sh

set -e
cd "$(git rev-parse --show-toplevel)"

DATASETS="opencell_map4_ch0 opencell_map4_ch1 opencell_lmnb1_ch0 opencell_lmnb1_ch1 kidney_dapi kidney_actin cells3d_nuclei cells3d_membrane organoid_ch0 celegans_t100 tribolium acto3d_heart_nuclei"

echo "================================================================"
echo "  FIGURE GENERATION (no GPU — reads cached results only)"
echo "  Started: $(date -Iseconds)"
echo "================================================================"
echo ""

# ──────────────────────────────────────────────────────────────────────
# 1. PER-DATASET ANALYSIS FIGURES (from TSVs)
# ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  1. Per-dataset analysis figures                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"

for ds in $DATASETS; do
    echo ""
    echo "--- $ds ---"

    # Convergence curves
    hatch run python manuscript/analysis/convergence/plot_convergence.py --dataset "$ds" 2>&1 || echo "  WARNING: $ds convergence plotting failed"

    # Progressive vs single-pass comparison
    hatch run python manuscript/analysis/progressive_vs_single/plot_results.py --dataset "$ds" 2>&1 || echo "  WARNING: $ds progressive plotting failed"

    # Splat count vs quality curves
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_results.py --dataset "$ds" 2>&1 || echo "  WARNING: $ds quality plotting failed"

    # Cross-validation curves
    hatch run python manuscript/analysis/splat_count_vs_quality/plot_noise2self.py --dataset "$ds" 2>&1 || echo "  WARNING: $ds noise2self plotting failed"
done

# ──────────────────────────────────────────────────────────────────────
# 2. COMPRESSION & DENOISED COMPARISON FIGURES
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  2. Compression & denoised comparison figures               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

hatch run python manuscript/analysis/compression_comparison/plot_results.py 2>&1 || true
hatch run python manuscript/analysis/compression_comparison/plot_baselines.py 2>&1 || true
hatch run python manuscript/analysis/denoised_compression/plot_results.py 2>&1 || true
hatch run python manuscript/analysis/denoised_compression/plot_n2s_baselines.py 2>&1 || true

# ──────────────────────────────────────────────────────────────────────
# 3. PREPRINT COMPOSITE FIGURES (Fig 1-3, table, supp)
#    These scripts read TSVs only — no computation.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3. Preprint composite figures                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

hatch run python manuscript/analysis/progressive_vs_single/plot_summary_figure.py 2>&1 || true
hatch run python manuscript/preprint/scripts/generate_figures.py 2>&1
hatch run python manuscript/preprint/scripts/generate_fig1_code.py 2>&1 || true
hatch run python manuscript/preprint/scripts/generate_fig3_combined.py 2>&1 || true

# Visual comparison: uses cached .npz slices (no GPU).
# Falls back gracefully if slices don't exist yet.
hatch run python manuscript/preprint/scripts/generate_visual_from_montages.py 2>&1 || {
    echo "  WARNING: visual_comparison.pdf not generated (needs .npz slices from analysis phase)"
}

# ──────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  FIGURE GENERATION COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo ""
echo "  Analysis figures:"
figure_count=$(find manuscript/analysis -name "fig_*.pdf" 2>/dev/null | wc -l | tr -d ' ')
echo "    $figure_count PDFs in manuscript/analysis/"
echo ""
echo "  Preprint figures:"
for pdf in manuscript/preprint/figs/**/*.pdf; do
    if [ -f "$pdf" ]; then
        echo "    ${pdf#manuscript/preprint/}"
    fi
done
echo ""

# Check for known missing figures
missing=0
for expected in overview/overview.pdf visual_comparison/visual_comparison.pdf quantitative_analysis/quantitative_analysis.pdf suppfig/progressive.pdf; do
    if [ ! -f "manuscript/preprint/figs/$expected" ]; then
        echo "  MISSING: figs/$expected"
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo ""
    echo "  $missing figure(s) missing — these may require .npz slices from the GPU analysis phase."
fi
echo "================================================================"
