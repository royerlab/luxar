#!/bin/bash
# Master build script: regenerates all manuscript content from scratch.
#
# Runs four phases in order:
#   1. Analysis computations (~6h with GPU)  — manuscript/analysis/run_all.sh
#   2. Figure generation (from TSVs)         — manuscript/build_figures.sh
#   3. Supplementary document PDFs           — manuscript/build_supp_docs.sh
#   4. Main preprint PDF                     — manuscript/build_preprint.sh
#
# Each phase can be run independently — see the individual scripts.
# If analysis TSVs already exist, skip phase 1 and start from phase 2.
#
# Usage:
#   bash manuscript/build_all.sh
#   bash manuscript/build_all.sh 2>&1 | tee manuscript/build_all.log

set -e
cd "$(git rev-parse --show-toplevel)"

echo "================================================================"
echo "  MANUSCRIPT BUILD — FULL REGENERATION"
echo "  Started: $(date -Iseconds)"
echo "================================================================"
echo ""

# ──────────────────────────────────────────────────────────────────────
# 1. RUN ALL ANALYSES (rate-distortion, cross-validation, progressive, convergence)
# ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  1. ANALYSES                                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash manuscript/analysis/run_all.sh

# ──────────────────────────────────────────────────────────────────────
# 2. GENERATE ALL FIGURES (from TSVs → PDF figures)
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  2. FIGURES                                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash manuscript/build_figures.sh

# ──────────────────────────────────────────────────────────────────────
# 3. BUILD ALL SUPPLEMENTARY DOCUMENTS
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  3. SUPPLEMENTARY DOCUMENTS                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash manuscript/build_supp_docs.sh

# ──────────────────────────────────────────────────────────────────────
# 4. BUILD MAIN PAPER
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  4. MAIN PAPER                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash manuscript/build_preprint.sh

# ──────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  MANUSCRIPT BUILD COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo ""
echo "  Generated PDFs:"
for pdf in manuscript/preprint/*.pdf manuscript/supp_doc/*/*.pdf; do
    if [ -f "$pdf" ]; then
        size=$(du -h "$pdf" | cut -f1)
        echo "    $pdf ($size)"
    fi
done
echo "================================================================"
