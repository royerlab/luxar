#!/bin/bash
# Master build script: regenerates all manuscript content from scratch.
#
# Delegates to:
#   1. manuscript/analysis/run_all.sh — runs all analyses + builds supp PDFs inline
#   2. manuscript/supp_doc/*/build.py  — (re)builds each supplementary document
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
# 1. RUN ALL ANALYSES (rate-distortion, N2S, progressive, convergence)
#    This also builds the supp PDFs inline after each analysis phase.
# ──────────────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  1. ANALYSES                                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash manuscript/analysis/run_all.sh

# ──────────────────────────────────────────────────────────────────────
# 2. (RE)BUILD ALL SUPPLEMENTARY DOCUMENTS
#    Even though run_all.sh builds them inline, this ensures they're
#    up to date if any figures were regenerated after the analysis.
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  2. SUPPLEMENTARY DOCUMENTS                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"

for build_script in manuscript/supp_doc/*/build.py; do
    if [ -f "$build_script" ]; then
        echo ""
        echo "--- $(dirname "$build_script" | xargs basename) ---"
        hatch run python "$build_script"
    fi
done

# ──────────────────────────────────────────────────────────────────────
# 3. BUILD MAIN PAPER (if it exists)
# ──────────────────────────────────────────────────────────────────────
MAIN_TEX="manuscript/preprint/luxar_preprint.tex"
if [ -f "$MAIN_TEX" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  3. MAIN PAPER                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    PAPER_DIR=$(dirname "$MAIN_TEX")
    PAPER_NAME=$(basename "$MAIN_TEX" .tex)
    cd "$PAPER_DIR"
    pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" > /dev/null 2>&1
    bibtex "$PAPER_NAME" > /dev/null 2>&1 || true
    pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" > /dev/null 2>&1
    pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" > /dev/null 2>&1
    cd "$(git rev-parse --show-toplevel)"
    if [ -f "$PAPER_DIR/$PAPER_NAME.pdf" ]; then
        echo "  Built: $PAPER_DIR/$PAPER_NAME.pdf"
    else
        echo "  WARNING: $PAPER_DIR/$PAPER_NAME.pdf not produced"
    fi
else
    echo ""
    echo "  (Main paper not found at $MAIN_TEX — skipping)"
fi

# ──────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  MANUSCRIPT BUILD COMPLETE"
echo "  Finished: $(date -Iseconds)"
echo ""
echo "  Generated supplementary documents:"
for pdf in manuscript/supp_doc/*/*.pdf; do
    if [ -f "$pdf" ]; then
        size=$(du -h "$pdf" | cut -f1)
        echo "    $pdf ($size)"
    fi
done
echo "================================================================"
