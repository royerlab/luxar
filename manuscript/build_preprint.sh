#!/bin/bash
# Build the main preprint PDF from LaTeX sources.
#
# Assumes figures already exist in preprint/figs/.
# Does NOT run analyses or build supplementary documents.
#
# Usage:
#   bash manuscript/build_preprint.sh

set -e
cd "$(git rev-parse --show-toplevel)"

MAIN_TEX="manuscript/preprint/luxar_preprint.tex"

if [ ! -f "$MAIN_TEX" ]; then
    echo "ERROR: $MAIN_TEX not found"
    exit 1
fi

PAPER_DIR=$(dirname "$MAIN_TEX")
PAPER_NAME=$(basename "$MAIN_TEX" .tex)

echo "Building: $PAPER_DIR/$PAPER_NAME.pdf"
cd "$PAPER_DIR"

BUILD_LOG="$PAPER_NAME.build.log"
rm -f "$BUILD_LOG"

pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" >> "$BUILD_LOG" 2>&1 || true
bibtex "$PAPER_NAME" >> "$BUILD_LOG" 2>&1 || true
pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" >> "$BUILD_LOG" 2>&1 || true
pdflatex -interaction=nonstopmode "$PAPER_NAME.tex" >> "$BUILD_LOG" 2>&1 || true

if [ -f "$PAPER_NAME.pdf" ]; then
    size=$(du -h "$PAPER_NAME.pdf" | cut -f1)
    echo "  Built: $PAPER_NAME.pdf ($size)"
    warnings=$(grep -c "Warning\|Error" "$BUILD_LOG" 2>/dev/null || true)
    warnings=${warnings:-0}
    if [ "$warnings" -gt 0 ]; then
        echo "  $warnings warnings/errors — see $BUILD_LOG for details"
    fi
else
    echo "  ERROR: $PAPER_NAME.pdf not produced — see $BUILD_LOG for details"
    exit 1
fi
