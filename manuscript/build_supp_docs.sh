#!/bin/bash
# Build all supplementary document PDFs from existing analysis results.
#
# Assumes analysis results already exist (run analysis/run_all.sh first).
# Does NOT run analyses or build the main preprint.
#
# Usage:
#   bash manuscript/build_supp_docs.sh            # Build all
#   bash manuscript/build_supp_docs.sh convergence # Build one by name

set -e
cd "$(git rev-parse --show-toplevel)"

FILTER="${1:-}"
built=0
skipped=0

echo "Building supplementary documents..."
echo ""

for build_script in manuscript/supp_doc/*/build.py; do
    if [ ! -f "$build_script" ]; then
        continue
    fi

    doc_name=$(dirname "$build_script" | xargs basename)

    # If a filter was given, skip non-matching documents
    if [ -n "$FILTER" ] && [ "$doc_name" != "$FILTER" ]; then
        skipped=$((skipped + 1))
        continue
    fi

    echo "--- $doc_name ---"
    hatch run python "$build_script"
    echo ""
    built=$((built + 1))
done

if [ "$built" -eq 0 ] && [ -n "$FILTER" ]; then
    echo "ERROR: No supplementary document matching '$FILTER'"
    echo "Available:"
    for build_script in manuscript/supp_doc/*/build.py; do
        echo "  $(dirname "$build_script" | xargs basename)"
    done
    exit 1
fi

echo "Built $built supplementary document(s)."

# List generated PDFs
for pdf in manuscript/supp_doc/*/*.pdf; do
    if [ -f "$pdf" ]; then
        size=$(du -h "$pdf" | cut -f1)
        echo "  $pdf ($size)"
    fi
done
