#!/bin/bash
# Force-refit every gsplat demo with the current MAX_SPLATS / SEEDS_PER_TILE
# values (after running calibrate_gsplat_demos.py + update_demo_max_splats.py).
#
# Usage:
#   bash scripts/refit_gsplat_demos.sh                 # foreground, sequential
#   nohup bash scripts/refit_gsplat_demos.sh > /tmp/refit.log 2>&1 &  # background

set -eu
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/luxar-refit-${USER:-$(id -u)}.XXXXXX")"

echo "=========================================="
echo "$(date): Starting gsplat demo refit sequence"
echo "Logs under: $LOGDIR/"
echo "=========================================="

# Demos in dependency-friendly order:
#   - small/cached datasets first (fast feedback)
#   - large downloads last (in case of issues)
DEMOS_DIR=packages/luxar/src/luxar/demos
declare -a DEMOS=(
    "organoid_dapi:$DEMOS_DIR/demo_gsplats_3d_organoid_dapi_nuclei.py"
    "cells3d_multichannel:$DEMOS_DIR/demo_gsplats_3d_cells3d_multichannel.py"
    "kidney_layers:$DEMOS_DIR/demo_gsplats_3d_kidney_multichannel_layers.py"
    "kidney_toggles:$DEMOS_DIR/demo_gsplats_3d_kidney_multichannel_toggles.py"
    "organoid_multi:$DEMOS_DIR/demo_gsplats_3d_organoid_multichannel.py"
    "opencell_map4:$DEMOS_DIR/demo_gsplats_3d_opencell_map4.py"
    "tribolium_embryo:$DEMOS_DIR/demo_gsplats_3d_tribolium_embryo.py"
    "acto3d_heart:$DEMOS_DIR/demo_gsplats_3d_acto3d_heart.py"
    "celegans_4d:$DEMOS_DIR/demo_gsplats_4d_celegans_tracking.py"
    "zebrafish_4d:$DEMOS_DIR/demo_gsplats_4d_zebrafish_timelapse.py"
    "cmu1_pathology:$DEMOS_DIR/demo_gsplats_2d_cmu1_pathology.py"
    "codex_pancreas:$DEMOS_DIR/demo_gsplats_2d_codex_pancreas.py"
)

OK_NAMES=()
FAIL_NAMES=()
for entry in "${DEMOS[@]}"; do
    name="${entry%%:*}"
    script="${entry##*:}"
    log="$LOGDIR/$name.log"

    echo
    echo "--- $name ($(date)) ---"
    if hatch run python "$script" --no-napari --no-serve --recompute > "$log" 2>&1; then
        echo "$(date): OK $name (log: $log)"
        OK_NAMES+=("$name")
    else
        echo "$(date): FAILED $name (log: $log)"
        tail -n 20 "$log" || true
        FAIL_NAMES+=("$name")
    fi
done

echo
echo "=========================================="
echo "$(date): Refit run complete"
echo "  OK     (${#OK_NAMES[@]}): ${OK_NAMES[*]:-(none)}"
echo "  FAILED (${#FAIL_NAMES[@]}): ${FAIL_NAMES[*]:-(none)}"
echo "=========================================="
