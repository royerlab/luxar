#!/bin/bash
# Sequential demo recompute — forces re-fitting from scratch.
# Usage: nohup bash scripts/run_demo_recompute.sh > /tmp/demo_recompute.log 2>&1 &

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=========================================="
echo "$(date): Starting demo recompute sequence"
echo "=========================================="

run_demo() {
    local name="$1"
    local script="$2"
    echo ""
    echo "--- $name ---"
    echo "$(date): Starting $name"
    if ! hatch run python "$script" --no-napari --no-serve --recompute 2>&1; then
        echo "FAILED: $name"
    fi
    echo "$(date): Finished $name"
}

DEMOS=packages/luxar/src/luxar/demos

run_demo "kidney_layers"    "$DEMOS/demo_gsplats_3d_kidney_multichannel_layers.py"
run_demo "kidney_toggles"   "$DEMOS/demo_gsplats_3d_kidney_multichannel_toggles.py"
run_demo "zebrafish_4d"     "$DEMOS/demo_gsplats_4d_zebrafish_timelapse.py"
run_demo "organoid_dapi"    "$DEMOS/demo_gsplats_3d_organoid_dapi_nuclei.py"
run_demo "opencell_map4"    "$DEMOS/demo_gsplats_3d_opencell_map4.py"
run_demo "organoid_multi"   "$DEMOS/demo_gsplats_3d_organoid_multichannel.py"
run_demo "acto3d_heart"     "$DEMOS/demo_gsplats_3d_acto3d_heart.py"
run_demo "cells3d_multi"    "$DEMOS/demo_gsplats_3d_cells3d_multichannel.py"
run_demo "cells3d_toggles"  "$DEMOS/demo_gsplats_3d_cells3d_multichannel_toggles.py"
run_demo "tribolium_embryo" "$DEMOS/demo_gsplats_3d_tribolium_embryo.py"

echo ""
echo "=========================================="
echo "$(date): All demos complete"
echo "=========================================="
