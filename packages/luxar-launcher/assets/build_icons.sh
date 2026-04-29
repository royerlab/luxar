#!/usr/bin/env bash
# Generate AppIcon.icns from luxar-logo.png using macOS native tools.
#
# Outputs:
#   luxar-launcher/assets/AppIcon.icns          — packed icon for macOS .app
#
# Requires: macOS (uses /usr/bin/sips and /usr/bin/iconutil).
# Re-run only when the source PNG changes; the icns is committed to the repo.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Both source and output live in the package-bundled asset dir so the
# rendered files ship with `pip install luxar`.
ASSETS_DIR="${SCRIPT_DIR}/../../luxar/src/luxar/cli/_launcher_assets"
SRC_PNG="${ASSETS_DIR}/luxar-logo.png"
OUT_ICNS="${ASSETS_DIR}/AppIcon.icns"

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "build_icons.sh: macOS-only (need iconutil); skipping on $(uname -s)" >&2
    exit 0
fi
if [[ ! -f "$SRC_PNG" ]]; then
    echo "build_icons.sh: missing $SRC_PNG; run build_logo.py first" >&2
    exit 1
fi
mkdir -p "$ASSETS_DIR"

ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
trap 'rm -rf "$(dirname "$ICONSET")"' EXIT

# macOS .icns expects exactly these (size, @1x/@2x) variants.
declare -a SIZES=(
    "16:icon_16x16.png"
    "32:icon_16x16@2x.png"
    "32:icon_32x32.png"
    "64:icon_32x32@2x.png"
    "128:icon_128x128.png"
    "256:icon_128x128@2x.png"
    "256:icon_256x256.png"
    "512:icon_256x256@2x.png"
    "512:icon_512x512.png"
    "1024:icon_512x512@2x.png"
)

for entry in "${SIZES[@]}"; do
    size="${entry%%:*}"
    name="${entry##*:}"
    /usr/bin/sips -z "$size" "$size" "$SRC_PNG" --out "$ICONSET/$name" >/dev/null
done

/usr/bin/iconutil -c icns "$ICONSET" -o "$OUT_ICNS"
echo "Wrote $OUT_ICNS"
