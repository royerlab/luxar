#!/bin/bash
# Build Luxar WASM module using wasm-pack
#
# This script compiles the Rust code to WebAssembly and generates JavaScript bindings
#
# Usage:
#   ./build-wasm.sh          # Production build (optimized)
#   ./build-wasm.sh --dev    # Development build (faster compilation)

set -e  # Exit on error

# Source cargo environment if available (needed for wasm-pack and rustc)
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

# Parse arguments
BUILD_MODE="--release"
MODE_DESC="production (optimized)"

if [[ "$1" == "--dev" ]]; then
    BUILD_MODE="--dev"
    MODE_DESC="development (faster compilation)"
fi

echo "🦀 Building Luxar WASM module (${MODE_DESC})..."

# Navigate to WASM source directory
cd "$(dirname "$0")/../src/wasm/rust"

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ Error: wasm-pack is not installed"
    echo ""
    echo "To install Rust and wasm-pack, run from the project root:"
    echo "  make install-rust"
    echo ""
    echo "Or install manually:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "  source ~/.cargo/env"
    echo "  cargo install wasm-pack"
    exit 1
fi

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Error: Rust is not installed"
    echo ""
    echo "To install Rust, run from the project root:"
    echo "  make install-rust"
    echo ""
    echo "Or install manually from: https://rustup.rs/"
    exit 1
fi

echo "📦 Running wasm-pack build..."

# Build with wasm-pack
# - target web: For browser ES modules
# - out-dir: Output to public/wasm for Vite to serve
# - release/dev: Build mode based on flag
build_wasm_once() {
  # `-- --locked` forwards to cargo so the committed Cargo.lock is enforced
  # (reproducible WASM: fail instead of silently resolving newer transitive deps).
  wasm-pack build \
    --target web \
    --out-dir ../../../public/wasm \
    ${BUILD_MODE} \
    --scope luxar \
    -- --locked
}

# wasm-pack downloads its own wasm-opt (binaryen) binary and caches it under
# ~/.cache/.wasm-pack. That download is occasionally corrupt on CI runners,
# producing a non-deterministic `wasm-opt: invalid global index` parse error
# even though the Rust source and pinned wasm-bindgen are unchanged (the same
# wasm-pack version both passes and fails — see CI history). Retry a few
# times, clearing the wasm-opt cache between attempts so a corrupt binary is
# re-downloaded rather than reused. `until` is used so `set -e` doesn't abort
# on the first failed attempt.
attempt=1
max_attempts=3
until build_wasm_once; do
  if [ "${attempt}" -ge "${max_attempts}" ]; then
    echo "❌ wasm-pack build failed after ${max_attempts} attempts" >&2
    exit 1
  fi
  echo "⚠️  wasm-pack build attempt ${attempt} failed (likely a flaky wasm-opt download); clearing cache and retrying..." >&2
  rm -rf "${HOME}/.cache/.wasm-pack" 2>/dev/null || true
  attempt=$((attempt + 1))
  sleep 5
done

# wasm-pack emits a .gitignore in its output directory. Drop it so it does
# not ride into the Vite build output and ultimately the Python wheel.
rm -f ../../../public/wasm/.gitignore

echo "✅ WASM module built successfully!"
echo "📁 Output: packages/luxar-viewer/public/wasm/"
echo ""
echo "Generated files:"
ls -lh ../../../public/wasm/
echo ""
