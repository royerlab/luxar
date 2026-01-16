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
wasm-pack build \
  --target web \
  --out-dir ../../../public/wasm \
  ${BUILD_MODE} \
  --scope luxar

echo "✅ WASM module built successfully!"
echo "📁 Output: packages/luxar-viewer/public/wasm/"
echo ""
echo "Generated files:"
ls -lh ../../../public/wasm/
echo ""
