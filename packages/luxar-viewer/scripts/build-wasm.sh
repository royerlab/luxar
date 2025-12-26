#!/bin/bash
# Build Luxar WASM module using wasm-pack
#
# Phase 3: WASM Acceleration
# This script compiles the Rust code to WebAssembly and generates JavaScript bindings

set -e  # Exit on error

echo "🦀 Building Luxar WASM module..."

# Navigate to WASM source directory
cd "$(dirname "$0")/../src/workers/wasm"

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ Error: wasm-pack is not installed"
    echo "Install with: cargo install wasm-pack"
    echo "Or visit: https://rustwasm.github.io/wasm-pack/installer/"
    exit 1
fi

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "❌ Error: Rust is not installed"
    echo "Install from: https://rustup.rs/"
    exit 1
fi

echo "📦 Running wasm-pack build..."

# Build with wasm-pack
# - target web: For browser ES modules
# - out-dir: Output to public/wasm for Vite to serve
# - release: Optimized build
wasm-pack build \
  --target web \
  --out-dir ../../../public/wasm \
  --release \
  --scope luxar

echo "✅ WASM module built successfully!"
echo "📁 Output: packages/luxar-viewer/public/wasm/"
echo ""
echo "Generated files:"
ls -lh ../../../public/wasm/
echo ""
echo "🎯 WASM module ready for Phase 3 integration"
