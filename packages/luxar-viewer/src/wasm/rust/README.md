# luxar-viewer/src/wasm/rust

Rust source for the Luxar WASM compute kernels (crate `luxar-wasm`): nD
visibility, projection, line clipping, Gaussian-splat processing, and array
decoding. Compiled to WebAssembly with `wasm-pack`/`wasm-bindgen` and loaded
by the viewer's data workers; mirrored function-for-function by the pure
TypeScript implementation in [`../typescript/`](../typescript/).

## Directory Layout

```
rust/
├── Cargo.toml           — crate manifest (cdylib + rlib, wasm-bindgen; release
│                          profile: opt-level 3, LTO, panic = "abort", strip)
├── Cargo.lock           — committed lockfile (build enforces it via --locked)
├── rust-toolchain.toml  — pinned toolchain (1.92.0 + wasm32-unknown-unknown)
└── src/                 — kernel modules (see src/README.md for the full
                           module map, buffer ABI, and optimization notes)
    ├── lib.rs                — crate root: module declarations + WASM re-exports
    ├── common.rs             — shared constants, validate_ndim, packed_index
    ├── lines_clipping.rs     — Liang-Barsky nD clipping + attribute interpolation
    ├── mesh_culling.rs       — whole-triangle nD slab culling for indexed surfaces
    ├── gsplats_processing.rs — Mahalanobis / marginal Cholesky / attenuation
    ├── effective_radii.rs    — radius shrinkage when slicing through hidden dims
    ├── projection.rs         — 3D extraction through display_dims
    ├── depth_sort.rs         — back-to-front splat ordering (depth-sorting Phase 2)
    └── decode.rs             — quantized / log / LUT / broadcast decoders
```

## Build & Test

From the project root:

```bash
make build-wasm    # wasm-pack build → packages/luxar-viewer/public/wasm/
make test-wasm     # native Rust unit tests (cargo test)
```

Or from `packages/luxar-viewer/`:

```bash
pnpm build:wasm        # bash scripts/build-wasm.sh (release)
pnpm build:wasm:dev    # bash scripts/build-wasm.sh --dev (fast compile, unoptimized)
pnpm test:wasm         # cd src/wasm/rust && cargo test
```

`scripts/build-wasm.sh` runs `wasm-pack build --target web --out-dir
../../../public/wasm --release --scope luxar -- --locked` (`--dev` replaces
`--release` in dev mode) and
writes the ES-module glue (`luxar_wasm.js`), the binary
(`luxar_wasm_bg.wasm`), and TypeScript definitions to `public/wasm/` so Vite
can serve them. The toolchain is pinned by `rust-toolchain.toml` for
reproducible codegen; run `make install-rust` if `rustc`/`wasm-pack` are
missing.

## 16-Dimension Limit & panic = "abort"

Kernels allocate fixed-size stack arrays (`[f32; 16]`) for speed, so every
dimension-sensitive function is guarded by `validate_ndim` (in
`src/common.rs`), which **panics** above `MAX_SUPPORTED_DIMS = 16`. The
release profile sets `panic = "abort"`, so such a panic kills the WASM
instance — these kernels must never be called above 16D. In practice they
aren't: `pickBackend(ctx, ndim)` in `src/workers/data-worker/state.ts`
routes every `ndim > 16` operation to the TypeScript backend instead, making
TypeScript the (uncapped, slower) production path for >16D data.

## See also

- [`../README.md`](../README.md) — integration story: `WasmModule`
  interface, `initWasm` loader, fallback behavior, and the full API listing.
- [`src/README.md`](./src/README.md) — detailed per-module kernel docs,
  buffer ABI, and hot-loop optimization patterns.
- [`../typescript/README.md`](../typescript/README.md) — the 1:1 TypeScript
  reference implementation and parity contract.
