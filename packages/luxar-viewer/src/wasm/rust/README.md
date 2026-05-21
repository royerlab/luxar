# WASM Rust Source

Rust source code for the Luxar WASM module. See the parent [wasm/README.md](../README.md) for build instructions, architecture, and usage details.

## Contents

- `Cargo.toml` — `luxar-wasm` crate manifest (cdylib + rlib, `wasm-bindgen 0.2`, release profile tuned for size and speed: `opt-level=3`, `lto=true`, `codegen-units=1`, `panic="abort"`, `strip=true`; `wasm-opt` flags enable SIMD, bulk-memory, non-trapping float-to-int, and sign-ext).
- `Cargo.lock` — Pinned dependency graph (committed for reproducible WASM builds).
- `.cargo/config.toml` — Sets `target-feature=+simd128` for `wasm32-unknown-unknown` so the per-kernel modules can rely on `core::arch::wasm32` SIMD intrinsics.
- `.gitignore` — Ignores `target/`, `pkg/` (wasm-pack output goes to `public/wasm/`), and editor backups.
- `LICENSE` — Symlink to the repository root `LICENSE` (BSD-3-Clause).
- `src/` — Per-kernel Rust source modules (see Subpackages).
- `target/` — Cargo build artifacts (gitignored).

## Build

This crate is built via `wasm-pack` from the parent `wasm/` folder; see [../README.md](../README.md). It is not consumed as a standalone Rust crate.

## Subpackages

- [src](./src/README.md) — Kernel implementations (spatial queries, nD visibility, projection, decoding, gsplats/lines/points helpers) plus the `wasm-bindgen` entrypoints in `lib.rs`.
