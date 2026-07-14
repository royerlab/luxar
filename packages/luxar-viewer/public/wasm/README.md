# public/wasm

Build output directory for the compiled WASM module. `make build-wasm` (or
`pnpm build:wasm` → `scripts/build-wasm.sh`) writes `luxar_wasm.js`,
`luxar_wasm_bg.wasm`, and `luxar_wasm.d.ts` here so the Vite dev server can
serve them. The directory is gitignored — never commit artifacts here.
See `src/wasm/README.md` for build instructions and the integration story.
