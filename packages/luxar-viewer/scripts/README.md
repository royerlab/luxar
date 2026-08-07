# Viewer Build & Quality Scripts

Developer scripts that sit outside `src/` — invoked by `pnpm` commands, CI,
and humans pasting perf tables into commit bodies. They compile the Rust→WASM
module, sanity-check the embeddable library bundle, ratchet TypeDoc warnings,
audit JSDoc coverage, keep the pnpm security pins single-sourced, and diff
perf-bench JSON captures into a Markdown table.

## Contents

```
scripts/
├── build-wasm.sh             # Rust → WASM build via wasm-pack (pnpm build:wasm[:dev])
├── check-lib-exports.mjs          # Post-build sanity check on dist/lib/
├── check-jsdoc-coverage.ts        # Standalone JSDoc coverage report
├── check-overrides.mjs            # pnpm overrides single-source guard
├── check-typedoc-warnings.mjs     # Baseline-driven TypeDoc warning ratchet
├── check-typedoc-warnings-tests.mjs # node:test coverage for the warning ratchet
├── perf-diff.mjs                  # Markdown delta table from perf-bench JSON
├── perf-diff.test.mjs             # tests for perf-diff's buildPerfDiff()
└── perf/                          # perf-bench capture fixtures / helpers
```

## Build

### `build-wasm.sh`

Compiles `src/wasm/rust/` to WebAssembly with `wasm-pack` and writes the
generated ES-module glue + `.wasm` to `public/wasm/` so Vite can serve it.
Sources `~/.cargo/env` if present, fails fast with an `make install-rust`
hint when `wasm-pack` or `rustc` is missing, and removes the `.gitignore`
that wasm-pack drops in the output dir (otherwise it would ride into the
Vite build and ultimately the Python wheel).

```bash
pnpm build:wasm        # → bash scripts/build-wasm.sh         (release)
pnpm build:wasm:dev    # → bash scripts/build-wasm.sh --dev   (fast compile, unoptimized)
```

The script is also runnable directly: `./scripts/build-wasm.sh [--dev]`.

### `check-lib-exports.mjs`

Post-build smoke test for `pnpm build:lib` — verifies the embeddable
library output in `dist/lib/` actually satisfies the contract documented
in the parent README's "Embedding" section. Checks:

1. The expected files exist (`luxar-viewer.js`, `luxar-viewer.css`,
   `types/index.d.ts`).
2. The JS bundle re-exports the public symbols — `LuxarApp`,
   `bootstrapStandalone`, `readUrlParams`, `StorageKeys`.
3. Dynamic-importing the bundle does **not** patch the host `console.log`
   (the side-effect-free guarantee from `src/index.ts`).
4. `three` is externalized: the bundle text must not contain a
   `class WebGLRenderer` definition. A bundled `three` would be a
   multi-MB regression and break peer-dep semantics.

Exits non-zero on any failure with a per-issue diagnostic.

```bash
pnpm build:lib:check   # → node scripts/check-lib-exports.mjs
```

## Quality

### `check-typedoc-warnings.mjs`

Runs TypeDoc conversion and validation without emitting HTML, normalizes
checkout-specific paths, and compares the warning multiset with
`../typedoc-warnings-baseline.json`. New warnings and increased duplicate
counts fail; removed warnings are reported so the baseline can be tightened.
TypeDoc conversion/compiler errors always fail independently of the baseline.

```bash
pnpm run typedoc:check-warnings
pnpm --silent run typedoc:check-warnings -- --json
pnpm run typedoc:check-warnings -- --update-baseline
pnpm run test:typedoc-warnings
```

The warning baseline is an explicit reviewed allowlist, not a target count:
the checker compares normalized warning messages as well as the total.

### `check-jsdoc-coverage.ts`

Walks `src/**/*.ts` (excluding `*.test.ts`, `*.spec.ts`, `__tests__/`) and
counts exported `function | class | interface | type | const | enum`
declarations preceded within 15 lines by a `/**` block. Reports overall
coverage, per-file failures below the threshold, and a per-package
breakdown (grouped by top-level dir under `src/`). Exits non-zero when
overall coverage is under the threshold.

This remains a manual package-wide estimator. The required PR gate uses the
file-level baseline policy in `scripts/check_documentation.py`; do not add this
second percentage policy to CI.

```bash
npx tsx scripts/check-jsdoc-coverage.ts                       # default threshold 70%
npx tsx scripts/check-jsdoc-coverage.ts --threshold=80 --verbose
```

Flags:

- `--threshold=<N>` — minimum overall coverage % to pass (default `70`).
- `--verbose` — also list the top 10 fully-documented files.

### `check-overrides.mjs`

Keeps the pnpm security-advisory pins single-sourced. They may legally live
in `pnpm-workspace.yaml` **or** in `package.json`'s `pnpm.overrides`, and
when both exist the two silently disagree, because their consumers read
different files: pnpm (any version that can read `pnpm-workspace.yaml` at
all, i.e. ≥10.6) prefers `package.json`, while Dependabot's updater reads
`pnpm-workspace.yaml`. The result is a lockfile that fails every
`pnpm install --frozen-lockfile` with `ERR_PNPM_LOCKFILE_CONFIG_MISMATCH`,
plus one pin set that is not actually applied. Checks:

1. `package.json` declares no `pnpm.overrides` — `pnpm-workspace.yaml` owns them.
2. `pnpm-lock.yaml`'s recorded `overrides` match `pnpm-workspace.yaml`'s,
   printing both blocks side by side rather than pnpm's opaque error code.
3. At least one pin exists, so the block cannot vanish unnoticed (checks 1
   and 2 both hold trivially at zero, and `pnpm audit` is
   `continue-on-error` in CI, so nothing else would catch it).

Exits non-zero on any failure. Reads only the three files — no
`node_modules` — so CI runs it _before_ `pnpm install --frozen-lockfile`,
which would otherwise abort first and hide the explanation.

```bash
pnpm check:overrides   # → node scripts/check-overrides.mjs   (also inside pnpm check:ci)
```

### `perf-diff.mjs`

Takes two JSON files produced by `src/tests/e2e/line-perf-bench.spec.ts`
or `src/tests/e2e/gsplat-perf-bench.spec.ts` and prints a Markdown delta
table suitable for pasting into a commit body or PR description. Joins
scenarios on `${scenarioId}/${backend}`, marks NEW / DROPPED rows when a
key only appears on one side, and tags any row that moves ≥5% with 🟢
(faster) or 🔴 (slower). Emits a separate GPU-time table when at least
one scenario has `gpu.supported === true`.

Beyond the JS frame-timing and GPU-time tables, it emits additional
sections for the gsplat-bench campaign metrics — each shown ONLY when at
least one scenario (on either side) carries the field (lower is better
throughout; a missing/null metric renders `—`, never a phantom `0.0%`):

- **Depth-sort stages** — per scenario, the worker-stage medians
  (`depthSort.kernelMsMedian` / `queueMsMedian` / `boundaryMsMedian`,
  all optional) and end-to-end sort latency
  (`depthSort.sortLatencyMedianMs` / `sortLatencyP95Ms`, may be null).
  Only the columns some scenario actually carries are shown.
- **L8 sort-tail (p99)** — `sortAdjacentP99Ms` and `idleOrbitP99Ms`
  (top-level; present on the 10M scenario only, may be null).
- **Ladder load** — `ladder.wallMsToLadderComplete` (visible-human
  ladder scenario only). When `ladder.observedGrowth` is `false` the
  wall time is a lower bound, not a measurement: the cell is marked
  `(lb)` and no delta is computed for that row.

The markdown-building logic is the exported pure function
`buildPerfDiff(base, next)` (unit-tested in `perf-diff.test.mjs`); all
CLI behaviour runs inside `main()`, which only executes when the script
is invoked directly (standard ESM main-module guard), so the module can
be imported without side effects.

The `api` column appends ` (sw)` when either side of a row ran on a
software rasterizer (`softwareRenderer`, gsplat bench only) — the bench
marks such runs as not comparable to a GPU run, so every delta in that
row must be discounted.

The `api` column appends ` (webgl-bk)` when a WebGPURenderer run fell
back to its internal WebGL2 backend — otherwise that fallback would look
like a clean `webgpu` row.

```bash
node scripts/perf-diff.mjs baseline.json new.json > diff.md
```

Both arguments are required; the script also echoes the absolute paths
it consumed to stderr so you can spot a stale baseline.
