# Viewer Build & Quality Scripts

Developer scripts that sit outside `src/` — invoked by `pnpm` commands, CI,
and humans pasting perf tables into commit bodies. They compile the Rust→WASM
module, sanity-check the embeddable library bundle, ratchet TypeDoc warnings,
audit JSDoc coverage, keep the pnpm security pins single-sourced, and diff
perf-bench JSON captures into a Markdown table.

## Contents

```
scripts/
├── ab-webgpu-vs-webgl.mjs         # Real-WebGPU vs WebGL screenshot A/B (SSIM/NCC), the
│                                  #   MESH_PHYSICAL_MATERIALS_SPEC §3.6 acceptance test
├── bake-env.mjs                   # Headless driver for `luxar env bake`: opens ?bakeEnv,
│                                  #   waits for the capture, writes the .env.bin container
├── build-wasm.sh             # Rust → WASM build via wasm-pack (pnpm build:wasm[:dev])
├── check-build-identity.mjs       # Asserts a built bundle carries its build stamp
├── check-build-identity.test.mjs  # unit coverage for the stamp gate
├── check-coverage-slack.mjs       # Coverage floor/baseline drift gate and refresh printer
├── check-lib-exports.mjs          # Post-build sanity check on dist/lib/
├── check-node-types-version.mjs   # Node runtime/@types major-version declaration guard
├── check-node-types-version.test.mjs # unit coverage for the Node types guard
├── check-three-types-version.mjs  # Three runtime/types/embed minor-version declaration guard
├── check-three-types-version.test.mjs # unit coverage for the Three types guard
├── check-jsdom-version.mjs        # jsdom Blob-bridge pin and Dependabot policy guard
├── check-jsdom-version.test.mjs   # unit coverage for the jsdom version guard
├── public-api-exports.json        # The public barrel's value-export list (shared with the barrel unit test)
├── check-jsdoc-coverage.ts        # Standalone JSDoc coverage report
├── check-overrides.mjs            # pnpm overrides single-source guard
├── check-typedoc-warnings.mjs     # Baseline-driven TypeDoc warning ratchet
├── check-typedoc-warnings-tests.mjs # node:test coverage for the warning ratchet
├── generate-third-party-licenses.mjs # Build redistributed dependency notices
├── generate-third-party-licenses.test.mjs # Tests for complete notices
├── generate-lod-visual-ab-fixture.py # Deterministic LOD visual A/B fixture
├── lod-visual-ab.mjs              # Opt-in coarse-vs-finest rendered acceptance bench
├── lod-visual-ab-thresholds.json  # Per-bench visual thresholds and provenance
├── visual-ab-core.mjs             # Shared screenshot capture and image scoring
├── visual-ab-core.test.mjs        # Tests for shared visual scoring
├── perf-diff.mjs                  # Markdown delta table from perf-bench JSON
├── perf-diff.test.mjs             # tests for perf-diff's buildPerfDiff()
├── run-ci-checks.mjs              # Aggregate viewer static + coverage merge gates
└── perf/                          # perf-bench capture fixtures / helpers
```

## LOD visual A/B

`pnpm test:lod-visual-ab` generates a deterministic fixture, captures the
recorded coarse and finest levels at the same pinned camera with LOD fading and
stream-energy compensation disabled, and writes PNGs plus `summary.json` under
`test-results/lod-visual-ab/`. The command is deliberately opt-in and separate
from `test:perf:e2e`: it gates image similarity, not timing. Thresholds are
recorded for Linux Chromium, so the command refuses other platforms. Multi-bench
fixtures hide every non-active LOD group during capture so each score measures
only the named geometry.

Each `(geometry, blending mode)` bench owns its thresholds in
`lod-visual-ab-thresholds.json`: minimum SSIM and coarse/finest mean-luma ratio,
maximum mean CIE76 DeltaE, and maximum increase in full-resolution blown-pixel
fraction. The finest arm must also contain a recorded minimum fraction of blown
pixels so the clipping check cannot silently become inert. A threshold change
should include the old and new measured values in the PR description and refresh
the recorded provenance. The top-level `comparisons` array records cross-bench
margins; re-derive those margins alongside the per-bench values whenever the
fixture or thresholds change.

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
2. The JS bundle's runtime exports are EXACTLY the list in
   `public-api-exports.json` — the same list the source barrel is held to
   by `src/tests/unit/api/barrel-side-effects.test.ts`, so a lost export
   fails here and a leaked one fails in both places. `VIEWER_VERSION` must
   also equal `package.json`'s version (the `define` is silent when it
   goes wrong).
3. Dynamic-importing the bundle does **not** patch the host `console.log`
   (the side-effect-free guarantee from `src/index.ts`).
4. `three` is externalized: the bundle text must not contain a
   `class WebGLRenderer` definition. A bundled `three` would be a
   multi-MB regression and break peer-dep semantics.
5. Every emitted chunk that spells out a bundle-relative WASM shim
   specifier can actually reach `dist/lib/wasm/luxar_wasm.js` from its own
   directory. The entry chunk sits at the output root and the worker
   chunks under `assets/`, so one relative literal cannot serve both
   (#1649), and getting it wrong is silent — the import 404s and the
   viewer drops to the TypeScript fallback. At least one specifier must
   appear somewhere in the build, since this is a text-level scan that
   would otherwise pass vacuously once the paths stop being literals.
   Which chunk carries them is deliberately not pinned: that is a
   code-splitting detail, and any placement is still verified by the
   per-chunk reachability check.

Exits non-zero on any failure with a per-issue diagnostic.

```bash
pnpm build:lib:check   # → check-lib-exports.mjs && check-build-identity.mjs dist/lib
```

### `check-build-identity.mjs`

Reads the emitted artifacts and fails when they do not carry a build stamp
that agrees with `package.json`'s version. `define:` substitution is silent
when it goes wrong — dropping the `define` block, renaming the constant or
adding a third Vite config that forgets it leaves every unit test passing
and ships an artifact with no revision on it. So the check scans the built
`.js` files (and, for the app build, the `<meta name="luxar-build">` tag in
`index.html`) rather than trusting the config. Both sides come from the
tree: expected version from `package.json`, observed from the artifact.

Escaping-insensitive by design — the minified app build emits the stamp as an
unescaped payload inside a template literal while the library build
backslash-escapes its quotes, so a grep written against one spelling reports a
false negative against the other. Covered by `check-build-identity.test.mjs`,
which runs with the viewer unit suite (`pnpm test --run`).

```bash
pnpm build:check       # → check-eager-chunks.mjs && check-build-identity.mjs dist
pnpm build:lib:check   # → check-lib-exports.mjs && check-build-identity.mjs dist/lib
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

A `⚠️ JS frame timing not comparable` line follows the frame-timing table
when a measured row's two sides disagree on whether they carry
`excludedResolveIntervals`. The line bench drops the frame interval after
each GPU-timestamp resolve (readback latency, not scene work) and records
the count in that field; because the field postdates the exclusion, its
one-sided absence means the older side's frame stats still include that
latency wherever that run resolved timestamps, and the row's deltas are
instrument drift rather than a rendering change. The test is
presence-only and conservative: a run that never resolved (any WebGL row,
or a run whose timestamp queries produced no samples) leaves no trace
perf-diff can read, so those rows are listed too — the warning text says
which case is which.

```bash
node scripts/perf-diff.mjs baseline.json new.json > diff.md
```

Both arguments are required; the script also echoes the absolute paths
it consumed to stderr so you can spot a stale baseline.
