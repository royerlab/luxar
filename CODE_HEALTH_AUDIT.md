# Luxar code-health audit

**Date**: 2026-09-02 · **SHA**: `0ceb6c6dd` · **Branch**: `feat/viewer-auto-dolly`
**Scope**: shipping surface first — `luxar` library + `luxar-viewer` + Rust/WASM + build/CI/docs.
`demos/`, `scripts/` and the test trees got a structural pass only (except dimension A10, for
which tests were the deep target).

**Not in scope**: release *logistics* — Zenodo records, git-LFS payload size, demo hosting,
cold-fetch gates. Those live in `RELEASE_CHECKLIST.md` and are deliberately not repeated here.
This document is about code health.

---

## Verdict

**The codebase is in good health and there is no architectural blocker to release.**

Of 122 candidate findings raised across 14 review dimensions plus a completeness critic,
adversarial verification **confirmed 51, corrected 50, and killed 12**; I personally verified
the critic's 9 and refuted its one claimed blocker. After severity correction there are **zero
blockers**, **20 major** findings and **90 minor/nit** findings. Not one is a correctness defect
in shipped behaviour; the majors are concentrated in *gate coverage*, *parameter-schema
duplication*, *documentation drift*, and *shipping-artifact verification*.

Three things are genuinely notable for a codebase this size (~460K LOC of production source
across three languages, written largely by AI agents over months):

- **No dead-code problem.** One orphan production module in 527. One unreferenced exported
  function in 1,534 viewer exports. Two `TODO`/`FIXME` markers in the entire production tree.
  Zero suppressed tests.
- **No unfinished-work problem.** All 33 `NotImplementedError` sites are legitimate typing/MRO
  stubs. Every documented CLI flag is actually parsed. Mesh's gaps are documented deliberate
  omissions with PR numbers.
- **The cross-language contracts hold.** The Rust and TypeScript kernels export the same 33/34
  functions with a 3,419-line CI-enforced parity suite; `format-contract/contract.yaml`
  genuinely single-sources the on-disk vocabulary for both languages under a drift gate.

The real theme is narrower and consistent: **the gates are strong where they exist and absent
in specific, nameable places** — and where a gate is absent, debt has quietly accumulated in
proportion to how long it has been absent.

### The five root causes

A completeness critic, given every dimension's findings and asked what they all missed,
identified five root causes that between them explain most of the 100+ individual findings.
This is the most useful paragraph in the document:

1. **The gate surface is the map, and the codebase is the territory.** Almost every dimension's
   headline finding sits *precisely* in the gate's complement — ruff's missing rule families,
   eslint's missing type-aware rules, import-linter's 2 contracts, the absent TS complexity
   gate, the non-recursive docs gate, bandit skipping `demos/` and `scripts/`, knip being
   report-only, `exclude_lines` deleting statements, E2E off in CI, and (the critic's additions)
   5,774 LOC of build/gate config outside eslint, no C++/CUDA arm, one browser configured, and
   launcher/npm artifacts unverified pre-tag. **AI authors optimise to the gate; wherever the
   gate stops, quality stops within about one commit.** This is why *"add gates"* beats
   *"fix findings"* here.
2. **Additive authoring with no deletion pass.** Dead code, over-abstraction, stale compat and
   incomplete work are four views of one behaviour: a new path is added, the superseded one is
   left, and nothing gates absence-of-consumers.
3. **Duplication is the default response to a second requirement.** 4 geometry types × 2
   backends, Rust/TS, GLSL/TSL, 7 restatements of the fit schema, three spatial-index loaders,
   plus torch/CUDA/Metal. Notably, the one place duplication is deliberate (WASM/TS, for >16D)
   has a real parity gate — which plausibly normalised copy-as-a-pattern everywhere it is not.
4. **Docs and code are two sources of truth with only spot reconciliation.** The doc gate checks
   *presence and paths*, never *claims* — which is why a four-browser support matrix tested on
   one browser and a v3.3/v3.4 split both survive a green gate.
5. **Shipping-artifact discipline is thinner than source discipline.** The source tree is
   genuinely well-tended; what actually *leaves* it is verified in exactly one place (the viewer
   dist). The wheel, the launcher binaries, the npm library bundle and third-party licence
   notices are not.

---

## Phase 0 — the test-and-coverage prerequisite (gating)

Nothing in the refactor plan below starts until this phase is done. A refactor without a
trustworthy safety net converts *invisible* debt into *visible* breakage, which is worse.

### 0.1 The measured baseline

Both suites were run in full for this audit.

| Suite | Result | Coverage | Enforced floor | Slack |
|---|---|---|---|---|
| Python (`hatch run test-cov-all`) | 15,950 passed, 258 skipped, 0 failed, 28m24s | **92%** (3,259/41,347 missed) | `fail_under = 80` | **12 pts** |
| Viewer (`pnpm test:coverage`) | pass | **88.1% lines / 81.7% branch / 84.5% func** | 71 / 61 / 74 | **17 pts** |

**Both gates are decorative.** The viewer could lose a third of its tests and CI would stay
green. Raising each floor to just under measured is a ~10-minute change and is the single
highest-value item in this entire document, because every subsequent refactor is protected by
it. Do this first.

### 0.1b Corrected baseline (measured after the Phase 0 measurement fixes)

The §0.1 numbers were measured against a *broken* denominator. After anchoring
the Python `exclude_lines` regexes, dropping the `*/__init__.py` omit, and giving
the viewer an explicit `coverage.include`, the real baseline is:

| Suite | Statements | Coverage | Floor now | Slack |
|---|---:|---|---:|---:|
| Python `hatch run test-cov` (what CI gates) | 42,847 (was 41,347) | **91.58%** | `fail_under = 89` | 2.58 pts |
| Viewer `pnpm vitest run --coverage` | 686 files (was 648) | **88.09 L / 87.22 S / 84.41 F / 81.72 B** | 86 / 85 / 82 / 80 | ~2 pts |

Two things this measurement settled:

- **The `"pass"` regex was hiding well-tested code, not thin code.** Once visible,
  `cli/gsplat_ops/fitting/fit.py` reads 84%, `cli/lod.py` 91%,
  `cli/gsplat_ops/batch/submit.py` 90%, `gsplats/fit_progressive_gsplats.py` 91%.
  The defect was in the *report*, not the coverage — which is worse in one way
  (nobody could see those bodies) and better in another (no test debt hid behind it).
- **The viewer `include` fix was worth more than the percentage suggests.** The
  headline barely moved, but the report grew by 38 files, and `core/main.ts` and
  `workers/sort-worker/initialize.ts` went from *invisible* to *visibly 0%*.
  Before it, adding an untested module could raise the reported number.

Every subtree now carries its own floor (13 glob keys) so no area can decay to the
global number, and `scripts/check-coverage-slack.mjs` fails `check:ci` once any of
the 43 floors falls more than 3 points behind measured. Max slack today: 2.9 pts.

Two gate bugs found while doing this, both verified empirically:

- A vitest glob threshold matching **zero files** reports `pct: "Unknown"`, and
  `"Unknown" < 86` is `false` — so a renamed directory silently converts its gate
  into one that inspects nothing. Same family as the three fail-open gates in A9-04.
  `check-coverage-slack.mjs` asserts every glob key is non-empty.
- `test_no_zarr_2_create_dataset_calls_remain` scans gitignored scratch directories,
  so an untracked `delme/*.py` file belonging to another session reddens the suite.
  Fail-noisy rather than fail-open, but the same class of gate-scoping bug.

### 0.2 Classify the gap before writing a single test

4,211 viewer lines and 3,259 Python statements are uncovered. **They are not all missing
tests**, and treating them as if they were would generate a large amount of harmful busywork.
Three distinct categories:

| Category | Share | Example | Right action |
|---|---|---|---|
| **Genuinely missing tests** | ~2,190 viewer lines | `ui/layers-panel.ts` at 64% branch | Write them (0.3 below) |
| **Instrumentation-blind, actually tested** | **~2,021 viewer lines (48% of the viewer gap)** + a visible slice of Python | TSL shader cone; Numba JIT kernels | Exclude from the metric; verify via the suite that *does* cover them |
| **Hardware / environment-gated** | 10 Python modules below 60% | `_cuda_build.py` 45%, `gpu_profile.py`, Slurm submit | Accept, or measure on the GPU box |

Two worked examples of category 2, both of which look like alarming gaps and are not:

- **The TSL/shader cone** — 90 modules, 5,018 lines, 2,021 uncovered — is exercised in a real
  browser by `tsl-shader-parity.spec.ts`, which drives the page through Playwright and is
  therefore invisible to vitest instrumentation. You cannot meaningfully unit-test a TSL shader
  graph in Node. Writing vitest tests here would be wasted effort.
- **`io/_ordering/curves/hilbert.py` at 39%** is the *default* spatial ordering, and lines
  26–70 are the body of a `@numba.njit` kernel. Coverage cannot instrument compiled bytecode.
  Verified live: `numba 0.66.0`, kernel compiles and returns on first call. The obvious "fix"
  — testing the slow `hilbertcurve` fallback — would turn the number green while leaving the
  kernel that actually runs in production still untested. **Do not do this.**

### 0.3 Real unit-test gaps (write these)

Branch coverage is consistently worse than line coverage here, which is exactly the profile
that lets a refactor break an edge case silently.

| Module | LOC | uncov. lines | uncov. branches | lines% | branch% | func% |
|---|---:|---:|---:|---:|---:|---:|
| `data/points/points-spatial-index-loader.ts` | 1422 | 98 | 87 | 72.5 | 66.9 | 68.4 |
| `ui/layers/layers-panel.ts` | 1250 | 97 | 99 | 80.6 | **64.1** | **60.0** |
| `ui/dimension-sliders.ts` | 1503 | 90 | 107 | 83.4 | **60.5** | 74.6 |
| `ui/data-loading-monitor.ts` | 1730 | 83 | 79 | 82.0 | **63.6** | 76.4 |
| `ui/debug-console.ts` | — | 65 | 38 | 69.9 | **47.9** | — |
| `data/scene-loader.ts` | 2184 | 61 | 27 | 85.6 | 85.7 | 76.2 |
| `input/input-handler.ts` | 1104 | 59 | 18 | **67.9** | 66.0 | **54.2** |
| `ui/rendering-controls.ts` | — | 53 | 38 | 78.5 | **54.8** | — |
| `data/array-decoder/decoder.ts` | 1282 | 51 | 108 | 86.3 | 80.1 | 100 |
| `scene/scene-manager.ts` | 1546 | 47 | 24 | 82.7 | 82.3 | **62.4** |

`input-handler.ts` at 54% function coverage — nearly half its functions never invoked by a unit
test — is the one that most needs characterization tests before anything moves.

Only 6 viewer modules have literally zero coverage, totalling 158 lines. A non-issue.

### 0.4 Coverage is necessary, not sufficient

An executed line is *covered*; a line whose mutation fails a test is *tested*. For the specific
modules you intend to restructure, run a scoped mutation pass (`mutmut` on the Python target,
Stryker on the viewer target) rather than trusting the percentage. Two supporting findings:

- **[A10-04, minor]** Of 11,589 viewer test blocks containing an assertion, **1,091 assert only
  that a spy was called** and ~490 only weak predicates (`toBeDefined`, `toBeTruthy`). That is
  an upper bound on behaviour-blind tests — some spy assertions are legitimate — but it is
  concentrated in seven orchestration files, which are exactly the god-objects on the refactor
  list. *Verifier note: a material fraction is legitimate; treat as a prioritised list, not a
  1,581-item backlog.*
- **[A10-01, minor — downgraded from major]** `[tool.coverage.report] exclude_lines` contains
  the unanchored regex `"pass"` (pyproject.toml:774). coverage.py searches it in every line, so
  `def bypass_cache():` or a Typer `help="…passes…"` string excludes the whole clause. Measured
  against the real data: **1,122 statements hidden, of which 114 are genuinely uncovered**.
  Fix is one line — `"^\s*pass\s*$"`. *I verified this myself: the headline stays 92% either
  way, so the finding's claim that it invalidates the number does not survive measurement.*
  The value of fixing it is that `--cov-report=term-missing` currently cannot show you those
  114 lines.
- **[A10-06, nit]** The blanket `*/__init__.py` omit, justified as "re-export shims", also
  removes real logic — `gsplats/doctor/__init__.py` alone is 66 statements.

### 0.5 The E2E question — the real prerequisite for rendering work

**[A4-01, minor]** The 15 production dual-backend `ShaderSource` values (~11K LOC across both
backends) have exactly one equivalence check: `tsl-shader-parity.spec.ts`. It does not run in
CI, and the E2E baseline is known red-and-flaky. **[A4-08, minor]** Two of the 15
(`BLOOM_DOWNSAMPLE_SOURCE`, `BLOOM_UPSAMPLE_SOURCE`) have no harness entry at all, and nothing
asserts registry completeness.

This matters more than the raw coverage numbers: **the rendering cone's safety net is a suite
nobody runs.** Note the parity harness renders 8×8 fullscreen passes through
`WebGPURenderer({ forceWebGL: true })` — far cheaper and less flaky than the
rendering-dependent specs the `if: false` comment was written about. A narrow CI job running
*only* `tsl-shader-parity.spec.ts` + `tsl-codegen-snapshot.spec.ts` is likely affordable and
would close this.

### 0.6 Phase 0 checklist

Status as of PR #2471. Two items were changed on contact with evidence rather than executed as
written; both are noted, because a checklist that quietly edits itself is worth less than one
that records where it was wrong.

1. **Done.** Both floors raised to just under measured — Python `fail_under` 80 → 89 (91.58
   measured), viewer 71/71/74/61 → 86/85/82/80 plus 13 per-subtree floors. Every floor was
   verified to fire by overshooting it first.
2. **Done, and the estimate was wrong in an interesting direction.** Anchoring `"pass"` and
   dropping the `*/__init__.py` omit moved the denominator 41,347 → 42,847. But the regex was
   hiding *well-tested* code: `cli/lod.py` reads 91%, `batch/submit.py` 90%,
   `fit_progressive_gsplats.py` 91%. The value was making them visible, not finding a gap.
3. **Deliberately NOT done — decision reversed.** The plan was to exclude the TSL cone so the
   metric stops lying. Checking first showed the proposed glob would have hidden nine files the
   parity harness never imports (it imports 5 `shader-tsl*` + 6 `*.tsl`, and zero
   `material-tsl`), one of them at 29.2% lines / 19.2% functions — and that
   `tsl-shader-parity.spec.ts` appeared in no workflow at all, so "gated by the parity spec" was
   a false premise. Excluding code on the strength of a gate that never fires is how a metric
   starts lying. The cone stays in, capped by an explicit debt ceiling
   (`src/rendering/**` at 72/73/68), and item 5 supplies the missing gate. Revisit after the
   parity job has run green in CI for a while. The Numba half stands as written and is unbuilt.
4. **Partly done.** See §0.7 — but the selection rule changed: *low branch coverage INTERSECT
   named refactor target*, not the §0.3 table alone. Most of that table is on no refactor list.
   `input-handler.ts` (44 of 96 functions never invoked) and the nine `material-tsl.ts` files
   remain outstanding.
5. **Done for the job**, non-required, verified locally at 105 passed in 1.4 min. Promoting it to
   required is a four-file change and belongs in its own PR. The two missing bloom harness
   entries (`BLOOM_DOWNSAMPLE_SOURCE`, `BLOOM_UPSAMPLE_SOURCE`) are **not** done.
6. **Done for the target touched**, by hand rather than by tool — six mutants, six kills (§0.7).
   Still to do per-target as Phase 3 reaches each one.

Added during execution, and not in the original list: an anti-decay guard
(`check-coverage-slack.mjs`). Without it this is a one-time correction that rots back to the
same 12–17 pts of slack, which is how the 71/61 floors happened in the first place.

### 0.7 Characterization tests landed (Step 8, 2026-09-02)

Target selection was by **low branch coverage INTERSECT named refactor target**, not by low
coverage alone — most thin modules are on no refactor list, and testing those is the busywork
§0.2 warns about one category down. That narrowed the list to the three spatial-index loaders
(refactor target A2-07), whose measured asymmetry is itself the evidence for the finding:

| Loader | LOC | Branches | Before (fn / br) | After (fn / br) |
|---|---:|---:|---|---|
| `points-spatial-index-loader.ts` | 1,422 | **263** | 68.4% / 62.7% | **82.5% / 64.3%** |
| `lines-spatial-index-loader.ts` | 1,046 | 142 | 65.5% / 77.5% | **80.0% / 79.6%** |
| `gsplats-spatial-index-loader.ts` | 1,123 | 143 | 78.8% / 84.6% | **94.2% / 87.4%** |

Points carries **nearly double its siblings' decision surface for the same job**, and all three
share a positionally identical 7-argument constructor. That is A2-07 restated as a measurement.

**Two files added.** Both go through observable consequences rather than spy-call assertions,
per the §0.4 caution:

1. `src/tests/unit/data/loaders/l0-cache-wiring.test.ts` — differential across all three
   loaders. The 21 `wrapWithCache(...)` call sites had **zero** coverage: `this.l0Cache` was
   falsy in every existing loader test, so the whole wrapping path — including twelve
   `() => this._activeProbe` / `() => this._activeSignal` thunks — never executed. The wrapper's
   own behaviour was never the gap (`cached-zarr-array.ts` is at 97.9% lines / 100% functions
   from its own test); the loaders' *wiring* of it was. So the seam is the call site and the
   real implementation still runs. The load-bearing assertion is a **property, not a golden
   list**: *no successfully-opened array bypasses the cache.* That survives adding an array and
   is the acceptance criterion the A2-07 extraction gets checked against.
2. `src/tests/unit/data/points/spatial-extend-dims.test.ts` — the derivation deciding, per
   non-displayed dimension, whether points extend as hyperspheres (spatial) or must match
   exactly (categorical). Uncovered because the sibling tests build the loader with no
   `zarrStore`, so the method returned `null` on its first line every time.

**Mutation-verified, not assumed.** Six mutants against the production source, each killed by
exactly the intended test, source restored byte-identical (md5-checked) after each:

| Mutant | Killed by |
|---|---|
| one array skips the L0 wrap | *routes every opened array through the L0 cache* |
| probe passed as a snapshot, not `() => …` | *hands the wrapper live probe/signal accessors* |
| cache key loses its `node.path` prefix | *keys every wrap under the node path* |
| `dim.spatial ?? false` → `?? true` | *an UNSET spatial flag defaults to categorical* |
| displayed dims no longer forced spatial | *a displayed dimension is spatial regardless* |
| root-read failure rethrows | *falls back without throwing* (+1) |

This answers Step 10's actual question — "did the tests kill mutants, or did I write more spy
assertions?" — directly and on the exact contracts claimed, which a Stryker score would not.
A broad Stryker pass remains worthwhile but was **not** run: it needs a dev dependency installed
into a tree several agents are editing.

**Two corrections worth recording**, both found by measurement rather than review:

- My first attempt asserted that a derived config and the no-store fallback produce *different*
  query tolerances. They do not — the fallback is documented to mirror the config path branch
  for branch, and reproduces the **permissive** `spatial: true` reach. The test now pins that
  fact, which is the more useful one: *a scene whose root attrs cannot be read queries a
  categorical axis as though it were spatial.*
- The helper initially read `builderCtor.mock.calls[0]`, but it is called several times per
  test — so every case silently re-read the first case's arguments and agreed trivially. Caught
  because a probe showed all seven fixtures returning byte-identical tolerances. A textbook
  vacuous-pass; the fix scopes the capture per load.

**The ratchet then fired on its own.** The new tests lifted `src/data/**` functions 89.2 → 91.2,
and `check-coverage-slack.mjs` failed the build with *"floor 87 is 4.2 pts under measured 91.19
(budget 3). Raise it to 90."* The floor was raised in a separate change from the tests, which is
the intended loop: tests move the measurement, the guard moves the floor. Viewer global after:
**88.22 L / 87.34 S / 84.85 F / 81.80 B**, `pnpm run check:ci` exit 0.

### 0.8 Mutation testing: what the coverage number does not say (Step 10, 2026-09-02)

Stryker is installed and committed, **report only** (`thresholds.break: null`). Coverage says a
line RAN; a mutation score says a change to that line FAILS something. §0.4 predicted the two
would diverge wherever tests assert on spies rather than behaviour. They do, by a lot.

| File | Line cov | Branch cov | **Mutation score** | Killed | Survived | No cov | Runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| `points-spatial-index-loader.ts` | ~72% | ~64% | **33.7%** (44.4% of covered) | 252 | **315** | 180 | 1m45s / 747 mutants |
| `input-handler.ts` | 86.8% | 75.5% | **48.8%** (54.1% of covered) | 138 | **117** | 28 | 58s / 284 mutants |

Read the first row carefully: on the loader, **roughly half the code the tests execute is not
actually checked.** That is A10-04 (1,091 spy-only assertions) confirmed by measurement rather
than by counting occurrences.

**Measurement caveat, stated so the number is not over-read.** Stryker scoped the test pool via
the module graph — 1,185 of 13,734 tests for the loader, 310 for the input handler — not the
whole suite. That is the correct pool (a test that never imports the file cannot kill its
mutants), but the denominator is the selected pool, not the entire suite.

**Two survivors worth acting on directly.** These are not statistical; each is a specific
untested behaviour:

- `isNotFoundError` can be replaced by an **empty body** and every test still passes.
  *Corrected on inspection, and the correction is the more interesting result.* My first reading
  was that this decides whether a missing optional array is genuinely absent or a real failure
  being swallowed. That is true of `zarr.isNotFoundError` — the shared one in `data/zarr.ts`,
  which the gsplats loader uses at two sites to gate `throw e`. But the surviving mutant is a
  **second, local copy** at `points-spatial-index-loader.ts:92`, and all four of its call sites
  gate nothing but a `log.info` line. Emptying it changes log noise, not behaviour: an
  effectively equivalent mutant, and not worth a test.

  What it does expose is a duplicate: the local copy tests only `'404'` and `'Not Found'`, while
  the shared one also accepts a `zarrita.NotFoundError` instance, `'Node not found'` and a
  lowercase `'not found'`. The duplicate is the weaker of the two. Harmless where it currently
  sits (logging only), but it is the same predicate implemented twice with different semantics,
  and it should collapse into the shared one during the A2-07 extraction. **Read every survivor
  before believing it — a mutation score counts equivalent mutants as failures.**
- `registerBounds` likewise empties clean, so prefetcher bounds registration is unverified.
- In `input-handler.ts`, `() => window.removeEventListener('keydown', onKeyDown)` mutates to
  `() => undefined` and survives — for both the keydown and keyup teardowns. **Listener cleanup
  on dispose is untested**, which is precisely the leak class CLAUDE.md calls out under "Event
  Listener Memory Leaks". Worth a test on its own merits.

**It also caught a weakness in the characterization test written earlier the same day**, which
is the strongest argument for keeping the tool. `ui-actions-surface.test.ts` invokes all 27
command thunks and asserts they do not throw — but 34 `ArrowFunction` mutants survived,
including `getScaleBar: () => this.scaleBar` becoming `() => undefined`. Executing a thunk while
asserting nothing about its result is a spy assertion wearing different clothes. (The identity
and CustomEvent assertions in the same file are sound — hand-built mutants killed them.)

Fixed, and the fix was verified the same way rather than assumed. A new arm captures the five
panel getters at registration, attaches panels afterwards through the public setters, and
asserts the same getter objects observe the change — the real contract, since panels attach
after `init()` and a snapshotting getter would hand the key bindings `undefined` forever. The
invocation arm now states in the file that it is a robustness check that cannot stand alone.

| `input-handler.ts` | Score | Killed | `ArrowFunction` survivors |
|---|---:|---:|---:|
| as first measured | 48.76% | 138 | 34 |
| + 3 panel getters asserted | 50.18% | 142 | 31 |
| + all 5 asserted | **50.88%** | 144 | 29 |

Five targeted assertions, five mutants killed, +2.1 points. Small — and that is the honest
shape of this work. The wider lesson stands: **a test can raise coverage 29 points and still not
test much**, and only a mutation run distinguishes the two.

**Three environment facts, each of which cost a failed run** (all commented in
`stryker.conf.json` so nobody rediscovers them):

1. `plugins` must name `@stryker-mutator/vitest-runner` explicitly — Stryker finds plugins by
   scanning `node_modules` for `@stryker-mutator/*`, and pnpm's symlinked store hides them
   ("no TestRunner plugins were loaded").
2. Runs need `--inPlace`. The vitest `globalSetup` regenerates zarr fixtures via Python, those
   fixtures are gitignored, so Stryker's sandbox copy omits them and regeneration fails inside
   the sandbox. In-place instruments the file once with every mutant behind an activation
   switch, so concurrency is unaffected, and restores the original afterwards.
3. Clear `node_modules/.vite` between in-place runs. A second consecutive run failed its dry run
   with `Failed to resolve import "./debug-state"` for a file that exists and is tracked —
   Vite's transform cache reacting to the in-place file swap, the same family as the known
   stale-transform-after-checkout trap. It is transient, not damage; verify with `git status`
   before believing otherwise.

**How to use it**: `pnpm test:mutation`, or `--mutate <path>` for one investigation. Read it
before restructuring a module, not on every push — runtime is O(mutants x suite) and a score
means nothing without a baseline. The committed `mutate` list is narrow on purpose; Stryker's
default would mutate the whole 218K-LOC tree.

---

## Findings

Severity is post-verification. `S` < 1h · `M` ≈ half a day · `L` ≈ a few days · `XL` ≈ a week+.

### Theme 1 — Gate coverage gaps (the dominant theme; 6 of 20 majors)

Where a gate exists it works. These are the places one does not.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A12-05** | major | L | **ruff selects only `E,W,F,I`.** A wider select finds **834 production findings** (tests/demos excluded), ~366 after dropping the RUF unicode and B008-Typer false-positive families. The correctness families: **67× `B905`** (`zip()` without `strict=`) concentrated in `gsplats/_data`, `gsplats/multiscale`, `gsplats/lod`, `io/_compiler` — silent truncation in a package whose job is producing numeric arrays a second language then reads; **55× `B904`** (lost exception chaining, 39 in `cli/gsplat_ops`); 5× `RUF012` mutable class default; 2× `B006`. *Independently reproduced: I measured 835 with ruff 0.16.2 against their 834 with 0.14.0, every sub-count matching.* Land `B905` and `B006` first. |
| **A12-03** | major | M | **ESLint enables no typescript-eslint recommended rules and no type-aware rules — while already paying for type info** (`parserOptions.project` is set). Measured yield: 10 `no-floating-promises`, 6 `no-misused-promises`, 8 `no-base-to-string`, 261 `no-unnecessary-type-assertion`. In a worker/loader/cache-heavy viewer with 449 `async` occurrences, an unhandled rejection shows up as a silently stalled load, not a crash — and E2E does not gate either. |
| **A12-04** | major | S | **CI's Python lint is narrower than the project's own `hatch run lint`**, so `scripts/` (121 files, 35K LOC — including the gate implementations themselves), `stats/` and `hatch_build.py` get no E/W/F/I check in CI. Separately **`ruff format --check` is gated nowhere** — the only enforcement is an opt-in pre-commit hook. |
| **A2-05** | major | M | **218K LOC of production TypeScript has no size, nesting, parameter-count or complexity gate at all.** 36 methods ≥120 lines and 13 ≥200 are invisible to every gate. The Python side has a ratchet that demonstrably shrank debt 228→198 in three weeks; the shipping viewer has nothing stopping the next 672-line method. |
| **A12-01** | major | M | **Nothing builds, inspects or installs the Python wheel until a release tag is pushed.** `hatch build` appears exactly once in the repo, in `publish.yml` on a `v*` tag. Nothing asserts what must *not* be in the wheel — the `**/demos/data/**` exclude guards ~394 MB of Git-LFS payload and `publish.yml`'s checkout has no `lfs: true`, so a regressed glob ships 130-byte pointer files. The project already invented exactly the right gate for npm (`release-readiness`) and did not mirror it. *Verifier correction: not release-blocking — a `workflow_dispatch` `dry_run` path publishes to TestPyPI, so it can be exercised manually.* |
| **A11-01** | major | M | **The documentation gate never recurses.** `check_documentation.py` uses `glob("*.py")` / `glob("*.ts")`, one level deep, so **79% of Python and 79% of TypeScript source is never scanned**. Re-running the gate's own rules over the skipped files finds 54 Python and 132 TypeScript files below its own 70% threshold. The near-empty baseline is not evidence of health. Fix: two `glob` → `rglob` changes plus a baseline regeneration. |
| A12-06 | minor | M | import-linter enforces only 2 forbidden edges; no `layers` and no cycle contract, where the TypeScript side has 8 layer rules plus `no-circular` at error. |
| A12-02 | minor | S | `cargo clippy` never runs in CI, and `make check-rust` cannot fail on macOS (clippy's exit status is swallowed by a backslash-joined recipe). *`cargo check` itself does run, via `pnpm build` → wasm-pack.* |
| A9-04 | minor | S | **Three gates report success when they inspected nothing** — `check_demo_ladders.py:696` and `check_scene_credits.py:155/122` return 0 on an empty inventory, and the directory they glob is gitignored and empty on any machine that has not built demos. |
| A9-03 | minor | S | (Same root as A12-03, viewed from the error-handling side.) 16 floating/misused promise sites. |
| A13-03 | minor | M | Bandit excludes `demos/` on the stated rationale that demos are "not shipped library code paths" — **factually wrong; the `.py` files ship in the wheel**. `scripts/` is never scanned at all. |

### Theme 2 — Parameter-schema duplication on the primary entry points (3 majors)

The largest maintenance liability in the Python package, and the one place drift has already
been *measured* rather than merely predicted.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A2-02** | major | L | **The gsplat fit parameter schema is hand-restated 7 times along one call chain**: `run_fit_volume` 63 typer options → `FitPipelineCtx` 58 fields → `run_content_fit` 39 → `fit_gaussian_splats` 50 → `GaussianSplatFitter.fit` 45 → `prepare_fit_config` 45 → `FitConfig` 55 fields. 44 of `prepare_fit_config`'s 45 params are literally `FitConfig` field names. **It has already drifted**: `asymmetric_penalty` defaults to `10.0` in `prepare_fit_config` and `1.0` in the three others — and `prepare_fit_config` is public API (`gsplats/fitting/__all__`), so an external caller genuinely gets the divergent default. Adding one fit knob means editing seven places with no gate. Fix: make `FitConfig` the single schema and thread the object. Collapsing `prepare_fit_config`'s guard block into a declarative table also drops the repo's highest complexity score (56) in one change. |
| **A2-03** | major | M | **`batch-fit run` forwards 56 parameters to a function with a byte-identical 56-parameter signature** (AST set-equality: empty symmetric difference). 54 of the 56 are *also* declared independently on `batch-fit submit` (70 options). This is the exact shape that produces "the flag works on `submit` but is silently ignored on `run`". The four config dataclasses that would fix it already exist in `planning.py`. |
| A4-07 | minor | L | `build_plan_configs` exists but only 1 of its 2 callers uses it; `run_orchestration.py` inlines the same four dataclass constructions field-for-field. |
| A2-01 | major | M | `runOfflineCaptureLoop` is a **672-line method at 12 levels of nesting** — 73% of its module — producing the gallery stills and videos that front the public release. Five phases are already self-marked in the code. *Verifier correction: the finding's justification ("only reachable through E2E") is false — a 1,385-line unit test file exists. Size is the issue, not testability.* |

### Theme 3 — Architecture: unenforced layering and one real cycle (2 majors)

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A1-03** | major | L | **`core` and `io` are mutually dependent, and 17 production modules outside `luxar/io` reach into the private `io._compiler` tree.** The cycle is module-level in one direction (`io/_compiler/node_common.py:26` and `gsplat_assembly.py:25` import from `core.group.compositing`) and papered over with four function-local imports in the other (`compositing.py:835/891/934/978`). The leading underscore says "private to `luxar.io`" while 17 external modules depend on it — de facto public API with none of the stability or documentation. The deferred imports make the cycle invisible to mypy *and* to both import-linter contracts. Fix is well-bounded: move two appearance constants to a leaf module and the four `validate_*` functions into `luxar.validation`. |
| **A1-01** | major | M | **No layering contract.** import-linter declares exactly two `forbidden` contracts and reports 2/2 kept over 1015 files — green while the graph carries **9 mutually-cyclic subpackage pairs** (`core↔gsplats` 21/10, `core↔io` 14/25, `io↔typing_utils` 31/1, `gsplats↔io` 19/7, …), paid for with **867 function-local intra-luxar imports across 153 files**. Add one `layers` contract and let it fail; a contract that starts with 60 dated `ignore_imports` is strictly better than no contract. |
| A1-02 | minor | S | `typing_utils`, the declared foundation package, imports `validation`/`encoding`/`io`. |
| A1-04 | minor | S | `luxar/__init__.py:107` rebinds `validation` to `validation.base`, so `import luxar; luxar.validation.<X>` reaches only a third of the package. *(The documented `from luxar.validation import X` form works.)* |
| A1-07 | minor | M | Viewer `src/types/` — the declared bottom layer — holds mutable process-global session state. |

### Theme 4 — Documentation drift (3 majors)

The docs are extensively hand-maintained and largely accurate (see "What's healthy"). These are
the specific places they are not.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A11-03** | major | M | **Mesh — a first-class geometry type — is missing from every public documentation index surface.** No `docs/api/mesh.rst`, no `shading.rst`; absent from the `docs/index.rst` Packages Overview; the string `luxar mesh` appears **zero times in CLAUDE.md's 1,473 lines** despite a 546-line gsplat CLI manual; absent from the root README's CLI Reference and the viewer README's feature line. A visitor reading the API reference concludes Luxar renders three geometry types. |
| **A11-02** | major | S | **The 3.3→3.4 gsplats format bump is N-1-of-N.** The constant is `"3.4"` and `GSPLATS_ZARR_FORMAT.md` agrees, but `FORMAT_AND_MIGRATION.md:14` — the Sphinx-published page whose entire job is a table of current on-disk versions — says **v3.3**, as do `gsplats/io/README.md:3` and `:532`, `gsplats/README.md:774`, and CLAUDE.md contradicts itself at `:542` vs `:747`. This is the kind of error that gets copied into someone else's reader. |
| **A8-06** | major | S | `luxar.mesh` and `luxar.shading` are public packages with `__all__` and zero Sphinx presence — while `docs/api/cli.rst` carries 13 `automodule::` entries for CLI *internals*. `bake_ambient_occlusion` is the sanctioned way to make emissive geometry read as 3D and a user cannot find it. |
| A11-08 | minor | L | `CHANGELOG.md` has **no released-version section at all** — 490 KB under a single `## [Unreleased]` — and 400 pending fragments (952 KB) will fold into one month heading. *The fold mechanism itself works cleanly; verified zero duplicate titles.* |
| A11-09 | minor | L | CLAUDE.md is 1,473 lines / 87 KB loaded into every agent context, **61% of it Quick Reference**, with `### GSplat CLI` alone spanning 546 lines (37% of the file) duplicating a CLI manual its own doc index omits. |
| A11-04 | minor | M | `gsplats/README.md`'s 138-line Package Structure tree omits 3 subpackages (`doctor/`, `interop/`, `planner/`) and 13 modules. |
| A11-05 | minor | M | The path-reference gate scans only `packages/**/README.md`, leaving `docs/` (28 pages, ~19K lines), the 1,083-line root README, and 61 KB of viewer contributor docs unchecked. |
| A11-06 | minor | S | `NETWORK_SIMULATION_SPEC.md` ships to the docs site marked "✅ Implemented" and then presents a 29-item unchecked implementation checklist. |
| A11-07 | nit | M | `VOLUMETRIC_BLENDING_SPEC.md` is an append-only stack of 5 status banners over superseded math. |
| A11-10 | nit | S | Four nested Python packages (all under `cli/`) and one viewer directory have no README — 37 modules, unseen by the non-recursive gate. |

### Theme 5 — Duplication, over-abstraction and API surface

None of these is urgent; together they are the bulk of the "would make the code nicer" work.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| A4-03 | minor | M | Six identical appearance methods copy-pasted across all 8 material classes (4 geometry types × 2 backends) — 48 duplicated bodies, 4,925 LOC. The two backends differ only in `needsUpdate = true` vs `rebuildGraph()`. |
| A4-05 | minor | M | The four per-geometry `lod-refinement` modules are **83–91% identical** after name normalization (729 LOC total) — four parallel adapters around an already-extracted shared loop, each carrying ~50 duplicated lines of abort/backoff/toast. |
| A4-04 | minor | S | `applyPartialExtendTolerance` is hardcoded at 12 call sites though `GEOMETRY_DESCRIPTORS` is the declared single source of truth and 2 sites already read it properly. |
| A2-07 | minor | L | Three spatial-index loaders each carry a 200–355 line `load<Geom>Internal` with an identical skeleton. |
| A2-08 | minor | M | The encoding taxonomy is a 15-arm `if/elif` chain plus a parallel 8-arm chain that must be kept in sync. |
| A5-01 | minor | M | `ZarrWriterProtocol`: a 417-line one-implementation Protocol whose 4 core methods are `# type: ignore[override]` — i.e. the protocol does not actually type-check its only implementation. |
| A5-04 / A3-02 | minor | S | `typing_utils/config.py` — a 248-line "centralized configuration" layer where **24 of 27 public names have zero production readers**. Reduce to the three that do. *Verifier caution: `check_dataset_size_warning` reads two of the constants the recommendation lists for deletion.* |
| A5-02 | minor | S | `typing_utils/protocols.py`: 4 Protocols + 4 TypeVars with no consumer; the one use is a cast to `Any`. |
| A5-06 | minor | M | Mixin stub-bases re-type full signatures mypy never verifies — a 30-parameter `filter_by`, a 15-parameter `encode`. |
| A8-05 | minor | L | **No exception hierarchy** — ~1,500 bare `ValueError` raises, no `luxar.exceptions`. *Counts did not reproduce exactly (1,503 vs the claimed 1,320) and `ValidationError` IS importable via `from luxar.validation import`.* |
| A8-08 | minor | M | The on-disk encoding vocabulary mixes `_uint8`/`_uint16` and `_u8`/`_u16` for the same concept across 18 names — and the contract file itself carries a warning that they are not interchangeable. Normalise **before** the format is published. |
| A8-10 | minor | M | `add_gsplats` is the only geometry adder without `additive_lod=`/`substitutive_lod=`. |
| A8-09 | minor | S | `GSplatData` (authoring) and `GSplatsData` (read-back) are two different public classes one plural apart. |
| A8-02 | minor | M | `LuxarScene` cannot read back `scalars`, `labels` or `keys` that the write API accepts and stores, while advertising "round-trip testing". |
| A8-03 | minor | M | `luxar gsplat additive` uses a spelling that `gsplat lod --recipe` actively **rejects as legacy**. |
| A8-04 | minor | M | `--spatial-dims` carries different contracts in sibling commands (order-significant in `transform`, order-insensitive in `filter`). |
| A8-07 | minor | S | The published viewer entry point's TSDoc — which TypeDoc renders as the API landing page — tells consumers to `import from 'luxar-viewer'`, a package name that does not exist (`@royerlab/luxar-viewer`). |
| A8-11 | minor | S | No stated API stability or deprecation policy for the *code* API. *(The on-disk format does have one.)* |
| A8-12 | nit | S | `--reveal-centre` is the single British spelling among 260 CLI options, next to `--center` in the same group. |
| A4-09 / A5-08 / A2-09 / A2-06 / A2-04 | minor–nit | S–L | Verbatim copy-pasted `join` refusal property in 3 node classes; `compressor` carried twice in two write contexts; `validation/base.py` 51% mesh-only; six UI builder methods over 150 lines; `depth-sort-coordinator.ts` with 23 mutable module-level globals. |

### Theme 6 — Error handling, lifecycle, performance, security

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A14-01** | major | M | **`poisson-disk` LOD ordering is O(N²) at coarse radii** and a pure-Python per-element loop, while its docstring and README claim O(N). `buckets` holds *all input indices* per cell rather than Bridson's accepted-samples grid, and at the coarsest level the grid collapses to ≤5 cells/axis so the ±2 neighbourhood covers everything. At 1M points the coarsest level alone is ~10¹² interpreted operations — it does not finish, it hangs. It is a selectable public authoring option on a project targeting 100K–10M elements. |
| **A14-02** | major | M | **`luxar gsplat info` decodes a whole partition/nested store into RAM up to three times to print attributes** — once thrown away, once as a structural probe, once for the summary. No lazy handle, no size check. The project's own operational record documents this shape reaching 116 GB RSS on a 13 GB store. This is an *inspect* command. |
| A9-01 | minor | M | Points and Lines loaders treat **any** optional-array open failure as "attribute absent" and render defaults — Points logs the absence message precisely when the error was *not* not-found; Lines uses a bare `catch` with no discrimination. The house guard (`if (!zarr.isNotFoundError(e)) throw e`) is used correctly three modules over. |
| A9-02 | minor | S | 18 CLI paths discard the traceback for unexpected exceptions with no `--traceback` or env escape hatch (`typer.Exit` means the `from e` chain is never rendered). |
| A9-05 | minor | S | `inspect_gsplats_zarr` leaks the `ZipStore` file handle on its flat-zip fast path. |
| A14-05 | minor | M | Two paths quadratic in node count: the always-on profiler merge (per view update) and the scene-graph internal-prefix scan (per load). |
| A14-04 | minor | S | Points projection allocates a boxed `number[]` per visible point on every nD view update, in a function that otherwise documents a zero-allocation invariant. |
| A14-06 | minor | S | Layers panel runs a sort + `JSON.stringify` over the failed-loads set **every animation frame** while open. |
| A14-03 | minor | S | The whole-array per-channel dequantization branch is the one decode path not routed through the shared WASM/TS kernels, paying an `i % cols` and a closure per element. |
| A13-02 | minor | S | `robust_download` forwards custom credential headers across cross-host redirects — `requests` strips only `Authorization`, and the docstring explicitly advertises `extra_headers` as the place to put an API key. |
| A13-01 | nit | S | The LAN-exposure warning is AND-gated on wildcard CORS, but CORS is browser-only and cannot restrict `curl`; a non-loopback bind alone is what reaches the network. |
| A9-06 | nit | S | A Rust comment claims four sibling buffer checks are `assert!`; they are `debug_assert!` and are compiled out of the shipped WASM. |

### Theme 7 — Stale compat and historical cruft (low value, listed for completeness)

Genuinely small. The project's "no backwards-compatibility burden" rule is being followed.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| A6-01 | minor | L | Two "Phased Implementation Plan" documents are the canonical rendering architecture reference, and ~84 production source comments cite their phase numbers. *Downgraded from major: the specs do precisely define the vocabulary the comments use.* |
| A6-06 | minor | S | `TODO.md` ships publicly with a "legacy tags" index for tags (R10/R17/R19) that have **zero** remaining occurrences, and a DONE entry naming a removed command (`slurm-fit` → `batch-fit`). |
| A6-03 | minor | S | `SUPPORTED_SCENE_VERSIONS` advertises scene formats `0.2` and `0.3` that nothing writes, reads or branches on. |
| A6-07 | nit | L | 104 module docblocks lead with refactor provenance ("Extracted from X…"). *Verifier: most state the responsibility first and add provenance as a secondary clause — lower value than it sounds.* |
| A6-04 | nit | S | `_DictCompatMixin` is labelled "for backward compatibility" but dict-style access **is** the idiom `io/README.md` teaches — so it is live API, not cruft. Fix the label, not the code. |
| A6-05 / A7-04 | nit | S | Two renderer-selection env flags, one a literal no-op. *Verifier: documented-intentional aliases; drop the "vestigial" framing.* |
| A6-08 | nit | S | `centroidMaxReachSphere` is named "legacy" but is a live per-frame correctness constraint. Rename to `conservative`/`centroidBound`. |
| A3-01 | minor | S | `OptimConfig`/`LossConfig`/`ConstraintConfig` carry docstring examples calling a `fit_gaussian_splats(volume, optim=cfg)` API that does not exist — **and the module is `automodule`'d at `docs/api/gsplats.rst:72`, so the broken examples are published**. |
| A3-05 / A3-06 / A3-07 / A3-08 / A3-10 | minor–nit | S–M | Four unread viewer config subtrees; 107 knip-reported unused exports/types in internal barrels (the CI knip variant excludes exports/types); seven Python helpers called only by their own tests, two duplicating live inline code; one dead function + 12 dead constants; `$LUXAR_PROFILES_PATH` documented with no reader. |
| A7-02 | minor | M | `coarsen_dims` barrier provenance is stamped by `lod --recipe levels` but not by `overview` or `adaptive` — verified empirically, and the issue the code comment points at (#1600) is closed. |

### Theme 8 — Gaps no dimension covered (from the completeness critic; all verified by me)

The critic examined directories no coverage_note mentioned: the CUDA/Metal backends, the Go
launcher, `packages/luxar-viewer/{tools,scripts}` and the repo-root configs, `examples/`,
`stats/`, and the release pipeline's *artifact* steps. **I verified each of the four majors
directly** — the critic was the one stage with no adversarial verifier downstream.

| ID | Sev | Eff | Finding |
|---|---|---|---|
| **A15-05** | major | **S** | **One unseeded RNG draw makes every default gsplat fit non-reproducible.** The dynamic-ops path is *deliberately* deterministic — `DynamicOpsConfig.seed = 42`, threaded to the per-tile keep decision at `operations.py:181` as `seed=cfg.seed`. But `_select_weak_splats` at `operations.py:372` calls `torch.randint(...)` with **no `generator=`**, drawing from the ungoverned global torch RNG, and there is **zero** `torch.manual_seed` anywhere in production Python. *Verified.* Compounding it: `gsplat fit` has `--seeds` (splat count K) and `--seed-method`, but **no RNG `--seed`** — the only `--seed` in the whole CLI is on `decimate` ("Reduction seed"), a genuine vocabulary collision. Fitted stores back published figures; one line breaks bit-reproducibility of every default fit. **Highest value-per-hour item in this document.** |
| **A15-03** | major | L | **4,409 LOC of shipped native code is built, tested and linted by nothing.** 3,372 LOC CUDA (`.cu/.cuh/.cpp/.h`) + 1,037 LOC Metal (`.mm/.metal`) ride into the wheel (the exclude list drops only demo data and three research demo trees), and the Metal package compiles itself on first use. **Zero** CI mentions of `nvcc`, `xcrun metal` or `clang-format`; `make check-all` has no native arm. *Verified.* This is a second and third independent implementation of the same splat-rasterization math as the torch reference — exactly the 1:1-sync hazard the project already gates for the Rust/TS pair — with none of the protection, and a silent divergence produces slightly wrong splats rather than a crash. Cheapest fix first: a compile-only arm (`nvcc -fsyntax-only`, `xcrun metal -c`) needs no GPU. |
| **A15-04** | major | M | **The README promises four browsers; only Chromium is even configured.** `README.md:981-986` asserts Chrome/Firefox/Safari/Edge "fully supported"; `playwright.config.ts` defines exactly one project — `firefox` and `webkit` are commented out. *Verified.* The untested engines are precisely where the code branches hardest (heap-budget probing, WebGL/WebGPU capability paths). Compounds the fact that zero E2E runs in CI. Pick an honest position: enable a webkit/firefox smoke subset on the GPU runner, or rewrite the table to say what is actually verified. |
| **A15-02** | major | M | **No third-party licence notices travel with any redistributed bundle.** Only 3 licence files exist, all Luxar's own BSD-3; no `NOTICE`, no `THIRD_PARTY_LICENSES`, and the production Vite build runs **zero plugins** so nothing generates one. *Verified.* The minified bundle is redistributed four ways (npm, the wheel's `_viewer_dist`, `luxar export` folders, native bundles), and MIT/Apache-2.0/BSD all require notices to accompany binary redistribution. Also affects the baked matplotlib / Google-turbo / ColorBrewer colormap LUTs. This is what a downstream packager (conda-forge, Debian) raises first. *Note: **dataset** licensing is exemplary and gated — see What's healthy. This is the orthogonal **code** axis.* |
| A15-08 | minor | L | The Python library writes to stdout unconditionally: **547 `aprint` calls below the CLI layer** (gsplats 314, io 146, core 75), **zero** uses of `logging`, and no `verbose`/`quiet` seam on `LuxarZarrCompiler.save()` or `fit_gaussian_splats()`. For a library driven from notebooks and napari this output cannot be silenced, routed or levelled. |
| A15-09 | minor | S | **5,774 LOC of viewer build/gate/config TS+JS sits outside eslint, prettier and tsc** — 31 files including all five Playwright configs and the five `.mjs` gate scripts. Adversely selected: these are the scripts that enforce the other gates. (A9-04 already found three gates that report success having inspected nothing — unlinted, untyped JS is that bug's natural habitat.) |
| A15-07 | minor | S | The npm **library** bundle (`build:lib` + `check-lib-exports`) is built for the first time only on a release tag. The viewer **app** bundle is properly gated per-PR by `wheel-viewer` — which is what makes the sibling's absence conspicuous. Different failure modes (externalising `three`, `.d.ts` emission, side-effect-free barrel). |
| A15-06 | minor | S | The shipped viewer carries **no build or version identifier** — not in the UI, `window.__luxarDebug`, the wheel bundle, or exported folders. No `define:` block in `vite.config.ts`. A bug report ("the viewer renders black") cannot be tied to a revision across five redistribution channels. |
| ~~A15-01~~ | **nit** | S | *Claimed blocker, refuted by me.* `luxar export --native` raises `LauncherNotBuiltError` in a PyPI wheel because `publish.yml` never builds the Go launchers and the binaries are gitignored. **This is documented deliberate design**: `publish.yml:11-12` — *"Native launchers are host-specific and intentionally NOT bundled — the wheel stays py3-none-any and `luxar export --native` tells users to build them"* — restated in `_launchers/README.md`, and the prerequisite appears in CLAUDE.md and at `docs/tutorials/distributing_scenes.rst:28,122`. Bundling CGO+WebView binaries would forfeit `py3-none-any`. **Residual nit only**: `README.md:75` and `:842` advertise `--native` without naming the prerequisite (`:849` does link the tutorial). |

---

## What's healthy (verified, not assumed)

Stated because a release audit that only lists problems misrepresents the codebase, and because
several of these are things the refactor must **not** break.

**Dead code / completeness**
- One orphan production Python module out of 527 (`conftest.py`, which is pytest's).
- Of 1,534 exported viewer symbols, 26 had no external textual reference; hand-checking every
  one left exactly **one** genuinely unused (`setSortWorkerUrl`).
- **Two** `TODO`/`FIXME`/`XXX`/`HACK` markers in the entire production tree.
- All 33 `NotImplementedError` sites are legitimate documented MRO/typing stubs.
- **Zero** suppressed tests — no `@pytest.mark.skip`, no `xfail`, no `it.skip` — across 465
  Python and 631 TS test files. All 123 `skipif` uses are capability gates with reasons.
- Every documented CLI flag is parsed: 148 flags from CLAUDE.md + 38 from CLI_REFERENCE diffed
  against 323 real flags, zero unparsed. 54 of 57 real commands are documented; the 3 that
  aren't are hidden from `--help`.
- All 284 viewer config keys, all 26 URL params, all 19 `LuxarAppOptions` fields, and all 75
  `ViewerConfig` wire fields have live consumers.
- **The gauntlet caught real false positives**: `read_gsplat_node`'s BSP partitioners, the LOD
  recipe builders (string-keyed dispatch), `LuxarDeltaV3.compute_encoded_size` (a zarr codec
  API override called by zarr itself), and the 14 `decode_*_perchannel_*` re-exports knip
  flags (invoked as `wasmModule.decode_…` through the WASM interface) all look dead and are
  not.

**Cross-language contracts**
- Rust and TypeScript export the same 33–34 free functions, verified by diffing sorted name
  lists, guarded by a **3,419-line CI-enforced** parity suite with `LUXAR_REQUIRE_WASM_TESTS=1`
  set job-wide so it cannot silently skip.
- The 1D/2D display-dimension hazard is handled **identically** on both backends — both
  synthesize the phantom Cholesky diagonal as the geometric mean of the real pivots, exactly as
  CLAUDE.md requires.
- `format-contract/contract.yaml` genuinely single-sources the on-disk vocabulary into both
  languages under a `check-contract` drift gate.
- The `erf` approximation is the model for how shader duplication *should* be handled: one
  coefficient array feeds the CPU reference, the generated GLSL and the TSL node graph.

**Architecture**
- The `_zarr_compat` facade holds: precise greps for direct `zarr.open_group|open|group|
  create_array|save|consolidate_metadata` across production Python return only 5 sites.
- Viewer layering is real and green: `depcruise` reports **no violations** over 690 modules /
  1,848 dependencies, with `no-circular` and 8 layer rules at error severity.
- Four Python subpackages are back-edge-free leaves (`encoding`, `colormaps`, `shading`,
  `mesh`); `encoding` imports neither `io`, `core` nor `gsplats`.
- No plugin/middleware/extension-point machinery anywhere — the project's minimum-viable rule
  is being followed.
- **The apparent god-objects are largely false alarms.** `scene-manager.ts` is 1,547 lines but
  only 588 are method bodies with 2 methods ≥40 lines; `io/compiler.py` has already been
  decomposed into `io/_compiler/`; `io/optimise.py` and `core/group/partition.py` are flat
  collections of ~43 small named functions, not god objects.

**Error handling & resources**
- **Zero** instances of the classic `addEventListener(x.bind(this))` leak across 193 sites.
- All 3 `setInterval` sites have a matching `clearInterval` on a wired teardown path.
- The async-init dedup pattern CLAUDE.md warns about is done **correctly** everywhere checked.
- `Group._transactional_add` restores `parent_node.children` on any `BaseException`, and all 7
  `add_*` methods route through it.
- All 7 archive temp-dir callers release in a `finally`.
- Rust kernels: zero `unwrap()`, zero `.expect()`; the single `panic!` is the documented
  `validate_ndim` guard the worker routes around above 16D.

**Security** — genuinely clean, and clearly deliberately so
- `_archive.py` validates **every** member before writing a byte (rejects backslash/absolute
  names, symlinks, hardlinks, device/FIFO members, and resolves each target under the root).
  All 6 remaining raw `extractall` sites checked — no zip-slip anywhere.
- Zero `shell=True` / `os.system` in production; all 44 subprocess sites pass argv lists; the
  sbatch generator uses `shlex.quote`.
- Servers bind loopback and constrain their root; the launcher listens on `127.0.0.1:0`.
- Dataset-supplied HTML goes through a real `sanitizeHtml` (tag allowlist, attribute allowlist,
  scheme denylist, CSS tokenisation).
- Demo-data integrity is *enforced* against manifest sha256, not merely recorded, with
  quarantine on mismatch.
- All 13 GitHub Actions pinned to commit SHAs; no `pull_request_target`, no `workflow_run`, no
  `issue_comment` triggers; least-privilege permissions. Dependabot covers all five ecosystems.

**Testing**
- Python over-mocking is essentially absent: 11 `MagicMock` and 111 `mock.patch` across a
  265K-LOC suite, every patch target a genuine trust boundary.
- **Zero viewer tests mock their own subject** — every relative `vi.mock` target was resolved
  against the test's module-under-test.
- Determinism: an autouse conftest fixture seeds numpy per test; 1,944 tolerance assertions
  with only 10 looser than 0.1.
- No orphan or uncollected tests in either language.

**Areas the critic expected to be holes and were not**
- **Viewer accessibility** — 299 `aria-*` references, 49 `role` assignments, 64 `tabIndex`, 123
  programmatic `.focus()` calls, 23 `prefers-reduced-motion` references. No a11y gate and no axe
  run, but the code was clearly written with keyboard and screen-reader users in mind.
- **Dataset licensing and attribution** — *exemplary, and enforced.* `README.md:1004-1050`
  carries a per-dataset citation and licence for every demo, and `scripts/check_scene_credits.py`
  is a real gate reading the `citation` attr off **built** stores (both zarr layouts) rather than
  trusting the source — written after a store shipped uncredited on 2026-08-21. The strongest
  single piece of release hygiene in the repo.
- **Locale/timezone correctness** — all 39 `toLocaleString`/`Intl` uses are display-only, never
  round-tripped through a parser; **zero** `datetime.now()`/`utcnow()` in production Python, so
  no written manifest carries a naive local timestamp.
- **Clone-to-running-viewer without a Rust toolchain** — `pnpm dev` does *not* hard-require
  wasm-pack; `build:wasm` runs only in `build`, and the WASM shim falls back to the TypeScript
  backend, so a newcomer gets a running (degraded) dev server with no Rust installed.
- **Untrusted-store admission asymmetry is justified, not a geometry-symmetry defect** — Mesh
  alone has a metadata preflight because Mesh alone is a whole-node loader; Points/Lines/GSplats
  stream through the spatial index under the shared adaptive GPU byte budget. Chased hard and
  dropped as a false positive.

**Build & repo hygiene**
- Every junk directory on disk is untracked and gitignored — verified per-directory with
  `git ls-files`: `delme/` (90 entries) 0 tracked, `.playwright-mcp/` (113) 0, `dist/` 0,
  `coverage/` 0, `.idea/` 0. No `.DS_Store`, no `__pycache__`.
- Version consistency is genuinely gated across pyproject / package.json / CITATION.cff.
- A core `pip install luxar` stays importable — an AST walk from `luxar.cli.main` finds zero
  paths to `torch` or `scipy`.
- The npm package has a proper PR-time packaging gate (`npm pack --dry-run` + asset assertions).
- `scripts/` is not a junk drawer: 39 root files each with a purpose row in `scripts/README.md`.
- The CI change-detector is fail-safe by construction (`!= 'false'`, so a failed classifier
  selects the full suite).

---

## Sequenced refactor plan

Each step is gated by the step before it. Effort is engineering time, not calendar time.

### Phase 0 — Safety net (gating; nothing below starts first)
See §Phase 0. **Total ≈ 1 week.** Items 1–3 are hours and should land immediately.

### Phase 1 — Close the gate gaps (cheap, high leverage, mostly mechanical)
Do this before the structural work so the structural work is protected.

1. `ruff` select `B, C4, RUF, NPY, SIM` with `RUF001/2/3, B008` ignored; land **B905** and
   **B006** first — they are the families that produce wrong numbers rather than crashes. **L**

   *Measured 2026-09-02, and it reframes the size of this item.* The headline "834 production
   hits" is dominated by cosmetic rules: a wider select over the package reports ~6,700
   annotation modernizations (`UP045`/`UP006`/`UP037`/`UP035`/`UP007`), ~1,000 ambiguous-unicode
   findings in docstrings (`RUF001/2/3` — mostly deliberate em-dashes and arrows), and 939
   `TID252` relative-imports, which is a style the codebase clearly chose. Enabling everything
   would be a very large diff for very little safety.

   The **defect-bearing** subset on the shipping surface (tests and demos excluded) is only
   **129**:

   | Rule | Hits | Why it matters |
   |---|---:|---|
   | `B905` zip-without-explicit-strict | 67 | silent truncation to the shortest input |
   | `B904` raise-without-from-inside-except | 55 | destroys the exception cause chain |
   | `RUF012` mutable-class-default | 5 | shared mutable state across instances |
   | `B006` mutable-argument-default | 2 | the classic |

   `B008` is 83 hits and **all 83 are `typer.Argument` / `typer.Option` under `cli/`** — the
   mandatory Typer idiom, not a defect. Configure
   `lint.flake8-bugbear.extend-immutable-calls` rather than ignoring the rule wholesale, so the
   rule still fires on a genuine mutable call elsewhere.

   **`B905` is not a mechanical fix, which is the argument for ratcheting rather than
   bulk-fixing.** `strict=True` raises when the inputs differ in length, so applying it blindly
   changes behaviour at any site that is *currently* relying on truncation — possibly correctly.
   Each of the 67 needs a per-site decision between `strict=True` and an explicit
   `strict=False`. Land the rules with a baseline (the `scripts/complexity_baseline.json`
   pattern), so new code is blocked immediately and the 129 burn down deliberately.
2. ESLint: spread `recommendedTypeChecked`, or at minimum land `no-floating-promises`,
   `no-misused-promises`, `await-thenable`, `no-base-to-string` (24 production sites). **M**
3. CI lint parity + `ruff format --check`. **S**
4. TypeScript complexity/size ratchet mirroring `scripts/check_complexity.py`. **M**
5. `wheel-python` CI job: build, assert wheel contents, install into a bare venv, `luxar --help`. **M**
6. Recursive documentation gate (two `glob` → `rglob`). **M**
7. Fix the three fail-open gates. **S**
8. `cargo clippy` in CI; fix the swallowed exit status in `make check-rust`. **S**

### Phase 1b — Do these first regardless (hours, disproportionate value)
- **`A15-05`: seed the `torch.randint` at `operations.py:372` from `cfg.seed`.** One line, and it
  restores bit-reproducibility to every default fit. Add a real `--seed` to `gsplat fit` (and
  resolve the `--seeds`/`--seed` collision while there). **S**
- `A15-09`: widen eslint/prettier scope to the package root. **S**
- `A15-07`: add `build:lib && build:lib:check` to the existing `wheel-viewer` job. **S**
- `A15-06`: `define:` block in `vite.config.ts`; surface in `__luxarDebug` and exports. **S**

### Phase 2 — Release-visible documentation and artifacts (do before announcing)
9. Mesh + shading into the API reference, Packages Overview, CLI docs, CLAUDE.md, READMEs. **M**
10. The v3.3 → v3.4 sweep across 6 surfaces; add a docs assertion to the format-contract test. **S**
11. Fix `OptimConfig`/`LossConfig`/`ConstraintConfig` — the broken examples are published. **S**
12. Cut `CHANGELOG.md` into version sections and fold the 400 fragments. **L**
13. Normalise the `_uintN`/`_uN` encoding vocabulary **before the format is published**. **M**
13b. **`A15-02`: generate `THIRD_PARTY_LICENSES.txt`** in the production build and force-include
    it into the wheel and `luxar export` output; add colormap LUT provenance. **M**
13c. **`A15-04`: reconcile the browser matrix** — enable a webkit/firefox smoke subset on the GPU
    runner, or rewrite `README.md:981-986` to state what is actually verified. **M**
13d. `A15-01` residual nit: name the `make build-launchers` prerequisite at `README.md:75/:842`. **S**

### Phase 3 — Structural work (only where Phase 0 says the net is green)
14. `A1-03`: break the `core`↔`io._compiler` cycle (move 2 constants + 4 validators). Well
    bounded, and it unlocks the layering contract. **L**
15. `A1-01`: add the `layers` import-linter contract, seeded with dated `ignore_imports`. **M**
16. `A2-02`: collapse the 7-way fit schema onto `FitConfig`. Highest maintenance value in the
    Python package; also drops the repo's worst complexity score. **L**
17. `A2-03` + `A4-07`: hoist the four batch config dataclasses; delete `run.py`'s 56-param
    forward. **M**
18. `A14-01`: fix the poisson-disk quadratic (bucket accepted samples only) + correct the
    docstring. **M**
19. `A14-02`: attrs-only reader for `gsplat info`; decoded-bytes preflight on
    `load_gsplat_node`. **M**
20. `A2-01`: extract `runOfflineCaptureLoop` into four collaborators. **M**

### Phase 3b — Native backends (independent track, can run in parallel)
20b. **`A15-03`**: compile-only CI arm for CUDA + Metal (`nvcc -fsyntax-only`, `xcrun metal -c`,
     no GPU needed), then `clang-format --dry-run --Werror`, then decide whether the CUDA/Metal
     rasterizers get a numeric parity test against the torch reference or an explicit
     "unverified, opt-in" status. **L**

### Phase 4 — Consolidation (post-release, opportunistic)
21. Material appearance-method mixin (`A4-03`); parameterise the four `lod-refinement` modules
    (`A4-05`); route the 12 hardcoded `applyPartialExtendTolerance` sites through the
    descriptor (`A4-04`). **M each**
22. Prune `typing_utils/config.py` and `protocols.py`; shrink `ZarrWriterProtocol`. **S–M**
23. `luxar.exceptions` hierarchy. **L**
24. Makefile `include` split + shellcheck. **L**
25. `A15-08`: a `luxar.set_verbosity()` seam (or route library messages through `logging`). **L**
26. The remaining nits.

---

## Method

14 specialist finder agents (one per issue class) fanned out over the shipping surface, each
required to ground every finding in code actually read, to run a false-positive gauntlet
(dynamic dispatch / public API / cross-language contract / test-only callers), and to report
clean areas as evidence the gauntlet ran. Each dimension's findings then went to an independent
adversarial verifier instructed to **default to REFUTED** and to check every cited `file:line`
in the real code. A final completeness critic was given all fourteen reports and asked only
what they missed.

29 agents · 5.55M tokens · 2,238 tool calls · 57 minutes · 0 errors.

**113 raw findings → 51 CONFIRMED, 50 ADJUSTED, 12 REFUTED**, plus 9 from the critic. The 44%
correction rate is the point: verifiers caught inflated severities, miscounted scopes, one
recommendation that would have broken working code, and several claims that were true in
mechanism but wrong in impact.

**A known weakness in this method**: the completeness critic had no verifier downstream, and it
produced the audit's only false-positive blocker (`A15-01`) by failing the very gauntlet guard
the finders were held to — *is this documented as deliberate?* It was, twice. I verified all
nine critic findings by hand for that reason. Any future run of this workflow should put the
critic behind a verifier too.

Findings marked *"Verifier correction"* above carry a correction I have kept in the text rather
than silently absorbing, because the correction is often more informative than the finding.

Measurements in Phase 0 and Themes 1–2 that are attributed to me (`I verified`, `independently
reproduced`) were run directly against this working tree, not taken from an agent report.

**The 12 refuted findings** — recorded so they are not re-raised: `setSortWorkerUrl` as a major
dead-code finding (it is a nit); `aiohttp` as an undeclared dead dependency; `luxar.gsplats.
multiscale` as an unwired subpackage; two independent PLY parsers; `EncodingMode.CUSTOM` as a
speculative hook; `GEOMETRY_CAPABILITIES` as over-abstraction; the CLI vocabulary-rename compat
layer; `CHANGELOG.md`'s single `[Unreleased]` section as a major; `label_ids` as an authored-but
-unconsumed channel; the claim that zero E2E tests run in CI; and two viewer-layering claims.
