# TypeScript Data Unit Tests -- Code Review

**Date**: 2026-03-31
**Scope**: `packages/luxar-viewer/src/tests/unit/data/` and `packages/luxar-viewer/src/tests/unit/data/loaders/`
**Reviewer**: Claude Opus 4.6 (1M context)

**Severity Legend**:
- **CRITICAL**: Incorrect behavior being enforced, or test that silently passes regardless of correctness
- **HIGH**: Missing coverage for important code paths or edge cases
- **MEDIUM**: Suboptimal test design, mild over-mocking, or minor gaps
- **LOW**: Style, naming, or minor redundancy issues
- **INFO**: Observations, no action needed

---

## Summary

| Category | Count |
|----------|-------|
| Files reviewed | 22 |
| CRITICAL issues | 3 |
| HIGH issues | 8 |
| MEDIUM issues | 11 |
| LOW issues | 6 |
| INFO observations | 5 |

Overall the test suite is **above average** for a viewer codebase. The strongest files are the pure-logic tests (gsplats-processor, lines-clipping, nd-transform, transferable-accumulator, spatial-query-builder) which test real code with meaningful assertions. The weakest are the integration-ish tests (zarr-loader, scene-loader, data-loading-integration) which over-mock to the point where they primarily verify mock wiring rather than real behavior.

---

## Per-File Reviews

### 1. `array-decoder.test.ts`

**Verdict**: Strong -- real fixture-based integration tests with Python-generated zarr data.

| Severity | Finding |
|----------|---------|
| LOW | Multiple tests have `// TODO: Blosc decompression error` comments but are not `.skip`-ed. These comments are misleading -- the tests clearly pass (since the suite runs). Either the TODO is stale or the comment is inaccurate. Clean up. |
| MEDIUM | `it.skip('should handle empty arrays gracefully')` at line 548 is permanently skipped. The comment says "mocking zarr.Array interface is too complex." This is a real edge case that deserves coverage -- consider creating a small fixture via `generate_test_data.py` instead of mocking. |
| HIGH | **Missing negative test for corrupted LUT indices within a real zarr fixture.** The `decodeLUTIndices` error path is tested with synthetic data (line 514-526) but never with an actual corrupt fixture. A round-trip fixture would catch Python/TS desync. |
| INFO | The test for `decodeLUTIndices` out-of-range (line 514) calls a method directly (`decoder.decodeLUTIndices`). This tests a public method that may be refactored to private -- consider testing through `decode()` instead for resilience. |

---

### 2. `data-accumulator.test.ts`

**Verdict**: Solid coverage of all three accumulator types with meaningful assertions.

| Severity | Finding |
|----------|---------|
| MEDIUM | The growth test at line 40-45 only checks `>= 5000` but does not verify the *exact* growth factor (1.5x). The comment says `1000 -> 1500 -> 2250 -> 3375 -> 5062.5 (rounded to 5063)` but only asserts `>= 5000`. This means a 2x growth factor would also pass. If the 1.5x factor is a deliberate design decision, assert it precisely. |
| LOW | `LinesDataAccumulator` segment capacity growth test (line 175-181) uses imprecise comment "estimated as ceil(1200/1.5) = 800" but asserts `>= 800`. The estimation logic is the critical part -- a tighter assertion or a direct test of the estimation formula would be better. |
| INFO | Memory calculation test (line 138-143) hardcodes `32 bytes per point`. If the accumulator adds a new attribute, this test silently becomes wrong. Consider computing expected from the same constants the source uses. |

---

### 3. `data-loading-integration.test.ts`

**Verdict**: Heavy over-mocking. Tests primarily verify mock wiring, not real logic.

| Severity | Finding |
|----------|---------|
| CRITICAL | **The entire SceneLoaderManager is mocked (line 159-210), and so is THREE.js (line 32-123).** Per CLAUDE.md guidelines: "Mock external dependencies, not your own code." `SceneLoaderManager` and `SceneLoader` are owned code. As a result, these tests verify that the `data/index.ts` facade correctly calls mock methods -- they cannot catch any regression in the actual loader pipeline. |
| MEDIUM | `getMockLoader()` helper (line 262-286) is created in `beforeEach` but never actually used (the mock is declared then immediately discarded: `void getMockLoader()`). Dead code. |
| MEDIUM | The concurrent operations test (line 494-530) asserts that three concurrent `loadScene` calls all succeed and `scenes[2]` is defined, but never verifies that the first two were actually superseded. The comment says "Only last one should be active" but nothing enforces this. |
| LOW | The test imports `SimpleDims` from `../../../types/dims` at line 28 but only uses it in one test. Minor coupling. |

---

### 4. `data-loading-monitor.test.ts`

**Verdict**: Good -- tests real DOM interactions with the monitor UI.

| Severity | Finding |
|----------|---------|
| MEDIUM | Button click tests (lines 211-232) use `if (minimizeBtn)` guards that silently skip assertions when buttons are not found. If the DOM structure changes, these tests pass vacuously. Use `expect(minimizeBtn).not.toBeNull()` before clicking. |
| LOW | The "should not expose any global window variables" test (line 238-252) checks for specific string patterns in `Object.keys(window)`. This is brittle if names change. Also, vitest/jsdom may pollute `window` differently than a real browser. |
| INFO | The polling architecture (events are queued until `forceUpdate()`) is well-tested. Good pattern. |

---

### 5. `data-monitor-integration.test.ts`

**Verdict**: Decent -- tests real `DataMonitorManager` singleton and `SceneLoader` construction.

| Severity | Finding |
|----------|---------|
| MEDIUM | `SceneLoader` integration test (line 198-223) creates a `SceneLoader` and then asserts `connectSpy` was NOT called, with comment "we verify the monitor is ready to receive connections." This test asserts almost nothing -- it only confirms that `connectLoader` was not called without loading a scene. The actual connection logic is untested. |
| LOW | Rapid state cycling test (line 517-529) asserts `isVisible=true, isExpanded=false` after 10 cycles (10 % 3 = 1). This is a correct and clever edge case test. |

---

### 6. `dtype-support.test.ts`

**Verdict**: Weak -- tests language/platform features rather than application code.

| Severity | Finding |
|----------|---------|
| HIGH | **Tests TypedArray `instanceof` checks (lines 11-28) and `BufferAttribute.normalized` behavior (lines 31-60) which are JavaScript/THREE.js platform behaviors, not application code.** These will always pass as long as the JS runtime and THREE.js work correctly. They do not test any Luxar code path. |
| MEDIUM | The memory efficiency test (lines 188-221) computes a savings percentage from hardcoded byte sizes. It tests arithmetic, not code. If the actual code uses different dtypes, this test won't catch it. |
| INFO | The `LoadedPointsData` construction tests (lines 63-141) are essentially type-checking tests. They verify that the TypeScript interfaces accept certain data shapes. These have marginal value in a `.test.ts` file since TypeScript compilation already enforces this. |

---

### 7. `encoded-range-extraction.test.ts`

**Verdict**: Excellent -- directly tests a critical bug fix path with real zarr fixtures.

| Severity | Finding |
|----------|---------|
| INFO | The `extractRangesFromDecoded` helper (line 62-79) is a local copy of the production logic. If the production code changes, this test won't catch the regression since it tests its own copy. Consider importing the actual function. However, the comment (line 55-58) acknowledges this is deliberate for documentation of the bug. Acceptable. |
| LOW | The bug simulation test (line 165-182) is valuable for documentation but could be a `describe` comment rather than a test that asserts incorrect behavior is incorrect. |

---

### 8. `geometry-update-manager.test.ts`

**Verdict**: Good coverage of the GeometryUpdateManager with real code tested.

| Severity | Finding |
|----------|---------|
| MEDIUM | `validateColorMode` test (line 369-377) asserts that calling with mismatched types does not throw, but **never actually asserts the warning was logged**. The `consoleSpy` is created and restored but never checked with `expect(consoleSpy).toHaveBeenCalled()`. This makes it a no-op test. |
| HIGH | `createPointsMaterial` tests (lines 388-402) only assert `instanceof THREE.ShaderMaterial`, which is always true since THREE.ShaderMaterial is mocked. They never verify that opacity, gamma, or blending mode parameters are passed through correctly. |

---

### 9. `gsplats-chunk-spatial-index.test.ts`

**Verdict**: Strong -- thorough testing of spatial query logic with clear edge cases.

| Severity | Finding |
|----------|---------|
| LOW | `computeGSplatsTolerance` test (line 187-188) asserts hidden dim tolerance is `step * 3.0` ("default tolerance") but the magic number 3.0 is not explained. It should reference a named constant if one exists. |

---

### 10. `gsplats-processor.test.ts`

**Verdict**: Excellent -- one of the best test files. Tests real math (Mahalanobis distance, Cholesky extraction, attenuation), has workspace reuse safety tests, and covers discrete/continuous dimension handling.

| Severity | Finding |
|----------|---------|
| INFO | The cross-correlated hidden dimension test (line 469-516) with detailed math comments is exemplary. This is the gold standard for how nD gsplat tests should be written. |

---

### 11. `lines-clipping.test.ts`

**Verdict**: Excellent -- comprehensive clipping tests with precise numerical assertions.

| Severity | Finding |
|----------|---------|
| HIGH | **WASM comparison tests are gated by `wasmFilesExist` (line 20) but no test actually uses this flag in the visible code.** If there are WASM comparison tests later in the file, they may be silently skipped when WASM is not built. This should be surfaced explicitly (e.g., `it.skipIf(!wasmFilesExist)`). |
| INFO | The 2D display padding test (line 172-184) correctly verifies Z=0 padding. Good edge case. |

---

### 12. `loader-orchestrator.test.ts`

**Verdict**: Good -- clean tests with proper mock isolation for the orchestrator pattern.

| Severity | Finding |
|----------|---------|
| MEDIUM | All three loader implementation modules are mocked (lines 16-49). Per CLAUDE.md, this is acceptable since the orchestrator tests should isolate orchestration logic. However, the mocked `PointSpatialIndexLoader` (line 17-25) returns a fixed mock regardless of constructor args. This means the test can't verify that correct config is passed to the loader constructor. Consider adding a spy on the constructor args. |
| LOW | The `getLoaderType` tests (lines 300-322) are straightforward map-lookup verification. Not high value but acceptable as regression guards. |

---

### 13. `nd-transform.test.ts`

**Verdict**: Excellent -- pure function tests with clear mathematical verification.

| Severity | Finding |
|----------|---------|
| HIGH | **Missing test for `composeNdTransforms` with mixed affine + permutation on the same dimension.** What happens if parent has `{scale: 2, offset: 10}` and child has `{permutation: [1, 0]}`? This is an edge case that could cause silent data corruption. |
| INFO | The sibling backtracking test (line 217-225) is a smart edge case -- verifies tree traversal doesn't leak sibling transforms. |

---

### 14. `point-spatial-index-loader.test.ts`

**Verdict**: Moderate -- tests real loader but with heavy zarr mocking.

| Severity | Finding |
|----------|---------|
| MEDIUM | The mock for `zarr.get` (line 130-135) returns zero-filled arrays regardless of the slice range. This means tests verify that data flows through the pipeline but cannot catch off-by-one errors in range calculations or incorrect slice parameters. |
| HIGH | **Missing test for cache invalidation behavior.** The loader likely caches chunks -- there is no test verifying that changing the viewState causes re-fetching of new chunks and eviction of old ones. |

---

### 15. `range-loader.test.ts`

**Verdict**: Extremely minimal -- only 2 tests covering `detectEncoding`.

| Severity | Finding |
|----------|---------|
| CRITICAL | **Only 27 lines of tests for what appears to be a significant module (`RangeLoader`).** The `RangeLoader` class handles loading ranges from zarr arrays with various encodings (array_ref, broadcasted, lut, quantized, direct). Only the `detectEncoding` static method is tested, and only for the `array_ref` case. All actual range loading logic is completely untested. |
| HIGH | Missing tests for: `detectEncoding` with all encoding types (broadcasted, lut_uint8, bounded_scalar_uint8, rgb_uint8, float32, direct), `loadRange` method, error handling for malformed metadata. |

---

### 16. `scene-graph-builder.test.ts`

**Verdict**: Good for utility methods, weak for async building.

| Severity | Finding |
|----------|---------|
| MEDIUM | The comment at line 132-138 explicitly acknowledges that `buildSceneGraph` with child node construction is NOT tested due to zarrita ESM issues. This is a significant gap since `buildSceneGraph` is the primary entry point. The static utility methods (`countNodeTypes`, `findNodesByType`, `findNodeByPath`) are well-tested. |
| LOW | The `edge cases in utility methods` test (line 345-363) tests a node with `children` property missing entirely (not just empty array). This is defensive and good. |

---

### 17. `scene-loader.test.ts`

**Verdict**: Heavy over-mocking. Limited real code execution.

| Severity | Finding |
|----------|---------|
| CRITICAL | **THREE.js is fully mocked (100+ lines of mock, lines 14-179), zarrita is fully mocked, PointSpatialIndexLoader is fully mocked, material manager is fully mocked, DataMonitorManager is fully mocked.** The SceneLoader's core logic is to orchestrate these dependencies, so when everything is mocked, the test verifies almost nothing. Most test assertions are `expect(THREE.Group).toHaveBeenCalled()` which will always pass given the mocking setup. |
| HIGH | No test verifies that transforms are actually applied to THREE.js objects, that materials receive correct parameters, or that the scene graph hierarchy is correctly constructed in THREE.js. The test comments acknowledge this: "These attrs are tested indirectly via E2E tests" (line 582-588). |

---

### 18. `view-state-manager.test.ts`

**Verdict**: Excellent -- thorough testing of ViewState initialization and validation logic.

| Severity | Finding |
|----------|---------|
| LOW | The test at line 257 asserts `slicePosition[0] === 0` with comment "floor(0.5) = 0 (starts at minimum, floored)". This is correct but the comment is confusing -- `Math.floor(0.5)` is `0`, but the actual logic might be `starts at range minimum = 0.5, then floored = 0`. The comment conflates two behaviors. |
| INFO | The `preserve spatial, cyclic, categories, and description fields` test (line 306-347) is a good regression guard for field passthrough. |

---

### 19. `zarr-loader.test.ts`

**Verdict**: Extensive but suffers from the same over-mocking problem as scene-loader.test.ts.

| Severity | Finding |
|----------|---------|
| HIGH | **Most tests assert only `expect(scene).toBeTruthy()` or `expect(THREE.Group).toHaveBeenCalled()` (e.g., lines 278-281, 298-301, 338-340, 388-389, 471-473, 508-509).** These assertions pass trivially given the mock setup. No test verifies the actual scene graph structure, node names, transform values, or dimension parsing results. |
| MEDIUM | The `PointSpatialIndexLoader` mock path is wrong: `vi.mock('../data/point-spatial-index-loader')` (line 189) uses a relative path from the test file but should be `../../../data/point-spatial-index-loader`. This mock may not actually intercept the real import, though vitest's module resolution might resolve it anyway. Verify this mock is actually being used. |
| MEDIUM | The translate matrix at line 482 is `[1,0,0,10, 0,1,0,20, 0,0,1,30, 0,0,0,1]` which is **row-major** format. Per CLAUDE.md Critical Gotchas, THREE.js uses column-major. This test may be silently encoding a wrong assumption, or the loader may transpose it. Either way, the test should assert the resulting position to verify correctness. |

---

### 20. `loaders/integration-example.test.ts`

**Verdict**: Good -- tests the integration example loader with real projection logic.

| Severity | Finding |
|----------|---------|
| LOW | The worker pool mock (line 8-19) is set up but `config.dataLoading.performance.useWebWorkers` is `false`, so the worker path is never exercised. The mock is dead code. |
| MEDIUM | `queryVisibleRangesExample` tests (line 187-245) test a function whose name contains "Example" -- suggesting it may be demonstration code rather than production code. If so, these tests have limited value for regression prevention. |

---

### 21. `loaders/spatial-query-builder.test.ts`

**Verdict**: Excellent -- thorough testing of the spatial query builder with clear edge cases.

| Severity | Finding |
|----------|---------|
| INFO | The `executeSpatialQuery` synchronous test (line 396-411) explicitly verifies the result is not a Promise. Good defensive test for API stability. |
| LOW | The `shouldExtendVisibility` tests (lines 212-269) cover all major paths including `undefined` and empty array. Well done. |

---

### 22. `loaders/transferable-accumulator.test.ts`

**Verdict**: Excellent -- comprehensive coverage of the transferable buffer lifecycle.

| Severity | Finding |
|----------|---------|
| INFO | The detach/adopt cycle tests (lines 88-139) thoroughly cover the ownership transfer pattern including error cases. This is a model for how buffer management tests should be written. |
| LOW | The `createGSplatsAccumulator` test (line 259-280) tests with k=6 (3D) and k=10 (4D) but not k=1 (1D) or k=21 (6D). Minor gap. |

---

## Cross-Cutting Issues

### 1. Over-Mocking Pattern (HIGH)

Files affected: `data-loading-integration.test.ts`, `scene-loader.test.ts`, `zarr-loader.test.ts`

These three files mock virtually every dependency, including owned code (SceneLoaderManager, SceneLoader, PointSpatialIndexLoader, GeometryUpdateManager). This violates the CLAUDE.md testing principle: "Mock external dependencies (network, file system), not your own code." The result is tests that verify mock wiring rather than real behavior. These tests will NOT catch regressions in:
- Scene graph construction
- Transform application
- Material parameter passthrough
- Dimension parsing
- Spatial index integration

**Recommendation**: Either refactor these into true integration tests that use real objects with only zarr/network mocked, or accept that they provide limited value and rely on E2E tests (which the comments already acknowledge).

### 2. Assertion-Free or Trivially-True Assertions (HIGH)

Files affected: `zarr-loader.test.ts`, `scene-loader.test.ts`

Many tests only assert `expect(scene).toBeTruthy()` or `expect(THREE.Group).toHaveBeenCalled()`. Given the mock setup, these always pass. They provide a false sense of security.

**Recommendation**: Each test should assert at least one *specific* property of the output that would fail if the tested behavior regressed. For example, instead of `expect(scene).toBeTruthy()`, assert `expect(scene.userData.sceneDimensions.dimensions).toHaveLength(4)`.

### 3. Duplicated Test Logic (MEDIUM)

`chunkIndicesToSplatRanges` in `gsplats-chunk-spatial-index.test.ts` and `chunkIndicesToRanges` in `spatial-query-builder.test.ts` test nearly identical range conversion logic with identical test cases (same input values, same expected outputs). This suggests either code duplication in the source or two separate implementations of the same algorithm.

**Recommendation**: Verify whether these are the same function exported from different modules. If so, consolidate. If they are genuinely different implementations, add a test that explicitly compares their outputs for identical inputs.

Similarly, `mergeRanges` is tested in both `gsplats-chunk-spatial-index.test.ts` and `spatial-query-builder.test.ts` with identical test cases.

### 4. Missing Error Path Coverage (MEDIUM)

Across the test suite, error paths are underrepresented:
- No tests for malformed Cholesky factors (wrong length for ndim)
- No tests for NaN/Infinity in positions or amplitudes
- No tests for concurrent dispose + load race conditions
- No tests for out-of-memory scenarios during accumulator growth

---

## Recommendations (Prioritized)

1. **[CRITICAL] Add real tests for `RangeLoader`** -- the current 27-line test file is grossly insufficient for a module that handles 6+ encoding types.

2. **[HIGH] Reduce over-mocking in `zarr-loader.test.ts` and `scene-loader.test.ts`** -- either make these real integration tests or add assertion specificity.

3. **[HIGH] Add cache invalidation tests for `PointSpatialIndexLoader`** -- the current tests don't verify re-fetching behavior.

4. **[HIGH] Add mixed affine+permutation test for `composeNdTransforms`** in `nd-transform.test.ts`.

5. **[MEDIUM] Fix validateColorMode test** in `geometry-update-manager.test.ts` -- actually assert the warning was logged.

6. **[MEDIUM] Fix button click tests** in `data-loading-monitor.test.ts` -- replace `if` guards with explicit `expect().not.toBeNull()`.

7. **[MEDIUM] Replace `dtype-support.test.ts`** with tests that exercise actual application code paths, not platform features.

8. **[LOW] Clean up stale TODO comments** in `array-decoder.test.ts` about Blosc decompression errors.
