# TypeScript Unit Test Review: UI, Cache, and Miscellaneous Areas

**Reviewer:** Claude Opus 4.6 (1M context)
**Date:** 2026-03-31
**Scope:** 47 test files across 12 directories in `packages/luxar-viewer/src/tests/unit/`
**Files Reviewed:** All test files and their imports/source references

---

## Status

**Last updated:** 2026-03-31
**Related PRs:** None of the issues in this report were addressed in PR #53 or any subsequent PR.
**All findings below remain UNFIXED.**

| Severity | Count | Fixed | Remaining |
|----------|-------|-------|-----------|
| CRITICAL | 0 | 0 | 0 |
| HIGH | 3 | 0 | **3** |
| MEDIUM | 5 | 0 | **5** |
| LOW | ~10 | 0 | **~10** |

---

## Severity Legend

| Rating | Meaning |
|--------|---------|
| **CRITICAL** | Tests enforcing wrong behavior or hiding real bugs |
| **HIGH** | Significant gaps in coverage or test quality issues |
| **MEDIUM** | Missing edge cases or moderate improvements needed |
| **LOW** | Style, redundancy, or minor quality improvements |
| **INFO** | Observations, no action needed |

---

## 1. UI Tests

### 1.1 debug-console.test.ts
**Rating: MEDIUM**

- **Rigor:** Good. Tests XSS prevention, memory leak prevention (bound handler cleanup), and dispose behavior with meaningful assertions.
- **Completeness:** Missing tests for: message filtering, auto-scroll behavior, show/hide toggling, keyboard shortcuts, and message buffer overflow. Only covers critical fixes, not general functionality.
- **Over-mocking:** Mocks both `console-interceptor` and `log` entirely. The `console-interceptor` mock is necessary (external dependency), but it means message rendering tests exercise `formatArgAsDOMElement` without verifying it receives real interceptor data.
- **Weakness:** Tests access private methods via `any` cast (`formatArgAsDOMElement`, `renderMessage`). These tests will silently pass even if the method is renamed, since the cast bypasses type checking.

### 1.2 event-queue.test.ts
**Rating: LOW**

- **Rigor:** Excellent. Covers basic operations, drain, peek, clear, capacity/overflow, and concurrent simulation with proper assertions.
- **Completeness:** Comprehensive for the data structure's API surface.
- **Observation:** The "concurrent usage simulation" is synchronous (single-threaded JS), so it tests interleaving rather than true concurrency. This is fine for correctness, but the test name is slightly misleading.

### 1.3 dimension-sliders.test.ts
**Rating: MEDIUM**

- **Rigor:** Good memory-leak-focused tests (event listener cleanup on dispose, cleanup on rebuild).
- **Completeness:** Binary toggle tests are thorough. Missing: slider interaction tests (verifying `sceneDimsManager.setDimensionValue` is called when slider moves), keyboard navigation via `keydown`, and update behavior when dims change externally.
- **Over-mocking:** The `sceneDimsManager` mock means no verification that slider changes actually propagate to the scene dimension manager.
- **Weakness:** The `removeEventListener` spy technique (monkey-patching `removeEventListener`) is fragile -- it depends on the order of operations between spy setup and dispose.

### 1.4 helpers.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests idempotency (no duplicate overlays/errors), cleanup of global click listeners, ARIA attributes.
- **Completeness:** Covers showHelpOverlay, hideHelpOverlay, showError, clearError adequately. Missing: `showToast` tests (imported but not tested here).

### 1.5 polling-loop.test.ts
**Rating: LOW**

- **Rigor:** Excellent. Thoroughly tests start/stop idempotency, tick execution, interval changes, tickNow, statistics, error handling (continues running on throw), and restart behavior.
- **Completeness:** Very comprehensive for the class's API.
- **Minor:** The `tickNow` test does not verify that a tickNow during a running loop doesn't reset or interfere with the scheduled interval.

### 1.6 recording-panel.test.ts
**Rating: MEDIUM**

- **Rigor:** Good breadth. Covers screenshot capture, debouncing, toBlob failure, video recording lifecycle, confirmation dialog, turntable, EXR HDR, ffmpeg script generation, control visibility, bitrate calculation, and MIME type detection.
- **Over-mocking:** Heavily mocks GUI, config, helpers, log, scene-dims-manager, canvas, and URL APIs. The test is essentially verifying the orchestration logic of `RecordingPanel` against mocked dependencies. The GUI mock (`createMockController`) returns mock objects for every `add()` call, meaning the test cannot verify correct GUI structure.
- **Completeness:** Missing tests for: actual video recording flow (MediaRecorder start->ondataavailable->onstop), dimension-sweep recording, frame sequence download (ZIP packaging), and resize behavior during recording.
- **Weakness:** `canvasToBlobOverride` global variable pattern is fragile and could leak between tests if not reset properly. The test does reset it, but the pattern is error-prone.

### 1.7 scale-bar.test.ts
**Rating: LOW**

- **Rigor:** Excellent pure-function testing for `computeNiceValue` and `formatScaleValue`. Good component rendering tests.
- **Completeness:** Good coverage of edge cases (zero height, zero distance). The integration test verifying "clean labels" is a nice addition. Missing: orthographic camera behavior, unit display for non-um units.

### 1.8 ui-component.test.ts
**Rating: LOW**

- **Rigor:** Good lifecycle testing for the abstract base class via a concrete TestComponent subclass.
- **Completeness:** Tests lifecycle, visibility, event listener management (add/remove/duplicate prevention/cleanup), theme subscription, and disposal. Comprehensive.
- **Minor:** The `wasRenderCalled` / `wasAttachEventListenersCalled` workaround for initialization order is documented but adds complexity to the test fixture.

### 1.9 layers/layer-state.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests `computeUniforms`, `computeDisplayRange`, round-trip consistency, and `LayerStateManager` operations (selection modes, mutations, events, dispose).
- **Completeness:** Thorough for the layer state API. The round-trip test for uniform<->displayRange is particularly valuable.
- **Observation:** The `makeSceneGraph` helper correctly filters `group` type nodes, testing the real filtering behavior.

### 1.10 layers/colormap-layer-state.test.ts
**Rating: LOW**

- **Rigor:** Good targeted tests for colormap feature integration in LayerStateManager.
- **Completeness:** Covers reading colormap from attrs, supportsColormap detection (gsplats, has_scalars), scalarDataRange for both gsplats and points, setColormap mutation and notification. Adequate for the feature.

### 1.11 GUI Controller Tests (boolean, number, option, function, string)
**Rating: LOW**

- **Rigor:** All five controller test files follow a consistent pattern: constructor, setValue/getValue, updateDisplay, onChange/onFinishChange, name(), show/hide, dispose. Assertions are meaningful.
- **Completeness:** Good API coverage. NumberController tests include clamping, slider sync, custom updateDisplay override, and chaining.
- **Correctness:** The tests correctly verify that `setValue()` does NOT trigger `onChange` callbacks (matching lil-gui behavior), which is important for preventing infinite loops.
- **Minor:** FunctionController tests verify `this` context binding, which is a real-world concern.

### 1.12 gui/core/folder.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests constructor (root vs non-root), add() with auto-detection of types, addFolder() nesting, controllersRecursive(), open/close, show/hide, dispose.
- **Completeness:** Tests the `throw` for unsupported types (Symbol). Good depth for recursive controller collection.

### 1.13 gui/core/gui.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests constructor options, show/hide, add() for all types, addFolder, controllersRecursive, destroy.
- **Completeness:** Adequate. Integration test with complex nested structure is valuable.

### 1.14 gui/dom/event-manager.test.ts
**Rating: LOW**

- **Rigor:** Excellent. The "Memory Leak Prevention" section tests the critical `bind()` pattern, multiple elements, and window/document listeners -- all real leak sources.
- **Completeness:** Comprehensive for the EventManager API.

---

## 2. Cache Tests

### 2.1 lru-cache.test.ts
**Rating: LOW**

- **Rigor:** Excellent. Covers basic ops, size tracking (including key replacement), LRU eviction (single, multiple, oversized rejection), LRU ordering (move to end on access), edge cases, custom size functions, stress test, hit/miss counters, and eviction counters.
- **Completeness:** Very thorough. The stress test verifying O(1) behavior (1000 ops < 100ms) is a good regression gate.
- **Correctness:** All assertions match expected LRU behavior. The "reject oversized items without evicting" test is particularly important.

### 2.2 segmented-lru-cache.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests metadata file detection patterns, segment size calculation, segment isolation (metadata protected from chunk eviction), basic operations across segments, statistics, edge cases (partial matches, concurrent access), real-world zarr patterns, hit/miss/eviction counter aggregation.
- **Completeness:** Very comprehensive. The "partial match" edge case (e.g., `my.zmetadata.backup` not being metadata) is excellent.

### 2.3 decompressed-chunk-cache.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests basic operations, LRU eviction, statistics (hits/misses/evictions), key generation (makeKey/parseKey), and data type support (Float32, Uint8, Uint32).
- **Completeness:** Adequate. The `parseKey` round-trip test is valuable.

### 2.4 cached-zarr-array.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests proxy interception, caching behavior, property pass-through, cache key isolation, double-wrap prevention, error propagation (errors not cached), data integrity, and private field compatibility.
- **Completeness:** The "Private Field Compatibility" tests are critical -- they verify the `Reflect.get(target, ...)` fix for zarrita's private `#e` field access pattern. This is a regression test for a real production bug.
- **Minor strength:** `isCachedArray` and `unwrapCachedArray` have good null/undefined/primitive handling tests.

### 2.5 chunk-prefetcher.test.ts
**Rating: MEDIUM**

- **Rigor:** Good coverage of chunk parsing (v2/v3), adjacent chunk calculation, concurrency limiting, deduplication, configuration, error handling, statistics.
- **Weakness:** Heavy reliance on `setTimeout` for async assertions (`await new Promise(resolve => setTimeout(resolve, 50))`). These are timing-dependent and may flake. Consider using `vi.advanceTimersByTime()` or explicit promise resolution instead.
- **Completeness:** Missing: bounds-based filtering tests (what happens when neighbor would be out of bounds beyond the registered shape), and the interaction between `registerArrayBounds` and chunk generation for edge chunks.

### 2.6 opfs-store.test.ts
**Rating: MEDIUM**

- **Rigor:** Good. Tests init, get/set/delete/clear, LRU eviction, quota management, content hash, metadata persistence, edge cases.
- **Weakness:** The mock file system is simplified (all `getDirectoryHandle` calls return the same handle), which means nested path handling is not truly tested even though there's a test claiming to test it. The mock's `removeEntry` deletes from flat maps, not bucketed paths, so the eviction test may not match real behavior.
- **Completeness:** The bucketing algorithm tests (separate describe block) are excellent -- they test distribution, consistency, and known values independently from OPFS mocks.

### 2.7 two-level-caching-store.test.ts
**Rating: MEDIUM**

- **Rigor:** Good breadth. Tests initialization, L1->L2->HTTP cascade, content hash validation (including the bypass-cache fix), cache management, statistics, error handling, dispose, URL parameter handling, metadata routing, real-world scenarios, prefetcher integration, and URL construction (triple-slash prevention).
- **Completeness:** The content hash validation tests are particularly strong -- they verify the critical fix where validation was reading `.zattrs` from cache instead of HTTP.
- **Weakness:** Some tests have complex mock setup that's hard to follow. The `l1MaxSize: 20 * 1024 * 1024` comment notes it "must exceed SegmentedLRU MIN_METADATA_SIZE of 10MB" -- this coupling to internal implementation details is fragile.
- **Observation:** The "no-cache" URL parameter test could verify that both the first AND second `get()` produce HTTP fetches, but the assertion is `>= 1` rather than `== 2`.

---

## 3. nDim Tests

### 3.1 effective-radius-calculator.test.ts
**Rating: LOW**

- **Rigor:** Excellent. Mathematical assertions are precise (using `toBeCloseTo`). Tests cover: exact slice plane, mixed spatial/non-spatial dimensions, non-spatial dimension filtering, boundary conditions, multiple points with different radii, complex 6D scenarios, tolerance-based discrete matching, ALL discrete dimensions must match, combined discrete+spatial, non-zero slice positions, edge cases (empty arrays, beyond boundary, all dims displayed).
- **Completeness:** Very comprehensive. The "require ALL discrete dimensions to match" test is critical for correctness.
- **Correctness:** All mathematical expectations are verified against hand-calculated values.

### 3.2 ndim-calculation-projectTo3D.test.ts
**Rating: LOW**

- **Rigor:** Excellent regression test suite. Documents the exact bug (ndim=3 fallback for 4D data) with before/after comparisons. Shows the wrong values the buggy code would produce.
- **Completeness:** Tests 2D, 3D, 4D, 5D projections, dimension swapping, zero-padding, edge cases (single point, zero points, 1-2 display dims).
- **Observation:** The pure-function extraction (`calculateNdim`, `projectTo3D`) mirrors exact lines from source code. This is a strong pattern for regression testing.

### 3.3 nd-navigation-utils.test.ts
**Rating: LOW**

- **Rigor:** Good. Uses a `DimensionsBuilder` test helper for creating test data, which is clean.
- **Completeness:** Covers cycling, boundary wrapping, step size calculation (with shift/ctrl modifiers, discrete minimums, custom configs), position calculation (clamping, wrapping, discrete rounding, zero-range), key mapping, value formatting, and navigation help text.

---

## 4. WASM Tests

### 4.1 wasm-comparison.test.ts (partial read - large file)
**Rating: MEDIUM**

- **Rigor:** Appropriately uses `describe.skipIf(!wasmFilesExist)` to handle missing WASM builds.
- **Completeness:** Based on structure, tests WASM vs TypeScript output parity for multiple functions.
- **Weakness:** Could not fully review due to file size, but the pattern of comparing WASM and TypeScript outputs for identical inputs is correct.

### 4.2 wasm-performance.test.ts
**Rating: INFO**

- **Observation:** These are performance benchmarks, not correctness tests. They measure WASM vs TypeScript speedup and assert only `speedup > 0` (meaning WASM is at least as fast). The 30-second timeout and coverage instrumentation note show awareness of CI environment issues.
- **Weakness:** The `expect(result.speedup).toBeGreaterThan(0)` assertion is extremely weak -- it would pass even if WASM was 100x slower. Consider `toBeGreaterThanOrEqual(0.5)` as a minimum.

### 4.3 wasm-vs-typescript.test.ts (partial read - large file)
**Rating: INFO**

- **Observation:** Similar structure to wasm-comparison, focused on output equivalence.

---

## 5. Workers Tests

### 5.1 worker-pool.test.ts
**Rating: HIGH**

- **Rigor:** Tests are appropriately skipped when Worker API is unavailable (`describe.skipIf(!hasWorkerAPI)`).
- **Weakness:** In a Node.js/jsdom test environment, `typeof Worker === 'undefined'` is typically true, meaning **all tests in this file are skipped** in the standard `pnpm test` run. This effectively means worker pool logic has zero unit test coverage.
- **Recommendation:** Either: (a) mock the Worker API to enable these tests in Node, or (b) ensure these tests run in a browser environment (Playwright). Currently they serve as documentation more than testing.

---

## 6. Utils Tests

### 6.1 escape-html.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests HTML tags, ampersands, double quotes, safe strings, empty string.
- **Completeness:** Missing: single quotes (`'`), backticks, null bytes, unicode, very long strings. The function may or may not handle these, but they're common XSS vectors.

### 6.2 hdr-color-conversion.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests buffer sizes, return type, known values (black -> Y=64, Cb/Cr=512), brightness ordering, limited-range bounds for Y and Cb/Cr, neutral gray achromatic center.
- **Completeness:** Good coverage of the I420P10 pipeline. Does not test non-power-of-2 dimensions or odd width/height (which would affect chroma subsampling).

---

## 7. Types Tests

### 7.1 gsplats.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests type guards (`isGSplatsMetadata`, `isGSplatsUserData`), Cholesky size calculation, and compile-time interface validation.
- **Completeness:** Thorough for the type guard API. The CHOLESKY_SIZES constant validation is a good consistency check.

### 7.2 lines.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests `isLinesMetadata`, `isLinesUserData`, `isValidLineType` with positive and negative cases.
- **Completeness:** Good compile-time interface tests. Tests invalid types (123, null as any).

### 7.3 points.test.ts
**Rating: LOW**

- **Rigor:** Adequate. Tests `isPointsUserData` type guard.
- **Completeness:** Tests null, undefined, non-objects, wrong nodeType, missing nodeType, optional fields. Sufficient for the simple type guard.

---

## 8. Themes Tests

### 8.1 theme-manager.test.ts
**Rating: LOW**

- **Rigor:** Excellent. Tests singleton pattern (including reset), theme registration, theme switching (CSS variables, data-theme attribute, error on invalid), observer pattern (multiple observers, unsubscribe, error resilience), persistence (localStorage save/load/fallback), CSS variable injection (colors, typography, spacing, variable clearing on switch), and disposal.
- **Completeness:** Very comprehensive. The observer error resilience test (one observer throwing doesn't block others) is valuable.

---

## 9. Architecture Tests

### 9.1 directory-navigator.test.ts
**Rating: MEDIUM**

- **Rigor:** Good. Tests zarr detection, JSON/HTML/index listing strategies, path operations, error handling.
- **Weakness:** The fetch mock (`(globalThis as any).fetch = vi.fn()`) is set at module level, not per-test, which means mock state could leak. The `DOMParser` mock is created but not meaningfully used. The HTML parsing fallback test doesn't actually test HTML parsing (it mocks the response as failing).
- **Completeness:** Missing: WebDAV strategy tests (only tested as "fails"), actual HTML link extraction, entry type classification logic.

### 9.2 global-state.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests SceneLoaderManager and DataMonitorManager singleton patterns, CRUD operations, default selection on destroy, reset, dispose integration, and window pollution prevention.
- **Completeness:** Comprehensive for the manager pattern.

---

## 10. Core Tests

### 10.1 app.test.ts
**Rating: HIGH**

- **Rigor:** Tests initialization order, component cross-linking, dataset detection, error handling, cleanup, focus handling, dataset browser, and debug interface.
- **Over-mocking:** The test file acknowledges this in a comment: "This test file mocks 7 internal modules... Tests initialization order and mock wiring, but not real component behavior." This is an honest assessment. The entire test verifies that `LuxarApp` calls mocked constructors and methods in the right order, but does not verify any real behavior.
- **Weakness:** `vi.stubGlobal('window', ...)` and `vi.stubGlobal('document', ...)` replace the entire window/document with minimal mocks. This means any code in `LuxarApp` that accesses window properties not in the mock will silently get `undefined`. The debug interface test creates a local `URLSearchParams` mock inside `vi.stubGlobal`, which is a brittle pattern.
- **Recommendation:** As the TODO suggests, integration tests with fewer mocks would provide much higher confidence.

---

## 11. Integration Tests

### 11.1 accumulator-integration.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests that accumulators actually receive method calls, detect types on first fill, return subarrays (views not copies), and reuse buffers across loads.
- **Completeness:** The "subarrays prove views" test (modify data1, check data2) is excellent for proving zero-copy behavior.

### 11.2 gpu-pool-integration.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests geometry acquisition, update, and reuse verification with spies. Type-aware reuse (same type reuses, different type allocates new) is well-tested.
- **Completeness:** Good memory management tests (capacity growth, eviction).

### 11.3 worker-integration.test.ts
**Rating: HIGH**

- **Rigor:** Weak. Most tests merely verify data structure shapes or mock behavior patterns rather than actual integration.
- **Weakness:** The "should initialize worker when useWebWorkers=true" test just checks a config value exists. The "should verify worker query params structure" test creates an object literal and asserts its own properties. The "Fallback Behavior" tests catch a manually-thrown error. These tests provide almost no integration coverage.
- **Observation:** The Worker+WASM test creates a mock WASM module and calls mock methods. This verifies only that the mock was set up correctly.
- **Recommendation:** These tests should either be promoted to E2E (where real workers exist) or restructured to test the actual fallback code paths in the loaders, not mock-to-mock wiring.

---

## 12. Performance Tests

### 12.1 accumulator-performance.test.ts
**Rating: LOW**

- **Rigor:** Good regression tests. Zero-allocation verification (checking `stats.allocations` doesn't increase across getData calls), subarray/view proof, growth strategy validation (1.5x), type preservation (Uint8, Uint16), attribute presence tracking.
- **Completeness:** Comprehensive for the performance contract.

### 12.2 gpu-pool-performance.test.ts
**Rating: LOW**

- **Rigor:** Good. Tests reuse rates, allocation elimination, memory efficiency (active vs pooled buffers), eviction, type-aware behavior, and capacity growth strategy.
- **Completeness:** The 100-operation allocation tracking test is a good regression gate.

---

## Summary

### Overall Quality Assessment

| Area | Files | Quality | Key Concern |
|------|-------|---------|-------------|
| UI | 18 | **Good** | Recording panel over-mocking; dimension sliders missing interaction tests |
| Cache | 7 | **Very Good** | Best tested area; thorough edge cases and real-world patterns |
| nDim | 3 | **Excellent** | Mathematical rigor; great regression tests for ndim bug |
| WASM | 3 | **Adequate** | Conditional on WASM build; performance assertions too weak |
| Workers | 1 | **Poor** | All tests skipped in standard test environment |
| Utils | 2 | **Good** | Minor gaps in XSS edge cases |
| Types | 3 | **Good** | Appropriate for type guard testing |
| Themes | 1 | **Very Good** | Comprehensive observer + CSS variable testing |
| Architecture | 2 | **Good** | Directory navigator HTML parsing untested |
| Core | 1 | **Adequate** | Acknowledged over-mocking; tests wiring, not behavior |
| Integration | 3 | **Mixed** | Accumulator/GPU pool good; worker integration is mock-on-mock |
| Performance | 2 | **Good** | Effective regression gates for zero-allocation contracts |

### Top Issues by Severity

**CRITICAL:** None found. No tests enforce clearly wrong behavior.

**HIGH (3 issues):**
1. **worker-pool.test.ts** -- All tests skipped in Node.js; zero coverage of worker pool logic in standard test runs.
2. **app.test.ts** -- 7 mocked modules means the test verifies mock wiring, not application behavior. The TODO in the file acknowledges this.
3. **worker-integration.test.ts** -- Tests verify their own mock setup rather than real integration points.

**MEDIUM (5 issues):**
1. **recording-panel.test.ts** -- Heavy mocking of GUI obscures whether the panel actually builds the correct UI structure.
2. **chunk-prefetcher.test.ts** -- `setTimeout`-based async assertions are timing-dependent and may flake in CI.
3. **opfs-store.test.ts** -- Simplified mock FS doesn't test bucketed path handling that the real implementation uses.
4. **two-level-caching-store.test.ts** -- `l1MaxSize` coupling to internal `MIN_METADATA_SIZE` constant is fragile.
5. **directory-navigator.test.ts** -- HTML parsing and WebDAV strategies are only tested as "fails", not as working paths.

**LOW (numerous):**
- escape-html missing single-quote tests
- wasm-performance assertions too permissive (`> 0`)
- dimension-sliders missing actual slider interaction tests
- debug-console missing general functionality tests (filtering, auto-scroll)

### Positive Highlights

1. **Cache test suite** is the gold standard -- thorough edge cases, real-world zarr patterns, counter tracking, and segment isolation.
2. **ndim regression tests** (`ndim-calculation-projectTo3D.test.ts`) are a textbook example of regression testing: they document the exact bug, show what wrong values look like, and verify the fix.
3. **Accumulator/GPU pool performance tests** effectively guard the zero-allocation contract with statistical assertions.
4. **cached-zarr-array private field compatibility tests** prevent regression of a real production bug with zarrita's private fields.
5. **EventManager tests** properly test the `bind()` memory leak pattern that caused real issues.
6. **ThemeManager tests** comprehensively cover the observer pattern including error resilience.

---

## Recommended Next Batch

The following 5 issues are recommended for the next round of fixes, ordered by impact:

1. **[HIGH] worker-pool.test.ts -- Zero effective coverage** (Section 5.1)
   All tests are skipped in the standard Node.js/jsdom test environment because `Worker` is undefined. This means worker pool logic (concurrency, task distribution, error recovery) has no unit test coverage whatsoever. **Fix:** Mock the Worker API in the test setup so tests execute in `pnpm test`, or add equivalent Playwright-based tests that run in a real browser.

2. **[HIGH] worker-integration.test.ts -- Mock-on-mock tests** (Section 11.3)
   Tests verify their own mock setup rather than real integration points. The "initialize worker" test just checks a config value exists; the "worker query params" test asserts properties of a literal it just created. **Fix:** Restructure tests to exercise the actual fallback code paths in loaders (e.g., verify that when workers are unavailable, the main-thread fallback produces correct results).

3. **[HIGH] app.test.ts -- 7 mocked modules** (Section 10.1)
   The test file itself acknowledges it tests "mock wiring, not real component behavior." With 7 internal modules mocked, the test cannot catch regressions in initialization order, cross-component communication, or lifecycle management. **Fix:** Add at least one integration-level test that instantiates `LuxarApp` with fewer mocks (e.g., only mock WebGL context and fetch) to verify the real initialization flow.

4. **[MEDIUM] chunk-prefetcher.test.ts -- Timing-dependent assertions** (Section 2.5)
   Multiple tests use `await new Promise(resolve => setTimeout(resolve, 50))` for async assertions. These are flaky in CI under load. **Fix:** Replace `setTimeout`-based waits with `vi.useFakeTimers()` + `vi.advanceTimersByTime()`, or restructure to await explicit promise resolution from the prefetcher API.

5. **[MEDIUM] recording-panel.test.ts -- GUI mock obscures structure** (Section 1.6)
   The `createMockController` pattern returns a mock for every `add()` call, so the test cannot verify that the panel builds the correct GUI structure (correct labels, correct value ranges, correct order). **Fix:** Use a lightweight real `Folder`/`GUI` instance (from the custom gui implementation already in the codebase) instead of fully mocking it, and assert on the resulting controller tree.
