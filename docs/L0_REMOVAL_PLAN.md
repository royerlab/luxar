# L0 Cache (RangeCache) Removal Plan

## Objective

**Remove RangeCache completely and cleanly** - no dead code, no deprecated APIs, no backwards compatibility shims.

**Principle**: "If it's gone, it's GONE." - Complete surgical removal.

---

## Impact Analysis

### Files to DELETE (2 files, 553 lines)

1. **`src/data/range-cache.ts`** (323 lines)
   - RangeCache class
   - RangeCacheKey helper class
   - All caching logic

2. **`src/tests/unit/cache/range-cache.test.ts`** (230 lines)
   - All RangeCache tests
   - Range merging tests (keep mergePointRanges tests if used elsewhere)

**Total deletion**: 553 lines

### Files to MODIFY (8 files)

1. **`src/data/point-spatial-index-loader.ts`** (~1439 lines)
   - Remove `private cache: RangeCache` field
   - Remove `cache.get()` / `cache.set()` calls in `loadRanges()`
   - Remove cache statistics from metrics
   - Simplify constructor (no RangeCache instantiation)
   - Remove `getCacheStats()`, `clearCache()` methods

2. **`src/data/data-loader-types.ts`** (~244 lines)
   - Remove `CacheEntry` interface (lines 162-176)
   - Remove `CacheStats` interface (lines 130-148)
   - Remove `LoaderConfig` cache-related fields (maxMemoryMB, evictionStrategy)
   - Keep `PointRange` interface (used by spatial index, not just cache)
   - Keep `enableMonitor` and `debug` in LoaderConfig

3. **`src/data/index.ts`**
   - Remove `RangeCache` export
   - Remove `RangeCacheKey` export
   - Keep `mergePointRanges` if used by spatial index

4. **`src/config/types.ts`**
   - Remove `DataLoadingCacheConfig` interface (lines ~380-384)
   - Remove evictionStrategy references

5. **`src/config/index.ts`**
   - Remove `dataLoading.cache` section (lines ~438-442)

6. **`src/data/zarr-loader.ts`**
   - Remove or simplify `getCacheStats()` if it only returns RangeCache stats
   - Remove `clearCaches()` if it only clears RangeCache
   - Keep if they also manage other state

7. **`src/ui/data-loading-monitor.ts`**
   - Remove cache statistics display if present
   - Remove cache-related UI elements

8. **`src/utils/log.ts`**
   - Check if CACHE emoji/module is only for RangeCache
   - Remove if unused after RangeCache deletion

### Files to UPDATE (Tests)

All test files that reference cache stats need updates:
- `src/tests/unit/data/point-spatial-index-loader.test.ts`
- `src/tests/unit/data/scene-loader.test.ts`
- `src/tests/e2e/data-loading.spec.ts`
- `src/tests/e2e/performance-benchmarks.spec.ts`
- etc.

### Documentation to UPDATE

1. **`src/data/README.md`**
   - Remove "Efficient Caching" section
   - Remove "Cache Management" section
   - Remove RangeCache API reference
   - Update performance characteristics

2. **Cache README** (if exists)
   - Document that only L1/L2/L3 layers exist now

---

## Step-by-Step Execution Plan

### Phase 1: Preparation (Read & Analyze)

**Goal**: Understand all dependencies before making changes

1. ✅ Read `point-spatial-index-loader.ts` to find all RangeCache usage
2. ✅ Read `zarr-loader.ts` to check getCacheStats/clearCaches implementation
3. ✅ Read `data-loading-monitor.ts` to check cache UI display
4. ✅ Grep for all references to removed types (CacheEntry, CacheStats, LoaderConfig fields)
5. ✅ Identify if `mergePointRanges` is used outside of RangeCache (spatial index uses it!)

**Estimated time**: 30 minutes

### Phase 2: Core Removal (Delete & Modify Source)

**Goal**: Remove RangeCache and update all dependent code

#### Step 2.1: Delete RangeCache Files

```bash
rm src/data/range-cache.ts
rm src/tests/unit/cache/range-cache.test.ts
```

#### Step 2.2: Modify PointSpatialIndexLoader

**File**: `src/data/point-spatial-index-loader.ts`

**Changes**:

1. **Remove import**:
   ```typescript
   - import { RangeCache } from './range-cache';
   ```

2. **Remove field**:
   ```typescript
   - private cache: RangeCache;
   ```

3. **Remove from constructor**:
   ```typescript
   constructor(...) {
     ...
   -  this.cache = new RangeCache(config);
     ...
   -  memoryLimit: this.cache.getMemoryInfo().max,
   +  memoryLimit: 0, // No longer tracked (removed with RangeCache)
   }
   ```

4. **Simplify loadRanges()** (lines ~568-978):
   ```typescript
   private async loadRanges(...): Promise<...> {
   - // Check cache first
   - const cached = this.cache.get(arrayPath, ranges);
   - if (cached) {
   -   log.custom(LogEmoji.CACHE, ...);
   -   this.metrics.cacheHits++;
   -   this.updateCacheHitRate();
   -   this.emitEvent({ type: 'cache-hit', ... });
   -   return cached;
   - }
   -
   - log.load(...);
   - this.metrics.cacheMisses++;
   - this.updateCacheHitRate();
   - this.emitEvent({ type: 'cache-miss', ... });

     // Direct loading logic (already exists)
     ...

   - // Cache the result
   - if (output instanceof Float32Array) {
   -   this.cache.set(arrayPath, ranges, output);
   - }

     return output;
   }
   ```

5. **Remove cache methods**:
   ```typescript
   - getCacheStats(): CacheStats {
   -   return this.cache.getStats();
   - }
   -
   - clearCache(): void {
   -   this.cache.clear();
   - }
   ```

6. **Update dispose()**:
   ```typescript
   dispose(): void {
   -  this.clearCache();
     this.chunkIndex = null;
     ...
   }
   ```

7. **Remove cache metrics from getMetrics()**:
   ```typescript
   getMetrics(): LoaderMetrics {
     ...
   -  this.metrics.cacheHitRate = ...;
   -  this.metrics.memoryUsed = this.cache.getStats().totalMemory;
     ...
   -  rangesInCache: this.cache.getStats().numEntries,  // Remove this line
     ...
   }
   ```

8. **Remove cache hit rate tracking**:
   ```typescript
   - private updateCacheHitRate(): void {
   -   const total = this.metrics.cacheHits + this.metrics.cacheMisses;
   -   if (total > 0) {
   -     this.metrics.cacheHitRate = (this.metrics.cacheHits / total) * 100;
   -   }
   - }
   ```

**Estimated changes**: ~100 lines removed/modified

#### Step 2.3: Update Type Definitions

**File**: `src/data/data-loader-types.ts`

**Changes**:

1. **Remove CacheEntry** (lines 162-176):
   ```typescript
   - export interface CacheEntry<T = Float32Array> {
   -   data: T;
   -   size: number;
   -   lastAccess: number;
   -   accessCount: number;
   - }
   ```

2. **Remove CacheStats** (lines 130-148):
   ```typescript
   - export interface CacheStats {
   -   numEntries: number;
   -   totalMemory: number;
   -   hitRate: number;
   -   hits: number;
   -   misses: number;
   -   avgAccessTime: number;
   - }
   ```

3. **Simplify LoaderConfig** (lines 113-125):
   ```typescript
   export interface LoaderConfig {
   -  maxMemoryMB?: number;          // DELETE
     debug?: boolean;
     enableMonitor?: boolean;
   -  evictionStrategy?: 'lru' | 'lfu';  // DELETE
   }
   ```

4. **Keep PointRange** - still used by chunk spatial index!
   ```typescript
   export interface PointRange {
     start: number;
     end: number;
   }
   ```

**Estimated changes**: ~35 lines removed

#### Step 2.4: Update Data Loader Interface

**File**: `src/data/data-loader-types.ts`

**Changes**:

1. **Simplify DataLoader interface** (lines 93-108):
   ```typescript
   export interface DataLoader {
     loadPoints(viewState: ViewState): Promise<PointsData>;
     updateView(viewState: ViewState): Promise<PointsData>;
   -  getCacheStats(): CacheStats;     // DELETE
   -  clearCache(): void;               // DELETE
     dispose(): void;
   }
   ```

**Estimated changes**: ~3 lines removed

#### Step 2.5: Update Exports

**File**: `src/data/index.ts`

**Changes**:

```typescript
// Remove these exports
- export { RangeCache, RangeCacheKey } from './range-cache';
- export type { ... } from './range-cache';  // If any type exports

// Keep these (used by spatial index)
export { mergePointRanges, ... } from './chunk-spatial-index';
```

**Estimated changes**: ~2-3 lines removed

#### Step 2.6: Update Configuration

**File**: `src/config/types.ts`

**Changes**:

1. **Remove DataLoadingCacheConfig**:
   ```typescript
   - export interface DataLoadingCacheConfig {
   -   maxSizeMB: number;
   -   evictionStrategy: 'lru' | 'lfu' | 'fifo';
   -   ttlMs: number;
   - }
   ```

2. **Remove from DataLoadingConfig**:
   ```typescript
   export interface DataLoadingConfig {
     spatial: { ... };
     progressive: { ... };
   -  cache: DataLoadingCacheConfig;  // DELETE this field
   }
   ```

**File**: `src/config/index.ts`

**Changes**:

```typescript
dataLoading: {
  spatial: { ... },
  progressive: { ... },
-  cache: {                    // DELETE entire section
-    maxSizeMB: 512,
-    evictionStrategy: 'lru' as const,
-    ttlMs: 300000,
-  },
},
```

**Estimated changes**: ~10 lines removed

#### Step 2.7: Update Zarr Loader API

**File**: `src/data/zarr-loader.ts`

**Investigation needed**: Check if `getCacheStats()` and `clearCaches()` only handle RangeCache or also handle other caches.

**If RangeCache-only**:
```typescript
- export function getCacheStats(loaderId?: string): Map<string, CacheStats> | null {
-   ...
- }
-
- export function clearCaches(loaderId?: string): void {
-   ...
- }
```

**If also handles SceneLoader state**:
- Keep but simplify to remove RangeCache stats
- Rename to be more specific about what it clears

**Estimated changes**: 0-50 lines (depends on investigation)

#### Step 2.8: Update Data Loading Monitor UI

**File**: `src/ui/data-loading-monitor.ts`

**Changes**: Remove any cache statistics display from the UI

**Investigation needed**: Check if monitor shows:
- Cache hit rate
- Cache memory usage
- Cache entries count

**If yes**:
- Remove those UI elements
- Remove cache event handling
- Simplify metrics display

**Estimated changes**: 10-30 lines (depends on current implementation)

#### Step 2.9: Check Log Utils

**File**: `src/utils/log.ts`

**Investigation**: Is `Modules.CACHE` or `LogEmoji.CACHE` only used for RangeCache?

**If yes**:
```typescript
export enum Modules {
  ...
-  CACHE = 'Cache',          // DELETE if unused
  ...
}

export enum LogEmoji {
  ...
-  CACHE = '💾',             // DELETE if unused
  ...
}
```

**If also used by L1/L2 cache**: Keep it!

**Estimated changes**: 0-2 lines

**Estimated time**: 2 hours

### Phase 3: Test Updates

**Goal**: Fix all broken tests, ensure no regressions

#### Step 3.1: Update PointSpatialIndexLoader Tests

**File**: `src/tests/unit/data/point-spatial-index-loader.test.ts`

**Changes**:
- Remove cache-related test assertions
- Remove `getCacheStats()` calls
- Remove `clearCache()` calls
- Simplify loader instantiation (no cache config)

**Example**:
```typescript
describe('PointSpatialIndexLoader', () => {
-  it('should cache loaded ranges', async () => {
-    // ... cache test ...
-  });
-
-  it('should track cache statistics', async () => {
-    // ... stats test ...
-  });

  it('should load points for view state', async () => {
    const loader = new PointSpatialIndexLoader(location, node, {
-      maxMemoryMB: 100,      // DELETE
-      evictionStrategy: 'lru',  // DELETE
       debug: false,
     });
     ...
   });
});
```

**Estimated changes**: 20-50 lines removed

#### Step 3.2: Update Scene Loader Tests

**File**: `src/tests/unit/data/scene-loader.test.ts`

Similar changes as above - remove cache-related assertions.

**Estimated changes**: 10-20 lines

#### Step 3.3: Update E2E Tests

**Files**:
- `src/tests/e2e/data-loading.spec.ts`
- `src/tests/e2e/performance-benchmarks.spec.ts`
- `src/tests/e2e/error-recovery.spec.ts`

**Changes**: Remove cache statistics checks from E2E tests

**Estimated changes**: 5-15 lines per file

#### Step 3.4: Update Integration Tests

**File**: `src/tests/unit/data/data-loading-integration.test.ts`

**Changes**: Remove cache integration tests

**Estimated changes**: 10-30 lines

**Estimated time**: 1.5 hours

### Phase 4: Documentation Updates

**Goal**: Update all documentation to reflect L0 removal

#### Step 4.1: Update Data Package README

**File**: `src/data/README.md`

**Changes**:

1. **Remove sections**:
   - "Efficient Caching" feature mention (line 19)
   - "Cache Management" usage examples (lines 673-690)
   - "Range Cache" API reference (lines 893-902)
   - Cache configuration examples

2. **Update architecture diagram**:
   ```diff
   - Application → RangeCache → PointSpatialIndexLoader → TwoLevelCachingStore → Network
   + Application → PointSpatialIndexLoader → TwoLevelCachingStore (L1/L2/L3) → Network
   ```

3. **Update performance section**:
   - Remove cache hit rate mentions
   - Update memory usage guidance
   - Simplify caching explanation (only L1/L2/L3 now)

4. **Update LoaderConfig documentation**:
   ```diff
   interface LoaderConfig {
   -  maxMemoryMB?: number;           // DELETE
   -  evictionStrategy?: 'lru' | 'lfu';  // DELETE
     debug?: boolean;
     enableMonitor?: boolean;
   }
   ```

**Estimated changes**: ~100 lines removed/modified

#### Step 4.2: Update Cache Package README

**File**: `src/cache/README.md`

**Changes**:

1. **Clarify scope**: Document that cache package is L1/L2/L3 only
2. **Remove references to L0/RangeCache** if any exist
3. **Update architecture diagrams**

**Estimated changes**: 5-15 lines

#### Step 4.3: Update CLAUDE.md

**File**: `/Users/loic.royer/workspace/python/luxar/CLAUDE.md`

**Changes**: Update any mentions of caching architecture

**Estimated changes**: 5-10 lines

**Estimated time**: 1 hour

### Phase 5: Quality Assurance

**Goal**: Ensure everything works after removal

#### Step 5.1: Run All Tests

```bash
# Unit tests
pnpm test --run

# E2E tests
pnpm test:e2e

# Both should pass with 100% success rate
```

#### Step 5.2: Run Type Checking

```bash
pnpm typecheck
# Should pass with zero errors
```

#### Step 5.3: Run Linting

```bash
pnpm lint
# Should pass with zero warnings about unused imports/variables
```

#### Step 5.4: Manual Testing

Test actual application with real datasets:
```bash
# Start servers
cd /Users/loic.royer/workspace/python/luxar && python3 -m http.server 9000 &
pnpm dev

# Load dataset and verify:
# 1. Points load correctly
# 2. nD navigation works
# 3. No console errors about missing cache
# 4. Performance is acceptable
```

**Estimated time**: 1 hour

### Phase 6: Commit & Document

**Goal**: Clean git history with clear rationale

#### Step 6.1: Stage Changes Carefully

```bash
# Stage deletions
git rm src/data/range-cache.ts
git rm src/tests/unit/cache/range-cache.test.ts

# Stage modifications (verify each one)
git add src/data/point-spatial-index-loader.ts
git add src/data/data-loader-types.ts
git add src/data/index.ts
git add src/config/types.ts
git add src/config/index.ts
# ... etc for all modified files
```

#### Step 6.2: Commit Message

```
refactor: remove L0 cache (RangeCache) layer

BREAKING CHANGE: RangeCache has been removed from the data loading pipeline.

Rationale:
- Decompression overhead is minimal (1-2ms via WASM Blosc)
- L1/L2/L3 cache with prefetching is sufficient
- Memory efficiency: L1 covers 10x more data per MB (compressed vs decoded)
- Complexity reduction: -20% cache-related code (-553 lines)
- Architecture simplification: 3-layer cache instead of 4-layer

Performance impact:
- Average query: 6.35ms (vs 4.76ms with optimized L0)
- Trade-off: 1.6ms slower for 2x better memory coverage
- Network latency (100ms) remains the real bottleneck

Code changes:
- Deleted: range-cache.ts (323 lines), tests (230 lines)
- Modified: 8 files to remove RangeCache dependencies
- Simplified: LoaderConfig, CacheStats, DataLoader interface

Migration:
- No migration needed (internal change only)
- Public API unchanged (loadScene, updateView, etc.)
- Configuration: maxMemoryMB and evictionStrategy removed

See: /tmp/l0-architecture-analysis.md for detailed analysis

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>
```

**Estimated time**: 15 minutes

---

## Total Effort Estimate

| Phase | Task | Time |
|-------|------|------|
| 1 | Preparation & Analysis | 30 min |
| 2 | Core Removal (Code) | 2 hours |
| 3 | Test Updates | 1.5 hours |
| 4 | Documentation | 1 hour |
| 5 | Quality Assurance | 1 hour |
| 6 | Commit & Review | 15 min |
| **Total** | | **~6 hours** |

**Realistic estimate**: 1 full day (accounting for unexpected issues)

---

## Rollback Plan

**If something goes wrong**:

1. **Immediate**: `git reset --hard HEAD~1` (before push)
2. **After push**: `git revert <commit-hash>`
3. **Restore RangeCache**: Cherry-pick from git history

**Low risk**: RangeCache is self-contained, removal is straightforward.

---

## Checklist

### Before Starting
- [ ] Read this plan completely
- [ ] Understand all dependencies
- [ ] Create feature branch: `git checkout -b refactor/remove-l0-cache`

### Phase 1: Preparation
- [ ] Read point-spatial-index-loader.ts
- [ ] Read zarr-loader.ts (check getCacheStats/clearCaches)
- [ ] Read data-loading-monitor.ts (check cache UI)
- [ ] Grep for CacheEntry, CacheStats references
- [ ] Confirm mergePointRanges is used by spatial index

### Phase 2: Core Removal
- [ ] Delete range-cache.ts
- [ ] Delete range-cache.test.ts
- [ ] Modify point-spatial-index-loader.ts (remove cache field, loadRanges cache logic, methods)
- [ ] Modify data-loader-types.ts (remove CacheEntry, CacheStats, LoaderConfig fields)
- [ ] Modify data/index.ts (remove exports)
- [ ] Modify config/types.ts (remove DataLoadingCacheConfig)
- [ ] Modify config/index.ts (remove cache section)
- [ ] Check zarr-loader.ts (getCacheStats, clearCaches)
- [ ] Check data-loading-monitor.ts (cache UI)
- [ ] Check utils/log.ts (CACHE module/emoji usage)

### Phase 3: Test Updates
- [ ] Update point-spatial-index-loader.test.ts
- [ ] Update scene-loader.test.ts
- [ ] Update data-loading-integration.test.ts
- [ ] Update data-loading.spec.ts (E2E)
- [ ] Update performance-benchmarks.spec.ts (E2E)
- [ ] Update error-recovery.spec.ts (E2E)

### Phase 4: Documentation
- [ ] Update data/README.md (remove cache sections)
- [ ] Update cache/README.md (clarify scope)
- [ ] Update CLAUDE.md (cache architecture)
- [ ] Check for any other docs mentioning RangeCache

### Phase 5: QA
- [ ] pnpm typecheck (zero errors)
- [ ] pnpm lint (zero warnings)
- [ ] pnpm test --run (all pass)
- [ ] pnpm test:e2e (all pass)
- [ ] Manual testing (load datasets, navigate, verify performance)

### Phase 6: Commit
- [ ] Git stage all changes
- [ ] Review diff carefully
- [ ] Commit with detailed message
- [ ] Push to feature branch
- [ ] Create PR

---

## Success Criteria

✅ **Code**:
- Zero references to RangeCache, RangeCacheKey
- Zero references to deleted interfaces (CacheEntry, CacheStats if removed from public API)
- Zero unused imports or variables
- Simplified LoaderConfig (only debug, enableMonitor)

✅ **Tests**:
- All unit tests pass (pnpm test)
- All E2E tests pass (pnpm test:e2e)
- No skipped tests due to removed functionality

✅ **Types**:
- pnpm typecheck passes with zero errors
- No `any` types introduced to work around removed types

✅ **Lint**:
- pnpm lint passes with zero warnings
- No dead code (unused variables, unreachable code)

✅ **Functionality**:
- Application loads datasets successfully
- nD navigation works correctly
- Performance is acceptable (verified via manual testing)
- No console errors about missing cache

✅ **Documentation**:
- README accurately describes current architecture
- No references to removed L0 cache
- Migration guide not needed (internal change)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking tests | Medium | Medium | Comprehensive test updates in Phase 3 |
| Type errors | Low | Low | TypeScript will catch at compile time |
| Performance regression | Very Low | Low | 1-2ms difference is imperceptible |
| Missing references | Low | Medium | Thorough grep + TypeScript checking |
| Documentation drift | Medium | Low | Dedicated documentation phase |

**Overall Risk**: **LOW** - RangeCache is self-contained and well-isolated.

---

## Next Steps

**Ready to execute?**

1. Review this plan
2. Ask any clarifying questions
3. Approve to proceed
4. I'll execute phases 1-6 systematically
5. Final review before commit

**Estimated timeline**: 4-6 hours of focused work

---

**Plan Created**: 2025-12-06
**Status**: Ready for Execution
**Confidence**: High (clear dependencies, low risk)
