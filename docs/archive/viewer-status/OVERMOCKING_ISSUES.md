> **⚠️ Archived — point-in-time review, not re-verified.** Finding statuses below have **not** been re-verified against current `main`. Treat this as a historical snapshot, not a live task list; verify any finding against current code before acting on it. See [the archive README](../README.md).

# Over-Mocking Issues - Detailed Analysis

## Summary

Multiple test files mock internal modules instead of just external dependencies, reducing test effectiveness and creating brittle tests.

## Severity Levels

- **🔴 Critical**: Tests are essentially useless (mocking >5 internal modules)
- **🟡 Moderate**: Some useful testing but significant mocking issues (3-5 internal mocks)
- **🟢 Minor**: Mostly good, minor improvements possible (1-2 internal mocks)

---

## 🔴 Critical Issues

### 1. `scene/scene-manager.test.ts` (702 lines)

**Mocks:**
- `THREE` ✅ (external - correct)
- `PostProcessingManager` ❌ (internal)
- `ControlsManager` ❌ (internal)
- `../../../data` ❌ (internal - entire data module!)
- `../../../ui/helpers` ❌ (internal)
- `../../../utils/hdr-detection` 🟡 (utility, could use real with spies)

**Impact**: Tests verify mock behavior, not real scene management logic.

**Fix Priority**: HIGH
**Estimated Effort**: 4-6 hours
**Approach**: Use real implementations with test fixtures

---

### 2. `core/app.test.ts` (668 lines)

**Mocks** (from grep):
- `THREE` ✅ (external)
- `scene-manager` ❌ (internal)
- `animation-controller` ❌ (internal)
- `input-handler` ❌ (internal)
- `rendering-controls` ❌ (internal)
- `dataset-browser` ❌ (internal)
- `helpers` ❌ (internal)
- `scene-dims-manager` ❌ (internal)
- Plus several more...

**Impact**: App initialization tests don't test real initialization.

**Fix Priority**: HIGH
**Estimated Effort**: 5-7 hours
**Approach**: Mock only WebGL/DOM, use real application modules

---

### 3. `data/data-loading-integration.test.ts`

**Mocks:**
- `THREE` ✅ (external)
- `scene-loader` ❌ (internal)
- `scene-loader-manager` ❌ (internal)
- `point-spatial-index-loader` ❌ (internal)
- `material-manager` ❌ (internal)
- `data-loading-monitor` ❌ (internal)

**Impact**: Integration tests don't actually test integration!

**Fix Priority**: CRITICAL (it's an integration test!)
**Estimated Effort**: 3-4 hours
**Approach**: Use real loaders with test fixtures

---

## 🟡 Moderate Issues

### 4. `data/zarr-loader.test.ts`

**Mocks:**
- `THREE` ✅ (external)
- `zarrita` ✅ (external)
- `point-spatial-index-loader` ❌ (internal)
- `material-manager` ❌ (internal)

**Impact**: Doesn't test real integration with spatial index loader.

**Fix Priority**: MEDIUM
**Estimated Effort**: 2-3 hours

---

### 5. `data/scene-loader.test.ts`

**Mocks:**
- `THREE` ✅ (external)
- `zarrita` ✅ (external)
- `material-manager` ❌ (internal)
- `data-loading-monitor` ❌ (internal)
- `point-spatial-index-loader` ❌ (internal)
- `data-monitor-manager` ❌ (internal)

**Fix Priority**: MEDIUM
**Estimated Effort**: 2-3 hours

---

### 6. `data/point-spatial-index-loader.test.ts`

**Mocks:**
- `THREE` ✅ (external)
- `zarrita` ✅ (external)
- `chunk-spatial-index` ❌ (internal)
- `range-cache` ❌ (internal)

**Fix Priority**: MEDIUM
**Estimated Effort**: 2 hours

---

## 🟢 Good Examples (Learn From These)

### ✅ `rendering/postprocessing-manager.test.ts`

**Mocks:**
- `postprocessing` ✅ (external library only)

**Why Good**: Only mocks the external postprocessing library, tests real manager logic.

---

### ✅ `rendering/point-material.test.ts`

**Mocks:**
- `THREE.ShaderMaterial` ✅ (partial mock, keeps real THREE.js)

**Why Good**: Minimal mock, tests real shader generation logic.

---

## Refactoring Priorities

### Phase 1: Quick Wins (1-2 days)
1. ✅ Create `TESTING_GUIDELINES.md` - DONE
2. Create test fixture library
3. Fix `data-loading-integration.test.ts` (ironic integration test that doesn't integrate)

### Phase 2: High Impact (1 week)
4. Refactor `scene-manager.test.ts`
5. Refactor `app.test.ts`
6. Fix data loader tests (zarr-loader, scene-loader)

### Phase 3: Complete Cleanup (2-3 days)
7. Fix spatial index loader tests
8. Review all remaining tests
9. Update pre-commit hooks

---

## Test Fixture Strategy

### Required Fixtures

Create in `/test-fixtures/`:

```
test-fixtures/
├── basic_example.zarr/          # 100 points, 3D, simple colors
├── nd_example.zarr/              # 4D dataset for nD testing
├── hdr_example.zarr/             # HDR color values
├── categorical_example.zarr/     # Categorical dimensions
├── hierarchical_example.zarr/    # Scene graph with transforms
├── lines_example.zarr/           # Lines rendering
└── empty_example.zarr/           # Edge case: no points
```

### Fixture Generation Script

```python
# test-fixtures/generate_test_data.py
# Already exists! Just needs:
# 1. More comprehensive coverage
# 2. Documentation of each fixture
# 3. Versioning/checksums
```

---

## Refactoring Template

```typescript
// BEFORE: Over-mocked test
vi.mock('../data/scene-loader', () => ({
  SceneLoader: vi.fn().mockImplementation(() => ({
    loadScene: vi.fn().mockResolvedValue({ points: [] })
  }))
}));

test('loads scene', async () => {
  // Tests mock, not real code
  await manager.loadScene('test.zarr');
  expect(mockLoadScene).toHaveBeenCalled();
});

// AFTER: Real implementation with fixture
import { SceneLoader } from '../data/scene-loader';

test('loads scene from fixture', async () => {
  const loader = new SceneLoader();
  const scene = await loader.loadScene('test-fixtures/basic_example.zarr');

  // Tests real behavior
  expect(scene.points.length).toBe(100);
  expect(scene.points[0].positions.length).toBe(300); // 100 points * 3
});
```

---

## Metrics

### Current State
- **Total test files**: 37
- **Files with excessive mocking**: 6 (~16%)
- **Lines of mock code**: ~500 lines
- **Mock vs Real ratio**: ~30% mocks, 70% real

### Target State
- **Files with excessive mocking**: 0
- **Lines of mock code**: ~100 lines (only external deps)
- **Mock vs Real ratio**: ~5% mocks, 95% real
- **Test confidence**: HIGH (currently MEDIUM)

---

## Resources

- See `TESTING_GUIDELINES.md` for principles and patterns
- [Martin Fowler - Mocks Aren't Stubs](https://martinfowler.com/articles/mocksArentStubs.html)
- [Kent C. Dodds - Testing Implementation Details](https://kentcdodds.com/blog/testing-implementation-details)

---

## Action Items

- [ ] Generate comprehensive test fixtures
- [ ] Refactor `data-loading-integration.test.ts`
- [ ] Refactor `scene-manager.test.ts`
- [ ] Refactor `app.test.ts`
- [ ] Refactor data loader tests
- [ ] Add pre-commit hooks to prevent new over-mocking
- [ ] Update code review checklist
- [ ] Document fixture formats and usage

**Total Estimated Effort**: 2-3 weeks (can be parallelized)
**Expected Benefit**: 3-5x reduction in test brittleness, improved refactoring confidence
