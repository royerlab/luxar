# Missing E2E Tests - Playwright Capability Analysis

**Date**: January 2025
**Current E2E Tests**: 4 files, 24 tests (mostly smoke tests)
**Status**: Basic infrastructure in place, key functionality tests missing

---

## Executive Summary

While we have Playwright infrastructure and basic smoke tests, we're **missing critical E2E tests** that leverage Playwright's unique capabilities:
- Real browser WebGL rendering
- Actual Zarr dataset loading
- Visual verification via screenshots
- Performance monitoring
- State inspection

**Most Important Missing**: Real dataset loading, nD navigation, visual regression, performance benchmarks

---

## Critical Missing Tests (High Priority)

### 🔴 1. Real Dataset Loading & Verification

**Why Critical**: Current tests don't load actual Zarr files
**Playwright Advantage**: Can verify real data loading pipeline end-to-end

**What's Missing**:
```typescript
describe('Real Dataset Loading', () => {
  it('should load demo.zarr and display correct point count', async ({ page }) => {
    // Assumes demo.zarr exists in examples/
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 1000);

    const state = await getLuxarState(page);

    // Verify actual point count matches dataset
    expect(state.totalPoints).toBe(125000);  // Exact count from demo.zarr
    expect(state.pointClouds.length).toBe(1);
    expect(state.pointClouds[0].hasColors).toBe(true);
    expect(state.pointClouds[0].hasRadii).toBe(true);
  });

  it('should load 4D dataset and initialize dimensions correctly', async ({ page }) => {
    await page.goto('/?src=/data/4d_example.zarr&debug');
    await waitForLuxarReady(page);

    const dims = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      return debug.sceneDimsManager?.getDims();
    });

    expect(dims.ndim).toBe(4);
    expect(dims.displayed).toEqual([0, 1, 2]);  // xyz displayed
    expect(dims.metadata[3].name).toBe('time');  // 4th dim is time
  });

  it('should load dataset with spatial index and use it', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => {
      if (msg.text().includes('spatial index')) {
        consoleLogs.push(msg.text());
      }
    });

    await page.goto('/?src=/data/indexed_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Should log that spatial index was loaded
    const hasSpatialIndexLog = consoleLogs.some(log =>
      log.includes('Initialized with') && log.includes('occupied cells')
    );
    expect(hasSpatialIndexLog).toBe(true);
  });

  it('should load dataset without spatial index and create dummy index', async ({ page }) => {
    await page.goto('/?src=/data/3d_no_index.zarr&debug');
    await waitForLuxarReady(page);

    // Should still work, just with full loading
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);
  });

  it('should verify all point attributes loaded correctly', async ({ page }) => {
    await page.goto('/?src=/data/full_attributes.zarr&debug');
    await waitForLuxarReady(page);

    const attributes = await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      const pointCloud = debug.scene.children.find((obj: any) => obj.type === 'Points');

      return {
        hasPositions: !!pointCloud?.geometry?.attributes?.position,
        positionCount: pointCloud?.geometry?.attributes?.position?.count,
        hasColors: !!pointCloud?.geometry?.attributes?.color,
        colorCount: pointCloud?.geometry?.attributes?.color?.count,
        hasRadii: !!pointCloud?.geometry?.attributes?.radius,
        radiiCount: pointCloud?.geometry?.attributes?.radius?.count,
        hasSharpness: !!pointCloud?.geometry?.attributes?.sharpness,
        sharpnessCount: pointCloud?.geometry?.attributes?.sharpness?.count,
      };
    });

    // All attributes should be present and aligned
    expect(attributes.hasPositions).toBe(true);
    expect(attributes.hasColors).toBe(true);
    expect(attributes.hasRadii).toBe(true);
    expect(attributes.hasSharpness).toBe(true);

    // All should have same count (aligned)
    expect(attributes.colorCount).toBe(attributes.positionCount);
    expect(attributes.radiiCount).toBe(attributes.positionCount);
    expect(attributes.sharpnessCount).toBe(attributes.positionCount);
  });
});
```

**Priority**: CRITICAL
**Effort**: 6-8 hours (includes creating test datasets)
**Impact**: Validates entire data pipeline with real data

---

### 🔴 2. nD Navigation & Slicing Verification

**Why Critical**: Core feature not tested end-to-end
**Playwright Advantage**: Can simulate keyboard input and verify point counts change

**What's Missing**:
```typescript
describe('nD Navigation', () => {
  it('should navigate through 4D dataset time dimension', async ({ page }) => {
    await page.goto('/?src=/data/4d_timeseries.zarr&debug');
    await waitForLuxarReady(page);

    // Get initial point count
    const initialState = await getLuxarState(page);
    const initialPoints = initialState.totalPoints;

    // Press '4' to select 4th dimension (time)
    await page.keyboard.press('4');
    await page.waitForTimeout(200);

    // Press ']' to navigate forward in time
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);  // Wait for data loading

    // Verify points changed (different time slice)
    const newState = await getLuxarState(page);

    // Either point count changed OR positions changed
    const pointsChanged = newState.totalPoints !== initialPoints;
    expect(pointsChanged).toBe(true);  // Different slice should have different points
  });

  it('should filter points correctly based on slice position', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => consoleLogs.push(msg.text()));

    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Navigate through dimension
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);

    // Check console for query results
    const queryLogs = consoleLogs.filter(log => log.includes('Query result'));

    // Should see spatial index queries
    expect(queryLogs.length).toBeGreaterThan(0);

    // Should show cells → ranges → points
    const lastQuery = queryLogs[queryLogs.length - 1];
    expect(lastQuery).toMatch(/\d+ cells → \d+ ranges → \d+ points/);
  });

  it('should handle discrete dimensions with exact matching', async ({ page }) => {
    await page.goto('/?src=/data/channel_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Navigate to channel dimension
    await page.keyboard.press('4');  // Select channel dim
    await page.keyboard.press(']');  // Move to next channel
    await page.waitForTimeout(1000);

    // Should load different channel data
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);
  });

  it('should broadcast points across specified dimensions', async ({ page }) => {
    await page.goto('/?src=/data/broadcast_dataset.zarr&debug');
    await waitForLuxarReady(page);

    const initialPoints = (await getLuxarState(page)).totalPoints;

    // Navigate through broadcast dimension
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);

    // Points should stay the same (broadcasted)
    const newPoints = (await getLuxarState(page)).totalPoints;
    expect(newPoints).toBe(initialPoints);
  });
});
```

**Priority**: CRITICAL
**Effort**: 8-10 hours (includes test datasets)
**Impact**: Validates core nD visualization feature

---

### 🔴 3. Spatial Index Query Accuracy

**Why Critical**: Spatial index is performance-critical, must be correct
**Playwright Advantage**: Can verify queries return exact expected points

**What's Missing**:
```typescript
describe('Spatial Index Accuracy', () => {
  it('should return correct points for known slice position', async ({ page }) => {
    // Dataset with known structure: 10 points at specific positions
    await page.goto('/?src=/data/known_positions.zarr&debug');
    await waitForLuxarReady(page);

    // Query at known slice position
    const result = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;
      const loader = await debug.getSceneLoader();
      const defaultLoader = loader.getDefaultLoader();

      // Get spatial index info
      return {
        queryCells: /* capture from console */,
        loadedPoints: debug.getState().totalPoints,
      };
    });

    // Verify expected points were loaded
    expect(result.loadedPoints).toBe(expectedCount);
  });

  it('should handle edge cases in spatial queries', async ({ page }) => {
    await page.goto('/?src=/data/edge_cases.zarr&debug');
    await waitForLuxarReady(page);

    // Test queries at grid boundaries
    // Test queries with zero tolerance
    // Test queries outside data bounds
  });

  it('should merge adjacent ranges efficiently', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => consoleLogs.push(msg.text()));

    await page.goto('/?src=/data/clustered_points.zarr&debug');
    await waitForLuxarReady(page);

    // Check console for range merging
    const mergeLogs = consoleLogs.filter(log =>
      log.match(/\d+ cells → \d+ ranges/)
    );

    // Should merge adjacent ranges (fewer ranges than cells)
    expect(mergeLogs.length).toBeGreaterThan(0);
  });
});
```

**Priority**: HIGH
**Effort**: 6-8 hours
**Impact**: Ensures spatial indexing works correctly

---

### 🟡 4. Visual Regression Testing

**Why Important**: Catches rendering bugs
**Playwright Advantage**: Screenshot comparison detects visual changes

**What's Missing**:
```typescript
describe('Visual Regression', () => {
  it('should render demo dataset identically to baseline', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 10000);

    // Wait for render to stabilize
    await page.waitForFunction(
      () => (window as any).__luxarDebug?.renderer?.info?.render?.frame > 5,
      { timeout: 5000 }
    );

    // Take screenshot and compare to baseline
    await expect(page).toHaveScreenshot('demo-dataset-baseline.png', {
      maxDiffPixelRatio: 0.05,
      threshold: 0.2,
    });
  });

  it('should render with HDR multiplier changes', async ({ page }) => {
    await page.goto('/?src=/data/hdr_test.zarr&debug');
    await waitForLuxarReady(page);

    // Set HDR multiplier to known value
    await page.evaluate(() => {
      (window as any).__luxarDebug.sceneManager.updateHDRMultiplier(10.0);
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('hdr-multiplier-10.png', {
      maxDiffPixelRatio: 0.05,
      threshold: 0.2,
    });
  });

  it('should render with different blending modes', async ({ page }) => {
    // Test normal vs additive blending produces different visuals
    await page.goto('/?src=/data/blending_test.zarr&debug');
    await waitForLuxarReady(page);

    await expect(page).toHaveScreenshot('additive-blending.png');
  });

  it('should render different camera FOVs correctly', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Set FOV to wide angle
    await page.evaluate(() => {
      (window as any).__luxarDebug.camera.fov = 90;
      (window as any).__luxarDebug.camera.updateProjectionMatrix();
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('fov-90-wide.png');
  });
});
```

**Priority**: HIGH
**Effort**: 10-12 hours (create baselines)
**Impact**: Catches visual regressions automatically

---

### 🟡 5. Performance Benchmarking

**Why Important**: Detect performance regressions
**Playwright Advantage**: Real WebGL performance, accurate FPS measurement

**What's Missing**:
```typescript
describe('Performance Benchmarks', () => {
  it('should load 100k points in under 2 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/?src=/data/100k_points.zarr&debug');
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 100000);

    const loadTime = Date.now() - startTime;

    expect(loadTime).toBeLessThan(2000);  // Under 2 seconds
    expect(await getLuxarState(page)).toHaveProperty('totalPoints', 100000);
  });

  it('should maintain 50+ FPS with 500k points', async ({ page }) => {
    await page.goto('/?src=/data/500k_points.zarr&debug');
    await waitForLuxarReady(page);
    await waitForPointsLoaded(page, 500000);

    // Measure FPS over 5 seconds
    const fps = await page.evaluate(async () => {
      const debug = (window as any).__luxarDebug;

      let frames = 0;
      const startTime = Date.now();

      return new Promise<number>(resolve => {
        const counter = () => {
          frames++;
          if (Date.now() - startTime < 5000) {
            requestAnimationFrame(counter);
          } else {
            resolve(frames / 5);
          }
        };
        requestAnimationFrame(counter);
      });
    });

    expect(fps).toBeGreaterThan(50);  // At least 50 FPS
  });

  it('should not leak memory over multiple load cycles', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    const initialMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    // Load and unload 10 times
    for (let i = 0; i < 10; i++) {
      await page.evaluate(async () => {
        const { loadScene } = await import('../data');
        await loadScene('/data/demo.zarr');
      });

      await page.evaluate(() => {
        (window as any).__luxarDebug.sceneManager.clearSceneContent();
      });
    }

    const finalMemory = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    const leakMB = (finalMemory - initialMemory) / 1024 / 1024;
    expect(leakMB).toBeLessThan(50);  // Less than 50MB growth
  });

  it('should track cache hit rate', async ({ page }) => {
    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Navigate forward then back (should hit cache)
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);

    await page.keyboard.press('[');  // Go back
    await page.waitForTimeout(1000);

    // Check cache stats
    const cacheStats = await page.evaluate(async () => {
      const loader = await (window as any).__luxarDebug.getSceneLoader();
      const defaultLoader = loader.getDefaultLoader();
      // Get cache hit rate
      return {
        hits: /* from console logs */,
        hitRate: /* calculate */
      };
    });

    expect(cacheStats.hitRate).toBeGreaterThan(50);  // At least 50% hits
  });
});
```

**Priority**: MEDIUM-HIGH
**Effort**: 8-10 hours
**Impact**: Prevents performance regressions

---

### 🟡 6. Camera Controls & Navigation

**Why Important**: User interaction is core to 3D viewers
**Playwright Advantage**: Can simulate real mouse/keyboard, verify camera state

**What's Missing**:
```typescript
describe('Camera Controls', () => {
  it('should switch between orbit and fly controls', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Initial control type
    const initial = await page.evaluate(() => {
      return (window as any).__luxarDebug.controls.getControlType();
    });
    expect(initial).toBe('orbit');

    // Press 'V' to switch
    await page.keyboard.press('v');
    await page.waitForTimeout(200);

    const after = await page.evaluate(() => {
      return (window as any).__luxarDebug.controls.getControlType();
    });
    expect(after).toBe('fly');
  });

  it('should move camera with WASD in fly mode', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Switch to fly mode
    await page.keyboard.press('v');
    await page.waitForTimeout(200);

    // Get initial position
    const initialPos = await page.evaluate(() => {
      return (window as any).__luxarDebug.camera.position.z;
    });

    // Press 'W' to move forward
    await page.keyboard.down('w');
    await page.waitForTimeout(1000);  // Hold for 1 second
    await page.keyboard.up('w');
    await page.waitForTimeout(500);  // Let inertia settle

    const newPos = await page.evaluate(() => {
      return (window as any).__luxarDebug.camera.position.z;
    });

    // Z position should have changed
    expect(newPos).not.toBeCloseTo(initialPos, 1);
  });

  it('should center camera on scene with F key', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Press 'F' to center
    await page.keyboard.press('f');
    await page.waitForTimeout(500);

    // Camera should be repositioned
    const state = await getLuxarState(page);
    expect(state.cameraPosition).toBeDefined();
  });

  it('should update camera FOV with Shift+Scroll', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    const initialFOV = (await getLuxarState(page)).cameraFov;

    // Simulate Shift+Scroll (zoom FOV)
    await page.keyboard.down('Shift');
    await page.mouse.wheel(0, 100);  // Scroll down
    await page.keyboard.up('Shift');
    await page.waitForTimeout(200);

    const newFOV = (await getLuxarState(page)).cameraFov;

    expect(newFOV).not.toBe(initialFOV);
  });
});
```

**Priority**: MEDIUM-HIGH
**Effort**: 6-8 hours
**Impact**: Ensures navigation works correctly

---

### 🟡 7. Post-Processing Effects Verification

**Why Important**: Visual quality features need testing
**Playwright Advantage**: Can verify effects are applied via console/screenshots

**What's Missing**:
```typescript
describe('Post-Processing Effects', () => {
  it('should enable bloom and verify in logs', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => consoleLogs.push(msg.text()));

    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Enable bloom via debug interface
    await page.evaluate(() => {
      const debug = (window as any).__luxarDebug;
      debug.postProcessing.setBloomEnabled(true);
      debug.renderOnce();
    });

    await page.waitForTimeout(500);

    // Check logs for bloom activation
    const hasBloomLog = consoleLogs.some(log =>
      log.includes('Bloom') || log.includes('bloom')
    );

    // Visual verification
    await expect(page).toHaveScreenshot('with-bloom.png');
  });

  it('should switch tone mapping operators', async ({ page }) => {
    await page.goto('/?src=/data/hdr_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Test each tone mapping operator
    for (const operator of ['ACES', 'AgX', 'Reinhard']) {
      await page.evaluate((op) => {
        (window as any).__luxarDebug.postProcessing.setToneMappingOperator(op);
        (window as any).__luxarDebug.renderOnce();
      }, operator);

      await page.waitForTimeout(500);

      await expect(page).toHaveScreenshot(`tonemapping-${operator.toLowerCase()}.png`, {
        maxDiffPixelRatio: 0.1,
      });
    }
  });

  it('should apply SSAO correctly', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Without SSAO
    await expect(page).toHaveScreenshot('no-ssao.png');

    // With SSAO
    await page.evaluate(() => {
      (window as any).__luxarDebug.postProcessing.setSSAOEnabled(true);
      (window as any).__luxarDebug.renderOnce();
    });

    await page.waitForTimeout(500);

    await expect(page).toHaveScreenshot('with-ssao.png');

    // Pixels should be different
  });
});
```

**Priority**: MEDIUM
**Effort**: 8-10 hours
**Impact**: Ensures visual quality features work

---

### 🟡 8. Error Scenario Testing

**Why Important**: Error handling is critical for UX
**Playwright Advantage**: Can test real network failures, timeouts

**What's Missing**:
```typescript
describe('Error Scenarios', () => {
  it('should show error for 404 dataset', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', err => errors.push(err.message));

    await page.goto('/?src=/data/nonexistent.zarr&debug');

    // Wait for error handling
    await page.waitForTimeout(3000);

    // Should show error message (not crash)
    const errorVisible = await page.locator('.error-message').isVisible().catch(() => false);
    expect(errorVisible).toBe(true);

    // Should log error
    expect(errors.length).toBeGreaterThan(0);
  });

  it('should handle network timeout gracefully', async ({ page }) => {
    // Mock slow network via Playwright
    await page.route('**/slow_dataset.zarr/**', route => {
      setTimeout(() => route.abort('timedout'), 30000);
    });

    await page.goto('/?src=/data/slow_dataset.zarr&debug');

    // Should show timeout error
    await page.waitForTimeout(35000);

    const errorVisible = await page.locator('.error-message').isVisible();
    expect(errorVisible).toBe(true);
  });

  it('should handle corrupted Zarr gracefully', async ({ page }) => {
    await page.goto('/?src=/data/corrupted.zarr&debug');

    // Should not crash, should show error
    await page.waitForTimeout(3000);

    const state = await page.evaluate(() => {
      return (window as any).__luxarDebug?.getState?.() || null;
    });

    // App should still be initialized (graceful degradation)
    expect(state?.initialized).toBe(true);
  });

  it('should handle missing spatial index gracefully', async ({ page }) => {
    const consoleLogs: string[] = [];
    page.on('console', msg => consoleLogs.push(msg.text()));

    await page.goto('/?src=/data/no_index.zarr&debug');
    await waitForLuxarReady(page);

    // Should log that no index was found
    const noIndexLog = consoleLogs.some(log =>
      log.includes('No spatial index') || log.includes('3D dataset')
    );
    expect(noIndexLog).toBe(true);

    // Should still load points
    const state = await getLuxarState(page);
    expect(state.totalPoints).toBeGreaterThan(0);
  });
});
```

**Priority**: MEDIUM
**Effort**: 6-8 hours
**Impact**: Better error UX

---

### 🟡 9. Memory Leak Detection

**Why Important**: Critical for long-running sessions
**Playwright Advantage**: Can monitor real memory usage over time

**What's Missing**:
```typescript
describe('Memory Leak Detection', () => {
  it('should not leak memory on dataset switches', async ({ page }) => {
    await page.goto('/?debug');
    await waitForLuxarReady(page);

    // Get initial memory
    const getMemory = () => page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });

    const initialMemory = await getMemory();

    // Load and unload 20 times
    for (let i = 0; i < 20; i++) {
      await page.evaluate(async () => {
        await (window as any).__luxarDebug.sceneManager.loadSceneData('/data/demo.zarr');
      });

      await page.waitForTimeout(1000);

      await page.evaluate(() => {
        (window as any).__luxarDebug.sceneManager.clearSceneContent();
      });
    }

    // Force garbage collection if available
    await page.evaluate(() => {
      if ((globalThis as any).gc) {
        (globalThis as any).gc();
      }
    });

    const finalMemory = await getMemory();
    const leakMB = (finalMemory - initialMemory) / 1024 / 1024;

    expect(leakMB).toBeLessThan(100);  // Less than 100MB growth
  });

  it('should dispose WebGL resources on cleanup', async ({ page }) => {
    await page.goto('/?src=/data/demo.zarr&debug');
    await waitForLuxarReady(page);

    // Get WebGL memory info
    const beforeCleanup = await page.evaluate(() => {
      const info = (window as any).__luxarDebug.renderer.info.memory;
      return {
        geometries: info.geometries,
        textures: info.textures,
      };
    });

    expect(beforeCleanup.geometries).toBeGreaterThan(0);

    // Cleanup
    await page.evaluate(() => {
      (window as any).__luxarDebug.sceneManager.dispose();
    });

    // Memory should be freed
    const afterCleanup = await page.evaluate(() => {
      const info = (window as any).__luxarDebug.renderer.info.memory;
      return {
        geometries: info.geometries,
        textures: info.textures,
      };
    });

    expect(afterCleanup.geometries).toBe(0);
  });
});
```

**Priority**: MEDIUM
**Effort**: 6-8 hours
**Impact**: Prevents memory leaks in production

---

### 🟡 10. Cache Behavior Verification

**Why Important**: Cache is performance-critical
**Playwright Advantage**: Can monitor real cache hits/misses via console

**What's Missing**:
```typescript
describe('Cache Behavior', () => {
  it('should cache data on first load', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', msg => {
      if (msg.text().includes('Cache')) {
        cacheLogs.push(msg.text());
      }
    });

    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // First navigation - cache miss
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);

    const misses = cacheLogs.filter(log => log.includes('Cache miss') || log.includes('Loading'));
    expect(misses.length).toBeGreaterThan(0);
  });

  it('should hit cache on return navigation', async ({ page }) => {
    const cacheLogs: string[] = [];
    page.on('console', msg => {
      if (msg.text().includes('Cache') || msg.text().includes('cache')) {
        cacheLogs.push(msg.text());
      }
    });

    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Navigate forward
    await page.keyboard.press('4');
    await page.keyboard.press(']');
    await page.waitForTimeout(1000);

    cacheLogs.length = 0;  // Clear logs

    // Navigate back - should hit cache
    await page.keyboard.press('[');
    await page.waitForTimeout(1000);

    const hits = cacheLogs.filter(log => log.includes('Cache hit'));
    expect(hits.length).toBeGreaterThan(0);
  });

  it('should evict old entries when cache full', async ({ page }) => {
    await page.goto('/?src=/data/large_4d.zarr&debug');
    await waitForLuxarReady(page);

    // Navigate through many slices to fill cache
    for (let i = 0; i < 50; i++) {
      await page.keyboard.press(']');
      await page.waitForTimeout(200);
    }

    // Check cache stats
    const stats = await page.evaluate(async () => {
      const loader = await (window as any).__luxarDebug.getSceneLoader();
      return loader.getDefaultLoader()?.getCacheStats();
    });

    // Should have evicted some entries
    expect(stats?.evictions).toBeGreaterThan(0);
  });
});
```

**Priority**: MEDIUM
**Effort**: 4-6 hours
**Impact**: Validates caching works correctly

---

### 🟢 11. Multi-Dataset Scenarios

**Why Important**: Users switch between datasets
**Playwright Advantage**: Can test real dataset switching

**What's Missing**:
```typescript
describe('Multi-Dataset Scenarios', () => {
  it('should switch between datasets without errors', async ({ page }) => {
    // Load first dataset
    await page.goto('/?src=/data/dataset1.zarr&debug');
    await waitForLuxarReady(page);
    const count1 = (await getLuxarState(page)).totalPoints;

    // Load second dataset
    await page.evaluate(() => {
      (window as any).__luxarDebug.sceneManager.loadSceneData('/data/dataset2.zarr');
    });
    await page.waitForTimeout(2000);

    const count2 = (await getLuxarState(page)).totalPoints;

    // Should have different point counts
    expect(count2).not.toBe(count1);
  });

  it('should clear old scene when loading new one', async ({ page }) => {
    await page.goto('/?src=/data/dataset1.zarr&debug');
    await waitForLuxarReady(page);

    const children1 = await page.evaluate(() => {
      return (window as any).__luxarDebug.scene.children.length;
    });

    // Load different dataset
    await page.goto('/?src=/data/dataset2.zarr&debug');
    await waitForLuxarReady(page);

    const children2 = await page.evaluate(() => {
      return (window as any).__luxarDebug.scene.children.length;
    });

    // Old objects should be removed
    expect(children2).toBeGreaterThan(0);  // New scene loaded
  });
});
```

**Priority**: LOW-MEDIUM
**Effort**: 4-6 hours
**Impact**: Validates scene switching

---

### 🟢 12. Dimension Slider UI Testing

**Why Important**: Primary nD navigation UI
**Playwright Advantage**: Can interact with real DOM sliders

**What's Missing**:
```typescript
describe('Dimension Sliders', () => {
  it('should show sliders for nD dataset', async ({ page }) => {
    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Press 'N' to toggle dimension sliders
    await page.keyboard.press('n');
    await page.waitForTimeout(500);

    // Sliders should be visible
    const slidersVisible = await page.locator('.dimension-sliders').isVisible();
    expect(slidersVisible).toBe(true);
  });

  it('should update points when slider moved', async ({ page }) => {
    await page.goto('/?src=/data/4d_dataset.zarr&debug');
    await waitForLuxarReady(page);

    // Show sliders
    await page.keyboard.press('n');
    await page.waitForTimeout(500);

    const initialPoints = (await getLuxarState(page)).totalPoints;

    // Move time slider
    const slider = page.locator('input[type="range"]').nth(3);  // 4th slider = time
    await slider.fill('5');  // Move to frame 5
    await page.waitForTimeout(1000);

    const newPoints = (await getLuxarState(page)).totalPoints;

    // Points should change
    expect(newPoints).not.toBe(initialPoints);
  });
});
```

**Priority**: LOW-MEDIUM
**Effort**: 4-6 hours
**Impact**: UI component testing

---

## Priority Matrix

### CRITICAL (Must Have):
1. **Real Dataset Loading** (6-8 hours) - Core functionality
2. **nD Navigation** (8-10 hours) - Core feature
3. **Spatial Index Accuracy** (6-8 hours) - Correctness critical

**Total Critical**: ~20-26 hours
**Impact**: Validates core features work end-to-end

### HIGH (Should Have):
4. **Visual Regression** (10-12 hours) - Catches rendering bugs
5. **Performance Benchmarks** (8-10 hours) - Prevents regressions
6. **Camera Controls** (6-8 hours) - User interaction

**Total High**: ~24-30 hours
**Impact**: Quality and performance assurance

### MEDIUM (Nice to Have):
7. **Post-Processing** (8-10 hours) - Visual features
8. **Error Scenarios** (6-8 hours) - Better UX
9. **Memory Leak Detection** (6-8 hours) - Long-term stability
10. **Cache Behavior** (4-6 hours) - Performance verification

**Total Medium**: ~24-32 hours
**Impact**: Comprehensive coverage

### LOW (Future):
11. **Multi-Dataset** (4-6 hours)
12. **Dimension Sliders** (4-6 hours)

**Total Low**: ~8-12 hours
**Impact**: Edge case coverage

---

## Recommended Implementation Order

### Phase 1 (Week 1): Core Functionality - CRITICAL
1. **Real Dataset Loading** (Day 1-2)
   - Create 3-4 test Zarr datasets
   - Test loading with verification

2. **nD Navigation** (Day 3-4)
   - Test 4D/5D navigation
   - Verify slice updates

3. **Spatial Index Accuracy** (Day 5)
   - Verify queries are correct
   - Test edge cases

**Deliverable**: Core features validated

### Phase 2 (Week 2): Quality Assurance - HIGH
4. **Visual Regression** (Day 1-2)
   - Create baseline screenshots
   - Set up comparison tests

5. **Performance Benchmarks** (Day 3-4)
   - FPS tracking
   - Load time measurements
   - Memory monitoring

6. **Camera Controls** (Day 5)
   - Test all control modes
   - Verify camera updates

**Deliverable**: Quality and performance assured

### Phase 3 (Week 3): Comprehensive Coverage - MEDIUM
7-10. Error scenarios, post-processing, memory, cache

**Deliverable**: Comprehensive E2E coverage

---

## Test Dataset Requirements

To implement these tests, we need:

### Minimal Test Datasets (Required):
1. **demo.zarr** - 3D dataset, ~100k points, all attributes
2. **4d_timeseries.zarr** - 4D dataset with time dimension
3. **5d_dataset.zarr** - 5D dataset for complex nD testing
4. **extended_dataset.zarr** - With extend_to_all specified
5. **no_index.zarr** - 3D dataset without spatial index
6. **indexed_dataset.zarr** - With spatial index

### Extended Test Datasets (Nice to Have):
7. **100k_points.zarr** - Performance testing
8. **500k_points.zarr** - Stress testing
9. **hdr_dataset.zarr** - HDR colors for tone mapping tests
10. **known_positions.zarr** - Exact positions for accuracy testing

**Estimated effort to create**: 4-6 hours

---

## Conclusion

**Most Important Missing Tests** (leverage Playwright uniquely):

1. ✅ **Real dataset loading** - Only Playwright can load actual files
2. ✅ **nD navigation** - Only Playwright can simulate keyboard + verify changes
3. ✅ **Visual regression** - Only Playwright can take screenshots
4. ✅ **Performance benchmarks** - Only Playwright has real WebGL
5. ✅ **Memory leak detection** - Only Playwright can monitor real memory

**Recommended**: Start with items 1-3 (core functionality)

**Total Effort for Critical Tests**: ~20-26 hours
**Total Effort for All**: ~80-100 hours

---

**Next Step**: Create test datasets, then implement critical tests (1-3)
