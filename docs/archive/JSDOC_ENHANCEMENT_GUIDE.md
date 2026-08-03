> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# JSDoc Enhancement Guide
**Purpose**: Template and examples for adding @param, @returns, @throws, @example tags to existing JSDoc
**Target**: Phase 2, Task 2.1
**Status**: Template Ready

---

## Overview

Many TypeScript files have basic JSDoc comments but lack structured tags. This guide shows how to enhance them systematically.

## Complete JSDoc Template

```typescript
/**
 * [One-line summary of what the function does]
 *
 * [2-4 sentences providing additional context, explaining the approach,
 * and noting any important behavior or design decisions]
 *
 * [Optional: Numbered list of steps for complex operations]
 * 1. Step 1 description
 * 2. Step 2 description
 * 3. Step 3 description
 *
 * @param paramName - Description of parameter including:
 *                    - Purpose and usage
 *                    - Valid values/constraints
 *                    - Default behavior if optional
 *                    Example: 'maxSize - Maximum cache size in bytes (default: 64MB)'
 *
 * @param complexParam - For complex types, describe structure:
 *                       - field1: Purpose of field1
 *                       - field2: Purpose of field2
 *                       - field3: Optional field description
 *
 * @returns Description of return value including:
 *          - Type and meaning
 *          - Possible values and what they indicate
 *          - Object structure for complex returns
 *          Example: 'Promise resolving to loaded data, or null if not found'
 *
 * @throws {ErrorType} When this error occurs and why
 *         Example: '{ValidationError} If positions array has invalid dimensions'
 *
 * @example
 * ```typescript
 * // Basic usage with common scenario
 * const result = functionName(arg1, arg2);
 * console.log(result);
 * ```
 *
 * @example
 * ```typescript
 * // Advanced usage showing edge cases or options
 * const result = functionName(arg1, arg2, {
 *   optionA: true,
 *   optionB: 'custom'
 * });
 * ```
 *
 * @example
 * ```typescript
 * // Error handling pattern
 * try {
 *   const result = functionName(arg);
 * } catch (error) {
 *   console.error('Operation failed:', error);
 * }
 * ```
 *
 * @see {@link RelatedFunction} for related functionality
 * @see {@link README.md#section} for usage guide
 * @see {@link SPECIFICATIONS.md} Section X for algorithm details
 *
 * @performance O(n) time complexity, O(1) space (for performance-critical functions)
 * @internal (if function is internal-only)
 * @deprecated Use newFunction() instead (if deprecated)
 */
```

---

## Real Example: scene-loader.ts loadScene()

### Before Enhancement:
```typescript
/**
 * Load a complete scene from a zarr store
 */
async loadScene(url: string): Promise<THREE.Group>
```

### After Enhancement:
```typescript
/**
 * Load a complete scene from a Zarr store using chunk-based spatial indexing.
 *
 * Orchestrates the loading of hierarchical scene graphs, managing spatial indices,
 * attribute inheritance, and dimension metadata. Supports both points and lines
 * with automatic fallback for datasets without spatial ordering.
 *
 * The loading process:
 * 1. Opens Zarr store with optional two-level caching (L1 memory + L2 OPFS)
 * 2. Loads scene metadata and initializes dimensions
 * 3. Recursively constructs THREE.js scene graph from Zarr group hierarchy
 * 4. Creates spatial index loaders for efficient nD queries
 * 5. Connects loaders to data monitor for debugging
 *
 * @param url - Complete URL to the Zarr store. Can be:
 *              - HTTP URL: 'https://example.com/data.zarr'
 *              - Local path: '/path/to/data.zarr'
 *              - With query params: 'https://example.com/data.zarr?no-cache'
 *
 * @returns Promise resolving to a THREE.Group containing the complete scene graph.
 *          The group's userData contains:
 *          - sceneDimensions: Dimension metadata if available
 *          - bounds: AABB of all points
 *          - nodeCount: Total number of leaf nodes
 *
 * @throws {Error} If the Zarr store cannot be opened or is invalid
 * @throws {Error} If consolidated metadata (.zmetadata) is malformed
 * @throws {Error} If required arrays (positions) are missing from point nodes
 *
 * @example
 * ```typescript
 * // Load a scene from HTTP URL
 * const scene = await sceneLoader.loadScene('https://example.com/data.zarr');
 * threeScene.add(scene);
 * console.log(`Loaded ${scene.children.length} top-level nodes`);
 * ```
 *
 * @example
 * ```typescript
 * // Load with error handling
 * try {
 *   const scene = await sceneLoader.loadScene(url);
 *   if (scene.children.length === 0) {
 *     console.warn('Scene is empty');
 *   }
 * } catch (error) {
 *   console.error('Failed to load scene:', error);
 *   // Fallback to default visualization
 * }
 * ```
 *
 * @see {@link ../cache/two-level-caching-store.ts} for caching implementation
 * @see {@link SPECIFICATIONS.md} Section 4 for complete scene loading protocol
 */
async loadScene(url: string): Promise<THREE.Group>
```

**Key Improvements**:
- ✅ Expanded description with process steps
- ✅ Detailed @param with valid values
- ✅ Structured @returns with userData fields
- ✅ Multiple @throws for different error cases
- ✅ Three @example blocks showing different use cases
- ✅ Cross-references with @see

---

## Pattern Library

### Pattern 1: Simple Function with Options

```typescript
/**
 * Calculate optimal chunk size for the given array.
 *
 * Uses intelligent chunking strategy balancing memory and I/O performance.
 * Prefers power-of-2 sizes for alignment efficiency on most hardware.
 *
 * @param arraySize - Total size of array in elements
 * @param targetBytes - Target chunk size in bytes (default: 65536 = 64KB)
 * @param dtype - Data type for bytes-per-element calculation (default: 'float32')
 *
 * @returns Optimal chunk size in elements, always power of 2, clamped to [256, arraySize]
 *
 * @example
 * ```typescript
 * const chunkSize = calculateChunkSize(1000000, 65536, 'float32');
 * console.log(`Using ${chunkSize} elements per chunk`);
 * ```
 */
function calculateChunkSize(
  arraySize: number,
  targetBytes: number = 65536,
  dtype: string = 'float32'
): number
```

### Pattern 2: Method with Complex Parameters

```typescript
/**
 * Query chunks whose bounding boxes intersect the view region.
 *
 * Uses AABB (axis-aligned bounding box) intersection test in nD space.
 * A chunk intersects if its bounds overlap the query box in ALL dimensions.
 *
 * @param index - Chunk spatial index containing:
 *                - chunkBounds: Flat array of [min, max] for each chunk and dimension
 *                - numChunks: Total number of chunks
 *                - ndim: Number of dimensions
 *
 * @param slicePos - Current position in nD space, one value per dimension
 *                   Example: [0, 2.5, 1, 0, 0] for 5D dataset
 *
 * @param tolerance - Search radius per dimension in world units
 *                    Example: [0, 1.0, 0, 0, 0] = ±1.0 units in dimension 1
 *
 * @returns Array of chunk indices that intersect the query region.
 *          Empty array if no chunks intersect.
 *
 * @example
 * ```typescript
 * // Query chunks at position [0, 5.0, 0] with tolerance [0, 2.0, 0]
 * const chunks = queryChunksForView(
 *   spatialIndex,
 *   [0, 5.0, 0],
 *   [0, 2.0, 0]
 * );
 * console.log(`Found ${chunks.length} intersecting chunks`);
 * ```
 *
 * @performance O(numChunks × ndim), typically ~500-5000 operations for standard datasets
 */
function queryChunksForView(
  index: ChunkSpatialIndex,
  slicePos: number[],
  tolerance: number[]
): number[]
```

### Pattern 3: Async Method with Error Handling

```typescript
/**
 * Initialize the two-level cache and validate against remote content hash.
 *
 * Performs cache validation by fetching root .zattrs directly from HTTP
 * (bypassing cache) to detect dataset changes. If content_hash differs
 * from cached value, clears L2 cache completely and re-initializes.
 *
 * This is CRITICAL for correctness: we must fetch .zattrs bypassing cache
 * to avoid comparing cached hash to itself (false positive).
 *
 * @returns Promise that resolves when cache is initialized and validated
 *
 * @throws {Error} If HTTP fetch fails (network error, 404, etc.)
 * @throws {Error} If .zattrs is malformed or missing content_hash
 * @throws {QuotaExceededError} If OPFS storage quota is exceeded
 *
 * @example
 * ```typescript
 * const store = new TwoLevelCachingStore(url, options);
 * try {
 *   await store.init();
 *   console.log('Cache ready');
 * } catch (error) {
 *   console.error('Cache initialization failed:', error);
 *   // Fall back to direct HTTP
 * }
 * ```
 *
 * @see {@link validateCache} for validation algorithm
 * @performance First init: ~100ms (OPFS setup), Subsequent: ~10ms (validation only)
 */
async init(): Promise<void>
```

### Pattern 4: Getter/Accessor with Side Effects

```typescript
/**
 * Get material for point cloud rendering with specified properties.
 *
 * Materials are cached based on property hash to avoid redundant shader compilation.
 * Cache key includes: baseSize, blendingMode, hdrMode, useSharpness.
 *
 * **Note**: Returned material is shared - do NOT modify its properties directly.
 * Clone the material if you need custom per-instance modifications.
 *
 * @param props - Material properties including:
 *                - baseSize: Point size in pixels (typically 1-10)
 *                - blendingMode: THREE.js blending mode (Normal, Additive, etc.)
 *                - hdrMode: Enable HDR color encoding (requires float textures)
 *                - useSharpness: Use sharpness attribute for variable sizing
 *
 * @returns Shared PointMaterial instance from cache. Do not modify directly.
 *
 * @example
 * ```typescript
 * const material = materialManager.getPointMaterial({
 *   baseSize: 3.0,
 *   blendingMode: BlendingMode.Additive,
 *   hdrMode: false,
 *   useSharpness: true
 * });
 *
 * const points = new THREE.Points(geometry, material);
 * ```
 *
 * @see {@link PointMaterial} for material implementation
 * @performance O(1) cache lookup, ~50ms shader compilation on cache miss
 */
getPointMaterial(props: PointMaterialProperties): PointMaterial
```

### Pattern 5: Boolean Predicate

```typescript
/**
 * Check if port is available for binding.
 *
 * Attempts to create a temporary server on the specified port.
 * Returns immediately without waiting for server startup.
 *
 * @param port - Port number to check (1-65535)
 *
 * @returns true if port is available, false if already in use or invalid
 *
 * @example
 * ```typescript
 * if (check_port_available(8080)) {
 *   startServer(8080);
 * } else {
 *   console.log('Port 8080 is busy, trying 8081...');
 *   startServer(8081);
 * }
 * ```
 *
 * @performance ~1-5ms per check (fast, synchronous)
 */
function check_port_available(port: number): boolean
```

---

## Priority Files for Task 2.1

### High Priority (Most Used APIs):

**data/ package**:
- [x] ✅ `scene-loader.ts` - loadScene() DONE
- [ ] `scene-loader.ts` - Other methods (dispose, updateView, etc.)
- [ ] `zarr-loader.ts` - loadArray(), loadMetadata()
- [ ] `chunk-spatial-index.ts` - queryChunksForView(), buildChunkSpatialIndex()
- [ ] `point-spatial-index-loader.ts` - loadVisiblePoints(), loadRanges()
- [ ] `lines-spatial-index-loader.ts` - loadVisibleLines(), buildInstanceBuffers()

**cache/ package**:
- [ ] `two-level-caching-store.ts` - get(), init(), validateCache()
- [ ] `lru-cache.ts` - get(), set(), clear()
- [ ] `chunk-prefetcher.ts` - onAccess(), processQueue()

**core/ package**:
- [ ] `app.ts` - init(), dispose()
- [ ] `main.ts` - main() entry point

### Medium Priority:

**rendering/ package**:
- [ ] `material-manager.ts` - getPointMaterial(), getLineMaterial()
- [ ] `post-processing-manager.ts` - setupPasses(), updateBloomSettings()

**scene/ package**:
- [ ] `scene-manager.ts` - init(), updateView(), resetCamera()
- [ ] `animation-controller.ts` - start(), stop(), setFPS()

**controls/ package**:
- [ ] `controls-manager.ts` - setControlType(), dispose()
- [ ] `luxar-fly-controls.ts` - update(), handleInput()

**ui/ package**:
- [ ] `dimension-sliders.ts` - setDimensions(), updateSlider()
- [ ] `data-loading-monitor.ts` - connectLoader(), update()

---

## Verification Script

Use this to find methods with incomplete JSDoc:

```bash
#!/bin/bash
# find-incomplete-jsdoc.sh

cd packages/luxar-viewer/src

echo "=== Methods with JSDoc but missing @param tags ==="
for file in data/*.ts cache/*.ts core/*.ts; do
  if [ -f "$file" ]; then
    # Find methods with /** but no @param
    if grep -Pzo '\/\*\*[\s\S]*?\*\/\s*(async\s+)?\w+\s*\(' "$file" | grep -v "@param" > /dev/null; then
      echo "$file"
    fi
  fi
done

echo ""
echo "=== Methods with @param but no @returns ==="
for file in data/*.ts cache/*.ts core/*.ts; do
  if [ -f "$file" ]; then
    # Find methods with @param but no @returns
    if grep -Pzo '\/\*\*[\s\S]*?@param[\s\S]*?\*\/\s*(async\s+)?\w+\s*\(' "$file" | grep -v "@returns" > /dev/null; then
      echo "$file"
    fi
  fi
done

echo ""
echo "=== Async methods without @throws ==="
for file in data/*.ts cache/*.ts core/*.ts; do
  if [ -f "$file" ]; then
    # Find async methods without @throws
    if grep -Pzo '\/\*\*[\s\S]*?\*\/\s*async\s+\w+\s*\(' "$file" | grep -v "@throws" > /dev/null; then
      echo "$file"
    fi
  fi
done
```

---

## Checklist for Each Method

When enhancing JSDoc, verify you've added:

- [ ] **@param** for ALL parameters (required and optional)
  - Include type, purpose, valid values, defaults
  - Multi-line descriptions for complex parameters

- [ ] **@returns** for all non-void functions
  - Describe type and meaning
  - Explain possible return values (null, undefined, empty array, etc.)
  - Document object structure for complex returns

- [ ] **@throws** for methods that can throw
  - List all possible error types
  - Explain when each error occurs
  - Especially important for async methods and validation functions

- [ ] **@example** with realistic usage
  - Minimum 1 example for simple functions
  - 2-3 examples for complex APIs
  - Show error handling for error-prone operations
  - Include console.log for clarity

- [ ] **@see** for cross-references
  - Link to related functions
  - Link to README sections
  - Link to SPECIFICATIONS sections

- [ ] **@performance** for critical paths (optional but valuable)
  - Big-O complexity
  - Typical execution time
  - Memory usage notes

---

## Quick Tips

1. **Start with High-Impact APIs**: Focus on most-used functions first
2. **Copy Good Examples**: Use enhanced loadScene() as template
3. **Be Specific**: "Returns array" → "Returns array of chunk indices [0, 5, 12] or empty array if no matches"
4. **Show Real Values**: Use actual data examples, not placeholders
5. **Think Like User**: What would IDE tooltip need to show?
6. **Link Liberally**: Use @see to connect related code
7. **Test Examples**: Ensure example code actually works

---

## Common Mistakes to Avoid

❌ **Too Vague**:
```typescript
@param config - Configuration object
```

✅ **Specific**:
```typescript
@param config - Configuration containing:
               - maxSize: Maximum cache size in bytes
               - debug: Enable debug logging (default: false)
```

---

❌ **Missing Error Cases**:
```typescript
@returns The loaded data
```

✅ **Include Edge Cases**:
```typescript
@returns Loaded data as Float32Array, or null if chunk not found (404).
         Returns empty array if chunk exists but contains no points.
```

---

❌ **Generic Examples**:
```typescript
@example
const result = myFunction(param);
```

✅ **Realistic Examples**:
```typescript
@example
// Load scene from production server
const scene = await sceneLoader.loadScene('https://data.example.com/cells.zarr');
console.log(`Loaded ${scene.children.length} cell populations`);
```

---

## Automation Ideas (Future)

Consider creating:
1. **VS Code Snippet**: Insert JSDoc template with Tab-stops
2. **ESLint Rule**: Warn on missing @param/@returns
3. **Pre-commit Hook**: Check JSDoc completeness
4. **TypeDoc Config**: Generate API docs from enhanced JSDoc

---

**Status**: Template ready, 1 example completed (scene-loader.ts loadScene)
**Next**: Apply pattern to remaining high-priority methods
**Est. Time**: ~3 days for all high-priority methods
