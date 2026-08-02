> **⚠️ Archived — historical document, not maintained.** Kept for design history; it reflects the project state as of its original date and may not match current code. Do not treat it as current guidance. See [the archive README](README.md) for status labels and retention policy.

# Documentation Improvement Plan
**Project**: Luxar
**Generated**: 2025-12-13
**Total Estimated Effort**: 8-10 developer weeks
**Current Overall Grade**: A- (91/100)
**Target Grade**: A+ (96/100)

---

## Executive Summary

This plan addresses documentation gaps identified in the comprehensive review of 26 packages (15 Python, 11 TypeScript). The focus is on improving code-level documentation (JSDoc and docstrings) while maintaining the excellent quality of README and SPECIFICATIONS files.

**Priority Distribution**:
- 🔴 **Critical** (2 weeks): JSDoc coverage for TypeScript packages
- 🟠 **High** (2 weeks): Python docstring completeness, cross-package fixes
- 🟡 **Medium** (2 weeks): Inline comments, examples, standardization
- 🟢 **Low** (2 weeks): Enhancement, automation, polish

---

## 🔴 Phase 1: Critical Issues (Week 1-2)

### Task 1.1: Add JSDoc to TypeScript input/ Package
**Priority**: CRITICAL
**Effort**: 3 days
**Current Coverage**: 25%
**Target Coverage**: 70%
**Impact**: HIGH - Most used package for keyboard/mouse input

**Action Items**:
- [ ] `input-handler.ts`: Add JSDoc to all public methods (~15 methods)
  - `handleKeyDown()`, `handleKeyUp()`, `handleMouseMove()`, etc.
  - Include `@param`, `@returns`, `@example` tags
- [ ] `input-context-manager.ts`: Document context stack management (~10 methods)
- [ ] `input-handler-utils.ts`: Add JSDoc to navigation utilities (~8 functions)
  - `navigateForward()`, `navigateBackward()`, `clampPosition()`, etc.
- [ ] `types.ts`: Complete interface documentation (~5 interfaces)

**Example Template**:
```typescript
/**
 * Handle keyboard navigation in the specified dimension.
 *
 * Advances or retreats the slice position based on the dimension's step size.
 * Respects min/max bounds and emits position change events.
 *
 * @param dimIndex - Zero-based dimension index to navigate
 * @param direction - 1 for forward, -1 for backward
 * @returns true if navigation succeeded, false if at boundary
 *
 * @example
 * ```typescript
 * // Navigate forward in time dimension (index 3)
 * const success = inputHandler.navigateDimension(3, 1);
 * if (!success) console.log('Already at maximum time value');
 * ```
 */
navigateDimension(dimIndex: number, direction: 1 | -1): boolean
```

**Files**:
- `packages/luxar-viewer/src/input/input-handler.ts`
- `packages/luxar-viewer/src/input/input-context-manager.ts`
- `packages/luxar-viewer/src/input/input-handler-utils.ts`
- `packages/luxar-viewer/src/input/types.ts`

---

### Task 1.2: Add JSDoc to TypeScript ui/ Package
**Priority**: CRITICAL
**Effort**: 4 days
**Current Coverage**: 40%
**Target Coverage**: 70%
**Impact**: HIGH - User-facing UI components

**Action Items**:
- [ ] `dimension-sliders.ts`: Complete method JSDoc (~12 methods)
  - `updateSlider()`, `handleSliderChange()`, `setPosition()`, etc.
- [ ] `data-loading-monitor.ts`: Document monitoring methods (~8 methods)
- [ ] `performance-monitor.ts`: Add performance API docs (~6 methods)
- [ ] `rendering-controls.ts`: Document control methods (~10 methods)
- [ ] `scene-graph-tree.ts`: Add tree navigation JSDoc (~8 methods)
- [ ] `dataset-browser.ts`: Document browser methods (~7 methods)
- [ ] `hdr-display-controls.ts`: Add HDR control JSDoc (~5 methods)

**Priority Order**:
1. `dimension-sliders.ts` (most critical user interaction)
2. `data-loading-monitor.ts` (debugging essential)
3. `rendering-controls.ts` (visual quality control)
4. Others in parallel

**Files**: `packages/luxar-viewer/src/ui/*.ts` (7 files)

---

### Task 1.3: Add JSDoc to TypeScript scene/ Package
**Priority**: CRITICAL
**Effort**: 3 days
**Current Coverage**: 35%
**Target Coverage**: 70%
**Impact**: HIGH - Core scene management

**Action Items**:
- [ ] `scene-manager.ts`: Complete all method JSDoc (~20 methods)
  - `init()`, `loadScene()`, `updateView()`, `dispose()`, etc.
  - **Critical**: Document initialization sequence clearly
- [ ] `animation-controller.ts`: Document animation loop (~8 methods)
  - `start()`, `stop()`, `setFPS()`, `onFrame()`, etc.
- [ ] `scene-dims-manager.ts`: Add dimension management JSDoc (~10 methods)
  - **Note**: This file has GOOD JSDoc - use as example for others!

**Example from scene-dims-manager.ts to replicate**:
```typescript
/**
 * Initialize dimensions from the scene's metadata.
 *
 * Searches for dimension metadata in scene.userData, then in node metadata,
 * then falls back to legacy 'dims' property. If no metadata found, creates
 * default dimensions (X, Y, Z for 3D; X, Y for 2D).
 *
 * @param scene - The THREE.Group containing scene metadata
 * @returns DimensionMetadata array, or null if scene is invalid
 */
initFromScene(scene: THREE.Group): DimensionMetadata[] | null
```

**Files**: `packages/luxar-viewer/src/scene/*.ts` (3 main files)

---

### Task 1.4: Add JSDoc to TypeScript rendering/ and controls/ Packages
**Priority**: CRITICAL
**Effort**: 4 days
**Current Coverage**: rendering 40%, controls 35%
**Target Coverage**: 70%
**Impact**: HIGH - Core rendering and interaction

**Action Items - Rendering**:
- [ ] `post-processing-manager.ts`: Document effect management (~15 methods)
- [ ] `material-manager.ts`: Add material caching JSDoc (~8 methods)
- [ ] `point-material.ts`: Document shader configuration (~6 methods)
- [ ] `line-material.ts`: Document line rendering (~6 methods)

**Action Items - Controls**:
- [ ] `controls-manager.ts`: Document control switching (~10 methods)
- [ ] `luxar-fly-controls.ts`: Add physics-based control JSDoc (~12 methods)
- [ ] `luxar-orbit-controls.ts`: Document orbit controls (~8 methods)

**Files**:
- `packages/luxar-viewer/src/rendering/*.ts` (4 files)
- `packages/luxar-viewer/src/controls/*.ts` (3 files)

---

### Task 1.5: Verify and Add Missing Python Module Docstrings
**Priority**: CRITICAL
**Effort**: 1 day
**Current Coverage**: ~90% (estimated)
**Target Coverage**: 100%
**Impact**: MEDIUM - Completeness

**Action Items**:
Check and add module docstrings to these files (if missing):

**Python core**:
- [ ] `packages/luxar/src/luxar/core/datanode.py`
- [ ] `packages/luxar/src/luxar/core/gsplats.py`
- [ ] `packages/luxar/src/luxar/core/lines.py`
- [ ] `packages/luxar/src/luxar/core/__init__.py`

**Python cli**:
- [ ] `packages/luxar/src/luxar/cli/network_simulation.py`
- [ ] `packages/luxar/src/luxar/cli/__init__.py`

**Python encoding**:
- [ ] `packages/luxar/src/luxar/encoding/decoder.py`
- [ ] `packages/luxar/src/luxar/encoding/modes.py`
- [ ] `packages/luxar/src/luxar/encoding/registry.py`
- [ ] `packages/luxar/src/luxar/encoding/__init__.py`

**Template**:
```python
"""Module for [purpose].

[Brief description of module's role in the package, 2-3 sentences]

Key components:
- [Component 1]: [Purpose]
- [Component 2]: [Purpose]
"""
```

**Verification Command**:
```bash
# Check for missing module docstrings
for file in $(find packages/luxar/src/luxar -name "*.py" -not -path "*/tests/*"); do
  if ! grep -q '"""' "$file" | head -5; then
    echo "Missing docstring: $file"
  fi
done
```

---

## 🟠 Phase 2: High Priority Issues (Week 3-4)

### Task 2.1: Add `@param` and `@returns` Tags to Existing JSDoc
**Priority**: HIGH
**Effort**: 3 days
**Impact**: HIGH - Improves existing documentation

**Action Items**:
Many TypeScript methods have basic JSDoc but lack structured tags. Systematically add:
- `@param {type} name - Description` for all parameters
- `@returns {type} Description` for all return values
- `@throws {ErrorType} Description` for methods that can throw

**Target Packages** (in order):
1. `data/` - Most methods have JSDoc but lack tags
2. `cache/` - Good JSDoc but inconsistent tags
3. `core/` - Needs tag standardization

**Script to Find Incomplete JSDoc**:
```bash
# Find JSDoc without @param tags
cd packages/luxar-viewer/src
grep -r "\/\*\*" . --include="*.ts" | \
  while read -r line; do
    file="${line%%:*}"
    if grep -A 20 "\/\*\*" "$file" | grep -q "function\|method" && \
       ! grep -A 20 "\/\*\*" "$file" | grep -q "@param"; then
      echo "$file"
    fi
  done | sort -u
```

---

### Task 2.2: Fix Cross-Package Reference Errors
**Priority**: HIGH
**Effort**: 1 day
**Impact**: MEDIUM - User confusion

**Known Issues**:

1. **types/README.md references navigation utilities**:
   - **Issue**: Claims `navigateForward()`, `navigateBackward()` are in types/
   - **Reality**: These are in `input/input-handler-utils.ts`
   - **Fix**: Update types/README.md Section 4 to redirect to input package

2. **types/SPECIFICATIONS.md Section 4**:
   - **Issue**: Documents navigation utilities that don't exist in types/
   - **Fix**: Remove Section 4 or add clear note: "See input/SPECIFICATIONS.md"

3. **utils/README.md references slicing utilities**:
   - **Issue**: Mentions dimension slicing utilities (lines 40, 171-213)
   - **Fix**: Either implement these utilities or remove references

**Action Items**:
- [ ] Audit all README files for broken cross-references
- [ ] Update types/README.md Section 4
- [ ] Update types/SPECIFICATIONS.md Section 4
- [ ] Check utils/ for phantom utility references
- [ ] Add "Related Packages" sections to all READMEs

**Verification Script**:
```bash
# Find broken cross-references
grep -r "See.*SPECIFICATIONS" packages/luxar*/src/*/README.md | \
  while IFS=: read -r file ref; do
    target=$(echo "$ref" | sed -n 's/.*See \([^ ]*\).*/\1/p')
    if [ -n "$target" ] && [ ! -f "$target" ]; then
      echo "Broken reference in $file: $target"
    fi
  done
```

---

### Task 2.3: Add Docstrings to Python Private Methods
**Priority**: HIGH
**Effort**: 5 days
**Current Coverage**: ~70%
**Target Coverage**: 90%
**Impact**: MEDIUM - Maintainability

**Approach**: Add brief docstrings to private methods with significant logic

**Priority Files** (most complex private methods):
1. `io/compiler.py` - Spatial ordering logic
2. `io/ordering.py` - Morton encoding implementation
3. `validation/base.py` - Validation logic
4. `validation/nd.py` - Dimensional coverage
5. `typing_utils/config.py` - Configuration validation
6. `gsplats/fitting/optimization.py` - Training loop helpers
7. `gsplats/optim/per_splat_adam.py` - Optimizer internals

**Template for Private Methods**:
```python
def _compute_chunk_size(self, array_size: int, target_bytes: int) -> int:
    """Calculate optimal chunk size for the given array.

    Uses intelligent chunking strategy balancing memory and I/O.
    Prefers power-of-2 chunk sizes for alignment efficiency.
    """
    # Implementation...
```

**Not every private method needs docstrings**. Focus on:
- ✅ Methods with >10 lines of code
- ✅ Methods with complex algorithms
- ✅ Methods that other private methods call
- ❌ Simple getters/setters
- ❌ Obvious helper functions (<5 lines)

---

### Task 2.4: Add `Raises:` Sections to Python Functions
**Priority**: HIGH
**Effort**: 2 days
**Impact**: MEDIUM - Error handling clarity

**Action Items**:
Audit all validation functions and public APIs that can raise exceptions.

**Priority Packages**:
1. `validation/` - All validation functions need `Raises:` sections
2. `io/` - File I/O functions
3. `core/` - Scene construction functions
4. `encoding/` - Encoding/decoding functions

**Example**:
```python
def validate_positions(positions: np.ndarray) -> None:
    """Validate position array meets requirements.

    Args:
        positions: Array of shape (N, ndim) with float32 dtype

    Raises:
        ValidationError: If positions is not 2D array
        ValidationError: If positions dtype is not float32
        ValidationError: If positions contains NaN or Inf values
        ValidationError: If positions array is empty (N=0)
    """
```

**Verification Command**:
```bash
# Find functions in validation/ that raise but don't document it
cd packages/luxar/src/luxar/validation
for file in *.py; do
  echo "Checking $file..."
  grep -n "raise ValidationError" "$file" | while read -r line; do
    linenum="${line%%:*}"
    # Check if preceding docstring has "Raises:"
    if ! head -n $linenum "$file" | tail -20 | grep -q "Raises:"; then
      echo "  Line $linenum: Missing Raises: documentation"
    fi
  done
done
```

---

### Task 2.5: Add `@example` Tags to Complex Functions
**Priority**: HIGH
**Effort**: 3 days
**Impact**: MEDIUM - Developer productivity

**Strategy**: Add examples to the most-used and most-complex functions

**Priority Functions** (TypeScript):
1. `data/scene-loader.ts`: `loadScene()`
2. `cache/two-level-caching-store.ts`: `get()`
3. `rendering/material-manager.ts`: `getPointMaterial()`, `getLineMaterial()`
4. `scene/scene-manager.ts`: `init()`, `updateView()`
5. `ui/dimension-sliders.ts`: `setDimensions()`

**Priority Functions** (Python):
1. `io/compiler.py`: `LuxarZarrCompiler.compile()`
2. `io/ordering.py`: `sort_points_compound()`
3. `validation/nd.py`: `broadcast_to_all_slices()`
4. `encoding/encoder.py`: `ArrayEncoder.encode()`
5. `gsplats/fit_gsplats.py`: `fit_gaussian_splats()`

**Template**:
```typescript
/**
 * @example
 * ```typescript
 * // Load a scene from a Zarr dataset
 * const scene = await sceneLoader.loadScene('http://example.com/data.zarr');
 * threeScene.add(scene);
 *
 * // Scene now contains hierarchical groups with point clouds
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
 * }
 * ```
 */
```

---

## 🟡 Phase 3: Medium Priority Issues (Week 5-6)

### Task 3.1: Enhance Inline Comments in Complex Algorithms
**Priority**: MEDIUM
**Effort**: 4 days
**Impact**: MEDIUM - Code maintainability

**Target Areas**:

**Python**:
- [ ] `io/compiler.py`: Spatial ordering and compound key construction
- [ ] `io/ordering.py`: Morton bit-interleaving algorithm
- [ ] `validation/nd.py`: Dimensional coverage algorithm
- [ ] `gsplats/optim/per_splat_adam.py`: Gradient dilution compensation

**TypeScript**:
- [ ] `data/chunk-spatial-index.ts`: AABB intersection test
- [ ] `data/point-spatial-index-loader.ts`: Range merging logic
- [ ] `cache/chunk-prefetcher.ts`: Prefetch queue management
- [ ] `rendering/point-material.ts`: World-space sizing shader logic

**Inline Comment Guidelines**:
```typescript
// ✅ GOOD - Explains WHY and provides context
// Use dot product for squared distance (avoids sqrt, ~2x faster)
// This is safe because we're only comparing distances, not using absolute values
const distSq = dx*dx + dy*dy + dz*dz;

// ❌ BAD - Just describes WHAT the code does
// Calculate squared distance
const distSq = dx*dx + dy*dy + dz*dz;

// ✅ GOOD - Documents algorithm steps
// Morton encoding: interleave bits of coordinates
// For 3D: x, y, z → xyz_xyz_xyz (each letter is one bit)
// Example: x=5 (101b), y=3 (011b), z=2 (010b)
//          → 100_111_011b = 315
```

**Focus Areas**:
1. Algorithm steps (what's happening)
2. Performance optimizations (why this way)
3. Edge cases (what could go wrong)
4. Browser compatibility workarounds
5. Mathematical concepts (brief refresher)

---

### Task 3.2: Add Troubleshooting Sections to READMEs
**Priority**: MEDIUM
**Effort**: 2 days
**Impact**: MEDIUM - User support

**Action Items**:
Add troubleshooting sections to packages that currently lack them.

**Template** (based on rendering/README.md):
```markdown
## Troubleshooting

### Common Issues

#### Issue: Points not visible after loading
**Symptoms**: Scene loads but nothing appears on screen
**Causes**:
1. Camera positioned inside point cloud
2. Point sizes too small for current zoom level
3. Clipping planes excluding all points

**Solutions**:
```typescript
// Check point bounds
console.log(scene.userData.bounds);

// Reset camera to fit scene
sceneManager.resetCamera();

// Increase point size
materialManager.getPointMaterial(props).uniforms.baseSize.value = 5.0;
```

#### Issue: Performance degradation over time
...
```

**Priority Packages**:
- [ ] `scene/README.md` - Add WebGL context loss troubleshooting
- [ ] `controls/README.md` - Add input conflict resolution
- [ ] `data/README.md` - Add data loading failures
- [ ] `cache/README.md` - Add cache quota exceeded handling

---

### Task 3.3: Expand Module Docstrings with Key Concepts
**Priority**: MEDIUM
**Effort**: 2 days
**Impact**: LOW-MEDIUM - Developer understanding

**Action Items**:
Some module docstrings are only 1-2 lines. Expand to 5-10 lines with:
- Purpose (1 sentence)
- Key concepts (2-3 sentences)
- Main classes/functions (bulleted list)
- Related modules (optional)

**Example Expansion**:

**Before** (`utils/trils.py`):
```python
"""Pack/unpack lower triangular matrices."""
```

**After**:
```python
"""Lower triangular matrix packing utilities.

Lower triangular matrices (trils) are common in Gaussian splatting for
covariance matrices. This module provides efficient pack/unpack operations
to store only the N*(N+1)/2 unique values instead of the full N×N matrix.

Packing reduces memory by ~50% for covariance storage and improves cache
locality during rendering.

Key functions:
- tril_size(): Calculate number of unique elements in tril
- pack_tril(): Convert full matrix to packed 1D array
- unpack_tril(): Restore full matrix from packed representation
- calculate_gradient_dilution_factor(): Adjust learning rates for packed params

Related:
- See gsplats/models/ for covariance matrix usage
- See gsplats/optim/ for gradient dilution compensation
"""
```

**Target Files**:
- [ ] `utils/trils.py`
- [ ] `cli/network_simulation.py`
- [ ] `typing_utils/aliases.py`
- [ ] Any other 1-2 line module docstrings found

---

### Task 3.4: Add Quick Start Sections to READMEs
**Priority**: MEDIUM
**Effort**: 2 days
**Impact**: MEDIUM - Onboarding

**Action Items**:
Add "Quick Start" sections at the top of READMEs (after overview, before deep dive).

**Template**:
```markdown
## Quick Start

### Installation
```python
pip install luxar  # or: pnpm install luxar-viewer
```

### Minimal Example
```python
from luxar.core import Scene, Points
import numpy as np

# Create a simple point cloud scene
scene = Scene()
positions = np.random.randn(1000, 3).astype(np.float32)
points = Points(positions=positions)
scene.add(points)

# Save to zarr
scene.save('my_scene.luxar.zarr')
```

### Next Steps
- Read [Architecture](#architecture) for system design
- See [Usage Examples](#usage-examples) for advanced patterns
- Check [API Reference](#api-reference) for complete documentation
```

**Priority READMEs**:
- [ ] `core/README.md`
- [ ] `io/README.md`
- [ ] `encoding/README.md`
- [ ] `gsplats/README.md`
- [ ] `rendering/README.md`
- [ ] `data/README.md`

---

### Task 3.5: Add Big-O Complexity Analysis to Algorithms
**Priority**: MEDIUM
**Effort**: 2 days
**Impact**: LOW - Performance understanding

**Action Items**:
Add complexity analysis to major algorithms in SPECIFICATIONS.md files.

**Target Algorithms**:
1. **io/SPECIFICATIONS.md**:
   - Morton encoding: O(ndim * log(max_coord))
   - Hilbert encoding: O(ndim * 2^ndim)
   - Spatial ordering sort: O(N log N)

2. **data/SPECIFICATIONS.md**:
   - Chunk spatial index query: O(num_chunks * ndim)
   - Point loading: O(visible_points)
   - Lines instance buffer build: O(num_segments)

3. **cache/SPECIFICATIONS.md**:
   - LRU cache get: O(1) average
   - LRU cache eviction: O(1) amortized
   - Prefetch queue processing: O(queue_size)

4. **gsplats/fitting/SPECIFICATIONS.md**:
   - Fitting iteration: O(num_gaussians * num_pixels)
   - Seed generation: O(num_seeds * ndim)

**Template**:
```markdown
### Algorithm: Chunk Spatial Index Query

**Purpose**: Find chunks whose bounding boxes intersect the view region

**Complexity**:
- **Time**: O(C × D) where C = number of chunks, D = dimensionality
  - Typical: C ≈ 100-1000, D ≈ 3-5, so ~500-5000 operations
- **Space**: O(M) where M = number of matching chunks (typically << C)

**Implementation**:
```pseudocode
...
```
```

---

## 🟢 Phase 4: Low Priority / Enhancement (Week 7-8)

### Task 4.1: Add JSDoc Coverage Checking to CI/CD
**Priority**: LOW
**Effort**: 2 days
**Impact**: HIGH (long-term) - Prevents regression

**Action Items**:

1. **Install documentation coverage tool**:
```bash
cd packages/luxar-viewer
pnpm add -D typedoc documentation
```

2. **Create coverage script** (`scripts/check-jsdoc-coverage.js`):
```javascript
#!/usr/bin/env node
const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Find all .ts files (excluding tests)
const files = execSync(
  'find src -name "*.ts" -not -name "*.test.ts" -not -path "*/tests/*"',
  { encoding: 'utf-8' }
).trim().split('\n');

let totalFunctions = 0;
let documentedFunctions = 0;

files.forEach(file => {
  const content = fs.readFileSync(file, 'utf-8');

  // Count exported functions/methods
  const functionMatches = content.match(/export\s+(function|class|interface)/g);
  if (functionMatches) totalFunctions += functionMatches.length;

  // Count functions with JSDoc
  const jsdocMatches = content.match(/\/\*\*[\s\S]*?\*\/\s*export\s+(function|class)/g);
  if (jsdocMatches) documentedFunctions += jsdocMatches.length;
});

const coverage = (documentedFunctions / totalFunctions * 100).toFixed(1);
console.log(`JSDoc Coverage: ${coverage}% (${documentedFunctions}/${totalFunctions})`);

const threshold = 70; // Target threshold
if (coverage < threshold) {
  console.error(`❌ Coverage ${coverage}% is below threshold ${threshold}%`);
  process.exit(1);
} else {
  console.log(`✅ Coverage ${coverage}% meets threshold ${threshold}%`);
}
```

3. **Add to package.json**:
```json
{
  "scripts": {
    "check:jsdoc": "node scripts/check-jsdoc-coverage.js",
    "check": "pnpm typecheck && pnpm lint && pnpm check:jsdoc"
  }
}
```

4. **Add to GitHub Actions** (`.github/workflows/docs-check.yml`):
```yaml
name: Documentation Check
on: [pull_request]
jobs:
  check-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: pnpm install
      - run: pnpm check:jsdoc
```

---

### Task 4.2: Create Documentation Templates
**Priority**: LOW
**Effort**: 1 day
**Impact**: MEDIUM (long-term) - Consistency

**Action Items**:

Create template files in `docs/templates/`:

1. **JSDOC_TEMPLATE.md**:
```markdown
# JSDoc Style Guide

## File-Level Documentation
Every TypeScript file should have a file-level JSDoc comment:

```typescript
/**
 * Module for [purpose].
 *
 * [2-3 sentence description of module's role]
 *
 * Key components:
 * - [Component]: [Purpose]
 *
 * @module [package]/[module-name]
 */
```

## Function Documentation Template
```typescript
/**
 * [One-line description of what the function does]
 *
 * [Optional: 2-3 sentences of additional context, design decisions,
 * or important behavior notes]
 *
 * @param paramName - Description of parameter including constraints,
 *                    valid ranges, and any important behavior
 * @returns Description of return value, including possible values
 *          and what they mean
 * @throws {ErrorType} When this error is thrown and why
 *
 * @example
 * ```typescript
 * // Basic usage
 * const result = functionName(arg1, arg2);
 * ```
 *
 * @example
 * ```typescript
 * // Error handling
 * try {
 *   const result = functionName(arg);
 * } catch (error) {
 *   console.error('Failed:', error);
 * }
 * ```
 *
 * @see {@link RelatedFunction} for related functionality
 * @see {@link README.md#section} for usage guide
 */
```

[Continue with more examples...]
```

2. **DOCSTRING_TEMPLATE.md**:
```markdown
# Python Docstring Style Guide

Luxar uses Google-style docstrings with NumPy conventions.

## Module-Level Docstrings
```python
"""Module for [purpose].

[2-3 sentences describing module's role and key concepts]

Key components:
    [Component]: [Purpose]
    [Component]: [Purpose]

Example:
    Basic usage::

        from luxar.module import Component
        obj = Component()
        obj.method()

Related:
    - :mod:`luxar.related_module`: Related functionality
"""
```

[Continue with function/class templates...]
```

3. **INLINE_COMMENT_GUIDE.md**:
```markdown
# Inline Comment Guidelines

## When to Add Inline Comments

✅ **Always comment**:
- Complex algorithms (every 3-5 lines)
- Non-obvious performance optimizations
- Browser compatibility workarounds
- Edge case handling
- Magic numbers/constants
- Surprising behavior
- TODO/FIXME items (with issue references)

❌ **Don't comment**:
- Self-explanatory code
- Obvious operations
- Repeating what docstring says

## Good vs Bad Examples

### ❌ Bad: Stating the obvious
```python
# Increment counter
counter += 1
```

### ✅ Good: Explaining WHY
```python
# Increment counter to track cache misses for prefetch algorithm
counter += 1
```

[Continue with more examples...]
```

---

### Task 4.3: Generate API Documentation with TypeDoc/Sphinx
**Priority**: LOW
**Effort**: 2 days
**Impact**: MEDIUM - Discoverability

**Action Items**:

**For TypeScript** (TypeDoc):
```bash
cd packages/luxar-viewer
pnpm add -D typedoc

# Create typedoc.json
cat > typedoc.json << 'EOF'
{
  "entryPoints": ["src/"],
  "out": "docs/api",
  "excludePrivate": true,
  "excludeInternal": true,
  "readme": "README.md",
  "plugin": ["typedoc-plugin-markdown"],
  "theme": "markdown",
  "githubPages": false
}
EOF

# Generate docs
pnpm typedoc

# Add to package.json
"scripts": {
  "docs:generate": "typedoc",
  "docs:serve": "python -m http.server -d docs/api 8080"
}
```

**For Python** (Sphinx):
```bash
cd packages/luxar
hatch run pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme

# Initialize Sphinx
mkdir -p docs/api
cd docs/api
sphinx-quickstart --sep --project Luxar --author "Luxar Team"

# Configure conf.py for autodoc
# Add to docs/api/source/conf.py:
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Google/NumPy style docstrings
    'sphinx_autodoc_typehints',
]

# Generate API docs
sphinx-apidoc -o source/api ../../src/luxar

# Build HTML
make html
```

**Add to CI/CD**:
```yaml
# Deploy docs on main branch
- name: Generate API Docs
  run: |
    cd packages/luxar-viewer && pnpm docs:generate
    cd ../luxar && hatch run sphinx-build docs/api/source docs/api/build
```

---

### Task 4.4: Add Architectural Diagrams
**Priority**: LOW
**Effort**: 3 days
**Impact**: LOW - Visual understanding

**Action Items**:

Use ASCII art or Mermaid diagrams for key architectures.

**Priority Diagrams**:

1. **Overall Architecture** (add to main README.md):
```markdown
## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Luxar Ecosystem                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Python Side                      TypeScript Side          │
│  ┌──────────────────┐            ┌────────────────────┐   │
│  │                  │            │                    │   │
│  │  Scene Creation  │            │   Luxar Viewer     │   │
│  │  ┌────────────┐  │            │   ┌──────────┐    │   │
│  │  │ Core       │  │            │   │ Data     │    │   │
│  │  │ (Scene,    │  │            │   │ (Loaders)│    │   │
│  │  │  Nodes)    │  │            │   └─────┬────┘    │   │
│  │  └─────┬──────┘  │            │         │         │   │
│  │        │         │            │   ┌─────▼────┐    │   │
│  │  ┌─────▼──────┐  │            │   │ Cache    │    │   │
│  │  │ Encoding   │  │            │   │ (L1/L2)  │    │   │
│  │  │ (Array     │  │            │   └─────┬────┘    │   │
│  │  │  Encoder)  │  │            │         │         │   │
│  │  └─────┬──────┘  │            │   ┌─────▼────────┐   │
│  │        │         │            │   │ Rendering    │   │
│  │  ┌─────▼──────┐  │            │   │ (Materials,  │   │
│  │  │ I/O        │  │   HTTP     │   │  Post-FX)    │   │
│  │  │ (Zarr      │──┼───────────▶│   └─────┬────────┘   │
│  │  │  Writer)   │  │   Chunked  │         │         │   │
│  │  └────────────┘  │            │   ┌─────▼────┐    │   │
│  │                  │            │   │ Scene    │    │   │
│  │  Optional:       │            │   │ (Manager)│    │   │
│  │  ┌────────────┐  │            │   └─────┬────┘    │   │
│  │  │ GSplats    │  │            │         │         │   │
│  │  │ (Fitting)  │  │            │   ┌─────▼────┐    │   │
│  │  └────────────┘  │            │   │ UI/      │    │   │
│  │                  │            │   │ Controls │    │   │
│  └──────────────────┘            │   └──────────┘    │   │
│                                   └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

Legend:
  ──▶  Data flow
  │    Dependency
```
```

2. **Cache Architecture** (add to cache/README.md)
3. **Data Loading Pipeline** (add to data/README.md)
4. **GSplats Fitting Pipeline** (add to gsplats/fitting/README.md)

**Tool Options**:
- ASCII art (simple, version control friendly)
- Mermaid.js (renders in GitHub)
- Draw.io (export as SVG, commit to repo)

---

### Task 4.5: Add Missing Examples to READMEs
**Priority**: LOW
**Effort**: 2 days
**Impact**: LOW - Additional guidance

**Action Items**:

Add more usage examples to packages that have limited examples:

**Python**:
- [ ] `cli/README.md`: Add utility function examples (lines missing per review)
- [ ] `typing_utils/README.md`: Add more practical usage scenarios

**TypeScript**:
- [ ] `rendering/README.md`: Add more post-processing examples
- [ ] `controls/README.md`: Add custom control creation example
- [ ] `config/README.md`: Add custom configuration examples

**Example Addition** (cli/README.md):
```markdown
## Utility Functions

### `check_port_available()`

Check if a port is available before starting server:

```python
from luxar.cli.utils import check_port_available, find_available_port

# Check specific port
if check_port_available(8080):
    print("Port 8080 is available")
else:
    # Find alternative
    port = find_available_port(8080, 8090)
    print(f"Using port {port} instead")
```

### `open_browser()`

Automatically open viewer in browser after server starts:

```python
from luxar.cli.utils import open_browser
import time

# Start server in background
server = start_server(port=8080)
time.sleep(1)  # Wait for server to be ready

# Open browser
open_browser("http://localhost:8080", wait=0.5)
```
```

---

### Task 4.6: Document Migration Paths for Breaking Changes
**Priority**: LOW
**Effort**: 1 day
**Impact**: LOW - Historical context

**Action Items**:

Create migration guides for major breaking changes documented in changelogs.

**Create**: `docs/MIGRATION_GUIDES.md`

```markdown
# Migration Guides

This document helps users migrate between major versions with breaking changes.

## v1.4.0: Scalar Broadcasting Removal

**Breaking Change**: Removed automatic scalar broadcasting in array utilities.

**Before** (v1.3.x):
```python
# Scalar was automatically broadcast
result = utils.ensure_float32(42)  # Worked, returned array([42.])
```

**After** (v1.4.0+):
```python
# Explicitly convert scalar to array
result = utils.ensure_float32(np.array([42]))  # Required
```

**Migration**: Wrap scalar values in `np.array()` before passing to utils.

**Rationale**: Improved type safety and explicit behavior.

---

## v1.2.0: Config Consolidation

**Breaking Change**: Scattered config constants moved to unified config module.

**Before** (v1.1.x):
```typescript
import { FOV_DEFAULT } from './rendering/constants';
import { CACHE_SIZE } from './cache/defaults';
```

**After** (v1.2.0+):
```typescript
import { config } from './config';
const fov = config.rendering.fovDefault;
const cacheSize = config.cache.l1MaxSizeMB;
```

[Continue with more migrations...]
```

---

## Summary Tables

### Task Distribution by Package

**TypeScript** (Total: ~14 days):
| Package | Task | Effort | Priority |
|---------|------|--------|----------|
| input/ | Add JSDoc | 3 days | Critical |
| ui/ | Add JSDoc | 4 days | Critical |
| scene/ | Add JSDoc | 3 days | Critical |
| rendering/ | Add JSDoc | 2 days | Critical |
| controls/ | Add JSDoc | 2 days | Critical |

**Python** (Total: ~8 days):
| Package | Task | Effort | Priority |
|---------|------|--------|----------|
| All | Verify module docstrings | 1 day | Critical |
| All | Add private method docstrings | 5 days | High |
| All | Add Raises: sections | 2 days | High |

**Cross-Cutting** (Total: ~18 days):
| Task | Effort | Priority |
|------|--------|----------|
| Fix cross-references | 1 day | High |
| Add @param/@returns tags | 3 days | High |
| Add @example tags | 3 days | High |
| Enhance inline comments | 4 days | Medium |
| Add troubleshooting sections | 2 days | Medium |
| Add quick start sections | 2 days | Medium |
| Add complexity analysis | 2 days | Medium |
| CI/CD integration | 2 days | Low |
| Documentation templates | 1 day | Low |
| Generate API docs | 2 days | Low |
| Add diagrams | 3 days | Low |
| Additional examples | 2 days | Low |
| Migration guides | 1 day | Low |

---

## Progress Tracking

### Weekly Milestones

**Week 1**: Complete Tasks 1.1-1.3 (input, ui, scene JSDoc)
**Week 2**: Complete Tasks 1.4-1.5 (rendering, controls, Python modules)
**Week 3**: Complete Tasks 2.1-2.3 (@param/@returns, cross-refs, private methods)
**Week 4**: Complete Tasks 2.4-2.5 (Raises:, @examples)
**Week 5**: Complete Tasks 3.1-3.3 (inline comments, troubleshooting, module expansion)
**Week 6**: Complete Tasks 3.4-3.5 (quick starts, complexity)
**Week 7**: Complete Tasks 4.1-4.3 (CI/CD, templates, API docs)
**Week 8**: Complete Tasks 4.4-4.6 (diagrams, examples, migrations)

### Success Metrics

Track progress with these metrics:

**Coverage Metrics**:
- TypeScript JSDoc coverage: 50% → 70% (Current → Target)
- Python docstring coverage: 85% → 95%
- Inline comment density: 35% → 50% of complex functions

**Quality Metrics**:
- All public functions have `@param/@returns/@example` tags
- No broken cross-package references
- All validation functions document `Raises:`
- All modules have 5+ line docstrings

**Process Metrics**:
- JSDoc coverage enforced in CI/CD
- Documentation templates in use
- API docs auto-generated on main branch
- Migration guides updated for breaking changes

---

## Implementation Notes

### Parallelization Strategy

Many tasks can be done in parallel:

**Phase 1**: Can parallelize by package:
- Developer A: input/ + ui/ JSDoc
- Developer B: scene/ + rendering/ JSDoc
- Developer C: controls/ + Python modules

**Phase 2**: Can parallelize by task type:
- Developer A: Add @param/@returns tags
- Developer B: Fix cross-references + add private docstrings
- Developer C: Add Raises: sections + @example tags

**Phase 3-4**: Sequential or low priority, flexible scheduling

### Verification Commands

Use these commands to track progress:

**JSDoc Coverage**:
```bash
# TypeScript: Count functions with JSDoc
cd packages/luxar-viewer/src
for pkg in input ui scene rendering controls data cache core config types utils; do
  total=$(grep -r "^\s*\(export \)\?\(function\|class\)" $pkg --include="*.ts" | wc -l)
  documented=$(grep -B5 "^\s*\(export \)\?\(function\|class\)" $pkg --include="*.ts" | grep -c "/\*\*")
  pct=$((documented * 100 / total))
  echo "$pkg: $pct% ($documented/$total)"
done
```

**Python Docstring Coverage**:
```bash
# Python: Count functions with docstrings
cd packages/luxar/src/luxar
for pkg in core cli encoding io utils validation typing_utils gsplats; do
  total=$(grep -r "^\s*def " $pkg --include="*.py" | wc -l)
  documented=$(grep -A1 "^\s*def " $pkg --include="*.py" | grep -c '"""')
  pct=$((documented * 100 / total))
  echo "$pkg: $pct% ($documented/$total)"
done
```

**Cross-Reference Check**:
```bash
# Find potentially broken references
grep -r "See.*README" packages/luxar*/src/*/README.md | \
  grep -v "^Binary" | \
  cut -d: -f1 | sort -u | \
  while read file; do
    echo "Checking $file..."
    # Manual review needed
  done
```

---

## Questions / Clarifications Needed

Before starting implementation, clarify:

1. **JSDoc Coverage Target**: Is 70% acceptable, or aim higher?
2. **Private Method Documentation**: Should ALL private methods have docstrings, or only complex ones (>10 lines)?
3. **Inline Comment Density**: What % of code blocks should have inline comments?
4. **CI/CD Enforcement**: Should JSDoc coverage block PRs, or just warn?
5. **Timeline Flexibility**: Is 8-10 weeks acceptable, or need faster completion?
6. **Resource Allocation**: How many developers can work on this in parallel?

---

## Appendix: Example Before/After

### Example 1: TypeScript Method - Before/After

**Before** (input/input-handler.ts):
```typescript
// Handle dimension navigation
navigateDimension(dimIndex: number, direction: number) {
  const dims = this.dimsManager.getDims();
  if (!dims || dimIndex >= dims.displayDims.length) return false;

  const currentPos = dims.slicePosition[dimIndex];
  const step = dims.metadata[dimIndex].step || 1;
  const newPos = currentPos + direction * step;

  // Clamp to range
  const min = dims.metadata[dimIndex].min;
  const max = dims.metadata[dimIndex].max;
  const clamped = Math.max(min, Math.min(max, newPos));

  if (clamped === currentPos) return false;

  dims.slicePosition[dimIndex] = clamped;
  this.emit('dimensionChanged', dimIndex, clamped);
  return true;
}
```

**After**:
```typescript
/**
 * Navigate forward or backward in the specified dimension.
 *
 * Advances or retreats the slice position by the dimension's step size,
 * respecting min/max bounds. Emits 'dimensionChanged' event on success.
 * Frame-rate independent: step size determines travel distance.
 *
 * @param dimIndex - Zero-based index of dimension to navigate (0 = X, 1 = Y, etc.)
 * @param direction - Navigation direction: 1 for forward, -1 for backward
 * @returns true if navigation succeeded, false if already at boundary
 *
 * @throws {Error} If dims not initialized (call initFromScene first)
 *
 * @example
 * ```typescript
 * // Navigate forward in time dimension (typically index 3 for 4D data)
 * const success = inputHandler.navigateDimension(3, 1);
 * if (!success) {
 *   console.log('Already at last time point');
 * }
 * ```
 *
 * @example
 * ```typescript
 * // Navigate backward with arrow key
 * document.addEventListener('keydown', (e) => {
 *   if (e.key === 'ArrowLeft') {
 *     inputHandler.navigateDimension(currentDim, -1);
 *   }
 * });
 * ```
 *
 * @see {@link NavigationManager.clampPosition} for position clamping logic
 * @see {@link SceneDimsManager.getDims} for dimension metadata
 */
navigateDimension(dimIndex: number, direction: 1 | -1): boolean {
  const dims = this.dimsManager.getDims();
  if (!dims || dimIndex >= dims.displayDims.length) {
    throw new Error(`Invalid dimension index ${dimIndex}`);
  }

  const currentPos = dims.slicePosition[dimIndex];
  const step = dims.metadata[dimIndex].step || 1;
  const newPos = currentPos + direction * step;

  // Clamp to dimension range (min/max from metadata)
  const min = dims.metadata[dimIndex].min;
  const max = dims.metadata[dimIndex].max;
  const clamped = Math.max(min, Math.min(max, newPos));

  // Already at boundary, no change
  if (clamped === currentPos) return false;

  // Update position and notify listeners
  dims.slicePosition[dimIndex] = clamped;
  this.emit('dimensionChanged', dimIndex, clamped);
  return true;
}
```

**Key Improvements**:
- ✅ Complete JSDoc with description
- ✅ All parameters documented with constraints
- ✅ Return value meaning explained
- ✅ Two practical examples
- ✅ Error documentation
- ✅ Cross-references to related code
- ✅ Inline comments explain WHY not just WHAT

---

### Example 2: Python Function - Before/After

**Before** (validation/nd.py):
```python
def broadcast_to_all_slices(array, ndim):
    """Broadcast array to match nD dimensionality."""
    if array.ndim == 2 and array.shape[1] < ndim:
        # Extend with zeros
        padding = ndim - array.shape[1]
        zeros = np.zeros((array.shape[0], padding), dtype=array.dtype)
        return np.concatenate([array, zeros], axis=1)
    return array
```

**After**:
```python
def broadcast_to_all_slices(
    array: np.ndarray,
    ndim: int
) -> np.ndarray:
    """Broadcast array to match full nD dimensionality.

    Extends arrays that represent lower-dimensional data (e.g., 2D positions)
    to match the full dimensionality of the dataset by padding with zeros.
    This allows 2D data to be used in nD visibility queries.

    Common use case: 2D image points loaded into 4D dataset (X, Y, T, C).
    The 2D positions are extended to (X, Y, 0, 0) for nD slicing.

    Args:
        array: Position array of shape (N, current_dims) where current_dims <= ndim
        ndim: Target dimensionality to broadcast to

    Returns:
        Array of shape (N, ndim) with zero-padding if needed. If input already
        has ndim columns, returns input unchanged.

    Raises:
        ValidationError: If array is not 2D
        ValidationError: If current dimensions > target dimensions

    Example:
        >>> # Extend 2D positions to 4D
        >>> positions_2d = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> positions_4d = broadcast_to_all_slices(positions_2d, ndim=4)
        >>> print(positions_4d.shape)
        (2, 4)
        >>> print(positions_4d)
        [[1.0, 2.0, 0.0, 0.0],
         [3.0, 4.0, 0.0, 0.0]]

    Example:
        >>> # No change if already correct dimensionality
        >>> positions_3d = np.random.randn(100, 3)
        >>> result = broadcast_to_all_slices(positions_3d, ndim=3)
        >>> assert result is positions_3d  # Same object returned

    See Also:
        validate_dimensional_coverage: Validates positions cover all dimensions
        validate_positions: Validates position array before broadcasting
    """
    # Validate input shape
    if array.ndim != 2:
        raise ValidationError(
            f"Array must be 2D for broadcasting, got shape {array.shape}"
        )

    current_dims = array.shape[1]

    # Check if broadcasting is needed
    if current_dims == ndim:
        return array  # Already correct dimensionality

    if current_dims > ndim:
        raise ValidationError(
            f"Array has {current_dims} dimensions but target is {ndim}. "
            f"Cannot broadcast to lower dimensionality."
        )

    # Extend with zero-padding
    # This is safe because non-displayed dimensions will be filtered by tolerance
    padding_cols = ndim - current_dims
    zeros = np.zeros((array.shape[0], padding_cols), dtype=array.dtype)

    return np.concatenate([array, zeros], axis=1)
```

**Key Improvements**:
- ✅ Complete type hints
- ✅ Detailed Args/Returns/Raises sections
- ✅ Two practical examples with output
- ✅ Use case explanation
- ✅ See Also references
- ✅ Inline comments explain rationale
- ✅ Input validation with clear error messages

---

**END OF DOCUMENTATION IMPROVEMENT PLAN**
