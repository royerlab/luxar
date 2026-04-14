# Pre-Picking Refactoring Plan

> **STATUS: COMPLETED** — All four refactorings described below have been implemented. Shaders are in `rendering/shaders/`, geometry in `line-geometry.ts`/`gsplat-geometry.ts`, `camera-aware-material.ts` exists, and `node-factory.ts` exists. This document is retained for reference.

> **Purpose**: This document is a self-contained brief for an agent to execute four preparatory refactorings that unblock the implementation of GPU-based object picking (hover tooltips) in the Luxar viewer. Each refactoring is independent and can be done in any order, though the numbered order reflects priority.
>
> **Scope**: Read-only analysis is complete. This document describes exactly what to change and why. All file paths are relative to `packages/luxar-viewer/src/`.
>
> **Rules**: Follow the project's CLAUDE.md conventions. Run `pnpm test --run` and `pnpm typecheck` after each refactoring to verify nothing breaks. Do NOT change any runtime behavior — these are pure structural refactors.

---

## Context: Why These Refactorings Are Needed

We are about to implement a **GPU-based picking system** that lets users hover over points, lines, and Gaussian splats to see tooltip labels. The picking system works by:

1. Maintaining a **parallel "pick scene"** with shadow meshes that share the same `BufferGeometry` as the main scene but use special **picking materials** (which output encoded IDs instead of color).
2. Rendering this pick scene to a tiny offscreen `WebGLRenderTarget` (5x5 pixels) when the mouse stops moving.
3. Reading back the pixel data to identify which node and element is under the cursor.

This design requires:
- **Picking materials** that reuse the exact same vertex shader logic as the main materials (same world-space sizing, nD slicing, projection math) but with a trivial fragment shader that outputs IDs.
- **Geometry creation functions** that can be called independently of material creation (so the pick scene can share geometry but use different materials).
- **A common interface** for all camera-aware materials (main + picking) so the MaterialManager can broadcast camera updates uniformly.
- **A hook point** in scene construction where pick-scene shadow nodes can be created alongside main nodes.

The current code architecture makes all four of these harder than necessary. These refactorings fix that.

---

## Refactoring 1: Extract Vertex Shaders to Shared Constants

### Problem

Each material class (`PointMaterial`, `LineMaterial`, `GSplatMaterial`) defines its vertex and fragment shaders as `private static readonly` string properties. For picking materials, we need to reuse the **exact same vertex shader** but pair it with a different fragment shader. Currently there's no way to import or reference the vertex shader from outside the class.

### Files Involved

| File | Current Role |
|------|-------------|
| `rendering/point-material.ts` | Contains `PointMaterial.VERTEX_SHADER` (lines 44-106) and `PointMaterial.FRAGMENT_SHADER` (lines 114-165) as private static strings |
| `rendering/line-material.ts` | Contains `LineMaterial.VERTEX_SHADER` (lines 81-212) and `LineMaterial.FRAGMENT_SHADER` (lines 223-280) as private static strings |
| `rendering/gsplat-material.ts` | Contains `GSplatMaterial.VERTEX_SHADER` (lines 120-373) and `GSplatMaterial.FRAGMENT_SHADER` (lines 390-457) as private static strings |

### What to Do

**Step 1**: Create a new file `rendering/shaders/` directory (if it doesn't exist) with three files:

- `rendering/shaders/point-shaders.ts`
- `rendering/shaders/line-shaders.ts`
- `rendering/shaders/gsplat-shaders.ts`

Each file exports the vertex and fragment shader strings as named `export const` values.

**Step 2**: For each material file, move the shader strings out of the class and into the corresponding shaders file. Replace the `private static readonly` property with an import.

**Example for points** (apply the same pattern to lines and gsplats):

Create `rendering/shaders/point-shaders.ts`:
```typescript
/**
 * Shared vertex shader for point rendering.
 *
 * Used by both PointMaterial (main rendering) and PointPickingMaterial (GPU picking).
 * Contains world-space sizing, sharpness compensation, nD slicing, and projection logic.
 */
export const POINT_VERTEX_SHADER = /* glsl */ `
    precision highp float;
    // ... (entire vertex shader string, copied verbatim from PointMaterial.VERTEX_SHADER)
`;

/**
 * Fragment shader for standard point rendering.
 *
 * Computes Gaussian falloff, GOG color adjustment, and alpha output.
 * The picking system uses a different fragment shader (see picking-materials.ts).
 */
export const POINT_FRAGMENT_SHADER = /* glsl */ `
    precision highp float;
    // ... (entire fragment shader string, copied verbatim from PointMaterial.FRAGMENT_SHADER)
`;
```

Update `rendering/point-material.ts`:
```typescript
import { POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER } from './shaders/point-shaders';

export class PointMaterial extends THREE.ShaderMaterial {
  // Remove the private static readonly VERTEX_SHADER and FRAGMENT_SHADER properties entirely.
  // In the constructor's super() call, replace:
  //   vertexShader: PointMaterial.VERTEX_SHADER,
  //   fragmentShader: PointMaterial.FRAGMENT_SHADER,
  // with:
  //   vertexShader: POINT_VERTEX_SHADER,
  //   fragmentShader: POINT_FRAGMENT_SHADER,
  // ...
}
```

**Step 3**: Apply the identical pattern to `LineMaterial` and `GSplatMaterial`.

**Step 4**: Create a barrel export `rendering/shaders/index.ts`:

```typescript
export { POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER } from './point-shaders';
export { LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER } from './line-shaders';
export { GSPLAT_VERTEX_SHADER, GSPLAT_FRAGMENT_SHADER } from './gsplat-shaders';
```

The material files should import from the specific shader file (e.g., `./shaders/point-shaders`), not the barrel. The barrel is for external consumers like the upcoming picking system.

### Verification

- `pnpm test --run` — all existing material tests must pass unchanged.
- `pnpm typecheck` — no type errors.
- Grep for `VERTEX_SHADER` and `FRAGMENT_SHADER` in the material classes — they should no longer contain shader source strings, only import references.
- The shader files should each export exactly 2 constants: `*_VERTEX_SHADER` and `*_FRAGMENT_SHADER`.

### Desired Final State

```
rendering/
  shaders/
    index.ts             — barrel re-export of all 6 shader constants
    point-shaders.ts     — exports POINT_VERTEX_SHADER, POINT_FRAGMENT_SHADER
    line-shaders.ts      — exports LINE_VERTEX_SHADER, LINE_FRAGMENT_SHADER
    gsplat-shaders.ts    — exports GSPLAT_VERTEX_SHADER, GSPLAT_FRAGMENT_SHADER
  point-material.ts      — imports from ./shaders/point-shaders, no inline shader strings
  line-material.ts       — imports from ./shaders/line-shaders, no inline shader strings
  gsplat-material.ts     — imports from ./shaders/gsplat-shaders, no inline shader strings
```

Any external code can now `import { POINT_VERTEX_SHADER } from './shaders/point-shaders'` to create a picking material that reuses the vertex shader.

---

## Refactoring 2: Extract Geometry Creation from Material Files

### Problem

`line-material.ts` and `gsplat-material.ts` each contain exported functions that create and update THREE.js geometry. These are **not material logic** — they create `BufferGeometry`, `InstancedBufferGeometry`, compute bounding boxes, set buffer attributes, etc. They happen to live in the material files because they were originally written alongside the materials.

For picking, we need to share geometry between main materials and picking materials. Having geometry creation coupled to material files makes this awkward — you'd import geometry functions from a material file, then use a different material.

### Functions to Extract

**From `rendering/line-material.ts`** (lines 485-733):

| Function | Line | Description |
|----------|------|-------------|
| `createLineQuadGeometry()` | 502 | Creates base quad `BufferGeometry` for instanced lines |
| `createInstancedLinesMesh()` | 614 | Creates complete `THREE.Mesh` with instanced geometry + material |
| `updateInstancedLinesMesh()` | 687 | Updates existing mesh with new segment data |
| `computeLineBounds()` | 578 | Private helper — computes bounding box from segment positions |

Also move these **types** that the geometry functions depend on:
| Type | Line | Description |
|------|------|-------------|
| `InstancedLinesMeshConfig` | 536 | Config interface for line mesh creation/update |

**From `rendering/gsplat-material.ts`** (lines 771-1065):

| Function | Line | Description |
|----------|------|-------------|
| `createGSplatQuadGeometry()` | 788 | Creates base quad `BufferGeometry` for instanced splats |
| `packCholeskyForShader()` | 852 | Packs Cholesky factors into shader attribute format |
| `createInstancedGSplatsMesh()` | 920 | Creates complete `THREE.Mesh` with instanced geometry + material |
| `updateInstancedGSplatsMesh()` | 984 | Updates existing mesh with new splat data |
| `computeMaxCholeskyRowNorm()` | 887 | Private helper — computes max splat extent for bounding box |

Also move these **types**:
| Type | Line | Description |
|------|------|-------------|
| `InstancedGSplatsMeshConfig` | 822 | Config interface for gsplat mesh creation/update |

### What to Do

**Step 1**: Create two new files:

- `rendering/line-geometry.ts`
- `rendering/gsplat-geometry.ts`

**Step 2**: Move the functions and types listed above from the material files to the new geometry files. The private helper functions (`computeLineBounds`, `computeMaxCholeskyRowNorm`) become module-private (not exported) in the new files.

**Step 3**: The geometry files will need to import from the material files for the material type parameter in `createInstancedLinesMesh(config, material: LineMaterial)` and `createInstancedGSplatsMesh(config, material: GSplatMaterial)`. This is fine — geometry depends on material (to read `truncationRadius` uniform for bounding box expansion), but not vice versa.

**Step 4**: Update all existing import sites to point directly at the new geometry files (no re-exports — per project policy of "no backwards compatibility burden"):

| File | Old Import From | New Import From | Symbols |
|------|----------------|-----------------|---------|
| `data/scene-loader.ts` | `../rendering/line-material` | `../rendering/line-geometry` | `createInstancedLinesMesh`, `updateInstancedLinesMesh` |
| `data/scene-loader.ts` | `../rendering/gsplat-material` | `../rendering/gsplat-geometry` | `createInstancedGSplatsMesh`, `updateInstancedGSplatsMesh`, `packCholeskyForShader` |
| `data/scene-loader.ts` (geometry updates) | `../rendering/line-material` | `../rendering/line-geometry` | `updateInstancedLinesMesh` |
| `data/scene-loader.ts` (geometry updates) | `../rendering/gsplat-material` | `../rendering/gsplat-geometry` | `updateInstancedGSplatsMesh`, `packCholeskyForShader` |
| `tests/unit/rendering/line-material.test.ts` | `../../../rendering/line-material` | `../../../rendering/line-geometry` | geometry functions |
| `tests/unit/rendering/gsplat-material.test.ts` | `../../../rendering/gsplat-material` | `../../../rendering/gsplat-geometry` | geometry functions |

**Note**: Imports of the material *classes* themselves (`LineMaterial`, `GSplatMaterial`) stay pointing at the material files — only geometry function imports move.

### Verification

- `pnpm test --run` — all tests pass.
- `pnpm typecheck` — no type errors.
- The material files (`line-material.ts`, `gsplat-material.ts`) should now contain **only** the material class, its config interface, its uniforms interface, and the re-exports. No geometry creation logic.
- The new geometry files should contain all the geometry creation/update functions and their helper functions.

### Desired Final State

```
rendering/
  line-material.ts          — LineMaterial class + LineMaterialConfig (no geometry functions, no re-exports)
  line-geometry.ts           — createLineQuadGeometry, createInstancedLinesMesh, updateInstancedLinesMesh, InstancedLinesMeshConfig, computeLineBounds (private)
  gsplat-material.ts        — GSplatMaterial class + GSplatMaterialConfig (no geometry functions, no re-exports)
  gsplat-geometry.ts         — createGSplatQuadGeometry, packCholeskyForShader, createInstancedGSplatsMesh, updateInstancedGSplatsMesh, InstancedGSplatsMeshConfig, computeMaxCholeskyRowNorm (private)
```

---

## Refactoring 3: Define `CameraAwareMaterial` Interface

### Problem

`PointMaterial`, `LineMaterial`, and `GSplatMaterial` all implement an `updateCameraParams()` method, but with no shared interface. The `MaterialManager` uses duck typing with `(material as any).updateCameraParams(...)` to call it (see `material-manager.ts` lines 282-284 and 298-304). This is fragile and will get worse when we add 3 picking material classes.

### Current Signatures

```typescript
// PointMaterial
updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho?: boolean): void

// LineMaterial
updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho?: boolean): void

// GSplatMaterial
updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho?: boolean, nearCull?: number): void
```

The superset signature (GSplatMaterial's) is the one the MaterialManager already calls — it passes `nearCull` to all materials, and PointMaterial/LineMaterial silently ignore the extra argument because of how JS works. So the **unified interface** just needs the GSplatMaterial signature.

### What to Do

**Step 1**: Create a new file `rendering/camera-aware-material.ts`:

```typescript
/**
 * Interface for materials that need camera parameter updates.
 *
 * Implemented by all main rendering materials (PointMaterial, LineMaterial,
 * GSplatMaterial) and their picking counterparts. The MaterialManager
 * broadcasts camera changes to all registered CameraAwareMaterials.
 */
import * as THREE from 'three';

export interface CameraAwareMaterial {
  /**
   * Update camera-dependent uniforms (FOV, resolution, projection mode).
   *
   * @param fov - Field of view in radians (perspective) or frustum height in world units (ortho)
   * @param resolution - Viewport resolution in pixels
   * @param isOrtho - Whether camera is orthographic (default false)
   * @param nearCull - Near cull distance in world units (used by GSplats, ignored by others)
   */
  updateCameraParams(
    fov: number,
    resolution: THREE.Vector2,
    isOrtho?: boolean,
    nearCull?: number
  ): void;
}

/**
 * Type guard to check if a material implements CameraAwareMaterial.
 */
export function isCameraAwareMaterial(material: unknown): material is CameraAwareMaterial {
  return (
    typeof material === 'object' &&
    material !== null &&
    'updateCameraParams' in material &&
    typeof (material as Record<string, unknown>).updateCameraParams === 'function'
  );
}
```

**Step 2**: Update the three material classes to explicitly implement the interface:

In `rendering/point-material.ts`:
```typescript
import { CameraAwareMaterial } from './camera-aware-material';

export class PointMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  // updateCameraParams signature already matches — add the nearCull param (ignored):
  updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho: boolean = false, _nearCull?: number): void {
    // ... existing implementation unchanged (nearCull unused for points)
  }
}
```

In `rendering/line-material.ts`:
```typescript
import { CameraAwareMaterial } from './camera-aware-material';

export class LineMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho: boolean = false, _nearCull?: number): void {
    // ... existing implementation unchanged (nearCull unused for lines)
  }
}
```

In `rendering/gsplat-material.ts`:
```typescript
import { CameraAwareMaterial } from './camera-aware-material';

export class GSplatMaterial extends THREE.ShaderMaterial implements CameraAwareMaterial {
  // Already has the full signature — no change needed to the method itself.
}
```

**Step 3**: Update `MaterialManager` to use the interface instead of duck typing.

In `rendering/material-manager.ts`, change the `registeredMaterials` set type and the broadcast logic:

```typescript
import { CameraAwareMaterial, isCameraAwareMaterial } from './camera-aware-material';

export class MaterialManager {
  // Change from:
  //   private registeredMaterials = new Set<THREE.Material>();
  // To:
  private registeredMaterials = new Set<THREE.Material & CameraAwareMaterial>();

  // In updateCameraParams(), replace the duck-typed iteration:
  //   this.registeredMaterials.forEach((material) => {
  //     if ('updateCameraParams' in material && typeof (material as any).updateCameraParams === 'function') {
  //       (material as any).updateCameraParams(fov, resolution, isOrtho, nearCull);
  //     }
  //   });
  // With:
  updateCameraParams(fov: number, resolution: THREE.Vector2, isOrtho: boolean = false, nearCull?: number): void {
    this.currentFov = fov;
    this.currentResolution.copy(resolution);
    this.currentIsOrtho = isOrtho;
    this.currentNearCull = nearCull;

    for (const material of this.registeredMaterials) {
      material.updateCameraParams(fov, resolution, isOrtho, nearCull);
    }
  }

  // Similarly update register():
  register(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.add(material);
    material.updateCameraParams(
      this.currentFov,
      this.currentResolution,
      this.currentIsOrtho,
      this.currentNearCull
    );
  }

  // And unregister():
  unregister(material: THREE.Material & CameraAwareMaterial): void {
    this.registeredMaterials.delete(material);
    // ... rest of cache cleanup unchanged
  }
}
```

**Note**: The `getPointMaterial`, `getLineMaterial`, and `getGSplatMaterial` methods call `this.registeredMaterials.add(material)`. Since `PointMaterial`, `LineMaterial`, and `GSplatMaterial` now all implement `CameraAwareMaterial`, this will type-check correctly.

### Verification

- `pnpm typecheck` — the key check. All `registeredMaterials.add()` calls must type-check with the new `Set<THREE.Material & CameraAwareMaterial>` type.
- `pnpm test --run` — all tests pass.
- Grep for `as any` in `material-manager.ts` — there should be no more `(material as any).updateCameraParams` casts.

### Desired Final State

```
rendering/
  camera-aware-material.ts   — CameraAwareMaterial interface + isCameraAwareMaterial type guard
  material-manager.ts        — Uses CameraAwareMaterial instead of duck typing
  point-material.ts          — implements CameraAwareMaterial
  line-material.ts           — implements CameraAwareMaterial
  gsplat-material.ts         — implements CameraAwareMaterial
```

---

## Refactoring 4: Extract Node Factory from SceneLoader

### Problem

`data/scene-loader.ts` is **3030 lines** with ~30 methods. It's the largest file in the viewer and a classic God Object. Among its many responsibilities, it contains the logic for creating THREE.js scene nodes (Points, Lines, GSplats meshes) — including material selection, geometry creation, userData assignment, and transform application.

For the picking system, we need to **hook into node creation** to create parallel pick-scene shadow nodes that share geometry but use picking materials. Doing this inside the already-overloaded SceneLoader would make it worse.

### What to Extract

The goal is to extract a **`NodeFactory`** class that handles the creation of THREE.js objects from loaded data, separate from the loading and orchestration logic.

#### Methods to move from `SceneLoader` to `NodeFactory`:

1. **`createGeometry()`** (line 2340) — Creates `THREE.BufferGeometry` from loaded point data with position, color, radius, sharpness, scalar attributes. Also handles GPU buffer pool allocation.

2. **`createMaterial()`** (line 2476) — Creates a `PointMaterial` via `materialManager.getPointMaterial()` based on node attributes (blending mode, opacity, gamma, intensity, offset, radius/sharpness scale).

3. **`validateTransformFormat()`** (line 2522) — Validates a 16-element transform array.

4. **`applyTransform()`** (line 2550) — Applies a 4x4 transform matrix to a THREE.js object (with NumPy→THREE.js transpose).

5. **`validateLoadedPointsData()`** (line 2265) — Validates loaded point data arrays.

6. **`validateColorMode()`** (line 2669) — Validates color mode compatibility with data.

#### The node creation patterns to extract:

Currently in `loadPoints()` (lines 1649-1776), `loadLines()` (lines 1781-1942), and `loadGSplats()` (lines 1964-2143), each method follows this pattern:

```
1. Create loader                       ← stays in SceneLoader (orchestration)
2. Load data via loader                ← stays in SceneLoader (orchestration)
3. Validate data                       ← moves to NodeFactory
4. Create geometry from data           ← moves to NodeFactory
5. Create material from attrs          ← moves to NodeFactory
6. Create THREE.js object              ← moves to NodeFactory
7. Set userData (nodeType, loader, ...) ← moves to NodeFactory
8. Apply transform                     ← moves to NodeFactory
```

Steps 3-8 should be a single `NodeFactory` method per geometry type.

### What to Do

**Step 1**: Create `data/node-factory.ts` with a `NodeFactory` class:

```typescript
/**
 * NodeFactory - Creates THREE.js scene nodes from loaded data.
 *
 * Extracted from SceneLoader to separate node creation (geometry, material,
 * userData, transforms) from data loading and orchestration.
 *
 * This separation enables the picking system to hook into node creation
 * and create parallel pick-scene shadow nodes.
 */

import * as THREE from 'three';
import { materialManager, BlendingMode } from '../rendering/material-manager';
import { PointMaterial } from '../rendering/point-material';
import { createInstancedLinesMesh } from '../rendering/line-geometry';
import { createInstancedGSplatsMesh, packCholeskyForShader } from '../rendering/gsplat-geometry';
import { GPUBufferPool } from '../rendering/gpu-buffer-pool';
import { LoadedPointsData } from './data-loader-types';
import type { PointsMetadata, PointsUserData } from '../types/points';
import type { LinesMetadata, LinesUserData, LinesDataLoader, ProcessedLinesData } from '../types/lines';
import type { GSplatsMetadata, GSplatsUserData, GSplatsDataLoader } from '../types/gsplats';
import type { DataLoader } from './data-loader-types';
import { log, Modules } from '../utils/log';

export class NodeFactory {
  constructor(private gpuBufferPool: GPUBufferPool | null = null) {}

  /**
   * Create a THREE.Points object from loaded point data.
   * Handles geometry creation, material selection, userData, and transforms.
   */
  createPointsNode(
    path: string,
    attrs: PointsMetadata,
    data: LoadedPointsData,
    loader: DataLoader
  ): THREE.Points { ... }

  /**
   * Create a THREE.Mesh (instanced lines) from processed line data.
   */
  createLinesNode(
    path: string,
    attrs: LinesMetadata,
    processed: ProcessedLinesData,
    loader: LinesDataLoader
  ): THREE.Mesh { ... }

  /**
   * Create a THREE.Mesh (instanced gsplats) from processed splat data.
   *
   * For progressive loading, pass an onUpdate callback. NodeFactory wires
   * it into the mesh so SceneLoader can manage the progressive loader
   * lifecycle without NodeFactory knowing about loader internals.
   */
  createGSplatsNode(
    path: string,
    attrs: GSplatsMetadata,
    processed: ProcessedGSplatsData,
    loader: GSplatsDataLoader,
    onProgressiveUpdate?: (mesh: THREE.Mesh, newData: ProcessedGSplatsData) => void
  ): THREE.Mesh { ... }

  // Move these private helpers here:
  // - createGeometry() → createPointsGeometry()
  // - createMaterial() → createPointsMaterial()
  // - validateLoadedPointsData()
  // - validateColorMode()
  // - validateTransformFormat()
  // - applyTransform()
}
```

**Step 2**: In `SceneLoader`, replace the inline node creation code in `loadPoints()`, `loadLines()`, and `loadGSplats()` with calls to `NodeFactory`:

```typescript
// Before (in loadPoints):
const geometry = this.createGeometry(data, ...);
const material = this.createMaterial(node.attrs, ...);
const points = new THREE.Points(geometry, material);
points.name = node.path;
points.userData = { nodeType: 'points', loader, attrs, ... };
if (attrs.transform) this.applyTransform(points, attrs.transform);

// After:
const points = this.nodeFactory.createPointsNode(node.path, attrs, data, loader);
```

**Step 3**: The `SceneLoader` constructor should create a `NodeFactory` instance:

```typescript
export class SceneLoader {
  private nodeFactory: NodeFactory;

  constructor(...) {
    // ...existing constructor logic...
    this.nodeFactory = new NodeFactory(this.gpuBufferPool);
  }
}
```

### Important Details

- **Colormap handling**: The `loadLines()` and `loadGSplats()` methods contain colormap-related material cloning logic (checking `colormap_name`, cloning material, calling `updateColormapTexture`). This should also move to the NodeFactory methods since it's part of material setup.

- **The `gpuBufferPool`** is currently stored on `SceneLoader` and used in `createGeometry()`. Pass it to `NodeFactory` via constructor injection.

- **`materialManager`** is a module-level singleton (`import { materialManager } from '../rendering/material-manager'`), not a SceneLoader instance property. NodeFactory imports it the same way — this is an implicit dependency but matches the existing pattern. Document it in the NodeFactory class JSDoc.

- **`getColormapTexture()`** is imported from `rendering/colormap-textures` — the NodeFactory will need this import too.

- **Progressive GSplat loading**: `loadGSplats()` has two code paths — standard and progressive. Both paths should be extracted to NodeFactory. The progressive path creates the initial mesh AND sets up ongoing update callbacks that interact with `geometryUpdateManager` and `progressiveLoaders`. Use **callback injection**: `createGSplatsNode()` accepts an optional `onProgressiveUpdate` callback parameter. NodeFactory creates the node; SceneLoader passes a callback that handles the progressive loader lifecycle (registering with geometryUpdateManager, etc.). This keeps NodeFactory unaware of loader lifecycle management.

- **Error handling**: The try/catch in `loadPoints/loadLines/loadGSplats` should stay in SceneLoader. The NodeFactory methods should throw on validation errors, and SceneLoader catches and logs them.

### What Stays in SceneLoader

- All loading orchestration (`loadScene`, `updateView`, `loadSceneNodes`)
- Loader creation and management (`createLoader`, `loaders` map)
- nD view state management
- Data processing (worker projection, gsplat processing)
- Monitor integration
- Retry logic
- Dimension initialization
- Store enumeration and URL normalization

### Unit Tests for NodeFactory

Create `tests/unit/data/node-factory.test.ts` with tests for the extracted helpers. These are now independently importable pure-ish functions — test them directly:

| Function | What to Test |
|----------|-------------|
| `validateTransformFormat()` | Valid 16-element array, wrong length, row-major vs column-major detection |
| `applyTransform()` | Identity transform, translation, scale, NumPy→THREE.js transpose correctness |
| `validateLoadedPointsData()` | Valid data, mismatched array lengths, missing positions, zero-length arrays |
| `validateColorMode()` | Uint8 (SDR), Float32 (HDR), mismatched mode warnings |

The `createPointsNode`/`createLinesNode`/`createGSplatsNode` methods depend on THREE.js and the materialManager singleton — these are better covered by E2E tests than unit tests.

### Verification

- `pnpm test --run` — all tests pass, including new NodeFactory unit tests.
- `pnpm typecheck` — no type errors.
- `pnpm test:e2e` — run at minimum `basic-rendering.spec.ts` and `viewer-initialization.spec.ts` (mandatory — unit tests cannot catch WebGL regressions).
- `scene-loader.ts` should shrink by roughly 400-500 lines.
- The extracted methods should no longer exist in `SceneLoader` (except as delegating calls to `NodeFactory`).

### Desired Final State

```
data/
  node-factory.ts     — NodeFactory class with createPointsNode, createLinesNode, createGSplatsNode + validation/transform helpers
  scene-loader.ts     — Uses NodeFactory for node creation, retains all orchestration logic, ~500 lines smaller
tests/unit/data/
  node-factory.test.ts — Unit tests for validation and transform helpers
```

---

## Execution Order and Testing Strategy

### PR Strategy

**PR 1** — Refactorings 1, 2, 3 (three commits):
1. **Refactoring 1** (Extract shaders) — smallest, zero risk, no import changes
2. **Refactoring 2** (Extract geometry) — small, updates all import sites directly (no re-exports)
3. **Refactoring 3** (CameraAwareMaterial interface) — small, improves type safety

**PR 2** — Refactoring 4 (NodeFactory):
4. **Refactoring 4** (NodeFactory) — largest, touches the critical scene loading path, includes unit tests

Separating PR 2 ensures the most impactful change gets focused review and can be rolled back independently.

### Testing After Each Commit

```bash
# From packages/luxar-viewer/
pnpm typecheck          # Type checking (catches import/interface issues)
pnpm test --run         # Unit tests (catches behavioral regressions)
pnpm lint               # Lint (catches unused imports, style issues)
```

### E2E Tests (mandatory before merging each PR)

```bash
# Minimum required — catches WebGL rendering regressions that unit tests miss
npx playwright test basic-rendering.spec.ts viewer-initialization.spec.ts
```

### What NOT to Do

- **Do NOT change any runtime behavior.** These are structural refactors. The viewer should render identically before and after.
- **Do NOT add new features.** No picking materials, no pick scene, no tooltip. Just move code and add interfaces.
- **Do NOT leave stale imports** — when moving geometry functions out of material files (Refactoring 2), update all 6 import sites to point at the new geometry files. No re-exports.
- **Do NOT refactor the shader GLSL code** itself. Move it verbatim — character for character, comment for comment.
- **Do NOT rename any existing public APIs.** Method names, export names, and type names must remain the same.
- **Do NOT delete the validation/transform methods from SceneLoader** until the NodeFactory equivalents are wired in and tested.

---

## Summary of New Files Created

| File | Purpose | Approximate Size |
|------|---------|-----------------|
| `rendering/shaders/index.ts` | Barrel re-export of all 6 shader constants | ~6 lines |
| `rendering/shaders/point-shaders.ts` | Point vertex + fragment shader strings | ~130 lines |
| `rendering/shaders/line-shaders.ts` | Line vertex + fragment shader strings | ~210 lines |
| `rendering/shaders/gsplat-shaders.ts` | GSplat vertex + fragment shader strings | ~350 lines |
| `rendering/camera-aware-material.ts` | CameraAwareMaterial interface + type guard | ~30 lines |
| `rendering/line-geometry.ts` | Line geometry creation/update functions | ~220 lines |
| `rendering/gsplat-geometry.ts` | GSplat geometry creation/update/pack functions | ~290 lines |
| `data/node-factory.ts` | NodeFactory class for scene node creation | ~400 lines |
| `tests/unit/data/node-factory.test.ts` | Unit tests for validation/transform helpers | ~150 lines |

**Total new code**: ~1,780 lines (moved from existing files + new unit tests, no new runtime logic).

## Summary of Modified Files

| File | Change |
|------|--------|
| `rendering/point-material.ts` | Import shaders, implement CameraAwareMaterial, remove inline shader strings |
| `rendering/line-material.ts` | Import shaders, implement CameraAwareMaterial, remove inline shader strings, remove geometry functions |
| `rendering/gsplat-material.ts` | Import shaders, implement CameraAwareMaterial, remove inline shader strings, remove geometry functions |
| `rendering/material-manager.ts` | Use CameraAwareMaterial interface, remove duck typing |
| `data/scene-loader.ts` | Use NodeFactory, remove extracted methods (~400-500 lines removed) |
| `data/scene-loader.ts` | Import geometry functions from line-geometry/gsplat-geometry (not material files) |
| `data/scene-loader.ts` (geometry updates) | Import geometry functions from line-geometry/gsplat-geometry (not material files) |
| `tests/unit/rendering/line-material.test.ts` | Import geometry functions from line-geometry (not material file) |
| `tests/unit/rendering/gsplat-material.test.ts` | Import geometry functions from gsplat-geometry (not material file) |