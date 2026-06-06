# Node-factory helpers

Focused modules split out from the `NodeFactory` orchestrator one level
up at `rendering/node-factory.ts`. The orchestrator owns its caches,
the picking-system reference, and the public per-type entry points
(`createPointsNode`, `createLinesNode`, `createGSplatsNode`); these
helpers own the work — input validation, transform decomposition, and
per-geometry mesh / material construction.

Every helper here is pure over its arguments (aside from `log.*` side
effects). The orchestrator calls them directly and assigns the returned
geometry / material / mesh back onto the scene graph. That split keeps
the orchestrator file small and lets each helper be unit-tested without
a real scene, renderer, or picking system.

## Module map

| File                     | Role                                                                                                                                                                                                                                                                      |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validation.ts`          | `validateLoadedPointsData` (length/divisibility checks + structured log), `validateColorMode` (HDR Float32 vs SDR normalized-integer sanity), `validateTransformFormat` (row-major NumPy → throws)                                                                        |
| `transforms.ts`          | `applyTransform` — length-16 guard + row-major guard, then `Matrix4.fromArray().decompose()` onto `object.position / quaternion / scale`                                                                                                                                  |
| `create-points-node.ts`  | `createPointsGeometry` (one shared unit quad + per-instance `aCenter/aColor/aRadius/aSharpness/aScalar`, dtype-aware normalization, `instanceCount` + `drawRange(0,6)`, metadata bounds) and `createPointsMaterial` (materialManager lookup + scalar-colormap clone path) |
| `create-lines-node.ts`   | `createLinesNode` (line material + colormap clone + `sharpness=2` fast path + `createInstancedLinesMesh` + optional picking shadow) and `createEmptyLinesNode` placeholder                                                                                                |
| `create-gsplats-node.ts` | `createGSplatsNode` (gsplat material + colormap clone + `createInstancedGSplatsMesh` + optional picking shadow) and `createEmptyGSplatsNode` placeholder                                                                                                                  |

## How the orchestrator composes them

```
NodeFactory (class in rendering/node-factory.ts)
   │
   ├── points  ──► create-points-node.createPointsGeometry
   │               create-points-node.createPointsMaterial
   │               validation.validateLoadedPointsData
   │               validation.validateColorMode
   │
   ├── lines   ──► create-lines-node.createLinesNode
   │               create-lines-node.createEmptyLinesNode
   │
   ├── gsplats ──► create-gsplats-node.createGSplatsNode
   │               create-gsplats-node.createEmptyGSplatsNode
   │
   └── all     ──► transforms.applyTransform
                   validation.validateTransformFormat
```

## Key contracts

- **Producer-side row-major guard.** `validateTransformFormat` throws
  when a 4×4 transform looks row-major (non-zero at indices `[3,7,11]`
  with zeros at `[12,13,14]`). It also throws an "ambiguous" error when
  _both_ translation bands are non-zero — a correct column-major matrix
  has its last row `[3,7,11,15]` equal to `[0,0,0,1]`, so a non-zero
  `[3,7,11]` always signals a producer bug rather than letting geometry
  land in the wrong place. Python producers must transpose before
  storing: `matrix.T.ravel().tolist()`. This is the load-time refusal
  the project-root `CLAUDE.md` ("Critical Gotchas / Matrix Storage")
  references.
- **Colormap clone path.** `createPointsMaterial`, `createLinesNode`,
  and `createGSplatsNode` all share the same shape: when
  `nodeAttrs.colormap` is set (and, for points / lines, a scalar
  attribute is actually bound), the helper calls
  `materialManager.detachFromGlobalUpdates(material)`, clones the
  pooled material, re-registers the clone, and applies the colormap
  texture + scalar range. Without the detach step,
  `materialManager.disposeAll()` would dispose the pooled cache entry
  still serving other callers.
- **Scalar-attribute guard (points / lines).** Points consults
  `supportsScalarColormap('points', geometry)` when a geometry is
  supplied; lines checks for `startScalars`/`endScalars` on the
  processed config. If the guard fails, the colormap is suppressed and
  rendering falls back to vertex colors with a warning. GSplats do
  not gate on scalar binding — amplitudes are always present.
- **Sharpness fast path (lines).** `createLinesNode` calls
  `isAllSharpnessTwo(processed)` and forwards the result to both the
  visual and picking materials via `setSharpnessAllTwo(...)`. The
  shader then replaces `pow(x, vSharpness)` with `x*x`. The picking
  material **must** mirror the same toggle or pick rays will not match
  visible geometry.
- **Shared geometry across visual + pick.** The lines and gsplats
  picking shadow node reuses `mesh.geometry` directly — only the
  material differs. The pick material is registered with
  `materialManager.register(...)` so disposal flows through the same
  lifecycle path.
- **dtype-aware normalization (points).** `aRadius` / `aSharpness` /
  `aScalar` accept Float16 (widened to Float32 for the
  `InstancedBufferAttribute`), Uint8 (`normalized: true`, with a
  `radiusScale` / `sharpnessScale` baked into `geometry.userData` so
  the material can scale back to physical units), or Float32
  (unnormalized). The shader reads the normalized value times the
  per-geometry scale uniform.
- **Empty-buffer placeholders.** `createEmptyLinesNode` and
  `createEmptyGSplatsNode` build a fully-typed mesh with zero-length
  buffers so the scene graph can mount the node before its data
  arrives; the loader then commits real buffers in place.

## See also

- `../README.md` — rendering package overview, material-manager / mega
  shader / picking pipeline.
- `../node-factory.ts` — the orchestrator class these helpers serve.
- `../material-manager/README.md` — the material cache + global-update
  registry that the colormap clone path detaches from.
- `../line-geometry.ts`, `../gsplat-geometry.ts`, `../point-geometry.ts`
  — the lower-level mesh builders these helpers wrap.
- `../picking/picking-system.ts` — the picking-system reference threaded
  through the lines / gsplats creators for pick-shadow registration.
