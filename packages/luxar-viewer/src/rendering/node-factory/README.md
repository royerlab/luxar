# Node-factory helpers

Focused modules split out from the `NodeFactory` orchestrator one level
up at `rendering/node-factory.ts`. The orchestrator owns its caches,
the picking-system reference, and the public per-type entry points
(`createPointsNode`, `createLinesNode`, `createGSplatsNode`, and the
`createEmpty*Node` placeholders for all four types — a mesh node is only
ever created empty and filled by its first commit); these helpers own the
work — input validation, transform application, and
per-geometry mesh / material construction.

Every helper here is pure over its arguments (aside from `log.*` side
effects). The orchestrator calls them directly and assigns the returned
geometry / material / mesh back onto the scene graph. That split keeps
the orchestrator file small and lets each helper be unit-tested without
a real scene, renderer, or picking system.

## Module map

| File                     | Role                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validation.ts`          | `validateLoadedPointsData` (length/divisibility checks + structured log), `validateColorMode` (HDR Float32 vs SDR normalized-integer sanity), `validateTransformFormat` (row-major NumPy → throws)                                                                                                                                                                                                                      |
| `transforms.ts`          | `applyTransform` — length-16 + row-major guards, then installs the full matrix directly with auto-update disabled; TRS fields stay deliberately unpopulated so shear is preserved                                                                                                                                                                                                                                       |
| `create-points-node.ts`  | `createPointsGeometry` (one shared unit quad + the RGBA32F point texture / `aSortedIndex` storage pair, one fused dtype-aware texel write, `instanceCount` + `drawRange(0,6)`, metadata bounds), `createPointsMaterial` (per-node material + direct colormap application), `createPointsNode` (geometry + material + userData + optional picking shadow), and `createEmptyPointsNode` placeholder                       |
| `create-lines-node.ts`   | `createLinesNode` (line material + colormap + `createInstancedLinesMesh` + optional picking shadow) and `createEmptyLinesNode` placeholder                                                                                                                                                                                                                                                                              |
| `create-gsplats-node.ts` | `createGSplatsNode` (gsplat material + colormap + `createInstancedGSplatsMesh` + optional picking shadow) and `createEmptyGSplatsNode` placeholder                                                                                                                                                                                                                                                                      |
| `create-mesh-node.ts`    | `createEmptyMeshNode` (plain indexed geometry from `createEmptyMeshGeometry` + house or `physical` material + vertex-granular picking shadow), the viewer-side mode rules (`resolveRequestedMeshMode`: default `opaque`, `volumetric` degrades), and the appliers the commit path and the Layers panel call (`applyMeshShading`, `applyMeshTexture`, `applyMeshSide`, `applyMeshVertexAlpha`, `syncMeshPickAppearance`) |

## How the orchestrator composes them

```
NodeFactory (class in rendering/node-factory.ts)
   │
   ├── points  ──► create-points-node.createPointsNode
   │               create-points-node.createEmptyPointsNode
   │               create-points-node.createPointsGeometry
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
   ├── mesh    ──► create-mesh-node.createEmptyMeshNode
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
- **Colormap paths.** Every visual material is PER NODE (each carries
  its node's own element texture, or for a mesh its own vertex buffers),
  so the creators apply the colormap texture + scalar range directly to
  the node-owned material — there is no shared material cache and no
  clone.
- **Authored gain on a colormapped node is a WINDOW, not a gain.** All
  four factories agree (#936/#1081/#1082): when a colormap actually takes
  over, the post-LUT color GOG is left/reset at identity and the
  authored `intensity`/`offset` is re-expressed as the scalar LUT window
  via the shared `rendering/display-range.ts::resolveColormapWindow`.
  Applying it as both would double-apply, and would make the same
  attribute mean two different things depending on the `layer` flag
  (the layers panel already windows it). The identity-vs-window decision
  keys on the RAW LEAF gain so an ancestor-only gain folds onto the data
  range instead of replacing it — which is why points threads a
  `leafAttrs` param alongside its composed `attrs` (lines and gsplats
  already receive both). Direct-color nodes still get the gain.
- **Scalar-attribute guard (points / lines / mesh).** Points and mesh
  consult `supportsScalarColormap(type, geometry)` when a geometry is
  supplied; lines read the `userData.hasScalars` stamp
  `createInstancedLinesMesh` writes. If the guard fails, the colormap is suppressed and
  rendering falls back to vertex colors with a warning. GSplats do
  not gate on scalar binding — amplitudes are always present.
- **Shared geometry across visual + pick.** Every picking shadow node
  reuses `mesh.geometry` directly — only the
  material differs. The pick material is registered with
  `materialManager.register(...)` so disposal flows through the same
  lifecycle path.
- **dtype-aware normalization (points).** Radii, sharpness and scalars
  accept Float16 / Float32 (widened as-is into the point texture) or
  Uint8 (normalized ÷255, with a `radiusScale` baked into
  `geometry.userData` so the material can scale radii back to physical
  units). The shader reads the texel radius times the `radiusScale`
  uniform; sharpness needs no scale — it is authored natively in the
  `[0, 1]` knob range (a `uint8/255` value already lands in range).
- **Empty-buffer placeholders.** The four `createEmpty*Node` factories
  build a fully-typed mesh with zero-length
  buffers so the scene graph can mount the node before its data
  arrives; the loader then commits real buffers in place.

## See also

- `../README.md` — rendering package overview, material-manager / mega
  shader / picking pipeline.
- `../node-factory.ts` — the orchestrator class these helpers serve.
- `../material-manager/README.md` — material creation + the
  global-update registry every created material joins.
- `../line-geometry.ts`, `../gsplat-geometry.ts`, `../point-geometry.ts`,
  `../mesh-geometry.ts` — the lower-level geometry builders these helpers
  wrap.
- `../picking/picking-system.ts` — the picking-system reference threaded
  through every creator for pick-shadow registration.
