# Three.js version pinning notes

## Current pin

`packages/luxar-viewer/package.json`:

```jsonc
"three": "~0.185.1"        // peerDependency — the RUNTIME
"@types/three": "~0.185.1" // devDependency — the TYPES
```

The `~` (tilde) range allows **patch updates only** within a minor line. Minor
updates (`0.186.0`, `0.187.0`, …) require an explicit version bump after a
deliberate compatibility check.

Both pins sit on the same minor. That is new: the repo carried a deliberate
one-minor skew (`three@~0.184.0` against `@types/three@~0.185.1`) from the r185
type fix landing before the runtime could follow. The runtime caught up in
#1683, and the skew is closed — a future bump moves both together.
`pnpm check:three-types` enforces that shared minor across both declarations
and the embed importmaps. Reopening a deliberate skew therefore requires an
explicit change to the guard, not just different ranges in `package.json`.

## Why the types were allowed to lead, and why that is over

`three` ships **no `.d.ts` of its own**, so `@types/three` is the _sole_ type
description of the runtime. `@types/three@0.184.1` (the last r184 definitions
release) has an assignability defect in the TSL node types. It declares

```ts
interface Pow {
  (x: FloatOrNumber, y: FloatOrNumber): Node<'float'>;
  (x: Node<'vec3'>, y: Node<'vec3'>): Node<'vec3'>; // <- too narrow
}
```

and a `VarNode<"vec3">` is **not** assignable to `Node<"vec3">` there (the
recursive `NodeExtensions` chain resolves `label()` to `Node<"float">`).
Overload resolution therefore falls back to the float signature and rejects
`src/rendering/materials/line/shader-tsl-capsule.ts:618` — a `pow(vec3, vec3)`
call that is componentwise and correct in r184 and r185 alike. `0.185.x` fixes
it by widening the vector overloads to `Vec3OrFloat`.

So while the runtime sat at r184, the r185 _definitions_ described it more
accurately than the r184 definitions did. **Do not "fix" the history by pinning
`@types/three` back to `~0.184.1`** if you ever find yourself downgrading the
runtime — that reintroduces a typecheck failure whose only remedies are a cast
around a correct call or a downgrade of type safety at that site.

## The r185 runtime bump, and what it cost

Moving the runtime from r184 to r185 was attempted once and reverted: it broke
the TSL / WebGPU path in four measurable ways, most starkly `gsplat-pick-surface`,
where the TSL side rendered **zero pixels**. All four had one cause, and it was
ours rather than upstream's: the pick shaders shared fragment values across
`colorNode` and `depthNode`, so they depended on which entry point three built
first — and that order is not part of three's API. r184 built colour first, r185
builds depth first. #1697 fixed it (a `Fn(…).once()` prologue that assigns every
shared value in unconditional flow, called first from both entry points), and
#1683 then performed the bump.

Measured on the same machine, same specs, `three` the only variable:

| tree  | `three` | `tsl-shader-parity` + `tsl-codegen-snapshot` |
| ----- | ------- | -------------------------------------------- |
| #1697 | 0.184.0 | 101 passed                                   |
| #1697 | 0.185.1 | 101 passed                                   |

The **shaders still run clean at both revisions** — the fix removed the
dependency on build order rather than adapting to r185's. What the bump changed
is which one the snapshots pin.

### What moved in the 74 generated-shader snapshots

Regenerating `src/tests/__codegen__/` under r185 rewrote every file, and the
whole diff is presentational:

- the `// Three.js r184` header line becomes `r185`;
- a new (empty) `// structs` section is emitted above the uniforms;
- the `object` and `render` std140 blocks swap order, and `nodeUniformN` /
  `nodeVarN` renumber accordingly;
- `modelViewMatrix = cameraViewMatrix * modelMatrix` is emitted lazily at its
  first use instead of at the top of `main()`;
- `(!x)` is now printed `( ! x )`;
- and, in the six pick fragment shaders, the whole `depthNode` flow — the
  `uSurfaceDepth` select and the `gl_FragDepth` write — moves from the end of
  `main()` to above the discards, because r185 builds that entry point first.

That last one is the only _structural_ move, and it is the flip this whole bump
is about. It is semantically inert: a discarded fragment updates no buffer, depth
included, so writing `gl_FragDepth` above a `discard` changes nothing, and the
arithmetic hoisted along with it (a `dot`, a `sqrt`/`pow` on non-negative inputs)
is thrown away with the fragment. Worth naming because the multiset check below is
order-insensitive **by construction** and therefore cannot see it — the hazard scan
is what covers it.

Checked, not assumed: with `nodeUniformN` / `nodeVarN` / `nodeVaryingN`
normalized away, 58 of the 74 `main()` bodies are an identical multiset of
statements and the other 16 differ only by that `!` spacing. No expression
changed.

The flow reordering is the same hazard #1697 fixed, so it was re-checked the
same way: a scan for values first assigned inside a branch and later read at top
level returns the **same set** under r184 and r185 snapshots (only the variable
numbers differ). `tsl-codegen-snapshot.spec.ts` pins this structurally. Note that the
bump FLIPPED which half of that assertion is live rather than arming both: with
the depth flow built first, the prologue lands at brace depth 0 at the top of
`main()`, so the BRACE-DEPTH half is now the live one and the READ-ORDER half can
no longer be tripped from `colorNode`. One consequence is worth carrying here too:
`colorNode`'s own `fragmentPrologue()` call is now verified by nothing — removing
it leaves the generated GLSL byte-identical — and it is kept deliberately, so that
both entry points stay self-sufficient whichever one a future three builds first.
See the honesty note at that call site.

### Runtime behaviour changes r185 brings, and why none of them bite here

Checked against the two installed trees rather than the release notes:

- `WebGPUBackend` now installs `device.onuncapturederror -> renderer.onError`, and
  it does so **after** the "was a device supplied?" branch — so it applies to the
  pre-built device `createWebGPURenderer()` passes in. Three's default handler
  logs. A previously-silent uncaptured GPU error therefore becomes console output,
  which `LUXAR_E2E_STRICT_CONSOLE=1` would fail on. Nothing routes it through the
  viewer's `log` / notifier yet; if that becomes noisy, set `renderer.onError`
  deliberately rather than muting the console check.
- `WebGLProgram` calls `bindAttribLocation(program, 0, 'position')` for any
  geometry carrying a position attribute (r184 did it only for morph targets).
  Harmless here: no viewer GLSL uses an explicit `layout(location = …)`, so
  nothing can collide with the forced slot 0.
- `WebGLRenderer` moved `info.reset()` to before shadow rendering, so
  `renderer.info.render` now counts shadow draws. Harmless: nothing outside tests
  reads `renderer.info`.
- `WebGLRenderLists.sort()` gained a `reversedDepth` parameter. Harmless:
  `camera.reversedDepth` is never set.
- `WebGPUAttributeUtils` pads any `itemSize > 1` attribute whose byte stride is
  not a multiple of 4. Nothing the viewer builds hits it — `aQuadCorner` is an
  8-byte float32x2, mesh colours are already CPU-padded to four components. Note
  this means r185 partially closes the gap `docs/specs/MESH_NODE_SPEC.md` records
  as an r184 blocker; that spec is now describing a fixed upstream defect.
- Removed exports (`three/tsl`: `arrayBuffer`, `modInt`, `string`;
  `three/webgpu`: `BatchNode`, `InstanceNode`, `InstancedMeshNode`, `MorphNode`,
  `SkinningNode`) and the `NodeMaterial.setupLights` -> `setupMaterialLightings`
  rename touch nothing the viewer imports or overrides. Every pinned THREE enum
  value in `src/tests/e2e/blending-expected-state.ts` is unchanged at r185.

### The visual-regression suite is local evidence, not CI evidence

The E2E tree now has a complete Linux Chromium baseline for every `@visual`
test. The four Linux baselines that existed during the r185 upgrade
(`basic-rendering`, `blending-modes`, `custom-gui-library`, and
`post-processing-pipeline`) came out byte-unchanged, which remains real but
narrow evidence for that upgrade. The full corpus was recorded locally rather
than by CI: run `pnpm test:e2e:visual` and inspect intentional updates from
`pnpm test:e2e:visual:update`, but do not treat a green pull request as a pixel
comparison because CI still excludes visual tests.

## Why tilde, not caret

We previously used `^0.184.0`. With caret on a pre-1.0 package, npm / pnpm treats
the _minor_ component as the breaking-change boundary, so `^0.185.0` resolves to
anything in `>=0.185.0 <0.186.0` — equivalent to `~0.185.x` in practice. But the _intent_ with caret is "allow
non-breaking minor updates"; once Three.js ships a new minor, a fresh install on
a clean lockfile would silently pick it up.

Three.js's WebGPU surface (`WebGPURenderer`, `NodeMaterial`, TSL) is still marked
**"in development"**. The API shape has churned across past minor releases (TSL
function names renamed between r178 and r181, renderer constructor signature
reworked between r182 and r184, fragment entry-point emission order flipped
between r184 and r185). A floating minor pin would mean a fresh `pnpm install`
silently swapping the target API surface from under us.

`~0.185.1` is unambiguous: stay on r185 until we explicitly say otherwise. (It
takes its patch component from `@types/three`, which is published more often; the
runtime guard in `environment-guards.ts` compares MINORS only, so it admits any
`0.185.x` — the exact patch floor lives in the peer range alone.)

## What features we rely on (from r185)

These are the Three.js surfaces the viewer uses directly:

- `WebGLRenderer` for the default GLSL rendering path.
- `WebGPURenderer` (from `three/webgpu`) for the opt-in TSL rendering path —
  constructed in `src/scene/scene-manager/render-pipeline/renderer-setup.ts`
  once the backend is selected, and imported there lazily so a WebGL session
  never downloads it (see `src/rendering/tsl/README.md`). Three still defaults
  its adapter request to `featureLevel: 'compatibility'` at r185, exactly as it
  did at r184, so the core-adapter negotiation there is still required.
- `ShaderMaterial` with `glslVersion: THREE.GLSL3` for WebGL shaders.
- `NodeMaterial` / TSL for WebGPU shaders.
- `WebGLRenderTarget` with `HalfFloatType` for HDR scene and post-processing
  targets.
- `renderer.readRenderTargetPixelsAsync` for capture and picking readback.
- `EXRExporter` from `three/examples/jsm/exporters/EXRExporter.js`.
- Built-in tone-mapping support used by the WebGL mega-shader, with matching TSL
  math in the WebGPU mega-shader.

Note that several long-form comments elsewhere in the tree still describe
behaviour "in r184" — those are dated observations from when they were measured
(the mesh vertex-format gaps in `docs/specs/MESH_NODE_SPEC.md`, the depth-sorting
spike verdicts). They are kept as the record of what was checked and when; only
the statements about the _current pin_ were rewritten for r185.

## When to bump

Trigger an explicit `~0.186.0` (or higher) bump when one of:

1. Three.js releases notes for a stable WebGPU API surface, or
2. Luxar needs a specific rendering or TSL feature only present in a newer minor.

To perform the bump, edit the two ranges in `package.json` **by hand**.
Dependabot intentionally ignores `@types/three` minor and major updates because
it cannot update the peer runtime alongside them; patch updates within the
current minor remain automated. `three` lives under `peerDependencies`, and
`pnpm add -E` would both move it into `dependencies` (shipping a second copy of
three to every consumer of the published package) and replace the tilde with an
exact version:

```jsonc
"peerDependencies": { "three": "~0.186.0" }
"devDependencies":  { "@types/three": "~0.186.0" }
```

Then:

```bash
pnpm install
pnpm check:three-types
pnpm typecheck
pnpm test --run
# BEFORE regenerating anything — the failure set is the evidence:
npx playwright test tsl-shader-parity.spec.ts tsl-codegen-snapshot.spec.ts
LUXAR_UPDATE_SNAPSHOTS=1 npx playwright test tsl-codegen-snapshot.spec.ts
npx playwright test   # full E2E, not just the mega-shader specs
```

All must be clean, and the regenerated snapshot diff must be _read_ rather than
merely made green. Also bump the revision floor in
`src/core/app/init/environment-guards.ts` (`assertThreeRevision`), which tracks
the peer range's minor, and the pinned importmap URL in `examples/embed/`
(`index.html` and `README.md`). `pnpm check:three-types` verifies that both
importmaps stay on the peer range's minor before the example can drift.

The full E2E run is not ceremony: it is exactly what caught #1683, which
`pnpm typecheck` and all ~11,900 unit tests passed straight through.

## Why no upper-bound on major

Three.js has never shipped a `1.0.0`, so the major version is effectively pinned
at `0`. The `~0.185.1` constraint accepts `0.185.1` and later patches, refuses
`0.186.0+`, and that is exactly what we want.
