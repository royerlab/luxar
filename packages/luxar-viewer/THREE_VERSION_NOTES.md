# Three.js version pinning notes

## Current pin

`packages/luxar-viewer/package.json`:

```jsonc
"three": "~0.184.0"        // peerDependency — the RUNTIME
"@types/three": "~0.185.1" // devDependency — the TYPES
```

The `~` (tilde) range allows **patch updates only** within a minor line. Minor
updates (`0.185.0`, `0.186.0`, …) require an explicit version bump after a
deliberate compatibility check.

## Yes, the types deliberately lead the runtime by a minor

`three` ships **no `.d.ts` of its own**, so `@types/three` is the _sole_ type
description of the runtime. That makes the version skew above look like an
oversight. It is not — it is load-bearing, and closing it in the obvious
direction breaks the build:

`@types/three@0.184.1` (the last r184 definitions release) has an assignability
defect in the TSL node types. It declares

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
call that is componentwise and correct in r184 and r185 alike. `0.185.x` fixes it
by widening the vector overloads to `Vec3OrFloat`.

So the r185 _definitions_ describe the r184 _runtime_ more accurately than the
r184 definitions do. Only their version number leads.

**Do not "fix" this by pinning `@types/three` back to `~0.184.1`.** That
reintroduces a typecheck failure whose only remedies are a cast around a correct
call or a downgrade of type safety at that site.

**Do not fix it by bumping `three` to r185 either** — see below.

## Why the runtime is still on r184

The obvious resolution is to move the runtime up so both sit at r185. That was
attempted and **reverted**: r185 broke the TSL / WebGPU path. Measured on the
same machine, same specs, with the `three` version as the only variable — this
table predates the #1697 fix and describes the code as it was then:

| `three` | `tsl-shader-parity` + `tsl-codegen-snapshot` |
| ------- | -------------------------------------------- |
| 0.184.0 | **100 passed**, 0 failed                     |
| 0.185.1 | 96 passed, **4 failed**                      |

The failures were behavioural, not tolerance drift — most starkly
`gsplat-pick-surface`, where the TSL side rendered **zero pixels**. All four had
one cause: the pick shaders shared fragment values across `colorNode` and
`depthNode` and so depended on which entry point three built first, which is what
r185 flipped. #1697 fixed that, and re-ran the same A/B on the fixed code: 101
passed at both 0.184.0 and 0.185.1.

👉 **Tracking issue: #1683**, still open — it now covers the bump itself, which
also wants a full E2E pass, since a `three` minor can move pixels well outside
the picking shaders. Until someone does that, `three` stays at `~0.184.0`.

## Why tilde, not caret

We previously used `^0.184.0`. With caret on a pre-1.0 package, npm / pnpm treats
the _minor_ component as the breaking-change boundary, so `^0.184.0` resolves to
anything in `>=0.184.0 <0.185.0` — equivalent to `~0.184.x` in practice. But the
_intent_ with caret is "allow non-breaking minor updates"; once Three.js ships a
new minor, a fresh install on a clean lockfile would silently pick it up.

Three.js's WebGPU surface (`WebGPURenderer`, `NodeMaterial`, TSL) is still marked
**"in development"**. The API shape has churned across past minor releases (TSL
function names renamed between r178 and r181, renderer constructor signature
reworked between r182 and r184), and #1683 is simply the latest instance. A
floating minor pin would mean a fresh `pnpm install` silently swapping the target
API surface from under us.

`~0.184.0` is unambiguous: stay on r184 until we explicitly say otherwise.

## What features we rely on (from r184)

These are the Three.js surfaces the viewer uses directly:

- `WebGLRenderer` for the default GLSL rendering path.
- `WebGPURenderer` (from `three/webgpu`) for the opt-in TSL rendering path —
  constructed in `src/scene/scene-manager/render-pipeline/renderer-setup.ts`
  once the backend is selected, and imported there lazily so a WebGL session
  never downloads it (see `src/rendering/tsl/README.md`).
- `ShaderMaterial` with `glslVersion: THREE.GLSL3` for WebGL shaders.
- `NodeMaterial` / TSL for WebGPU shaders.
- `WebGLRenderTarget` with `HalfFloatType` for HDR scene and post-processing
  targets.
- `renderer.readRenderTargetPixelsAsync` for capture and picking readback.
- `EXRExporter` from `three/examples/jsm/exporters/EXRExporter.js`.
- Built-in tone-mapping support used by the WebGL mega-shader, with matching TSL
  math in the WebGPU mega-shader.

## When to bump

Trigger an explicit `~0.185.0` (or higher) bump when one of:

1. Three.js releases notes for a stable WebGPU API surface, or
2. Luxar needs a specific rendering or TSL feature only present in a newer minor.

The r185 parity failures are fixed (#1697), so what gates a bump now is the
checklist below rather than an open code defect. #1683 tracks that step.

To perform the bump, edit the two ranges in `package.json` **by hand**. `three`
lives under `peerDependencies`, and `pnpm add -E` would both move it into
`dependencies` (shipping a second copy of three to every consumer of the
published package) and replace the tilde with an exact version:

```jsonc
"peerDependencies": { "three": "~0.185.0" }
"devDependencies":  { "@types/three": "~0.185.0" }
```

Then:

```bash
pnpm install
pnpm typecheck
pnpm test --run
pnpm playwright test  # full E2E, not just the mega-shader spec
```

All must be clean. The full E2E run is not ceremony: it is exactly what caught
#1683, which `pnpm typecheck` and all 11,713 unit tests passed straight through.

## Why no upper-bound on major

Three.js has never shipped a `1.0.0`, so the major version is effectively pinned
at `0`. The `~0.184.0` constraint accepts `0.184.x` and refuses `0.185.0+`, which
is exactly what we want.
