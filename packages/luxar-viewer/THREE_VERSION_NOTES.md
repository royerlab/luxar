# Three.js version pinning notes

## Current pin

`packages/luxar-viewer/package.json` (peerDependency + devDependency):

```jsonc
"three": "~0.184.0"
"@types/three": "~0.184.0"
```

The `~` (tilde) range allows **patch updates only** within the
`0.184.x` line. Minor updates (`0.185.0`, `0.186.0`, …) require an
explicit version bump after a deliberate compatibility check.

## Why tilde, not caret

We previously used `^0.184.0`. With caret on a pre-1.0 package, npm /
pnpm treats the *minor* component as the breaking-change boundary, so
`^0.184.0` resolves to anything in `>=0.184.0 <0.185.0` — equivalent
to `~0.184.x` in practice. But the *intent* with caret is "allow
non-breaking minor updates"; once Three.js ships `0.185.0`, a fresh
install on a clean lockfile would silently pick it up.

Three.js's WebGPU surface (`WebGPURenderer`, `NodeMaterial`, TSL) is
still marked **"in development"** as of r184. The API shape has
churned across past minor releases (TSL function names renamed
between r178 and r181, renderer constructor signature reworked
between r182 and r184). A floating minor pin would mean a fresh
`pnpm install` mid-port silently swaps the target API surface from
under us.

`~0.184.0` is unambiguous: stay on r184 until we explicitly say
otherwise.

## What features we rely on (from r184)

These are the Three.js surfaces the viewer uses directly:

- `WebGLRenderer` for the default GLSL rendering path.
- `WebGPURenderer` for the opt-in TSL rendering path.
- `ShaderMaterial` with `glslVersion: THREE.GLSL3` for WebGL shaders.
- `NodeMaterial` / TSL for WebGPU shaders.
- `WebGLRenderTarget` with `HalfFloatType` for HDR scene and
  post-processing targets.
- `renderer.readRenderTargetPixelsAsync` for capture and picking readback.
- `EXRExporter` from `three/examples/jsm/exporters/EXRExporter.js`.
- Built-in tone-mapping support used by the WebGL mega-shader, with
  matching TSL math in the WebGPU mega-shader.

## When to bump

Trigger an explicit `~0.185.0` (or higher) bump only when:

1. Three.js releases notes for a stable WebGPU API surface, **or**
2. Luxar needs a specific rendering or TSL feature only present in a
   newer minor.

A bump means:

```bash
pnpm add three@~0.185.0 -E
pnpm add -D @types/three@~0.185.0 -E
pnpm typecheck
pnpm test --run
pnpm playwright test  # full E2E, not just the mega-shader spec
```

Both must be clean. If anything breaks, debug + fix before merging
the bump.

## Why no upper-bound on major

Three.js has never shipped a `1.0.0`, so the major version is
effectively pinned at `0`. The `~0.184.0` constraint accepts
`0.184.x` and refuses `0.185.0+`, which is exactly what we want.
