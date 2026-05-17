# Material wrapper layer — design decision for the WebGPU port

## Context

Every Luxar custom material today extends `THREE.ShaderMaterial`:

- Scene: `PointMaterial`, `LineMaterial`, `GSplatMaterial`.
- Post-processing: `MegaShaderMaterial`. Plus inline
  `new THREE.ShaderMaterial(...)` constructions inside
  `bloom-chain.ts` (three passes) and `fxaa-pass.ts` (one pass) —
  these are wrapped behind `BloomChain` and `FxaaPass` classes
  respectively, so the inheritance commitment is contained.
- Picking: `PointPickingMaterial`, `LinePickingMaterial`,
  `GSplatPickingMaterial`.

Under WebGPU, `WebGPURenderer` does not accept `ShaderMaterial` /
`RawShaderMaterial` (see `BROWSER_SUPPORT_POLICY.md`). Each material
must become a `NodeMaterial` instead. With the current inheritance
shape, that's a full rewrite of every material file.

This document records the chosen migration strategy for the
material wrapper layer. **No code change today**; the actual
refactor lands as part of port-PR-1.

## Options

### Option A — Composition wrapper

Define a `LuxarMaterial` interface that exposes
`getThreeMaterial(): THREE.Material`. Each material class composes
(rather than extends) a `THREE.Material` of the appropriate type.
Every consumer call site that does `mesh.material = pointMat`
becomes `mesh.material = pointMat.getThreeMaterial()`.

**Pros**:
- Clean swap of the inner `THREE.Material` at port time without
  touching the wrapper's own type.
- No inheritance commitment to a specific Three.js material class.
- Naturally supports multiple inner materials per wrapper if
  needed.

**Cons**:
- Touches roughly 40-60 consumer sites across `material-manager.ts`,
  `scene-setup/*`, factory functions, and test mocks — every
  `mesh.material = ...`, every `dispose()` chain, every place
  that reads material properties.
- Breaks Three.js convention (everyone reads `mesh.material as
  THREE.Material`). Future contributors expect that pattern.
- Material registration in `materialManager` (subscribes to
  `dispose` events) needs to track the wrapper plus the inner
  material — double bookkeeping.

### Option B — Conditional inheritance via factory

Keep `extends THREE.ShaderMaterial` today. At port time, introduce
a parallel `PointMaterialWebGPU extends NodeMaterial` and a factory
that picks the right class based on `RendererCapabilities.apiSurface`.
Every existing material class becomes WebGL-specific.

**Pros**:
- Keeps Three.js convention.
- No churn today.

**Cons**:
- Doubles the material surface during port. Both implementations
  must be maintained until the WebGL one is removed (which only
  happens if Option C in `BROWSER_SUPPORT_POLICY.md` succeeds and
  the fallback path is good enough).
- The two implementations must stay in sync — every PR that
  changes a uniform contract has to touch both.

### Option C — In-place rewrite at port time

Each material class flips from `extends ShaderMaterial` to
`extends NodeMaterial`. All call sites stay the same. The in-class
implementation (constructor body, the contents of `update*` /
`set*` setter methods) is rewritten in full.

**Pros**:
- Zero changes outside the material file itself. Smallest blast
  radius per-PR.
- Consumers stay typed against `THREE.Material` (the common
  base), which both `ShaderMaterial` and `NodeMaterial` satisfy.
- The `ShaderSource` registry from prep-Item-2 already isolates
  the actual GLSL → TSL conversion to the `webgpu` factory slot.
  The material wrapper just calls a different code path inside
  its constructor.

**Cons**:
- Each material file is rewritten as a single atomic change. Hard
  to incrementally test a single material's TSL port while the
  others stay GLSL within the same PR.

## Decision

**Option C, with one mitigation.**

Rationale:

1. **`THREE.Material` is the contract that matters.** Both
   `ShaderMaterial` and `NodeMaterial` derive from
   `THREE.Material`, sharing `.transparent`, `.blending`,
   `.depthTest`, `.depthWrite`, `.toneMapped`, `.side`, `.opacity`,
   `.visible`, and the lifecycle (`dispose` event, `needsUpdate`).
   See `BLENDING_PORT_NOTES.md` for the blending-state inventory —
   *every* state we use lives on the base class. Flipping a
   material's `extends` declaration is surgically scoped to the
   class's own body.
2. **The `ShaderSource` registry already absorbed the hard part.**
   Per `src/rendering/shaders/shader-source.ts`, each material's
   shader source-of-truth is a `ShaderSource` value whose
   `webgpu` slot is reserved for a TSL factory. The material's
   constructor reads `source.webgl.vertex/fragment` today; at
   port time it reads `source.webgpu(uniforms)` instead. Five
   lines of change per material.
3. **Composition (Option A) buys flexibility we don't need.**
   Luxar picks a renderer at startup and stays with it. No
   per-instance backend switching is on the roadmap. The
   call-site churn of Option A buys nothing concrete.
4. **Conditional inheritance (Option B) doubles maintenance.**
   With Option C the WebGL material file is *deleted* once the
   TSL port is verified; with B it stays alive parallel to the
   TSL one. Given `BROWSER_SUPPORT_POLICY.md` chooses Option C
   for the *renderer* (single renderer, Three.js internal
   fallback), supporting two material classes serves no purpose.

### The mitigation: `buildMaterial` helper

Introduce a single helper alongside port-PR-1's first TSL port:

```ts
// src/rendering/material-builder.ts (NEW at port-PR-1 time)
import * as THREE from 'three';
import type { ShaderSource } from './shaders/shader-source';
import type { RendererCapabilities } from './renderer-capabilities';

interface BuildMaterialConfig {
  uniforms: Record<string, THREE.IUniform>;
  defines?: Record<string, string>;
  // … blending, transparency, depth fields …
}

export function buildMaterial(
  source: ShaderSource,
  config: BuildMaterialConfig,
  caps: RendererCapabilities
): THREE.Material {
  if (caps.apiSurface === 'webgpu' && source.webgpu) {
    return source.webgpu(config.uniforms) as THREE.Material;
  }
  return new THREE.ShaderMaterial({
    uniforms: config.uniforms,
    vertexShader: source.webgl.vertex,
    fragmentShader: source.webgl.fragment,
    glslVersion: THREE.GLSL3,
    defines: config.defines,
    // … rest …
  });
}
```

Each material class delegates inner construction to it:

```ts
// Before: super({ vertexShader: SOURCE.webgl.vertex, … })
// After:  super(); Object.assign(this, buildMaterial(SOURCE, config, caps).parameters);
```

The class hierarchy stays the same (`extends THREE.ShaderMaterial`
today; flips to `extends NodeMaterial` per-PR as TSL factories
land), but the **branching point is one helper, not 9 classes**.
This makes the "WebGPU-or-WebGL-this-instance" decision a
runtime concern rather than a compile-time one — useful if we
ever do want a fallback path at instance construction time.

### What this commits to

- No refactor today. The 7 materials keep their current
  `extends THREE.ShaderMaterial` shape.
- **Port-PR-1** (`fxaa`, per `MIGRATION_PROGRESS.md`) lands the
  `buildMaterial` helper.
- **Port-PR-N** (each subsequent shader) flips its material's
  `extends` declaration to the WebGPU base, fills in
  `ShaderSource.webgpu`, and delegates inner construction to
  `buildMaterial`.
- The last port-PR removes any remaining `extends
  THREE.ShaderMaterial` markers and the `webgl` field on
  `ShaderSource`.

## Implications for other prep artefacts

- `MIGRATION_PROGRESS.md` lists `buildMaterial` as a non-shader
  migration task and the rationale section above as its source.
- `BROWSER_SUPPORT_POLICY.md` chooses single-renderer + Three.js
  internal fallback. `buildMaterial`'s branching on `caps.apiSurface`
  agrees with that choice (the helper returns whichever material
  type the renderer prefers, not whichever the user requested).
- `BLENDING_PORT_NOTES.md` confirms every blending state we use is
  on `THREE.Material` base — supporting the surgical-scope
  argument above.
- `PICKING_DESIGN.md` Option-A (async readback) is independent
  of the wrapper-layer decision; it lands separately.

## Out of scope

- **Bundle-size implications of `three/webgpu`.** That build is
  bigger than `three`. Worth measuring once the first TSL port
  lands and the import path goes live, but not today.
- **`NodeMaterial` TSL factory typing.** Today `ShaderSource.webgpu`
  is typed as `(uniforms: Record<string, unknown>) => unknown` —
  pessimistic. The first TSL port will need to tighten this to a
  discriminated union or a more concrete shape; that design call
  belongs to port-PR-1.

## References

- `src/rendering/shaders/shader-source.ts` — the `ShaderSource`
  interface with the `webgpu` factory slot.
- `BROWSER_SUPPORT_POLICY.md` — companion policy doc.
- `MIGRATION_PROGRESS.md` — port-order recommendation and
  shader matrix.
- [Three.js manual — "Using `WebGPURenderer`"](https://threejs.org/manual/en/webgpurenderer.html)
- [Three.js source — Material base class](https://github.com/mrdoob/three.js/blob/dev/src/materials/Material.js)
