# The lazy TSL / WebGPU boundary

Three tiny modules whose only job is to keep the three.js node system off the
initial payload.

The production default backend is WebGL (`selectBackend()` in
`scene/scene-manager/render-pipeline/renderer-setup.ts`); WebGPU is opt-in per
session via `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`. But the TSL
material classes all `extend NodeMaterial` from `three/webgpu`, and until
issue #1679 twenty-three production modules imported that subpath as a value.
One such import anywhere the entry point reaches is enough to pin the whole
`three-webgpu` chunk — **~185 kB gzipped, about a quarter of the initial JS** —
into the eager graph, where every WebGL user downloaded and parsed a renderer
that never ran.

## Module map

| File          | Role                                                                                                                                                                                                                                                                                                                  |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `registry.ts` | The designated entry to the lazy cone: the `*-tsl` / `*.tsl` modules that name `three/webgpu` / `three/tsl` as values are reachable only through it in production (the e2e TSL harness imports a few of them directly). Gathers the 9 TSL material classes and every TSL graph factory into one `TSL_REGISTRY` value. |
| `load.ts`     | The **only** module that imports `registry.ts` at runtime, via `await import()` (`slot.ts`'s import is type-only and erased). That single dynamic edge is what makes the chunk lazy. Memoized and concurrency-safe; re-loadable after a transient failure.                                                            |
| `slot.ts`     | Zero-runtime-import leaf holding the loaded registry. `requireTslMaterials()` for synchronous consumers.                                                                                                                                                                                                              |

## Why `slot.ts` is separate from `load.ts`

The `ShaderSource.webgpu` closures live next to their GLSL twin (they carry
shader-specific notes about reading build-time uniforms), so
`materials/*/shader-glsl.ts` needs `requireTslMaterials()`. If that accessor
lived in `load.ts`, those modules would gain a graph edge to
`load.ts → registry.ts → *-tsl.ts → *-glsl.ts` and close a cycle. The dynamic
import breaks it at runtime, but `no-circular` in `.dependency-cruiser.cjs` is
error-severity and counts a dynamic edge like any other — rightly, since a cycle
that only works because of import timing is what that rule exists to prevent.
So the accessor sits in a leaf that imports nothing at runtime.

## The ordering contract

`loadTslMaterials()` must resolve before the first material is constructed on
the WebGPU path. `SceneManager.init` already guarantees this:

```
init()
  └── await setupRenderer()
        └── await setupWebGPURenderer()
              ├── await createWebGPURenderer()   ← await import('three/webgpu')
              ├── await loadTslMaterials()       ← the registry lands here
              └── materialManager.setCaps(caps)
  └── setupPostProcessing()                      ← FIRST material in the app
        └── resource-lifecycle.ts → createMegaShaderMaterial(), BloomChain, FxaaPass
```

Because the registry is installed by then, `buildMaterial` and the
`MaterialManager` dispatch tables stay **synchronous**. Node materials come far
later still, behind async dataset loading.

If the ordering is ever broken, `requireTslMaterials()` throws with a directive
message. It deliberately does **not** fall back to the GLSL classes: a
`ShaderMaterial` does not render under `WebGPURenderer` at all, so a silent
fallback would paint blank quads and look like a rendering bug instead of the
wiring bug it is.

## What keeps it from regressing

Two gates, because the two failure modes are different:

- **Source** — `@typescript-eslint/no-restricted-imports` in `eslint.config.js`
  bans value imports of `three/webgpu` / `three/tsl` outside `registry.ts` and
  the `*-tsl` modules it owns. `import type` stays legal (types are erased).
- **Build output** — `scripts/check-eager-chunks.mjs` (`pnpm build:check`, chained
  into `pnpm build`) asserts against `dist/` that no chunk reachable statically
  from the entry imports `three-webgpu`, and that the chunk still exists.

The second is not redundant. The last mile of #1679 was not a source problem at
all: `three.core.js` is shared by `three.module.js` and `three.webgpu.js`, and
with no chunk group of its own it landed such that the plain `three` chunk
imported `three-webgpu` — eager again, with every source file already clean.
Only an assertion on the built output can see that. Hence the `three-core` group
in `vite.config.ts`.

## Adding a TSL symbol

Add it to `TSL_REGISTRY` and read it through `requireTslMaterials()`. Do not add
an ESLint exemption — the exemption list covers the modules that are already
inside the lazy cone, and adding to it is how the eager edge comes back.
