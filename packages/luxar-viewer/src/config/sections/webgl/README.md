# webgl

WebGL configuration slice. Owns the context attributes used when acquiring the WebGL2 canvas context, the `THREE.WebGLRenderer` constructor options, and the post-processing render-target settings.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

Shared attributes (`antialias`, `powerPreference`, `preserveDrawingBuffer`, `premultipliedAlpha`) live on `context` only — the renderer call spreads them alongside `renderer`-specific fields rather than duplicating them.

## Contents

- `data.ts` — `webglConfig: WebGLConfig`. `context` sets `alpha=false`, `antialias=false` (the scene never renders to the backbuffer — everything goes through the HDR render target, so a multisampled backbuffer is a dead allocation; `renderTarget.samples` controls real MSAA), `depth=true`, `stencil=false`, `powerPreference='high-performance'`, `desynchronized=true`, `premultipliedAlpha=true`, and `failIfMajorPerformanceCaveat=false`. There is deliberately no `colorSpace` — it is not a WebGL context attribute (the old `'display-p3'` entry was silently ignored); output color handling lives in the HDR pipeline. `renderer` sets `precision='highp'`, `logarithmicDepthBuffer=false`, and a disabled `shadowMap` (`THREE.PCFShadowMap`, soft by default since three r182). `renderTarget` keeps `depthBuffer=true`, `stencilBuffer=false`, and `samples=0` (MSAA off — incompatible with the additive blending used for points/gsplats).
- `types.ts` — `WebGLConfig` plus the three nested interfaces `WebGLContextAttributes`, `WebGLRendererConfig`, and `WebGLRenderTargetConfig`. `precision` is the literal union `'highp' | 'mediump' | 'lowp'`; `powerPreference` is `'high-performance' | 'low-power' | 'default'`.
- `validate.ts` — `validateWebGL(config, errors, warnings)`. Errors on `context.powerPreference` or `renderer.precision` outside their enum sets; warns on `renderTarget.samples` not in `{0, 2, 4, 8}`.

## Public API

- `webglConfig` — re-exported through `../../index.ts` into `AppConfig.webgl`.
- `WebGLConfig`, `WebGLContextAttributes`, `WebGLRendererConfig`, `WebGLRenderTargetConfig` — re-exported through `../../types.ts`.
- `validateWebGL` — called from `../../validation.ts`.
