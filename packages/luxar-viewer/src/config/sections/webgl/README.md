# webgl

WebGL configuration slice. Owns the context attributes used when acquiring the WebGL2 canvas context, the `THREE.WebGLRenderer` constructor options, and the post-processing render-target settings.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

Shared attributes (`antialias`, `powerPreference`, `preserveDrawingBuffer`, `premultipliedAlpha`) live on `context` only — the renderer call spreads them alongside `renderer`-specific fields rather than duplicating them.

## Contents

- `data.ts` — `webglConfig: WebGLConfig`. `context` sets `alpha=false`, `antialias=true`, `depth=true`, `stencil=false`, `powerPreference='high-performance'`, `colorSpace='display-p3'`, `desynchronized=true`, `premultipliedAlpha=true`, and `failIfMajorPerformanceCaveat=false`. `renderer` sets `precision='highp'`, `logarithmicDepthBuffer=false`, and a disabled `shadowMap` (`THREE.PCFShadowMap`, soft by default since three r182). `renderTarget` keeps `depthBuffer=true`, `stencilBuffer=false`, and `samples=0` (MSAA off — incompatible with the additive blending used for points/gsplats).
- `types.ts` — `WebGLConfig` plus the three nested interfaces `WebGLContextAttributes`, `WebGLRendererConfig`, and `WebGLRenderTargetConfig`. `precision` is the literal union `'highp' | 'mediump' | 'lowp'`; `powerPreference` is `'high-performance' | 'low-power' | 'default'`.
- `validate.ts` — `validateWebGL(config, errors, warnings)`. Errors on `context.powerPreference` or `renderer.precision` outside their enum sets; warns on `renderTarget.samples` not in `{0, 2, 4, 8}` and on `context.colorSpace` outside `{'srgb', 'display-p3', 'rec2020'}`.

## Public API

- `webglConfig` — re-exported through `../../index.ts` into `AppConfig.webgl`.
- `WebGLConfig`, `WebGLContextAttributes`, `WebGLRendererConfig`, `WebGLRenderTargetConfig` — re-exported through `../../types.ts`.
- `validateWebGL` — called from `../../validation.ts`.
