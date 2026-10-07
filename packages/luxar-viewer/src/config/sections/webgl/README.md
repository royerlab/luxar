# webgl

WebGL configuration slice. Owns the context attributes used when acquiring the WebGL2 canvas context, the `THREE.WebGLRenderer` constructor options, and the post-processing render-target settings.

Conforms to the section-trio pattern documented in [../../README.md](../../README.md): `data.ts` exports the literal, `types.ts` defines the interface, `validate.ts` exports a section validator invoked by the central dispatcher.

Shared attributes (`antialias`, `powerPreference`, `preserveDrawingBuffer`, `premultipliedAlpha`) live on `context` only — the renderer call spreads them alongside `renderer`-specific fields rather than duplicating them.

## Contents

- `data.ts` — `webglConfig: WebGLConfig`. `context` sets `alpha=false`, `antialias=false` (the scene never renders to the backbuffer — everything goes through the HDR render target, so a multisampled backbuffer is a dead allocation; real MSAA is `renderingControls.msaaEnabled` / `msaaSamples`, applied by the post-processing manager), `depth=true`, `stencil=false`, `powerPreference='high-performance'`, `desynchronized=true`, `premultipliedAlpha=true`, and `failIfMajorPerformanceCaveat=false`. There is deliberately no `colorSpace` — it is not a WebGL context attribute (the old `'display-p3'` entry was silently ignored); output color handling lives in the HDR pipeline. `renderer` sets `precision='highp'` and `logarithmicDepthBuffer=false`; it is spread verbatim into the `THREE.WebGLRenderer` constructor (`scene/scene-manager/render-pipeline/renderer-setup.ts`), so only real constructor parameters belong in it.
- `types.ts` — `WebGLConfig` plus the two nested interfaces `WebGLContextAttributes` and `WebGLRendererConfig`. `precision` is the literal union `'highp' | 'mediump' | 'lowp'`; `powerPreference` is `'high-performance' | 'low-power' | 'default'`.
- `validate.ts` — `validateWebGL(config, errors, warnings)`. Errors on `context.powerPreference` or `renderer.precision` outside their enum sets.

## Public API

- `webglConfig` — re-exported through `../../index.ts` into `AppConfig.webgl`.
- `WebGLConfig`, `WebGLContextAttributes`, `WebGLRendererConfig` — re-exported through `../../types.ts`.
- `validateWebGL` — called from `../../validation.ts`.
