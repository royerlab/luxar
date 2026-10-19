# Browser support and renderer backend policy

Luxar Viewer supports a WebGL2 path and a WebGPU path, and picks the default per
platform. The
policy in this file describes how the viewer selects a backend, what browsers are
expected to run, and which invariants keep the two shader stacks in parity.

## Supported browser classes

- **WebGL2-capable browsers**: fully supported through the default
  `THREE.WebGLRenderer` path.
- **WebGPU-capable browsers**: get `WebGPURenderer` by default on a measured
  platform (see [Platform policy](#platform-policy)), and can opt in anywhere
  with `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`.
- **Browsers without WebGL2 or WebGPU**: unsupported. The viewer shows an
  informational error instead of constructing a renderer.

WebGPU is the intended renderer, but whether it beats WebGL2 depends on the
platform, so the default follows a table of measured platforms and is WebGL2
everywhere else. WebGL2 stays the broadest path and the parity baseline.

## Backend selection

`SceneManager.setupRenderer` resolves the active backend through these levels,
highest precedence first:

1. **URL parameter**: `?renderer=webgl` or `?renderer=webgpu`.
   `webgl2` is accepted as an alias for `webgl`. Add
   `?webgpuForceWebgl` alongside `?renderer=webgpu` to construct
   `WebGPURenderer({ forceWebGL: true })`; Luxar still uses TSL
   `NodeMaterial` shaders and the WebGPURenderer API surface, while Three.js
   routes draw calls through its internal WebGL2 backend.
2. **Environment variable**: `VITE_LUXAR_USE_WEBGPU=1` selects the WebGPU path
   at dev-server/build time. `VITE_LUXAR_USE_WEBGPU_RENDERER=1` is accepted as
   an alias.
3. **Platform policy** (`render-pipeline/backend-policy.ts`): WebGPU on a
   platform listed in `WEBGPU_PLATFORM_RULES`, WebGL2 everywhere else.

Only an explicit `?renderer=webgpu` (or the Settings › Advanced › Renderer
choice) may run TSL on Three.js's internal WebGL2 backend when the browser has
no WebGPU adapter. Every other WebGPU selection (the env var, or the platform
policy) requires a native device: without `navigator.gpu`, an adapter, or a
successful `requestDevice`, the session uses `THREE.WebGLRenderer` instead, and
never downloads the `three-webgpu` chunk.

`VITE_LUXAR_USE_LEGACY_WEBGL=1` forces WebGL2 over the platform policy.

## Platform policy

A row of `WEBGPU_PLATFORM_RULES` admits one platform to WebGPU by default:

| OS    | Browser  | GPU vendor | Evidence                                                                                                                                           |
| ----- | -------- | ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux | Chromium | NVIDIA     | #3146 go/no-go, RTX 3070 on Vulkan: WebGPU 2-9% faster GPU time on every heavy perf case; cold load about 0.5 s slower on a first visit (accepted) |

Platforms measured and left on WebGL2: macOS with an Apple GPU (Metal), where
WebGPU is up to ~20% slower on 10M lines and additive splats (#3224) and cold
load is ~0.4 s slower. Windows, Android, ChromeOS and Safari are unmeasured.

The match runs in two steps, so a session that cannot qualify pays nothing:

1. **OS and browser**, from the user agent, before any adapter request. No
   row, no `navigator.gpu` call and no WebGPU download.
2. **GPU vendor**, from `GPUAdapter.info.vendor`, as soon as the adapter
   answers and before `requestDevice` or the `three-webgpu` chunk. A software
   fallback adapter (`isFallbackAdapter`, e.g. SwiftShader) never matches. The
   vendor is the reliable key: user-agent strings misreport the hardware
   (Safari on Apple silicon reports "Intel Mac OS X").

The renderer log names the outcome (`WebGPU by platform policy: …` or `… is
not on a measured WebGPU platform; using WebGLRenderer.`).

Add a row only after running the render gate's `perf` and `hosted` suites with
`--backends webgl,webgpu` on that platform, and record the evidence in the row.
Note that stock Chrome on Linux (146) exposes no WebGPU adapter without flags,
so the Linux row takes effect where Chrome enables WebGPU on that hardware.

Common invocations:

```text
http://localhost:5173/                   # platform default (see above)
http://localhost:5173/?renderer=webgl    # pin WebGL2 explicitly
http://localhost:5173/?renderer=webgpu   # opt into WebGPU + TSL
http://localhost:5173/?renderer=webgpu&webgpuForceWebgl
                                         # WebGPURenderer + TSL via WebGL2 backend
```

## Material and shader invariants

`WebGPURenderer` does not accept `THREE.ShaderMaterial` or
`THREE.RawShaderMaterial`; WebGPU rendering must use Three.js node materials and
TSL. Luxar therefore keeps paired shader sources:

- WebGL2 path: GLSL3 `ShaderMaterial` wrappers.
- WebGPU path: TSL `NodeMaterial` wrappers.

`MaterialManager` dispatches on `RendererCapabilities.apiSurface`:

- `'webgl2'` → `PointMaterial`, `LineMaterial`, `GSplatMaterial`,
  `MegaShaderMaterial`, and GLSL picking materials.
- `'webgpu'` → `PointTSLMaterial`, `LineTSLMaterial`, `GSplatTSLMaterial`,
  `MegaShaderTSLMaterial`, and TSL picking materials.

The GLSL wrappers remain the parity baseline and are not removed while WebGL2 is
Luxar's default backend. `tsl-shader-parity.spec.ts` compares the generated TSL
shader output against the GLSL behavior for the production shader set.

Codebase rules:

- Do not introduce `onBeforeCompile` patches in `src/`; WebGPURenderer does not
  support them.
- Keep direct `material.uniforms.X.value = ...` writes inside the owning
  material or its shared helpers so the GLSL and TSL wrappers expose the same
  setter contract.
- Add both GLSL and TSL coverage when introducing a production shader.

## Capability semantics

`RendererCapabilities.apiSurface` reports the **renderer API surface**, not the
physical GPU backend:

- `'webgl2'`: Luxar constructed `THREE.WebGLRenderer`.
- `'webgpu'`: Luxar constructed `WebGPURenderer`, including when Three.js routes
  through its internal WebGL2 backend.

Branch on `apiSurface` to pick method signatures and resource layout rules
(readback shape, row padding, render target behavior).
`RendererCapabilities.backendKind` names the GPU API that actually runs:
`'webgl'`, `'webgpu-native'` or `'webgpu-gl'` (WebGPURenderer on the WebGL2
backend).

## Runtime detection

Which renderer Luxar actually constructed is reported by
`RendererCapabilities.apiSurface` (`'webgl2' | 'webgpu'`). Branch on it for
telemetry or UI messaging about the renderer API surface; the renderer choice
itself follows the selection rules above.
