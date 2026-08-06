# Browser support and renderer backend policy

Luxar Viewer supports a WebGL2 production path and an opt-in WebGPU path. The
policy in this file describes how the viewer selects a backend, what browsers are
expected to run, and which invariants keep the two shader stacks in parity.

## Supported browser classes

- **WebGL2-capable browsers**: fully supported through the default
  `THREE.WebGLRenderer` path.
- **WebGPU-capable browsers**: can opt into `WebGPURenderer` with
  `?renderer=webgpu` or `VITE_LUXAR_USE_WEBGPU=1`.
- **Browsers without WebGL2 or WebGPU**: unsupported. The viewer shows an
  informational error instead of constructing a renderer.

The default remains WebGL2 because it is the broadest, most stable path and is
still the performance baseline. WebGPU is available for testing and adoption on
scenes where it matches or exceeds the WebGL2 path.

## Backend selection

`SceneManager.setupRenderer` resolves the active backend through these levels,
highest precedence first:

1. **URL parameter**: `?renderer=webgl` or `?renderer=webgpu`.
   `webgl2` is accepted as an alias for `webgl`. Add
   `?webgpu-force-webgl` alongside `?renderer=webgpu` to construct
   `WebGPURenderer({ forceWebGL: true })`; Luxar still uses TSL
   `NodeMaterial` shaders and the WebGPURenderer API surface, while Three.js
   routes draw calls through its internal WebGL2 backend.
2. **Environment variable**: `VITE_LUXAR_USE_WEBGPU=1` selects the WebGPU path
   at dev-server/build time. `VITE_LUXAR_USE_WEBGPU_RENDERER=1` is accepted as
   an alias.
3. **Default**: WebGL2.

`VITE_LUXAR_USE_LEGACY_WEBGL=1` is harmless because WebGL2 is already the
default; it is accepted so older local scripts do not fail.

Common invocations:

```text
http://localhost:5173/                   # default WebGL2
http://localhost:5173/?renderer=webgl    # pin WebGL2 explicitly
http://localhost:5173/?renderer=webgpu   # opt into WebGPU + TSL
http://localhost:5173/?renderer=webgpu&webgpu-force-webgl
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
(readback shape, row padding, render target behavior). It is also the value to
use for telemetry or UI messaging about which renderer Luxar constructed.

## Runtime detection

Which renderer Luxar actually constructed is reported by
`RendererCapabilities.apiSurface` (`'webgl2' | 'webgpu'`). Branch on it for
telemetry or UI messaging about the renderer API surface; the renderer choice
itself follows the selection rules above.
