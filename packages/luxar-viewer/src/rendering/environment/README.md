# `rendering/environment` — the scene environment

The one lighting input Luxar's otherwise light-free scene has: `scene.environment`,
read only by three's lighting-model materials — in Luxar, only the `material="physical"`
mesh wrappers (`../materials/mesh-physical/`). Every house material is a
`ShaderMaterial` / `NodeMaterial` with its own fragment code that never samples an
environment, so nothing here changes how points, lines, splats or house meshes render.
Design: `docs/guides/specs/MESH_PHYSICAL_MATERIALS_SPEC.md` §3.3.

## Precedence

`SceneEnvironment` (`scene-environment.ts`) owns `scene.environment` and applies one
rule, lazily — nothing is built until the material manager's physical-material hook calls
`ensure()`, so a scene without a physical mesh keeps `scene.environment === null`:

1. **A baked map** attached to the store (`luxar env bake` / `env attach`) whose
   `scene_content_hash` matches the scene. `baked.ts` builds a half-float `CubeTexture`
   from the six stored faces; three prefilters (PMREM) it on assignment. Zero live cost.
2. **The authored source**, `viewer_config.environment.source`:
   - `scene` — an EXACT cube capture of the scene itself from the probe
     (`cube-capture.ts`): a `CubeCamera` renders six faces into a half-float cube target
     with the real shaders, so metals and glass reflect the data they sit in. Physical
     meshes are hidden for the six draws (no self-reflection; their previous `visible`
     is restored), the cube camera's params (90° fov, square buffer, pixel ratio 1) are
     pushed to the material manager so point and line footprints are right and the main
     camera's restored after, and `scene.environment` is set to the target's texture —
     three re-prefilters in place whenever `CubeCamera.update` bumps `pmremVersion`.
     Re-captured on a geometry commit, a slice change or an appearance change, once the
     loader has settled and after a 250 ms debounce (`markStale` / `tick`, wired by
     `core/app/init/environment-wiring.ts`). Camera motion is NOT a trigger: a cube map
     from a fixed probe is view-independent.
   - `hdri` — an equirectangular image at `url` (`hdri.ts`; `.hdr` through a dynamically
     imported `HDRLoader`, anything else as LDR). The room stands in until it loads.
3. **The room** — three's procedural `RoomEnvironment`, prefiltered once.

`probe.ts` resolves where a capture looks out from: `auto` (the committed bounds centre),
`node:<path>` (that node's world bounding-box centre — a marker shell around a cluster),
or a literal `x,y,z`.

Ordering caveat: the first physical material is created DURING the scene load, before
`load-dataset.ts` has handed over the authored config and any baked map, so the first
light is whatever the default rule gives (the room, or one live capture) and
`configure()` / `setBaked()` re-apply a few milliseconds later. That costs at most one
room prefilter or one small capture per load, never a wrong steady state.

## Baking

`bake.ts` runs one capture and reads the six faces back as half floats through
`readPixelsCompactAsync` (`faceIndex` threads into the two renderers' differing
argument slots), packs the `LXENV001` container (magic, u32 header length, JSON header,
`uint16` half bits in three's `px nx py ny pz nz` order, GL row order) and hands it to
`__luxarDebug.environment.lastBake` for the Node driver
(`scripts/bake-env.mjs`) plus a download. `luxar env attach` writes it into the store as
`environment/faces-<digest>`; `data/loaders/environment/` reads it back. The environment
group is excluded from the scene `content_hash` on the Python side, which is what makes
the `scene_content_hash` guard exact and lets a bake leave warm caches alone.

## Backends

The PMREM generator and the cube render target are backend-specific classes
(`three`'s `PMREMGenerator` / `WebGLCubeRenderTarget` vs `three/webgpu`'s
`PMREMGenerator` / `CubeRenderTarget`), so `createSceneEnvironment` injects factories and
the WebGPU ones come through `rendering/tsl/registry.ts`, keeping the lazy chunk lazy.
`CubeCamera` itself is backend-agnostic.

## Testing

`tests/unit/rendering/environment/`: the lazy contract and house-material invisibility,
precedence and fallbacks, the capture orchestration against a stub renderer (six faces,
params push/restore, physical meshes hidden and restored exactly), the stale/debounce/
settled gating, probe parsing and resolution, the container layout and face-index
threading. `tests/unit/core/app/init/environment-wiring.test.ts` pins the triggers and the
`?bake-env` one-shot; `tests/unit/data/loaders/environment/` the store contract.
