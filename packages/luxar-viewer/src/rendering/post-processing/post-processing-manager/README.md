# Post-processing manager helpers

Focused modules split out from the `PostProcessingManager` orchestrator
class (one level up at `../post-processing-manager.ts`). The
orchestrator owns the state fields and event dispatch; these helpers own
the work — resource construction and sizing, user-facing setter logic,
the per-frame pipeline run, and the HDR/EXR/ImageData capture paths.

Every helper here is pure over its argument bundle (no `this`
reference). The orchestrator passes in snapshots / ctx objects and
assigns returned values back to its private fields. That split lets
each module be unit-tested with a synthesised renderer and keeps the
orchestrator file small.

## Module map

| File                    | Role                                                                                                                                                                                                                                      |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resource-lifecycle.ts` | Sizing math (`computeEffectiveSize`, `getPhysicalSize`) + transient-resource construction and disposal (`buildTransientResources`, `buildBloomChain`, `disposeTransientResources`) + detector-noise DPR scaling                           |
| `settings.ts`           | User-toggle setter logic — bloom strength/radius/threshold/levels, MSAA validation, vignette, chromatic lens distortion (full + partial update + picking-system getter)                                                                   |
| `pipeline.ts`           | `runPipeline(ctx, opts)` — scene → HDR target (`renderSceneToHdr`, one pass or the refraction split's two) → optional bloom pyramid → mega-shader → optional FXAA → final target (or canvas). Save/restore of renderer target + autoClear |
| `capture.ts`            | `captureHDRPixels`, `captureHDRAsEXR`, `renderToImageData` — the three capture modes plus EXR encode and ImageData snapshot                                                                                                               |
| `refraction-split.ts`   | `DataRefractionSplit` — the WebGL two-pass scene render that lets `refract_data` glass refract the emissive data (spec MESH_PHYSICAL_MATERIALS §3.4 Phase 3); owns the pass-A copy target and the screen quad. WebGL only                 |

## How the orchestrator composes them

```
PostProcessingManager (class)
   │
   ├── construct/resize ──► resource-lifecycle.buildTransientResources
   │                         resource-lifecycle.getPhysicalSize
   │                         resource-lifecycle.disposeTransientResources
   │
   ├── setters ──────────► settings.updateBloomSettings
   │                         settings.clampBloomLevels
   │                         settings.validateMSAASamples
   │                         settings.setVignetteEnabled
   │                         settings.setChromaticLensDistortionEnabled
   │                         settings.updateChromaticLensDistortion
   │                         settings.getLensDistortionParams
   │
   ├── render() ─────────► pipeline.runPipeline
   │
   └── capture paths ────► capture.captureHDRPixels
                             capture.captureHDRAsEXR
                             capture.renderToImageData
                              (capture re-enters pipeline.runPipeline
                               with applyFxaa: false and an explicit
                               capture target)
```

## Key contracts

- **Pure-over-ctx.** No helper reads `this` or mutates orchestrator
  fields directly. The orchestrator owns state; helpers return values
  or write through supplied references on the ctx bundle.
- **Physical-pixel sizing.** `getPhysicalSize` returns
  `effectiveSize × renderer.getPixelRatio()`. Render targets and the
  mega-shader / bloom / FXAA passes are sized in physical pixels so they
  match `renderer.getDrawingBufferSize()`. A mismatch at DPR > 1
  silently brightens the scene.
- **Save/restore in pipeline.** `runPipeline` and the `raw-scene-hdr`
  branch of `captureHDRPixels` snapshot `renderer.getRenderTarget()` and
  `autoClear` in a `try/finally` so a caller invoking pipeline / capture
  while another target is bound doesn't get clobbered.
- **Capture mode contract** (see `capture.ts::CaptureMode`):
  - `raw-scene-hdr` — scene-only render, no bloom, no mega-shader.
  - `hdr-effects-pre-tone` — bloom kept (HDR-space), EOG / tone / vignette /
    detector noise / lens distortion bypassed via `LUXAR_CAPTURE_RAW_HDR`.
  - `visible-ldr` — full pipeline, post-tone-mapping, sRGB encoding
    skipped via `LUXAR_CAPTURE_LINEAR_LDR`.
- **EXR Y-orientation.** `captureHDRAsEXR` is the documented exception
  to the viewer-wide top-down read convention — EXR consumers (Nuke,
  Houdini, oiiotool) expect bottom-up rows, so the capture forces
  `flipY: true`.
- **Bloom chain rebuild on level change.** `clampBloomLevels` passes
  the current physical size to `chain.setLevels(...)`. Without that,
  `setLevels` would derive the size from a stale `mip[0]` if the canvas
  resized between the last `setSize` and the level change.
- **Context-restore allocation skip.** `buildTransientResources`
  accepts `allocateBloomFromDefaults`; the orchestrator passes `false`
  on the context-restore path so the bloom chain isn't allocated from
  config defaults before the user-toggle snapshot is restored.

## See also

- `../README.md` — full pipeline overview, mega-shader stage list, the
  capture-mode `#define` toggles, the context-restore protocol, and
  the unsupported-effects list.
- `../post-processing-manager.ts` — the orchestrator class these
  helpers serve.
- `../bloom/chain.ts`, `../fxaa/pass.ts`, `../fullscreen/pass.ts` —
  the GPU passes the helpers construct and run.

## The refraction split (WebGL, `refract_data` glass)

Three's WebGL transmission pass draws only the OPAQUE render list into the texture a
glass samples, and every Luxar data material is transparent, so a `refract_data` glass
cannot see the data in one pass. `renderSceneToHdr` therefore asks the
`DataRefractionSplit` first: when some visible glass asks to refract (the depth-sort
coordinator's `collectRefractingGlass`, injected), it renders everything but the glass
(pass A), blits the colour into its copy target, and renders only the glass plus a
screen-space quad textured with that copy into the same HDR target without clearing
(pass B). The quad is opaque, so the transmission pass sees the data; pass A's depth
survives, so opaque meshes still occlude the glass. Under MSAA the copy is load-bearing
(three invalidates the multisampled colour after every render), which is why the quad's
`depthTest/depthWrite/transparent/frustumCulled` flags are pinned by test. Glass layer
masks, the camera mask, `autoClear`, `transmissionResolutionScale` (config
`renderingControls.refraction`, pass B only) and the quad's scene membership are all
restored in a `finally`. The raw-HDR capture path shares `renderSceneToHdr`, so an EXR
refracts the data too. WebGPU has no split: there the glass samples the live
framebuffer and ranking it last in its band is the whole mechanism.
