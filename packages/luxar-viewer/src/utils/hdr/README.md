# HDR display detection + color conversion

Display-side HDR capability probes, the per-display headroom resolver for
true HDR canvas output (PROTOTYPE, WebGPU only), and the linear-sRGB →
BT.2020 PQ I420P10 pipeline used for 10-bit WebCodecs video capture.
Renderer-side probes (float-texture extension, color-buffer bit depth,
the unmasked GPU string) deliberately live in
`rendering/renderer-capabilities.ts` — the single seam where raw GL is
allowed — so nothing here touches GL. Most of the folder is pure helpers
over an `HDRCapabilities` snapshot; the exception is
`hdr-headroom.ts`, whose `HdrHeadroomController` is stateful (Window
Management permission API, `localStorage`, display-change listeners).

| File                      | Role                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hdr-detection.ts`        | `HDRCapabilities` interface + `detectDisplayCapabilities()` (P3 / Rec2020 / dynamic-range / 10-bit `matchMedia` probes), `configureHDRRenderer(renderer, caps)` (logs only — never touches `outputColorSpace` or `toneMapping`; `PostProcessingManager` owns both), `HDR_DISPLAY_QUERY` (the `(dynamic-range: high)` media query, shared with the headroom controller), `logHDRCapabilities(caps)`, `isHDRDisplay(caps)`, `getOptimalRenderTargetType(caps)` (returns `THREE.HalfFloatType` when float textures are available, else `THREE.UnsignedByteType`).                                                                                                                                              |
| `hdr-color-conversion.ts` | `rgbaFloatToI420P10(rgba, width, height)` — converts linear sRGB float RGBA (from WebGL `readPixels`) to BT.2020 PQ YCbCr I420P10 `Uint16Array`. Pipeline: sRGB→BT.2020 linear (3×3 matrix), linear→PQ (SMPTE ST 2084, scene-referred 1.0 = SDR white ≈ 100 nits), RGB→YCbCr (BT.2020 NCL), 10-bit limited-range quantization (Y: 64–940, Cb/Cr: 64–960, centered at 512), 4:2:0 chroma subsampling, planar `Uint16Array` layout (Y plane, U plane, V plane).                                                                                                                                                                                                                                               |
| `display-identity.ts`     | Identifies the display the window is on, for headroom purposes. `DisplaySignals` (platform via `platformName()`, WebGL unmasked GPU string, screen size + DPR, and with the Window Management permission the ScreenDetailed `label` / `isInternal`), `identifyDisplay(signals)` (model table: an Apple-silicon Pro/Max/Ultra chip on a MacBook-shaped screen, or a 14" MacBook Pro scaled size, is Liquid Retina XDR → 3x; seed table of external monitors by EDID label), `measuredHeadroom(screen)` (reads the ScreenDetailed `hdrHeadroom` in stops, else the legacy linear `highDynamicRangeHeadroom`), `collectDisplaySignals`, `displayKey(signals)` (`label:<EDID name>` or `screen:<w>x<h>@<dpr>`). |
| `hdr-headroom.ts`         | Headroom (how far above SDR white the display can go) resolved per display: URL pin > hand-set per display (`localStorage` `luxar.hdrHeadroom`, a JSON map keyed by `displayKey`) > measured > model guess > default 2x, or 1x `'sdr'` while the window is on a display that stops reporting HDR. `resolveHdrHeadroom(gpuRenderer)` is the synchronous best effort used at renderer construction; `HdrHeadroomController` (`start`, `detect`, `setManual`, `current`, `canMeasure`, `dispose`) follows display and brightness changes live and reports each `HdrHeadroomResolution` (`value`, `source`, `display`); `isHeadroomLocked(source)` is true for `url` / `sdr`.                                   |

`floatTextures` and `colorDepth` on the snapshot from
`detectDisplayCapabilities()` are placeholder defaults
(`false`, `{red: 8, green: 8, blue: 8}`); `createRendererCapabilities`
in `rendering/renderer-capabilities.ts` overwrites them with the real
GL-probed values before any consumer reads the snapshot.

The deep-color probe is intentionally belt-and-braces: per CSS Media
Queries L4 a true 10-bit display should match `(color: 10)`, but some
shipping browsers report the _total_ bit depth instead
(~30 for 10-bpc RGB, ~48 for 16-bpc), so both `(color: 30)` and
`(color: 48)` are checked.

`recommendedColorSpace` is derived from the gamut/HDR probes:
`rec2020` when both `rec2020Gamut` and `hdr` are true, else
`display-p3` when `p3Gamut` is true, else `srgb`. `isHDRDisplay()`
returns true only when `hdr && deepColor && floatTextures` AND a
wide gamut (P3 or Rec2020) is present — i.e. all four pillars line up.

## Usage

```typescript
import {
  detectDisplayCapabilities,
  isHDRDisplay,
  getOptimalRenderTargetType,
} from '../utils/hdr/hdr-detection';
import { rgbaFloatToI420P10 } from '../utils/hdr/hdr-color-conversion';

const caps = detectDisplayCapabilities();
// renderer-capabilities.ts then overwrites caps.floatTextures + caps.colorDepth
const rtType = getOptimalRenderTargetType(caps); // HalfFloatType | UnsignedByteType

// HDR video capture path: linear-sRGB float RGBA → I420P10 for WebCodecs
const i420p10 = rgbaFloatToI420P10(floatRGBA, width, height);
```

`hdr-headroom.ts` imports `cross-layer/event-group`, `log`,
`storage-keys`, `hdr-detection`, `display-identity` and
`types/hdr-output`; `display-identity.ts` imports `platform`. Consumers
of the headroom resolver: `renderer-setup.ts` (`resolveHdrHeadroom`) and
`ui/rendering-controls/setup/hdr-setup.ts` (`HdrHeadroomController`).

Consumers: `hdr-detection.ts` is consumed by
`rendering/renderer-capabilities.ts` (which calls
`detectDisplayCapabilities()` and then overwrites the float-texture /
color-depth fields with real GL probes) and by
`scene/scene-manager/render-pipeline/renderer-setup.ts` (which calls
`getOptimalRenderTargetType()` to pick the render-target type).
`hdr-color-conversion.ts` provides the linear-sRGB → I420P10 helper for
the 10-bit WebCodecs HDR video-capture path; it is currently a
standalone, unwired helper — no module imports `rgbaFloatToI420P10`
today (the live HDR capture path in
`rendering/post-processing/post-processing-manager/capture.ts` produces
EXR / ImageData, not WebCodecs I420P10 frames).

See also the adjacent `rendering/post-processing/hdr/` package for the
backend-aware pixel readback (`readPixelsCompactAsync` in
`pixel-utils.ts`) that would feed `rgbaFloatToI420P10` once the
WebCodecs capture path is wired up.
