/**
 * HDR capture + EXR export + ImageData snapshot helpers.
 *
 * The functions are pure over their `CaptureCtx` argument plus the
 * `runPipeline` helper — no `this` reference. The orchestrator's
 * `captureHDRPixels`, `captureHDRAsEXR`, and `renderToImageData`
 * methods bundle their state into the ctx and forward.
 *
 * @module rendering/post-processing/post-processing-manager/capture
 */

import * as THREE from 'three';
import { EXRExporter, ZIP_COMPRESSION } from 'three/examples/jsm/exporters/EXRExporter.js';
import { log, Modules } from '../../../utils/log';
import { halfFloatToFloat32, float32ToHalfFloat, readPixelsCompactAsync } from '../hdr/pixel-utils';
import { formatHDRExrLogLine } from '../hdr/capture';
import type { Renderer, RendererCapabilities } from '../../renderer-capabilities';
import type { LuxarMegaShaderMaterial } from '../../material-manager';
import { renderSceneToHdr, runPipeline, type PipelineCtx } from './pipeline';

export type CaptureMode = 'visible-ldr' | 'hdr-effects-pre-tone' | 'raw-scene-hdr';

/** Read-only references the capture paths need. */
export interface CaptureCtx {
  readonly renderer: Renderer;
  readonly capabilities: RendererCapabilities;
  readonly scene: THREE.Scene;
  readonly camera: THREE.Camera;
  readonly hdrTarget: THREE.WebGLRenderTarget;
  readonly ldrTarget: THREE.WebGLRenderTarget;
  readonly megaShader: LuxarMegaShaderMaterial;
  readonly pipelineCtx: PipelineCtx;
}

/**
 * Read raw HDR float pixel data in one of three capture modes.
 * See {@link captureHDRPixels} in the orchestrator for full mode docs.
 */
export async function captureHDRPixels(
  ctx: CaptureCtx,
  mode: CaptureMode = 'hdr-effects-pre-tone',
  opts: { flipY?: boolean } = {}
): Promise<{ pixels: Float32Array; width: number; height: number }> {
  if (mode === 'raw-scene-hdr') {
    // Scene render only — no bloom, no mega-shader. Save/restore the
    // renderer's current target + autoClear so a caller invoking
    // capture while another target is bound doesn't get clobbered.
    const prevTarget = ctx.renderer.getRenderTarget();
    const prevAutoClear = ctx.renderer.autoClear;
    try {
      // The pipeline's own stage (0), so a refract_data glass refracts the data in an
      // EXR capture exactly as it does on screen.
      renderSceneToHdr(ctx.pipelineCtx);
      const result = await readTarget(ctx, ctx.hdrTarget, opts);
      // Sanitize alpha: this mode reads the HDR target DIRECTLY — the
      // only capture path that bypasses the mega-shader's unconditional
      // alpha=1.0 write. Additive blending accumulates alpha unbounded
      // (points/lines emit real alpha over SrcAlpha+One; gsplats emit
      // 1.0 per overlapping splat), so the target's alpha channel is a
      // meaningless overdraw count that can reach HalfFloat Inf on
      // dense scenes. Exporting it (EXR alpha) hands external
      // compositors garbage — force opaque here, once, for every
      // geometry type. (This also lets the gsplat GLSL wrapper drop
      // its historical alpha-MaxEquation blend guard, whose only
      // remaining purpose was protecting this path.)
      for (let i = 3; i < result.pixels.length; i += 4) {
        result.pixels[i] = 1.0;
      }
      return result;
    } finally {
      ctx.renderer.setRenderTarget(prevTarget as THREE.WebGLRenderTarget | null);
      ctx.renderer.autoClear = prevAutoClear;
    }
  }

  if (mode === 'hdr-effects-pre-tone') {
    // Save state, disable LDR-space effects, set the RAW_HDR capture
    // define so the mega-shader bypasses EOG, tone mapping, vignette,
    // and sRGB encoding. Bloom is intentionally kept (USE_BLOOM left
    // as-is) — bloom is HDR-space and belongs in linear-HDR captures.
    const wasNoise = ctx.megaShader.isDetectorNoiseEnabled();
    const wasVignette = ctx.megaShader.isVignetteEnabled();
    const wasLens = ctx.megaShader.isLensDistortionEnabled();

    ctx.megaShader.toggleDetectorNoise(false);
    ctx.megaShader.toggleVignette(false);
    ctx.megaShader.toggleLensDistortion(false);
    ctx.megaShader.toggleRawHdrCapture(true);

    try {
      runPipeline(ctx.pipelineCtx, { applyFxaa: false, finalTarget: ctx.ldrTarget });
      return await readTarget(ctx, ctx.ldrTarget, opts);
    } finally {
      ctx.megaShader.toggleRawHdrCapture(false);
      if (wasNoise) ctx.megaShader.toggleDetectorNoise(true);
      if (wasVignette) ctx.megaShader.toggleVignette(true);
      if (wasLens) ctx.megaShader.toggleLensDistortion(true);
    }
  }

  // 'visible-ldr': full pipeline (every enabled effect, EOG, tone
  // mapping) but skip the final sRGB encoding so the captured pixels
  // are post-tone-mapping LINEAR LDR.
  ctx.megaShader.toggleLinearLdrCapture(true);
  try {
    runPipeline(ctx.pipelineCtx, { applyFxaa: false, finalTarget: ctx.ldrTarget });
    return await readTarget(ctx, ctx.ldrTarget, opts);
  } finally {
    ctx.megaShader.toggleLinearLdrCapture(false);
  }
}

async function readTarget(
  ctx: Pick<CaptureCtx, 'renderer' | 'capabilities'>,
  target: THREE.WebGLRenderTarget,
  opts: { flipY?: boolean } = {}
): Promise<{ pixels: Float32Array; width: number; height: number }> {
  const flipY = opts.flipY ?? false;
  const isHalfFloat = target.texture.type === THREE.HalfFloatType;
  if (isHalfFloat) {
    const {
      pixels: halfData,
      width,
      height,
    } = await readPixelsCompactAsync(ctx.renderer, ctx.capabilities, {
      target,
      kind: 'rgba16f',
      flipY,
    });
    return { pixels: halfFloatToFloat32(halfData), width, height };
  }
  const { pixels, width, height } = await readPixelsCompactAsync(ctx.renderer, ctx.capabilities, {
    target,
    kind: 'rgba32f',
    flipY,
  });
  return { pixels, width, height };
}

/** Encode the capture as an EXR byte stream. */
export async function captureHDRAsEXR(
  ctx: CaptureCtx,
  options?: { type?: THREE.TextureDataType; mode?: CaptureMode }
): Promise<Uint8Array> {
  const exrType: THREE.TextureDataType = options?.type ?? THREE.HalfFloatType;
  // EXR consumers (Nuke, Houdini, oiiotool) conventionally expect
  // rows in scene-space bottom-up order. The single documented
  // exception to the viewer-wide top-down contract.
  const { pixels, width, height } = await captureHDRPixels(ctx, options?.mode, { flipY: true });
  const data: Float32Array | Uint16Array =
    exrType === THREE.HalfFloatType ? float32ToHalfFloat(pixels) : pixels;
  const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat, exrType);
  texture.needsUpdate = true;
  try {
    const exporter = new EXRExporter();
    const exrData = await exporter.parse(texture, {
      type: exrType,
      compression: ZIP_COMPRESSION,
    });
    log.info(
      Modules.POST_PROCESSING,
      formatHDRExrLogLine(width, height, exrType === THREE.HalfFloatType, exrData.byteLength)
    );
    return exrData;
  } finally {
    texture.dispose();
  }
}

/** Render a full frame to an offscreen target and return as ImageData. */
export async function renderToImageData(
  ctx: CaptureCtx,
  physSize: { width: number; height: number },
  applyFxaa: boolean
): Promise<ImageData> {
  const captureTarget = new THREE.WebGLRenderTarget(physSize.width, physSize.height, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
    stencilBuffer: false,
  });
  captureTarget.texture.name = 'PostProcessing.captureTarget';
  try {
    runPipeline(ctx.pipelineCtx, { applyFxaa, finalTarget: captureTarget });
    // Read pixels in canonical top-down order. The unified primitive
    // hides the backend signature split, compacts WebGPU row padding,
    // and flips WebGL2's bottom-up rows to top-down — exactly the
    // layout ImageData wants. No further flip needed here.
    const { pixels } = await readPixelsCompactAsync(ctx.renderer, ctx.capabilities, {
      target: captureTarget,
      kind: 'rgba8',
    });
    const clamped = new Uint8ClampedArray(pixels.length);
    clamped.set(pixels);
    return new ImageData(clamped, physSize.width, physSize.height);
  } finally {
    captureTarget.dispose();
  }
}
