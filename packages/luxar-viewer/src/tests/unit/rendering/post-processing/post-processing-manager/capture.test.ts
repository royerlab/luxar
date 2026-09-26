/**
 * captureHDRPixels restores its global state as soon as the readback is
 * ISSUED, not after the transfer resolves.
 *
 * The capture modes flip global mega-shader flags (raw HDR, linear LDR,
 * effects off) or rebind the render target. The GPU copy is issued
 * synchronously by every renderer path, so the state can be put back before
 * awaiting the transfer; holding it across the await let any loop frame drawn
 * meanwhile go through the capture shader and flash a blown-out frame.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';

const pipeline = vi.hoisted(() => ({
  runPipeline: vi.fn(),
  renderSceneToHdr: vi.fn(),
}));
vi.mock(
  '../../../../../rendering/post-processing/post-processing-manager/pipeline',
  () => pipeline
);

const readback = vi.hoisted(() => ({
  resolve: null as null | ((v: unknown) => void),
  stateAtIssue: null as null | Record<string, unknown>,
  snapshot: (() => ({})) as () => Record<string, unknown>,
}));
vi.mock('../../../../../rendering/post-processing/hdr/pixel-utils', async (importOriginal) => {
  const actual = await importOriginal<Record<string, unknown>>();
  return {
    ...actual,
    readPixelsCompactAsync: vi.fn(() => {
      readback.stateAtIssue = readback.snapshot();
      return new Promise((resolve) => {
        readback.resolve = resolve;
      });
    }),
  };
});

import * as THREE from 'three';
import { captureHDRPixels } from '../../../../../rendering/post-processing/post-processing-manager/capture';

function makeCtx() {
  const flags = { raw: false, linear: false, noise: true, vignette: true, lens: true };
  const megaShader = {
    isDetectorNoiseEnabled: () => flags.noise,
    isVignetteEnabled: () => flags.vignette,
    isLensDistortionEnabled: () => flags.lens,
    toggleDetectorNoise: (v: boolean) => (flags.noise = v),
    toggleVignette: (v: boolean) => (flags.vignette = v),
    toggleLensDistortion: (v: boolean) => (flags.lens = v),
    toggleRawHdrCapture: (v: boolean) => (flags.raw = v),
    toggleLinearLdrCapture: (v: boolean) => (flags.linear = v),
  };
  const screenTarget = { name: 'screen' };
  const renderer = {
    target: screenTarget as unknown,
    autoClear: true,
    getRenderTarget() {
      return this.target;
    },
    setRenderTarget(t: unknown) {
      this.target = t;
    },
  };
  const target = (w: number) => ({
    width: w,
    height: 1,
    texture: { type: THREE.FloatType },
  });
  const ctx = {
    renderer,
    capabilities: { apiSurface: 'webgl2' },
    megaShader,
    hdrTarget: target(2),
    ldrTarget: target(2),
    pipelineCtx: {},
  };
  // What renderSceneToHdr would leave behind: the HDR target bound, autoClear off.
  pipeline.renderSceneToHdr.mockImplementation(() => {
    renderer.target = ctx.hdrTarget;
    renderer.autoClear = false;
  });
  readback.snapshot = () => ({ ...flags, target: renderer.target, autoClear: renderer.autoClear });
  return { ctx: ctx as never, flags, renderer, screenTarget };
}

const pixels = { pixels: new Float32Array(8).fill(0.5), width: 2, height: 1 };

describe('captureHDRPixels', () => {
  beforeEach(() => {
    readback.resolve = null;
    readback.stateAtIssue = null;
  });

  it('hdr-effects-pre-tone: flags are restored while the transfer is still pending', async () => {
    const { ctx, flags } = makeCtx();
    const done = captureHDRPixels(ctx, 'hdr-effects-pre-tone');

    // Issued with the capture state on...
    expect(readback.stateAtIssue).toMatchObject({
      raw: true,
      noise: false,
      vignette: false,
      lens: false,
    });
    // ...and put back before the transfer resolves.
    expect(flags).toEqual({ raw: false, linear: false, noise: true, vignette: true, lens: true });

    readback.resolve!(pixels);
    await expect(done).resolves.toMatchObject({ width: 2, height: 1 });
  });

  it('visible-ldr: the linear-LDR flag is cleared while the transfer is still pending', async () => {
    const { ctx, flags } = makeCtx();
    const done = captureHDRPixels(ctx, 'visible-ldr');

    expect(readback.stateAtIssue).toMatchObject({ linear: true });
    expect(flags.linear).toBe(false);

    readback.resolve!(pixels);
    await done;
  });

  it('raw-scene-hdr: the render target and autoClear are restored while the transfer is still pending', async () => {
    const { ctx, renderer, screenTarget } = makeCtx();
    const done = captureHDRPixels(ctx, 'raw-scene-hdr');

    expect((readback.stateAtIssue as { target: unknown }).target).not.toBe(screenTarget);
    expect(renderer.target).toBe(screenTarget);
    expect(renderer.autoClear).toBe(true);

    readback.resolve!({ pixels: new Float32Array(8).fill(3), width: 2, height: 1 });
    const result = await done;
    // Alpha is still forced opaque on the resolved pixels.
    expect(result.pixels[3]).toBe(1);
    expect(result.pixels[7]).toBe(1);
  });

  it('restores the flags even when the pipeline throws', async () => {
    const { ctx, flags } = makeCtx();
    pipeline.runPipeline.mockImplementationOnce(() => {
      throw new Error('pipeline failed');
    });

    await expect(captureHDRPixels(ctx, 'hdr-effects-pre-tone')).rejects.toThrow('pipeline failed');
    expect(flags).toEqual({ raw: false, linear: false, noise: true, vignette: true, lens: true });
  });
});
