// @vitest-environment jsdom
/**
 * Unit tests for `createRendererCapabilities`.
 *
 * The wrapper is the single seam between Luxar and the raw graphics
 * API. These tests pin its contract: probe values that exist on the GL
 * context flow through verbatim; probe values that come back as junk
 * fall back to sensible defaults so the rest of the renderer can keep
 * running.
 */
import * as THREE from 'three';
import { describe, it, expect, vi } from 'vitest';

import {
  createRendererCapabilities,
  detectFramebufferYDown,
  type Renderer,
} from '../../../rendering/renderer-capabilities';
import { log, Modules } from '../../../utils/log';

// MAX_SAMPLES and ALIASED_POINT_SIZE_RANGE constants
const MAX_SAMPLES = 0x8d57;
const MAX_TEXTURE_SIZE = 0x0d33;
const MAX_RENDERBUFFER_SIZE = 0x84e8;
const ALIASED_POINT_SIZE_RANGE = 0x846d;
const RED_BITS = 0x0d52;
const GREEN_BITS = 0x0d53;
const BLUE_BITS = 0x0d54;

type Probes = {
  maxSamples?: number | null;
  maxTextureSize?: number | null;
  maxRenderbufferSize?: number | null;
  pointSizeRange?: ArrayLike<number> | number;
  extensions?: string[];
  colorBits?: { red: number; green: number; blue: number };
  drawingBufferWidth?: number;
  drawingBufferHeight?: number;
};

function fakeRenderer(probes: Probes = {}): THREE.WebGLRenderer {
  const maxSamples = 'maxSamples' in probes ? probes.maxSamples : 8;
  const maxTextureSize = 'maxTextureSize' in probes ? probes.maxTextureSize : 16384;
  const maxRenderbufferSize = 'maxRenderbufferSize' in probes ? probes.maxRenderbufferSize : 8192;
  const {
    pointSizeRange = new Float32Array([1, 1024]),
    extensions = [],
    colorBits = { red: 8, green: 8, blue: 8 },
    drawingBufferWidth = 4,
    drawingBufferHeight = 2,
  } = probes;

  const readPixels = vi.fn();
  const fakeGL = {
    MAX_SAMPLES,
    MAX_TEXTURE_SIZE,
    MAX_RENDERBUFFER_SIZE,
    ALIASED_POINT_SIZE_RANGE,
    RED_BITS,
    GREEN_BITS,
    BLUE_BITS,
    RGBA: 0x1908,
    UNSIGNED_BYTE: 0x1401,
    getExtension: (name: string) => (extensions.includes(name) ? {} : null),
    getParameter: (param: number) => {
      if (param === MAX_SAMPLES) return maxSamples;
      if (param === MAX_TEXTURE_SIZE) return maxTextureSize;
      if (param === MAX_RENDERBUFFER_SIZE) return maxRenderbufferSize;
      if (param === ALIASED_POINT_SIZE_RANGE) return pointSizeRange;
      if (param === RED_BITS) return colorBits.red;
      if (param === GREEN_BITS) return colorBits.green;
      if (param === BLUE_BITS) return colorBits.blue;
      return 0;
    },
    drawingBufferWidth,
    drawingBufferHeight,
    readPixels,
  };

  return {
    // `isWebGLRenderer` is the canonical discriminator
    // `createRendererCapabilities` reads to pick the WebGL2 vs
    // WebGPU branch. The real `THREE.WebGLRenderer` constructor
    // sets this on `this`; mocks must opt-in explicitly.
    isWebGLRenderer: true,
    getContext: () => fakeGL,
    setRenderTarget: vi.fn(),
  } as unknown as THREE.WebGLRenderer;
}

describe('createRendererCapabilities', () => {
  it('captures api discriminator as webgl2', () => {
    const caps = createRendererCapabilities(fakeRenderer());
    expect(caps.apiSurface).toBe('webgl2');
  });

  it('reports framebufferYDown=false under WebGL2 (FBO row 0 = bottom)', () => {
    const caps = createRendererCapabilities(fakeRenderer());
    expect(caps.framebufferYDown).toBe(false);
  });

  it('passes through an explicit framebufferYDown override (test-only seam)', () => {
    const caps = createRendererCapabilities(fakeRenderer(), true);
    expect(caps.framebufferYDown).toBe(true);
  });
});

describe('detectFramebufferYDown', () => {
  it('returns false for WebGLRenderer (bottom-up FBO)', () => {
    expect(detectFramebufferYDown(fakeRenderer())).toBe(false);
  });

  it('returns true for WebGPURenderer on the real-WebGPU backend', () => {
    const fake = {
      isWebGPURenderer: true,
      backend: { isWebGPUBackend: true },
    } as unknown as Renderer;
    expect(detectFramebufferYDown(fake)).toBe(true);
  });

  it('returns true for WebGPURenderer running on the WebGL2 compat backend', () => {
    // Three.js's WebGPURenderer normalises Y internally so its
    // forceWebGL / compat-fallback output matches real WebGPU.
    // The discriminator is the renderer class, not the backend flag.
    const fake = {
      isWebGPURenderer: true,
      backend: { isWebGLBackend: true },
    } as unknown as Renderer;
    expect(detectFramebufferYDown(fake)).toBe(true);
  });

  it('returns true for WebGPURenderer when backend introspection is unavailable', () => {
    const fake = {
      isWebGPURenderer: true,
      backend: {},
    } as unknown as Renderer;
    expect(detectFramebufferYDown(fake)).toBe(true);
  });
});

describe('createRendererCapabilities (GL probes)', () => {
  it('forwards MAX_SAMPLES from the GL context', () => {
    const caps = createRendererCapabilities(fakeRenderer({ maxSamples: 16 }));
    expect(caps.maxMSAASamples).toBe(16);
  });

  it('defaults maxMSAASamples to 0 when the GPU returns a non-number', () => {
    const caps = createRendererCapabilities(fakeRenderer({ maxSamples: null }));
    expect(caps.maxMSAASamples).toBe(0);
  });

  it('forwards texture and renderbuffer dimension limits independently', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ maxTextureSize: 16384, maxRenderbufferSize: 8192 })
    );
    expect(caps.maxTextureSize).toBe(16384);
    expect(caps.maxRenderbufferSize).toBe(8192);
  });

  it('falls back to the WebGL2 floor when framebuffer limit probes are invalid', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ maxTextureSize: null, maxRenderbufferSize: 0 })
    );
    expect(caps.maxTextureSize).toBe(2048);
    expect(caps.maxRenderbufferSize).toBe(2048);
  });

  it('forwards aliased point size range as [min, max]', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ pointSizeRange: new Float32Array([2, 256]) })
    );
    expect(caps.pointSizeRange).toEqual([2, 256]);
  });

  it('falls back to [1, 1024] when point-size probe is malformed', () => {
    const caps = createRendererCapabilities(fakeRenderer({ pointSizeRange: 1024 }));
    expect(caps.pointSizeRange).toEqual([1, 1024]);
  });

  it('exposes hdr.floatTextures from GL extensions', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ extensions: ['EXT_color_buffer_float'] })
    );
    expect(caps.hdr.floatTextures).toBe(true);
  });

  it('readBackbufferPixels binds the canvas, then reads the backbuffer', async () => {
    const renderer = fakeRenderer({ drawingBufferWidth: 3, drawingBufferHeight: 2 });
    const caps = createRendererCapabilities(renderer);
    const result = await caps.readBackbufferPixels();

    expect(renderer.setRenderTarget).toHaveBeenCalledWith(null);
    expect(result.width).toBe(3);
    expect(result.height).toBe(2);
    expect(result.pixels).toBeInstanceOf(Uint8Array);
    expect(result.pixels.length).toBe(3 * 2 * 4);

    const ctx = renderer.getContext() as unknown as { readPixels: ReturnType<typeof vi.fn> };
    expect(ctx.readPixels).toHaveBeenCalled();
  });

  // -------------------------------------------------------------------
  // HDR renderer-side probes — raw-GL probing flows through
  // `createRendererCapabilities` and fills out `hdr.*` from the
  // GL context.
  // -------------------------------------------------------------------

  it('reads RED/GREEN/BLUE bits from the GL context into hdr.colorDepth', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ colorBits: { red: 10, green: 10, blue: 10 } })
    );
    expect(caps.hdr.colorDepth).toEqual({ red: 10, green: 10, blue: 10 });
  });

  it('reports hdr.floatTextures=true when EXT_color_buffer_float is present', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ extensions: ['EXT_color_buffer_float'] })
    );
    expect(caps.hdr.floatTextures).toBe(true);
  });

  it('reports hdr.floatTextures=true when only the half-float extension is present', () => {
    const caps = createRendererCapabilities(
      fakeRenderer({ extensions: ['EXT_color_buffer_half_float'] })
    );
    expect(caps.hdr.floatTextures).toBe(true);
  });

  it('reports hdr.floatTextures=false when no float-buffer extension is present', () => {
    const caps = createRendererCapabilities(fakeRenderer({ extensions: [] }));
    expect(caps.hdr.floatTextures).toBe(false);
  });
});

describe('createRendererCapabilities (WebGPU limits)', () => {
  it('uses the WebGPU texture dimension for render attachments', () => {
    const renderer = {
      isWebGPURenderer: true,
      backend: { device: { limits: { maxTextureDimension2D: 12288 } } },
    } as unknown as Renderer;

    const caps = createRendererCapabilities(renderer, true);

    expect(caps.maxTextureSize).toBe(12288);
    expect(caps.maxRenderbufferSize).toBe(12288);
  });

  it('keeps the smaller renderbuffer limit on the WebGL compatibility backend', () => {
    const gl = {
      MAX_TEXTURE_SIZE,
      MAX_RENDERBUFFER_SIZE,
      getParameter: (parameter: number) =>
        parameter === MAX_TEXTURE_SIZE ? 16384 : parameter === MAX_RENDERBUFFER_SIZE ? 8192 : 0,
    };
    const renderer = {
      isWebGPURenderer: true,
      backend: { gl },
    } as unknown as Renderer;

    const caps = createRendererCapabilities(renderer, true);

    expect(caps.maxTextureSize).toBe(16384);
    expect(caps.maxRenderbufferSize).toBe(8192);
  });

  it('uses the texture limit when the compatibility renderbuffer probe fails', () => {
    const warning = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const gl = {
      MAX_TEXTURE_SIZE,
      MAX_RENDERBUFFER_SIZE,
      getParameter: (parameter: number) =>
        parameter === MAX_TEXTURE_SIZE ? 16384 : parameter === MAX_RENDERBUFFER_SIZE ? null : 0,
    };
    const renderer = {
      isWebGPURenderer: true,
      backend: { gl },
    } as unknown as Renderer;

    const caps = createRendererCapabilities(renderer, true);

    expect(caps.maxTextureSize).toBe(16384);
    expect(caps.maxRenderbufferSize).toBe(16384);
    expect(warning).toHaveBeenCalledWith(
      Modules.RENDERER,
      'MAX_RENDERBUFFER_SIZE probe failed; using MAX_TEXTURE_SIZE (16384)'
    );
    warning.mockRestore();
  });
});
