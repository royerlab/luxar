// @vitest-environment jsdom
/**
 * `createWebGLRenderer` turns three's synchronous shader-error check off
 * outside debug mode.
 *
 * `checkShaderErrors` makes three call `getProgramInfoLog` (and the shader
 * logs) right after every program link — a synchronous GPU round trip that
 * stalled the first frame by 15-91 ms in real Chrome. With it off the driver
 * links in parallel and a broken shader still surfaces through the browser's
 * own WebGL warning; with `?debug` three's formatted error report stays on.
 *
 * `THREE.WebGLRenderer` is replaced by an inert stand-in carrying three's
 * `debug` object, so this pins the flag rather than the GL contract.
 */

import { describe, it, expect, vi } from 'vitest';

vi.mock('three', async (importOriginal) => {
  const actual = await importOriginal<typeof import('three')>();
  class FakeWebGLRenderer {
    // three's default.
    debug = { checkShaderErrors: true, onShaderError: null };
    constructor(_opts: unknown) {}
  }
  return { ...actual, WebGLRenderer: FakeWebGLRenderer };
});
vi.mock('../../../../../rendering/renderer-capabilities', () => ({
  createRendererCapabilities: vi.fn(() => ({
    apiSurface: 'webgl2',
    hdr: {},
    pointSizeRange: [1, 64],
    maxTextureSize: 4096,
  })),
}));
vi.mock('../../../../../rendering/upload-counters', () => ({ installUploadCounters: vi.fn() }));
vi.mock('../../../../../utils/hdr/hdr-detection', () => ({
  configureHDRRenderer: vi.fn(),
  logHDRCapabilities: vi.fn(),
}));

import { createWebGLRenderer } from '../../../../../scene/scene-manager/render-pipeline/renderer-setup';

function canvas(): HTMLCanvasElement {
  const el = document.createElement('canvas');
  // jsdom has no WebGL; the helper tolerates a null context.
  el.getContext = vi.fn(() => null) as unknown as HTMLCanvasElement['getContext'];
  return el;
}

type WithDebug = { debug: { checkShaderErrors: boolean } };

describe('createWebGLRenderer shader-error checking', () => {
  it('is off without debug', async () => {
    const { renderer } = await createWebGLRenderer(canvas());
    expect((renderer as unknown as WithDebug).debug.checkShaderErrors).toBe(false);
  });

  it('is off when debug is explicitly false', async () => {
    const { renderer } = await createWebGLRenderer(canvas(), { debug: false });
    expect((renderer as unknown as WithDebug).debug.checkShaderErrors).toBe(false);
  });

  it('stays on with debug', async () => {
    const { renderer } = await createWebGLRenderer(canvas(), { debug: true });
    expect((renderer as unknown as WithDebug).debug.checkShaderErrors).toBe(true);
  });
});
