/**
 * The native-WebGPU Playwright project's URL rules (#3144).
 *
 * `withWebGPURenderer` decides which `page.goto` URLs the `chromium-webgpu`
 * project rewrites onto `renderer=webgpu`, and `requiresNativeWebGPU` decides
 * which loaded pages `waitForLuxarReady` then holds to a native backend. Both
 * are pure, so they are pinned here rather than only through a GPU run. The
 * hazards are the two directions of a wrong answer: rewriting a spec that
 * pinned WebGL (or a harness page) silently changes what it tests, and not
 * rewriting a viewer URL silently runs WebGL inside the WebGPU project.
 */

import { describe, expect, it } from 'vitest';

import { requiresNativeWebGPU, withWebGPURenderer } from '../../e2e/webgpu-project';

const BASE = 'http://127.0.0.1:5173';

describe('withWebGPURenderer', () => {
  it.each([
    [
      '/?src=http://localhost:9000/a.zarr&debug',
      '/?src=http://localhost:9000/a.zarr&debug&renderer=webgpu',
    ],
    ['/', '/?renderer=webgpu'],
    ['/?', '/?renderer=webgpu'],
    ['/?debug&', '/?debug&renderer=webgpu'],
    ['/index.html?debug', '/index.html?debug&renderer=webgpu'],
    ['/?debug#camera=1', '/?debug&renderer=webgpu#camera=1'],
    ['http://localhost:5173/?debug', 'http://localhost:5173/?debug&renderer=webgpu'],
  ])('appends renderer=webgpu to the viewer URL %s', (url, expected) => {
    expect(withWebGPURenderer(url, BASE)).toBe(expected);
  });

  it.each([
    '/?renderer=webgl&debug',
    '/?debug&renderer=webgpu',
    '/?renderer&debug',
    '/tsl-harness.html',
    '/examples/embed.html?src=x',
    'https://example.org/?debug',
    'about:blank',
  ])('leaves %s untouched', (url) => {
    expect(withWebGPURenderer(url, BASE)).toBe(url);
  });

  it('keeps flag-only params verbatim instead of re-serializing them', () => {
    expect(withWebGPURenderer('/?debug&noOpfs&dpr=1', BASE)).toBe(
      '/?debug&noOpfs&dpr=1&renderer=webgpu'
    );
  });
});

describe('requiresNativeWebGPU', () => {
  it('holds a webgpu viewer page to a native backend', () => {
    expect(requiresNativeWebGPU(`${BASE}/?debug&renderer=webgpu`)).toBe(true);
  });

  it.each([
    `${BASE}/?debug&renderer=webgl`,
    `${BASE}/?debug`,
    `${BASE}/?renderer=webgpu&webgpuForceWebgl&debug`,
    `${BASE}/tsl-harness.html?renderer=webgpu`,
    'not a url',
  ])('does not require it of %s', (url) => {
    expect(requiresNativeWebGPU(url)).toBe(false);
  });
});
