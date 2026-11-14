/**
 * The renderer-backend switches are page-wide and last-writer-wins, so every
 * Luxar host on a page must render through ONE backend. Mixing them is not
 * enforced (a host may legitimately re-init on another backend after the
 * first is gone), but it is never silent: the first mismatch warns once.
 */

import { describe, it, expect, vi } from 'vitest';
import type {
  Renderer,
  RendererCapabilities,
} from '../../../../../rendering/renderer-capabilities';

vi.mock('../../../../../rendering/element-texture-layout', () => ({
  configureElementTextureLayout: vi.fn(),
}));
vi.mock('../../../../../rendering/element-texture-row-upload', () => ({
  installElementTextureRowUploads: vi.fn(),
}));
vi.mock('../../../../../rendering/renderer-backend-policy', () => ({
  configureRendererBackendPolicy: vi.fn(),
}));

const caps = (backendKind: RendererCapabilities['backendKind']): RendererCapabilities =>
  ({
    backend: backendKind === 'webgl' ? 'webgl' : 'webgpu',
    apiSurface: backendKind === 'webgl' ? 'webgl2' : 'webgpu',
    backendKind,
    maxTextureSize: 4096,
  }) as unknown as RendererCapabilities;
const renderer = {} as Renderer;

const mixWarnings = (spy: { mock: { calls: unknown[][] } }): number =>
  spy.mock.calls.filter((call) => String(call[1]).includes('backend')).length;

describe('configureRendererBackend on a multi-host page', () => {
  it('warns once when a host configures a different backend than an earlier one', async () => {
    vi.resetModules();
    const { configureRendererBackend } =
      await import('../../../../../scene/scene-manager/render-pipeline/renderer-setup');
    const { log } = await import('../../../../../utils/log');
    const warning = vi.spyOn(log, 'warning');

    configureRendererBackend(renderer, caps('webgl'));
    configureRendererBackend(renderer, caps('webgl'));
    expect(mixWarnings(warning)).toBe(0);

    configureRendererBackend(renderer, caps('webgpu-native'));
    expect(mixWarnings(warning)).toBe(1);
    expect(String(warning.mock.calls.at(-1)?.[1])).toMatch(/webgl.*webgpu-native/);

    configureRendererBackend(renderer, caps('webgl'));
    expect(mixWarnings(warning)).toBe(1);
    warning.mockRestore();
  });

  it('warns when native WebGPU switches to the WebGL bridge with the same API surface', async () => {
    vi.resetModules();
    const { configureRendererBackend } =
      await import('../../../../../scene/scene-manager/render-pipeline/renderer-setup');
    const { log } = await import('../../../../../utils/log');
    const warning = vi.spyOn(log, 'warning');

    configureRendererBackend(renderer, caps('webgpu-native'));
    configureRendererBackend(renderer, caps('webgpu-gl'));

    expect(mixWarnings(warning)).toBe(1);
    expect(String(warning.mock.calls.at(-1)?.[1])).toMatch(/webgpu-native.*webgpu-gl/);
    warning.mockRestore();
  });
});
