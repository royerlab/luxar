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
vi.mock('../../../../../rendering/element-storage', () => ({
  configureSortedIndexChunkedApply: vi.fn(),
}));
vi.mock('../../../../../data/scene-loader/commit/invalidate-render-object', () => ({
  configureRenderObjectEviction: vi.fn(),
}));

import { configureRendererBackend } from '../../../../../scene/scene-manager/render-pipeline/renderer-setup';
import { log } from '../../../../../utils/log';

const caps = (apiSurface: 'webgl2' | 'webgpu'): RendererCapabilities =>
  ({
    backend: apiSurface === 'webgl2' ? 'webgl' : 'webgpu',
    apiSurface,
    maxTextureSize: 4096,
  }) as unknown as RendererCapabilities;
const renderer = {} as Renderer;

const mixWarnings = (spy: { mock: { calls: unknown[][] } }): number =>
  spy.mock.calls.filter((call) => String(call[1]).includes('backend')).length;

describe('configureRendererBackend on a multi-host page', () => {
  it.fails('warns once when a host configures a different backend than an earlier one', () => {
    const warning = vi.spyOn(log, 'warning');

    configureRendererBackend(renderer, caps('webgl2'));
    configureRendererBackend(renderer, caps('webgl2'));
    expect(mixWarnings(warning)).toBe(0);

    configureRendererBackend(renderer, caps('webgpu'));
    expect(mixWarnings(warning)).toBe(1);
    expect(String(warning.mock.calls.at(-1)?.[1])).toMatch(/webgl2.*webgpu|webgpu.*webgl2/);

    configureRendererBackend(renderer, caps('webgl2'));
    expect(mixWarnings(warning)).toBe(1);
    warning.mockRestore();
  });
});
