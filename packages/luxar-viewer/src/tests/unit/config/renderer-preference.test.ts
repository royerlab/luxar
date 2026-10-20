// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { readUrlParams } from '../../../config/url-params';
import { loadUserSettings } from '../../../config/user-settings';
import {
  earlyRendererOverride,
  resolveRendererOverride,
  SETTINGS_VERSION,
  storedRendererPreference,
} from '../../../config/renderer-preference';
import { StorageKeys } from '../../../utils/storage-keys';
import { selectBackend } from '../../../scene/scene-manager/render-pipeline/backend-selection';

const storedCases = [
  { name: 'missing', raw: null },
  {
    name: 'WebGL',
    raw: JSON.stringify({ version: SETTINGS_VERSION, advanced: { renderer: 'webgl' } }),
  },
  {
    name: 'WebGPU',
    raw: JSON.stringify({ version: SETTINGS_VERSION, advanced: { renderer: 'webgpu' } }),
  },
  {
    name: 'auto',
    raw: JSON.stringify({ version: SETTINGS_VERSION, advanced: { renderer: 'auto' } }),
  },
  {
    name: 'invalid',
    raw: JSON.stringify({ version: SETTINGS_VERSION, advanced: { renderer: 'bogus' } }),
  },
  {
    name: 'old version',
    raw: JSON.stringify({ version: SETTINGS_VERSION + 1, advanced: { renderer: 'webgpu' } }),
  },
  { name: 'corrupt JSON', raw: '{bad' },
] as const;
const searches = [
  '',
  '?renderer=webgl',
  '?renderer=WEBGL2',
  '?renderer=WebGPU',
  '?renderer=bad',
  '?renderer=',
];

afterEach(() => {
  localStorage.clear();
  vi.unstubAllEnvs();
});

describe('early renderer preference', () => {
  for (const stored of storedCases) {
    for (const search of searches) {
      it(`matches bootstrap for ${stored.name} and ${search || 'no parameter'}`, () => {
        if (stored.raw === null) localStorage.removeItem(StorageKeys.settings);
        else localStorage.setItem(StorageKeys.settings, stored.raw);
        const fromBootstrap = resolveRendererOverride(
          readUrlParams(search).renderer,
          loadUserSettings().advanced.renderer
        );
        expect(earlyRendererOverride(search)).toBe(fromBootstrap);
      });
    }
  }

  it('stored WebGL wins over a WebGPU-default build', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    localStorage.setItem(
      StorageKeys.settings,
      JSON.stringify({ version: SETTINGS_VERSION, advanced: { renderer: 'webgl' } })
    );
    expect(selectBackend(earlyRendererOverride('')).backend).toBe('webgl');
    expect(
      selectBackend(resolveRendererOverride(null, loadUserSettings().advanced.renderer)).backend
    ).toBe('webgl');
  });

  it('does not read storage when the URL already selects a renderer', () => {
    const getItem = vi.fn(() => null);
    expect(earlyRendererOverride('?renderer=webgl', { getItem })).toBe('webgl');
    expect(getItem).not.toHaveBeenCalled();
  });

  it('treats blocked storage as auto', () => {
    expect(
      storedRendererPreference({
        getItem: () => {
          throw new Error('blocked');
        },
      })
    ).toBe('auto');
  });
});
