/**
 * Unit tests for the renderer-setup helpers extracted from
 * SceneManager in Step 9 of the scene-folder layout overhaul.
 *
 * The construction helpers (createWebGLRenderer / createWebGPURenderer)
 * touch THREE.WebGLRenderer and the dynamic three/webgpu import,
 * which is heavy to set up in jsdom; the end-to-end paths are
 * covered by the existing scene-manager.test.ts integration tests
 * plus the e2e renderer-init suite.
 *
 * This file covers the pure `selectBackend` ladder — straightforward
 * to test in isolation and the easiest thing to break by mistake.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { selectBackend } from '../../../../../scene/scene-manager/render-pipeline/renderer-setup';

describe('selectBackend', () => {
  const originalEnv = { ...import.meta.env };

  beforeEach(() => {
    // Clean env between cases.
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_WEBGPU;
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_WEBGPU_RENDERER;
    delete (import.meta.env as Record<string, string | undefined>).VITE_LUXAR_USE_LEGACY_WEBGL;
  });

  afterEach(() => {
    Object.assign(import.meta.env, originalEnv);
  });

  it('honours an explicit ?renderer=webgl URL override', () => {
    const r = selectBackend('webgl');
    expect(r).toEqual({ backend: 'webgl', source: 'url-param' });
  });

  it('honours an explicit ?renderer=webgpu URL override', () => {
    const r = selectBackend('webgpu');
    expect(r).toEqual({ backend: 'webgpu', source: 'url-param' });
  });

  it('respects VITE_LUXAR_USE_WEBGPU=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_WEBGPU_RENDERER=1 env var (legacy alias)', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU_RENDERER', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgpu', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('respects VITE_LUXAR_USE_LEGACY_WEBGL=1 env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_LEGACY_WEBGL', '1');
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'env-var' });
    vi.unstubAllEnvs();
  });

  it('URL override beats env var', () => {
    vi.stubEnv('VITE_LUXAR_USE_WEBGPU', '1');
    const r = selectBackend('webgl');
    expect(r).toEqual({ backend: 'webgl', source: 'url-param' });
    vi.unstubAllEnvs();
  });

  it('defaults to webgl when neither URL nor env vars are set', () => {
    const r = selectBackend(undefined);
    expect(r).toEqual({ backend: 'webgl', source: 'default' });
  });
});
