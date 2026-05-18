/**
 * Smoke tests for the GSplats handler — mirror of points/handler.test.ts
 * and lines/handler.test.ts (three-geometry-symmetry rule). Step 7 of
 * the god-object refactor.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/gsplats/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state-queue';
import type { GSplatsDataLoader } from '../../../../types/gsplats';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

describe('gsplats handler', () => {
  it('discriminates as kind="gsplats" with label="GSplats"', () => {
    expect(kind).toBe('gsplats');
    expect(label).toBe('GSplats');
  });

  it('returns null on derived.skip without calling the loader', async () => {
    const loader: GSplatsDataLoader = {
      loadGSplats: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as GSplatsDataLoader;
    const result = await loadAndStage('/g', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      extendedToleranceCache: new Map(),
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(result).toBeNull();
    expect(loader.updateView).not.toHaveBeenCalled();
  });
});
