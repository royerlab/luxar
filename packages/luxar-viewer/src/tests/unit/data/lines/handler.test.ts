/**
 * Smoke tests for the Lines handler — mirrors points/handler.test.ts
 * and gsplats/handler.test.ts so all geometry handlers share the same
 * basic contract.
 */

import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { kind, label, loadAndStage } from '../../../../data/lines/handler';
import { ViewStateQueue } from '../../../../data/scene-loader/view-state-queue';
import type { LinesDataLoader } from '../../../../types/lines';
import type { UpdateSession } from '../../../../profiling/update-profiler';

function makeSession(): UpdateSession {
  return {
    markSkipped: vi.fn(),
    setMetadata: vi.fn(),
    begin: vi.fn().mockReturnValue({ end: vi.fn() }),
    end: vi.fn(),
  } as unknown as UpdateSession;
}

describe('lines handler', () => {
  it('discriminates as kind="lines" with label="Lines"', () => {
    expect(kind).toBe('lines');
    expect(label).toBe('Lines');
  });

  it('returns null on derived.skip without calling the loader', async () => {
    const loader: LinesDataLoader = {
      loadLines: vi.fn(),
      updateView: vi.fn(),
      dispose: vi.fn(),
    } as unknown as LinesDataLoader;
    const result = await loadAndStage('/l', loader, makeSession(), {
      rootGroup: new THREE.Group(),
      viewStateQueue: new ViewStateQueue(),
      clearFailure: vi.fn(),
      currentVersion: 1,
      updateVersion: 1,
      deriveNodeViewState: () => ({ skip: 'extend_to_all' }),
    });
    expect(result).toBeNull();
    expect(loader.updateView).not.toHaveBeenCalled();
  });
});
