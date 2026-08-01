/**
 * Unit tests for LoaderRegistry.
 *
 * The registry is a thin wrapper around a kind-keyed loader store and a
 * failure log; we test it with stand-in loader objects that implement only the
 * dispose() method the registry calls. No zarr / DOM / WebGL involved.
 */

import { describe, it, expect, vi } from 'vitest';
import {
  LoaderRegistry,
  MAX_AUTO_RETRY_ATTEMPTS,
} from '../../../../../data/scene-loader/loaders/loader-registry';
import type { DataLoader } from '../../../../../data/data-loader-types';
import type { LinesDataLoader } from '../../../../../types/lines';
import type { GSplatsDataLoader } from '../../../../../types/gsplats';
import type { GeometryKind } from '../../../../../data/data-loader-types';
import { GEOMETRY_TYPES } from '../../../../../types/format-contract';

function makeStub<T>(): T & { dispose: ReturnType<typeof vi.fn> } {
  return { dispose: vi.fn() } as unknown as T & { dispose: ReturnType<typeof vi.fn> };
}

describe('LoaderRegistry — registration & lookup', () => {
  it('starts empty', () => {
    const r = new LoaderRegistry();
    expect(r.totalLoaderCount).toBe(0);
    expect(r.hasLoaders).toBe(false);
    expect(r.loaders.size).toBe(0);
    expect(r.linesLoaders.size).toBe(0);
    expect(r.gsplatLoaders.size).toBe(0);
  });

  it('registers points / lines / gsplats loaders into separate maps', () => {
    const r = new LoaderRegistry();
    r.registerPointsLoader('/p', makeStub<DataLoader>());
    r.registerLinesLoader('/l', makeStub<LinesDataLoader>());
    r.registerGSplatsLoader('/g', makeStub<GSplatsDataLoader>());

    expect(r.loaders.size).toBe(1);
    expect(r.linesLoaders.size).toBe(1);
    expect(r.gsplatLoaders.size).toBe(1);
    expect(r.totalLoaderCount).toBe(3);
    expect(r.hasLoaders).toBe(true);
  });

  it('overwrites an existing entry on re-registration', () => {
    const r = new LoaderRegistry();
    const first = makeStub<DataLoader>();
    const second = makeStub<DataLoader>();
    r.registerPointsLoader('/p', first);
    r.registerPointsLoader('/p', second);

    expect(r.loaders.size).toBe(1);
    expect(r.loaders.get('/p')).toBe(second);
  });
});

describe('LoaderRegistry.getLoaderType', () => {
  it('returns the geometry type for a registered path', () => {
    const r = new LoaderRegistry();
    r.registerPointsLoader('/p', makeStub<DataLoader>());
    r.registerLinesLoader('/l', makeStub<LinesDataLoader>());
    r.registerGSplatsLoader('/g', makeStub<GSplatsDataLoader>());

    expect(r.getLoaderType('/p')).toBe('points');
    expect(r.getLoaderType('/l')).toBe('lines');
    expect(r.getLoaderType('/g')).toBe('gsplats');
  });

  it('returns null for an unknown path', () => {
    const r = new LoaderRegistry();
    expect(r.getLoaderType('/missing')).toBeNull();
  });

  it('checks points first when a path collides across maps', () => {
    // Defensive: nothing prevents the same path from being registered in
    // two maps. Documented precedence is points > lines > gsplats.
    const r = new LoaderRegistry();
    r.registerLinesLoader('/x', makeStub<LinesDataLoader>());
    r.registerGSplatsLoader('/x', makeStub<GSplatsDataLoader>());
    expect(r.getLoaderType('/x')).toBe('lines');

    r.registerPointsLoader('/x', makeStub<DataLoader>());
    expect(r.getLoaderType('/x')).toBe('points');
  });
});

describe('LoaderRegistry — failure tracking', () => {
  it('records a failure with retryCount=0 on first occurrence', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/p', new Error('boom'));

    const info = r.failedLoaders.get('/p');
    expect(info).toBeDefined();
    expect(info!.error.message).toBe('boom');
    expect(info!.retryCount).toBe(0);
    expect(info!.autoRetryCount).toBe(0);
    expect(typeof info!.timestamp).toBe('number');
    expect(r.hasFailures()).toBe(true);
  });

  it('persists the classified error kind alongside the failure', () => {
    // The kind used to be computed for logging and thrown away, so retry policy
    // could not tell a transient failure from a deterministic one.
    const r = new LoaderRegistry();
    r.recordFailure('/net', new Error('fetch failed: http 503'));
    r.recordFailure('/bad', new Error('invalid chunk header'));

    expect(r.failedLoaders.get('/net')!.kind).toBe('Network');
    expect(r.failedLoaders.get('/bad')!.kind).toBe('Decode');
  });

  it('honors an explicitly supplied kind over the heuristic', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/p', new Error('fetch failed'), 'Validation');

    expect(r.failedLoaders.get('/p')!.kind).toBe('Validation');
  });

  describe('automatic-retry eligibility', () => {
    it('offers transient failures and withholds deterministic ones', () => {
      const r = new LoaderRegistry();
      r.recordFailure('/net', new Error('fetch failed'));
      r.recordFailure('/bad', new Error('invalid chunk'));

      expect(r.autoRetryablePaths()).toEqual(['/net']);
      expect(r.hasAutoRetryableFailures()).toBe(true);
    });

    it('does not drain the automatic-retry budget on ordinary recorded failures', () => {
      // recordFailure fires on every update sweep and manual retry. If it
      // charged the automatic budget, a few offline slice scrubs would use it
      // up before `online` fired. Only markAutoRetryAttempt may charge it.
      const r = new LoaderRegistry();
      for (let i = 0; i < MAX_AUTO_RETRY_ATTEMPTS + 5; i++) {
        r.recordFailure('/net', new Error('http 503'));
      }
      expect(r.failedLoaders.get('/net')!.retryCount).toBe(MAX_AUTO_RETRY_ATTEMPTS + 4);
      expect(r.failedLoaders.get('/net')!.autoRetryCount).toBe(0);
      expect(r.autoRetryablePaths()).toEqual(['/net']);
      expect(r.hasAutoRetryableFailures()).toBe(true);
    });

    it('bounds automatic retries via markAutoRetryAttempt, surviving re-records', () => {
      const r = new LoaderRegistry();
      r.recordFailure('/net', new Error('http 404'));
      for (let i = 0; i < MAX_AUTO_RETRY_ATTEMPTS; i++) {
        expect(r.autoRetryablePaths()).toEqual(['/net']); // still eligible
        r.markAutoRetryAttempt('/net'); // connectivity retry attempts it
        r.recordFailure('/net', new Error('http 404')); // ...and it fails again
      }
      // Budget exhausted: autoRetryCount reached the cap and re-records preserved it.
      expect(r.failedLoaders.get('/net')!.autoRetryCount).toBe(MAX_AUTO_RETRY_ATTEMPTS);
      expect(r.autoRetryablePaths()).toEqual([]);
      expect(r.hasAutoRetryableFailures()).toBe(false);
      // Still a failure — a MANUAL retry ignores both filters.
      expect(r.hasFailures()).toBe(true);
    });

    it('markAutoRetryAttempt is a no-op for an unknown path', () => {
      const r = new LoaderRegistry();
      expect(() => r.markAutoRetryAttempt('/nope')).not.toThrow();
      expect(r.hasFailures()).toBe(false);
    });

    it('reports no auto-retryable failures when there are none at all', () => {
      expect(new LoaderRegistry().hasAutoRetryableFailures()).toBe(false);
    });
  });

  it('increments retryCount on repeated failures for the same path', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/p', new Error('first'));
    r.recordFailure('/p', new Error('second'));
    r.recordFailure('/p', new Error('third'));

    const info = r.failedLoaders.get('/p');
    expect(info!.error.message).toBe('third');
    expect(info!.retryCount).toBe(2);
  });

  it('tracks failures independently per path', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/a', new Error('A'));
    r.recordFailure('/b', new Error('B1'));
    r.recordFailure('/b', new Error('B2'));

    expect(r.failedLoaders.get('/a')!.retryCount).toBe(0);
    expect(r.failedLoaders.get('/b')!.retryCount).toBe(1);
  });

  it('clearFailure() removes only the specified entry', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/a', new Error('A'));
    r.recordFailure('/b', new Error('B'));
    r.clearFailure('/a');

    expect(r.failedLoaders.has('/a')).toBe(false);
    expect(r.failedLoaders.has('/b')).toBe(true);
  });

  it('getFailedLoaders() returns a read-only view of the failure map', () => {
    const r = new LoaderRegistry();
    r.recordFailure('/p', new Error('boom'));
    const view = r.getFailedLoaders();
    expect(view.size).toBe(1);
    // Identity: same Map reference (ReadonlyMap is a typescript notion).
    expect(view).toBe(r.failedLoaders);
  });

  it('clearAllFailures() empties the map and logs once when something was cleared', () => {
    const r = new LoaderRegistry();
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined);
    try {
      r.recordFailure('/a', new Error('A'));
      r.recordFailure('/b', new Error('B'));
      r.clearAllFailures();
      expect(r.failedLoaders.size).toBe(0);
      expect(r.hasFailures()).toBe(false);
      expect(logSpy).toHaveBeenCalledTimes(1);

      // Clearing an already-empty map does not log.
      logSpy.mockClear();
      r.clearAllFailures();
      expect(logSpy).not.toHaveBeenCalled();
    } finally {
      logSpy.mockRestore();
    }
  });
});

describe('LoaderRegistry.disposeAll', () => {
  it('disposes every registered loader exactly once and clears every map', () => {
    const r = new LoaderRegistry();
    const p1 = makeStub<DataLoader>();
    const p2 = makeStub<DataLoader>();
    const l1 = makeStub<LinesDataLoader>();
    const g1 = makeStub<GSplatsDataLoader>();
    r.registerPointsLoader('/p1', p1);
    r.registerPointsLoader('/p2', p2);
    r.registerLinesLoader('/l1', l1);
    r.registerGSplatsLoader('/g1', g1);

    r.disposeAll();

    expect(p1.dispose).toHaveBeenCalledTimes(1);
    expect(p2.dispose).toHaveBeenCalledTimes(1);
    expect(l1.dispose).toHaveBeenCalledTimes(1);
    expect(g1.dispose).toHaveBeenCalledTimes(1);
    expect(r.totalLoaderCount).toBe(0);
    expect(r.hasLoaders).toBe(false);
  });

  it('is safe to call on an empty registry', () => {
    const r = new LoaderRegistry();
    expect(() => r.disposeAll()).not.toThrow();
  });

  it('does NOT clear failure tracking', () => {
    const r = new LoaderRegistry();
    r.registerPointsLoader('/p', makeStub<DataLoader>());
    r.recordFailure('/elsewhere', new Error('still tracked'));

    r.disposeAll();

    expect(r.totalLoaderCount).toBe(0);
    expect(r.failedLoaders.size).toBe(1);
  });
});

describe('LoaderRegistry — kind-keyed surface', () => {
  it('covers every geometry type in the format contract', () => {
    const r = new LoaderRegistry();
    for (const kind of GEOMETRY_TYPES) {
      expect(() => r.loadersOf(kind as GeometryKind)).not.toThrow();
      expect(r.loadersOf(kind as GeometryKind).size).toBe(0);
    }
  });

  it('throws on an unknown geometry kind rather than silently creating a bucket', () => {
    const r = new LoaderRegistry();
    expect(() => r.loadersOf('not_a_kind' as GeometryKind)).toThrow(/unknown geometry kind/);
  });

  it('the typed accessors are views onto the same buckets, not copies', () => {
    const r = new LoaderRegistry();
    r.register('points', '/p', makeStub<DataLoader>());
    r.register('lines', '/l', makeStub<LinesDataLoader>());
    r.register('gsplats', '/g', makeStub<GSplatsDataLoader>());

    expect(r.loaders.get('/p')).toBe(r.loadersOf('points').get('/p'));
    expect(r.linesLoaders.get('/l')).toBe(r.loadersOf('lines').get('/l'));
    expect(r.gsplatLoaders.get('/g')).toBe(r.loadersOf('gsplats').get('/g'));

    // Mutating through the typed view must be visible through the keyed one:
    // call sites hold `registry.loaders` directly and mutate it.
    r.loaders.set('/p2', makeStub<DataLoader>());
    expect(r.loadersOf('points').has('/p2')).toBe(true);
    expect(r.totalLoaderCount).toBe(4);
  });

  it('generic register/unregister match the named per-type methods', () => {
    const generic = new LoaderRegistry();
    const named = new LoaderRegistry();

    generic.register('lines', '/x', makeStub<LinesDataLoader>());
    named.registerLinesLoader('/x', makeStub<LinesDataLoader>());
    expect(generic.getLoaderType('/x')).toBe(named.getLoaderType('/x'));

    generic.unregister('lines', '/x');
    named.unregisterLinesLoader('/x');
    expect(generic.getLoaderType('/x')).toBeNull();
    expect(named.getLoaderType('/x')).toBeNull();
  });

  it('disposeAll disposes every kind and empties every bucket', () => {
    const r = new LoaderRegistry();
    const stubs = GEOMETRY_TYPES.map((kind) => {
      const stub = makeStub<DataLoader>();
      r.register(kind as GeometryKind, `/${kind}`, stub);
      return stub;
    });

    r.disposeAll();

    for (const stub of stubs) expect(stub.dispose).toHaveBeenCalledTimes(1);
    expect(r.totalLoaderCount).toBe(0);
    expect(r.hasLoaders).toBe(false);
  });
});
