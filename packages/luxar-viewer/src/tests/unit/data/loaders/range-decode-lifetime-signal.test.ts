import { describe, expect, it } from 'vitest';
import { PointsSpatialIndexLoader } from '../../../../data/points/points-spatial-index-loader';
import { LinesSpatialIndexLoader } from '../../../../data/lines/lines-spatial-index-loader';
import { GSplatsSpatialIndexLoader } from '../../../../data/gsplats/gsplats-spatial-index-loader';

const location = { resolve: (path: string) => path };
const node = { path: '/node', attrs: {} };

describe('range decode signal on initial node builds', () => {
  // geometry-subset: mesh has no spatial-index loader (it loads whole), so it owns no range decode
  it.each([
    ['points', PointsSpatialIndexLoader],
    ['lines', LinesSpatialIndexLoader],
    ['gsplats', GSplatsSpatialIndexLoader],
  ] as const)('%s uses its own lifetime when no update signal exists', async (_name, Loader) => {
    const first = new Loader(
      location as unknown as ConstructorParameters<typeof Loader>[0],
      node as ConstructorParameters<typeof Loader>[1]
    );
    const second = new Loader(
      location as unknown as ConstructorParameters<typeof Loader>[0],
      node as ConstructorParameters<typeof Loader>[1]
    );
    const firstInternals = first as unknown as {
      rangeLoader: { _getSignal: () => AbortSignal | null };
      _lifetime: {
        signal: AbortSignal;
        calls: { runWithSignal: <T>(signal: AbortSignal, load: () => Promise<T>) => Promise<T> };
      };
    };
    const secondInternals = second as unknown as typeof firstInternals;

    expect(firstInternals.rangeLoader._getSignal()).toBe(firstInternals._lifetime.signal);
    expect(secondInternals.rangeLoader._getSignal()).toBe(secondInternals._lifetime.signal);
    const update = new AbortController();
    await firstInternals._lifetime.calls.runWithSignal(update.signal, async () => {
      expect(firstInternals.rangeLoader._getSignal()).toBe(update.signal);
    });

    first.dispose();
    expect(firstInternals.rangeLoader._getSignal()?.aborted).toBe(true);
    expect(secondInternals.rangeLoader._getSignal()?.aborted).toBe(false);
    second.dispose();
  });
});
