/**
 * The monitor must report ladder rungs that are ON SCREEN, not the progressive
 * loader's cursor (#2426).
 *
 * The two diverge exactly when a user is trying to diagnose something. A loader
 * advances its cursor as each rung ARRIVES; the `committedLODCount` stamp is
 * written by the commit that puts geometry on screen. So a node whose commit
 * failed or was superseded keeps a cursor claiming rungs the viewer never drew.
 *
 * That is not hypothetical: on the hosted Cosmicflows/Laniakea demo, basins
 * that had run out of memory, stopped refining and frozen at a coarse prefix
 * reported "LOD 7/7 ~100%" in this very panel. It was the one surface that
 * could have shown the stall and it showed the opposite.
 */

import { describe, it, expect } from 'vitest';
import { createLODProgressProvider } from '../../../../../data/scene-loader/monitor/lod-progress-provider';

/** A loader stalled at rung 3 of 7 whose cursor ran on to 7. */
function strandedLoader() {
  return {
    totalLODCount: 7,
    loadedLODCount: 7, // cursor: every rung was FETCHED
    hasMoreLODs: false, // and the loader believes it is finished
    lastAllResident: true,
  };
}

describe('monitor LOD progress reports committed geometry', () => {
  it('reports the committed rung count, not the loader cursor', () => {
    const provider = createLODProgressProvider({
      loaderMaps: [new Map([['/Basin 2 streamlines', strandedLoader()]])],
      lodGroupRegistry: null,
      // Only 3 rungs ever reached the screen.
      committedLODCounts: () => new Map([['/Basin 2 streamlines', 3]]),
    });

    const state = provider.getLODStates().get('/Basin 2 streamlines');

    expect(state).toMatchObject({ kind: 'additive', loaded: 3, total: 7 });
  });

  it('falls back to the loader cursor when no stamp is available', () => {
    // Headless wiring and tests have no root group; losing the panel entirely
    // would be worse than reporting the cursor, which is the old behaviour.
    const provider = createLODProgressProvider({
      loaderMaps: [new Map([['/n', strandedLoader()]])],
      lodGroupRegistry: null,
    });

    expect(provider.getLODStates().get('/n')).toMatchObject({ loaded: 7 });
  });

  it('falls back per node, so one unstamped node does not blank the others', () => {
    const provider = createLODProgressProvider({
      loaderMaps: [
        new Map([
          ['/stamped', strandedLoader()],
          ['/unstamped', strandedLoader()],
        ]),
      ],
      lodGroupRegistry: null,
      committedLODCounts: () => new Map([['/stamped', 2]]),
    });

    const states = provider.getLODStates();
    expect(states.get('/stamped')).toMatchObject({ loaded: 2 });
    expect(states.get('/unstamped')).toMatchObject({ loaded: 7 });
  });

  it('reports a committed count of zero rather than falling back', () => {
    // A node that has loaded rungs but committed NONE is the starkest version
    // of the bug; `?? fallback` must not treat 0 as "no stamp".
    const provider = createLODProgressProvider({
      loaderMaps: [new Map([['/n', strandedLoader()]])],
      lodGroupRegistry: null,
      committedLODCounts: () => new Map([['/n', 0]]),
    });

    expect(provider.getLODStates().get('/n')).toMatchObject({ loaded: 0 });
  });
});
