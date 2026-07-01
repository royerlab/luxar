/**
 * Unit tests for the LOD-progress provider producer — verifies it maps
 * progressive loaders (additive) and LOD-group registry entries
 * (substitutive) into the `path → LODProgressState` snapshot the monitor
 * polls.
 */

import { describe, it, expect } from 'vitest';
import { createLODProgressProvider } from '../../../../../data/scene-loader/monitor/lod-progress-provider';
import type { LODGroupRegistry, LODGroupEntry } from '../../../../../scene/lod-group-registry';

/** Minimal progressive-loader stand-in exposing the duck-typed getters. */
function progressiveLoader(loaded: number, total: number, allResident: boolean) {
  return {
    loadedLODCount: loaded,
    totalLODCount: total,
    hasMoreLODs: loaded < total,
    lastAllResident: allResident,
  };
}

/** A registry stub exposing only `list()` (the producer's single dependency). */
function registryWith(entries: Partial<LODGroupEntry>[]): LODGroupRegistry {
  return {
    list: () => entries as LODGroupEntry[],
  } as unknown as LODGroupRegistry;
}

describe('createLODProgressProvider', () => {
  it('reports additive progressive loaders with loaded/total/refining/residency', () => {
    const gsplatLoaders = new Map<string, unknown>([['/splats', progressiveLoader(2, 4, false)]]);
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), gsplatLoaders],
      lodGroupRegistry: null,
    });

    const state = provider.getLODStates().get('/splats');
    expect(state).toEqual({
      kind: 'additive',
      loaded: 2,
      total: 4,
      refining: true,
      lastAllResident: false,
    });
  });

  it('marks an additive loader as not refining once all levels are loaded', () => {
    const loaders = new Map<string, unknown>([['/pts', progressiveLoader(3, 3, true)]]);
    const provider = createLODProgressProvider({
      loaderMaps: [loaders, new Map(), new Map()],
      lodGroupRegistry: null,
    });
    expect(provider.getLODStates().get('/pts')?.refining).toBe(false);
  });

  it('ignores single-LOD (non-progressive) loaders', () => {
    const loaders = new Map<string, unknown>([['/plain', { updateView: () => {} }]]);
    const provider = createLODProgressProvider({
      loaderMaps: [loaders, new Map(), new Map()],
      lodGroupRegistry: null,
    });
    expect(provider.getLODStates().has('/plain')).toBe(false);
  });

  it('reports substitutive LOD groups with active level + selector mode', () => {
    const registry = registryWith([
      {
        path: '/lod',
        children: [{}, {}, {}] as LODGroupEntry['children'],
        selectorMode: 'auto',
        activeChildIndex: 1,
      },
    ]);
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), new Map()],
      lodGroupRegistry: registry,
    });

    const state = provider.getLODStates().get('/lod');
    expect(state).toEqual({
      kind: 'lod',
      levelCount: 3,
      activeLevel: 1,
      selector: 'auto',
    });
  });

  it('reports the DISPLAYED level, not the aspiration, when they diverge (B2 finding 1)', () => {
    // During a slice scrub the registry shows a coarser fresh level
    // (displayedChildIndex) while activeChildIndex points at the stale fine
    // level being reloaded. The monitor must report what is ON SCREEN.
    const registry = registryWith([
      {
        path: '/lod',
        children: [{}, {}, {}] as LODGroupEntry['children'],
        selectorMode: 'auto',
        activeChildIndex: 2, // aspiration: fine level reloading
        displayedChildIndex: 0, // on screen: coarse fresh fallback
      },
    ]);
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), new Map()],
      lodGroupRegistry: registry,
    });
    // Fails if the production read reverts to bare `entry.activeChildIndex` (→ 2).
    expect(provider.getLODStates().get('/lod')?.activeLevel).toBe(0);
  });

  it('labels a locked selector mode', () => {
    const registry = registryWith([
      {
        path: '/lod',
        children: [{}, {}] as LODGroupEntry['children'],
        selectorMode: { lockLevel: 1 },
        activeChildIndex: 1,
      },
    ]);
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), new Map()],
      lodGroupRegistry: registry,
    });
    expect(provider.getLODStates().get('/lod')?.selector).toBe('locked L2');
  });

  it('emits a partition state per partition group from the snapshot', () => {
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), new Map()],
      lodGroupRegistry: null,
      partitionGroups: [
        { path: '/parts', partCount: 4 },
        { path: '/more', partCount: 2 },
      ],
    });
    const states = provider.getLODStates();
    expect(states.get('/parts')).toEqual({ kind: 'partition', partCount: 4 });
    expect(states.get('/more')).toEqual({ kind: 'partition', partCount: 2 });
  });

  it('emits no partition states when none are provided', () => {
    const provider = createLODProgressProvider({
      loaderMaps: [new Map(), new Map(), new Map()],
      lodGroupRegistry: null,
    });
    const kinds = [...provider.getLODStates().values()].map((s) => s.kind);
    expect(kinds).not.toContain('partition');
  });
});
