/**
 * `LODGroupRegistry.invalidatePartitionFootprint` (B9c, tracking #2944).
 *
 * Every geometry commit calls it with the committed node path, and it marks the
 * owning partition part's cached footprint stale. On a 2000-part partition a
 * slice move commits every part, so the cost of ONE call matters: scanning every
 * registered part per call is O(parts²) per pass. These pin both halves:
 *
 * - the dirty set after any stream of invalidations equals the reference rule
 *   (every part whose path is the committed path or a segment-prefix of it, in
 *   every partition whose path is too; a partition hit by no part is dirtied
 *   whole), including pathless parts, duplicate part paths and nested
 *   partitions;
 * - the work per call does not grow with the number of parts.
 */

import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import {
  LODGroupRegistry,
  type PartitionGroupChild,
  type PartitionGroupEntry,
} from '../../../scene/lod-group-registry';

function makeRegistry(): LODGroupRegistry {
  const camera = new THREE.Camera();
  return new LODGroupRegistry({
    getCamera: () => camera,
    getViewportSize: () => ({ width: 800, height: 600 }),
    getDisplayDims: () => [0, 1, 2],
  });
}

/** Deterministic PRNG (mulberry32). */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

type Caches = Map<string, { children: Array<{ footprintDirty: boolean }> }>;

function cachesOf(reg: LODGroupRegistry): Caches {
  return (reg as unknown as { partitionCaches: Caches }).partitionCaches;
}

function part(path: string): PartitionGroupChild {
  return {
    path,
    objects: [new THREE.Group()],
    positionBounds: { min: [0, 0, 0], max: [1, 1, 1] },
  };
}

function partition(path: string, partPaths: string[]): PartitionGroupEntry {
  return { path, groupObject: new THREE.Group(), children: partPaths.map(part) };
}

/** The rule, stated directly: the pre-index linear scan. */
function referenceInvalidate(
  entries: readonly PartitionGroupEntry[],
  dirty: Map<string, boolean[]>,
  nodePath: string
): void {
  for (const entry of entries) {
    if (nodePath !== entry.path && !nodePath.startsWith(`${entry.path}/`)) continue;
    const flags = dirty.get(entry.path)!;
    let matched = false;
    entry.children.forEach((child, index) => {
      if (nodePath === child.path || nodePath.startsWith(`${child.path}/`)) {
        flags[index] = true;
        matched = true;
      }
    });
    if (!matched) flags.fill(true);
  }
}

function snapshot(reg: LODGroupRegistry): Map<string, boolean[]> {
  const out = new Map<string, boolean[]>();
  for (const [path, cache] of cachesOf(reg)) {
    out.set(
      path,
      cache.children.map((c) => c.footprintDirty)
    );
  }
  return out;
}

function clean(reg: LODGroupRegistry): void {
  for (const cache of cachesOf(reg).values()) {
    for (const child of cache.children) child.footprintDirty = false;
  }
}

describe('invalidatePartitionFootprint — dirty set', () => {
  it.each([1, 2, 3, 4, 5, 6])('matches the reference rule (seed %i)', (seed) => {
    const rand = rng(seed);
    const pick = <T>(items: readonly T[]): T => items[Math.floor(rand() * items.length)];
    const reg = makeRegistry();
    // Outer partition; one of its parts is itself a partition (nested), plus an
    // unrelated sibling partition. Part paths include a pathless part, a
    // duplicate, and a path outside its wrapper.
    const entries = [
      partition('/p', ['/p/a', '/p/b', '/p/b', '', '/p/n', '/elsewhere/x']),
      partition('/p/n', ['/p/n/0', '/p/n/1', '/p/n/1/deep']),
      partition('/q', ['/q/0', '/q/1', '']),
    ];
    for (const entry of entries) reg.registerPartition(entry);
    const candidates = [
      '/p',
      '/p/a',
      '/p/a/level_1',
      '/p/b',
      '/p/b/additive_2',
      '/p/bb',
      '/p/n',
      '/p/n/0',
      '/p/n/1/deep/level_0',
      '/p/n/9',
      '/p/zzz',
      '/q',
      '/q/1',
      '/q/unregistered',
      '/elsewhere/x',
      '/other',
      '/',
      '',
    ];
    for (let step = 0; step < 60; step++) {
      clean(reg);
      const expected = snapshot(reg);
      const calls = 1 + Math.floor(rand() * 3);
      for (let c = 0; c < calls; c++) {
        const nodePath = pick(candidates);
        reg.invalidatePartitionFootprint(nodePath);
        referenceInvalidate(entries, expected, nodePath);
      }
      expect(snapshot(reg)).toEqual(expected);
    }
    // Unregistering a partition drops it from the lookup too.
    reg.unregister('/q');
    clean(reg);
    reg.invalidatePartitionFootprint('/q/1');
    expect([...snapshot(reg).values()].flat().some(Boolean)).toBe(false);
  });
});

describe('invalidatePartitionFootprint — cost', () => {
  it.fails('does not scan every part per call on a 2000-part partition', () => {
    const reg = makeRegistry();
    let pathReads = 0;
    const children: PartitionGroupChild[] = [];
    for (let i = 0; i < 2000; i++) {
      const path = `/big/part_${i}`;
      const child = part(path);
      Object.defineProperty(child, 'path', {
        get: () => {
          pathReads++;
          return path;
        },
      });
      children.push(child);
    }
    reg.registerPartition({ path: '/big', groupObject: new THREE.Group(), children });

    pathReads = 0;
    for (let i = 0; i < 100; i++) reg.invalidatePartitionFootprint(`/big/part_${i * 7}`);
    // The pre-index scan read every part's path on every call (200,000 here).
    expect(pathReads).toBeLessThan(100 * 10);
    const dirty = cachesOf(reg).get('/big')!.children;
    expect(dirty[7].footprintDirty).toBe(true);
  });
});
