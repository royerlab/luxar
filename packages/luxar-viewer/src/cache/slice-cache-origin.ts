/**
 * Slice-cache ORIGIN stamps — "this loaded-data object is exactly the content
 * of that S-cache entry".
 *
 * A post-projection stage cache (#2944 B2) may only reuse a projection when
 * its INPUT is provably the same slice data. Comparing typed arrays byte by
 * byte would cost as much as the projection it saves, so the proof is
 * structural instead: the loaders stamp the objects they hand downstream
 * whose content is a pure function of a restored S-cache payload, and the
 * stage cache keys on that payload's OBJECT IDENTITY (plus a signature of the
 * other stage inputs). S-cache payloads are never mutated after store (see
 * `data/loaders/progressive/slice-cache-helper.ts::cloneLodSnapshot`), so
 * identity implies content.
 *
 * Stamp only objects whose content cannot change afterwards: a restored
 * payload element, or a fresh concatenation built from restored elements and
 * nothing else. NEVER a freshly LOADED result — its arrays alias the loader's
 * reused accumulator, which the next load rewrites in place.
 *
 * Side table (a WeakMap), not a field on the data object, for the same reason
 * `prefix-lineage.ts` is one: the data can itself be an S-cache-owned
 * snapshot whose bytes were measured at store time and which must not be
 * mutated. The stamp does not keep the entry alive: a lookup re-checks that
 * the entry is still resident AND still holds the same payload
 * (`SliceCache.getStage`), so an evicted or upgraded entry simply misses.
 *
 * @module cache/slice-cache-origin
 */

import type { SliceCache, SliceStageOutput } from './slice-cache';

/** Where a loaded-data object's content came from. */
export interface SliceCacheOrigin {
  /** The S-cache holding the entry. */
  readonly cache: SliceCache;
  /** The entry's key (`SliceCache.makeKey`). */
  readonly key: string;
  /** The entry's payload object at restore time (identity is the proof). */
  readonly payload: unknown;
}

const origins = new WeakMap<object, SliceCacheOrigin>();

/** Stamp (or, with `null`, clear) the S-cache origin of `data`. */
export function setSliceCacheOrigin(data: object, origin: SliceCacheOrigin | null): void {
  if (origin === null) origins.delete(data);
  else origins.set(data, origin);
}

/** The S-cache origin stamped on `data`, if any. */
export function getSliceCacheOrigin(data: object): SliceCacheOrigin | undefined {
  return origins.get(data);
}

/**
 * The stage output cached for `data`'s origin entry under `sig`, or undefined
 * when `data` carries no origin, the entry is gone or was replaced, or the
 * output was produced under other parameters.
 */
export function lookupStageOutput(data: object, sig: string): unknown {
  const origin = origins.get(data);
  return origin ? origin.cache.getStage(origin.key, origin.payload, sig) : undefined;
}

/**
 * Whether a `bytes`-sized stage output for `data` could be admitted right now
 * (see `SliceCache.canAdmitStage`) — false when `data` carries no origin.
 */
export function canStoreStageOutput(
  data: object,
  bytes: number,
  opts?: { scan?: boolean }
): boolean {
  const origin = origins.get(data);
  return origin ? origin.cache.canAdmitStage(origin.key, origin.payload, bytes, opts) : false;
}

/**
 * Attach a stage output to `data`'s origin entry. Returns whether it was
 * admitted (false: no origin, entry gone/replaced, or no spare budget).
 * `opts.scan`: the caller is inside dimension playback (see
 * `SliceCache.setStage` for why a scan admits into free budget only).
 */
export function storeStageOutput(
  data: object,
  stage: SliceStageOutput,
  opts?: { scan?: boolean }
): boolean {
  const origin = origins.get(data);
  return origin ? origin.cache.setStage(origin.key, origin.payload, stage, opts) : false;
}
