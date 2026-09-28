/**
 * Post-projection stage cache helpers (#2944 B2), shared by the GSplats and
 * Lines data processors.
 *
 * Revisiting a slice during playback restores its decoded payload from the
 * S-cache, but the nD→3D projection used to re-run on every visit: a
 * structured clone of the whole slice into a worker, the kernel, a clone of
 * the projected buffers back, plus the main-thread garbage both clones leave.
 * The projection is a pure function of (slice data, projection parameters),
 * so its output is cached ON the slice's S-cache entry
 * (`SliceCache.setStage`) and reused while both halves are unchanged:
 *
 * - SLICE DATA is identified by the S-cache payload object the loader restored
 *   the data from (`cache/slice-cache-origin.ts`) — never by content, which
 *   would cost as much as the projection.
 * - PARAMETERS are the dispatcher's own params object with every typed array
 *   removed ({@link projectionStageSig}). Deriving the signature from the very
 *   object handed to the kernel — rather than from a hand-picked list of view
 *   fields — makes it complete by construction: a new kernel input cannot be
 *   forgotten, because it has to be in `params` to reach the kernel at all.
 *   The typed arrays are exactly the slice data covered by the payload
 *   identity (their lengths still enter the signature, belt and braces).
 *
 * The cached buffers are shared by every later commit of the slice, so they
 * are marked `sharedBuffers` and the commit must neither mutate nor transfer
 * them (the gsplats depth-sort hand-off copies instead — see
 * `commit-gsplats-geometry.ts`).
 *
 * @module data/scene-loader/process/projection-stage-cache
 */

/** Non-DataView ArrayBufferView (i.e. a TypedArray). */
function isTypedArray(v: unknown): v is ArrayBufferView & { length: number } {
  return ArrayBuffer.isView(v) && !(v instanceof DataView);
}

/**
 * JSON replacer: typed arrays become `{ta, n}` markers (their CONTENT is the
 * slice data the payload identity already covers), and the numbers JSON would
 * collapse (`NaN`/`±Infinity` → `null`, `-0` → `0`) keep distinct spellings,
 * so no two different parameter values can share a signature.
 */
function sigReplacer(_key: string, value: unknown): unknown {
  if (isTypedArray(value)) return { ta: value.constructor.name, n: value.length };
  if (typeof value === 'number') {
    if (Object.is(value, -0)) return '#-0';
    if (!Number.isFinite(value)) return `#${String(value)}`;
  }
  return value;
}

/**
 * Signature of every projection input other than the slice data: the
 * dispatcher params with typed arrays reduced to markers, tagged by geometry.
 */
export function projectionStageSig(kind: 'gsplats' | 'lines', params: object): string {
  return JSON.stringify([kind, params], sigReplacer);
}

/**
 * Retained bytes of a stage output: the sum of the DISTINCT backing buffers of
 * its typed arrays. Buffers, not views — a projection hands back prefix views
 * of worst-case-sized buffers, and the whole buffer is what stays alive.
 */
export function retainedBufferBytes(
  arrays: ReadonlyArray<ArrayBufferView | null | undefined>
): number {
  const seen = new Set<ArrayBufferLike>();
  let bytes = 0;
  for (const a of arrays) {
    if (!a || seen.has(a.buffer)) continue;
    seen.add(a.buffer);
    bytes += a.buffer.byteLength;
  }
  return bytes;
}

/** Sum of the typed arrays' own (view) byte lengths. */
function viewBytes(arrays: ReadonlyArray<ArrayBufferView | null | undefined>): number {
  let bytes = 0;
  for (const a of arrays) if (a) bytes += a.byteLength;
  return bytes;
}

/**
 * Whether retaining `arrays` as-is would keep more than 1/8 of dead tail
 * alive (a projection that culled much of its input returns short views of
 * input-sized buffers). The caller then stores tight copies instead.
 */
export function hasWastefulBuffers(
  arrays: ReadonlyArray<ArrayBufferView | null | undefined>
): boolean {
  return retainedBufferBytes(arrays) * 8 > viewBytes(arrays) * 9;
}

/** A tight copy of `a` (own buffer, exactly `a.length` elements), or `a` when absent. */
export function tightCopy<T extends Float32Array | Uint32Array | undefined>(a: T): T {
  return (a === undefined ? undefined : a.slice()) as T;
}

/**
 * True when every array is still attached and at least `minLength` long — a
 * guard against a cached output whose buffers were detached by a transfer
 * after it was stored (which the commit contract forbids, but a lookup that
 * trusted a detached output would commit an empty frame).
 */
export function buffersIntact(
  checks: ReadonlyArray<readonly [ArrayBufferView | null | undefined, number]>
): boolean {
  for (const [a, minLength] of checks) {
    if (minLength === 0) continue;
    if (!a || a.byteLength === 0 || (a as unknown as { length: number }).length < minLength) {
      return false;
    }
  }
  return true;
}
