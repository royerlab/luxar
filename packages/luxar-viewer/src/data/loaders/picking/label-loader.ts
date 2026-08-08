/**
 * Label Loader — Lazy CSR-style label fetching from zarr.
 *
 * Labels are stored per node as two zarr arrays:
 *   - label_offsets (uint64, N+1): byte offset of each label in label_bytes
 *   - label_bytes (uint8): concatenated UTF-8 encoded label strings
 *
 * Label i = label_bytes[offsets[i] : offsets[i+1]], decoded as UTF-8.
 * Empty labels (offsets[i] === offsets[i+1]) return null.
 *
 * Labels are fetched lazily on first hover per node, then cached.
 * Concurrent requests for the same node share a single in-flight promise.
 */

import * as zarr from '../../zarr';
import { log, Modules } from '../../../utils/log';

/**
 * Whether two byte ranges of `bytes` hold identical content.
 *
 * Length is checked first: unequal-length ranges cost a single integer
 * compare, and equal-length ones a byte scan that stops at the first
 * mismatch. An empty/uninitialised previous range never matches a non-empty
 * one, and a descending (negative-length) range never matches anything.
 */
function bytesEqual(
  bytes: Uint8Array,
  aStart: number,
  aEnd: number,
  bStart: number,
  bEnd: number
): boolean {
  const length = aEnd - aStart;
  if (length !== bEnd - bStart) return false;
  // Descending offsets (only reachable from a corrupt store) would otherwise
  // skip the loop entirely and report "equal", silently making this element
  // inherit the previous label instead of decoding to ''.
  if (length < 0) return false;
  for (let i = 0; i < length; i++) {
    if (bytes[aStart + i] !== bytes[bStart + i]) return false;
  }
  return true;
}

export class LabelLoader {
  /** Cache of decoded labels per node path. */
  private cache = new Map<string, string[]>();

  /** In-flight loading promises for request coalescing. */
  private inflight = new Map<string, Promise<string[]>>();

  /** UTF-8 text decoder (reused). */
  private decoder = new TextDecoder('utf-8');

  constructor(
    _store: zarr.Readable,
    private rootLoc: zarr.Location<zarr.Readable>
  ) {}

  /**
   * Get a single label by node path and element index.
   * Triggers lazy load of the entire node's labels on first access.
   * Returns null if: no labels exist, label is empty, or index is out of range.
   */
  async getLabel(nodePath: string, elementIndex: number): Promise<string | null> {
    let labels = this.cache.get(nodePath);
    if (!labels) {
      labels = await this.loadNodeLabels(nodePath);
    }
    if (elementIndex < 0 || elementIndex >= labels.length) return null;
    const label = labels[elementIndex];
    return label === '' ? null : label;
  }

  /**
   * Check if a node has labels based on its cached .zattrs metadata.
   */
  hasLabels(nodeAttrs: Record<string, unknown>): boolean {
    return nodeAttrs?.has_labels === true;
  }

  /** Clean up caches. */
  dispose(): void {
    this.cache.clear();
    this.inflight.clear();
  }

  // ---------------------------------------------------------------------------
  // Private
  // ---------------------------------------------------------------------------

  /**
   * Load all labels for a node from zarr (CSR decode).
   * Coalesces concurrent requests for the same node.
   */
  private async loadNodeLabels(nodePath: string): Promise<string[]> {
    // Check cache again (may have been populated by a coalesced request)
    const cached = this.cache.get(nodePath);
    if (cached) return cached;

    // Check if already loading (request coalescing)
    const existing = this.inflight.get(nodePath);
    if (existing) return existing;

    const promise = this.doLoadNodeLabels(nodePath);
    this.inflight.set(nodePath, promise);

    try {
      const result = await promise;
      this.cache.set(nodePath, result);
      return result;
    } finally {
      this.inflight.delete(nodePath);
    }
  }

  /**
   * Actually fetch and decode labels from zarr.
   */
  private async doLoadNodeLabels(nodePath: string): Promise<string[]> {
    try {
      // Strip leading "/" for zarr path resolution
      const cleanPath = nodePath.startsWith('/') ? nodePath.slice(1) : nodePath;

      // Open the two CSR arrays
      const offsetsLoc = this.rootLoc.resolve(`${cleanPath}/label_offsets`);
      const bytesLoc = this.rootLoc.resolve(`${cleanPath}/label_bytes`);

      const offsetsArr = await zarr.open(offsetsLoc, { kind: 'array' });
      const bytesArr = await zarr.open(bytesLoc, { kind: 'array' });

      // Get typed data
      const offsetsData = await zarr.readArray(offsetsArr);
      const bytesData = await zarr.readArray(bytesArr);

      // zarrita returns typed arrays; offsets are BigUint64
      const offsets = offsetsData.data as BigUint64Array;
      const bytes = bytesData.data as Uint8Array;

      const nElements = offsets.length - 1;
      const labels: string[] = new Array(nElements);

      // Consecutive-run reuse. `decode` mints a fresh string per element and
      // the decoded array is cached for the session, so a *broadcast* label
      // (one string repeated over every element — how a producer tags a whole
      // node, e.g. one tract name across 168k line vertices) would otherwise
      // be decoded and retained N times: ~39 MB for a 107-byte label at 168k
      // elements, which drops to ~1.3 MB when the run collapses to a single
      // instance. Comparing element i's bytes to element i-1's costs a length
      // compare, or at most one short byte scan when the lengths coincide
      // (fixed-width IDs), which is trivial next to the `decode` it replaces —
      // so the all-distinct case, the common one (embeddings, protein IDs,
      // edge labels), is not penalised. Deliberately only *consecutive* runs:
      // a general hash-and-pool of every value is a large net loss on distinct
      // labels, so interleaved duplicates are left undeduped.
      let prevStart = -1;
      let prevEnd = -1;

      for (let i = 0; i < nElements; i++) {
        const start = Number(offsets[i]);
        const end = Number(offsets[i + 1]);
        if (start === end) {
          labels[i] = '';
        } else if (bytesEqual(bytes, start, end, prevStart, prevEnd)) {
          labels[i] = labels[i - 1];
        } else {
          labels[i] = this.decoder.decode(bytes.subarray(start, end));
        }
        prevStart = start;
        prevEnd = end;
      }

      log.info(Modules.SCENE_LOADER, `Loaded ${nElements} labels for ${nodePath}`);
      return labels;
    } catch (error) {
      // A node with no labels at all is the ordinary case, not a failure: the
      // picker calls this for whatever it hit, and most nodes (every coarse
      // LOD level of a labelled ladder, for one) simply have no
      // `label_offsets` array. Demote that to info so it does not drown the
      // console — hovering one labelled tract's ladder would otherwise log a
      // warning per coarse level. `isNotFoundError` is the house guard for
      // this (see gsplats-spatial-index-loader / mesh preflight).
      //
      // Caveat: under MultiLevelCachingStore a retry-exhausted fetch also
      // surfaces as a missing key, so this branch can swallow a genuine
      // network death. The cache tier already logs its own warning for that,
      // and everything else — decode failures, corrupt chunks, bad metadata,
      // aborts — still reaches the warning below.
      if (zarr.isNotFoundError(error)) {
        log.info(Modules.SCENE_LOADER, `Node carries no labels: ${nodePath}`);
        return [];
      }
      log.warning(
        Modules.SCENE_LOADER,
        `Failed to load labels for ${nodePath}: ${error instanceof Error ? error.message : error}`
      );
      return [];
    }
  }
}
