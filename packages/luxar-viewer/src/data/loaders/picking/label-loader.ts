/**
 * Label Loader — Lazy CSR-style label fetching from zarr.
 *
 * Text channels are stored per node as two zarr arrays:
 *   - labels: label_offsets + label_bytes
 *   - keys: key_offsets + key_bytes
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
 * Length, then both END bytes, then the interior. The ends are probed before
 * the scan because a *full* compare of two equal-length ranges is not cheap
 * relative to what it saves — measured on this loader's decode loop, scanning
 * and decoding cost about the same per byte — so the compare only pays for
 * itself when a mismatch is found in a couple of byte tests. Distinct labels almost
 * always differ at one end (a trailing index or id, a leading code); without
 * the end probes, equal-length labels sharing a prefix cost ~1.7x the plain
 * decode-everything loop, which is exactly the regression this reuse is
 * supposed to avoid.
 *
 * An empty/uninitialised previous range never matches a non-empty one, and a
 * zero- or negative-length range never matches anything.
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
  // Zero length is the caller's empty-label branch and never arrives here.
  // Descending offsets (only reachable from a corrupt store) would otherwise
  // skip the loop entirely and report "equal", silently making this element
  // inherit the previous label instead of decoding to ''.
  if (length <= 0) return false;
  if (bytes[aEnd - 1] !== bytes[bEnd - 1]) return false;
  if (bytes[aStart] !== bytes[bStart]) return false;
  for (let i = 1; i < length - 1; i++) {
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

  /**
   * @param channel Which per-element string channel to read (issue #1917).
   *   `'labels'` (default) reads `label_offsets` / `label_bytes` — the
   *   human-readable string a tooltip shows. `'keys'` reads
   *   `key_offsets` / `key_bytes` — the machine-readable string a `link` /
   *   `copy` template substitutes. Identical CSR encoding, identical laziness
   *   and coalescing, identical spatial ordering; only the array names differ,
   *   so one loader serves both rather than a near-copy serving each.
   *   Mirrors `STRING_CHANNELS` in `luxar/io/_compiler/labels/text_labels.py`.
   */
  constructor(
    _store: zarr.Readable,
    private rootLoc: zarr.Location<zarr.Readable>,
    private channel: 'labels' | 'keys' = 'labels'
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
    return nodeAttrs?.[this.channel === 'keys' ? 'has_keys' : 'has_labels'] === true;
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
      const prefix = this.channel === 'keys' ? 'key' : 'label';
      const offsetsLoc = this.rootLoc.resolve(`${cleanPath}/${prefix}_offsets`);
      const bytesLoc = this.rootLoc.resolve(`${cleanPath}/${prefix}_bytes`);

      // Only THIS open may be absent innocently. A node with no labels at all
      // is the ordinary case, not a failure: the picker calls getLabel for
      // whatever it hit, and most nodes (every coarse LOD level of a labelled
      // ladder, for one) simply have no `label_offsets` array. Demote that to
      // info so it does not drown the console — hovering one labelled tract's
      // ladder would otherwise log a warning per coarse level.
      // `isNotFoundError` is the house guard for this (see
      // gsplats-spatial-index-loader / mesh preflight).
      //
      // Everything AFTER this point still warns, deliberately: a node whose
      // offsets exist but whose `label_bytes` do not is a corrupt store, not an
      // unlabelled node, and so are missing chunks, decode failures, bad
      // metadata and aborts.
      //
      // MultiLevelCachingStore rejects retry-exhausted fetches, so only a real
      // not-found or a protected 403/410 metadata probe reaches this optional
      // array branch (see HttpChunkSource).
      let offsetsArr: zarr.Array;
      try {
        offsetsArr = await zarr.open(offsetsLoc, { kind: 'array' });
      } catch (error) {
        if (!zarr.isNotFoundError(error)) throw error;
        log.info(Modules.SCENE_LOADER, `Node carries no ${this.channel}: ${nodePath}`);
        return [];
      }
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
      // instance. What this buys is RETAINED MEMORY, not time: comparing two
      // equal-length ranges costs about what decoding one does, so `bytesEqual`
      // rejects on length and on both end bytes before it scans, which is what
      // keeps the all-distinct case — the common one (embeddings, protein IDs,
      // edge labels) — at parity rather than ~1.7x. Deliberately only
      // *consecutive* runs: a general hash-and-pool of every value is a large
      // net loss on distinct labels, so interleaved duplicates are left
      // undeduped.
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

      log.info(Modules.SCENE_LOADER, `Loaded ${nElements} ${this.channel} for ${nodePath}`);
      return labels;
    } catch (error) {
      log.warning(
        Modules.SCENE_LOADER,
        `Failed to load ${this.channel} for ${nodePath}: ${error instanceof Error ? error.message : error}`
      );
      return [];
    }
  }
}
