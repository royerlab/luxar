/**
 * Scene-identity watchdog — detects when a tab's `?src=` address stops
 * serving the scene the tab loaded.
 *
 * Local demo/dev servers share ports and come and go: server A (scene A)
 * dies, server B (scene B) later binds the same port, and every tab that
 * loaded scene A now fronts scene B's data with full confidence — the
 * "I started demo B and got demo A" trap. The load-time cache validation
 * protects fresh loads; nothing protected a tab that simply stayed open.
 *
 * This watchdog re-fetches the dataset's root attrs document (cache-bypassing)
 * on an interval and whenever the tab regains focus/visibility — the exact
 * moment a user returns to a stale tab — and compares scene identity:
 *
 * The poll is CONDITIONAL: once a probe has confirmed identity, its `ETag` is
 * replayed as `If-None-Match`, so the steady state is a bodyless `304`. That
 * matters more than it sounds. Under zarr format 2 the root attrs document was
 * a small `.zattrs`; under format 3 it is the root `zarr.json`, which for a
 * Luxar store carries the ENTIRE consolidated metadata index — 248 kB (38 kB
 * brotli) for a 92-node scene. The format-3 migration silently turned a cheap
 * poll into a ~40 kB download every 15 s, for the lifetime of every open tab.
 * See {@link REMOTE_CHECK_INTERVAL_MS} for the matching cadence change.
 *
 * - `content_hash` when the loaded scene had one (every compiler-written
 *   scene does): the same identity stamp the L2 cache validates against.
 * - Otherwise (hash-less bare nodes) the canonicalized `.zattrs` JSON,
 *   baselined on the attrs actually LOADED — never on a probe, so even a
 *   swap before the first probe is caught.
 *
 * Verdicts surface through the cross-layer notifier as a persistent banner
 * (`ui/scene-identity-banner.ts`):
 *
 * - **changed** (different hash, unparseable body, or a conclusive HTTP error
 *   such as 404): terminal — polling stops, the banner offers Reload.
 *   An HTTP 404 is "changed" rather than "unreachable" because something IS
 *   answering the address; whatever it is, it no longer serves this scene.
 * - **unreachable** (fetch throws, times out, or the server answers with an
 *   inconclusive status — 401/403/408/425/429/5xx): shown only after two
 *   consecutive failures so a single blip stays silent, and cleared
 *   automatically when the server answers again (a recovered server that
 *   serves a different scene escalates straight to `changed`). A transient
 *   overload — or an expired credential on a presigned source — must never
 *   latch the terminal verdict: neither says anything about scene identity.
 *
 * Only `http(s)` sources are watched — there is nothing to race against on
 * an in-memory or file-backed store. Zipped stores (`.zarr.zip`) are excluded
 * too: their root attrs live inside the archive, so the probe URLs this class
 * builds cannot resolve (see `isWatchable`). All timers/listeners are removed by
 * `dispose()`, which the SceneLoader calls on dataset switch and teardown.
 */

import { ROOT_ATTR_DOCS, rootAttrDocOf, rootAttributes } from '../types/zarr-documents';
import { isZippedStoreUrl } from './zip/entries';
import { log, Modules } from '../utils/log';
import { notifier } from '../utils/cross-layer/notifier';

/**
 * Periodic probe cadence for a LOCAL dataset server. Focus/visibility probes
 * fire immediately regardless.
 *
 * This is the case the watchdog was built for: dev/demo servers share ports, so
 * server A can die and server B bind the same port within seconds. A tight
 * cadence is worth it there.
 */
const CHECK_INTERVAL_MS = 15_000;
/**
 * Cadence for a REMOTE dataset.
 *
 * A hosted store is published under an immutable, usually dated, prefix and is
 * replaced by publishing a NEW url, not by swapping bytes under the old one —
 * so the fast cadence buys almost nothing while costing a request every 15 s,
 * per open tab, for the lifetime of the tab. `If-None-Match` already reduces
 * each poll to a bodyless 304; this reduces how many of them there are.
 * With the two-failure threshold below, an unfocused hosted scene can take up
 * to roughly four minutes to show unreachable; focus/visibility probes remain
 * immediate once the spacing guard allows them.
 */
const REMOTE_CHECK_INTERVAL_MS = 120_000;
/** Minimum spacing between probes (guards focus+visibility double-fire). */
const MIN_PROBE_SPACING_MS = 2_000;
/** Consecutive fetch failures before the `unreachable` banner shows. */
const UNREACHABLE_THRESHOLD = 2;
/**
 * Hard cap on a single probe. A server that accepts the connection and then
 * never answers would otherwise leave the probe pending forever, wedging
 * `probeInFlight` and silently killing the watchdog for the tab's lifetime.
 */
const PROBE_TIMEOUT_MS = 10_000;

/**
 * HTTP statuses that say nothing about scene identity — "ask again later" or
 * "I won't tell you" — as opposed to "someone else is serving this address".
 *
 * - 408/425/429/5xx: an overloaded server or proxy.
 * - 401/403: an auth wall. Presigned/tokenized sources are explicitly
 *   supported and their credentials expire; a refused probe is NOT evidence
 *   that the scene changed, and the terminal banner's Reload cannot repair a
 *   stale credential anyway.
 *
 * None of these may latch the terminal verdict. A 404 is different: the
 * server answered, and this scene's root attrs are simply not there.
 */
function isInconclusiveStatus(status: number): boolean {
  return (
    status === 401 ||
    status === 403 ||
    status === 408 ||
    status === 425 ||
    status === 429 ||
    status >= 500
  );
}

/**
 * Serialize a value with object keys sorted recursively, so two encodings of
 * the same attributes compare equal regardless of key order.
 *
 * The hash-less baseline is taken from the attrs the loader read (which come
 * from the store's CONSOLIDATED metadata when the dataset has any), while a
 * probe reads the raw `.zattrs`. zarr writes both key-sorted today, but scene
 * identity must not hinge on that.
 *
 * @param value - Any JSON-serializable value (typically parsed root attrs).
 * @returns The canonical JSON encoding, or `undefined` for `undefined` input.
 */
export function canonicalJson(value: unknown): string | undefined {
  return JSON.stringify(sortKeysDeep(value));
}

/**
 * Build the URL of one root metadata document for a dataset base URL.
 *
 * Appends to the PATH rather than to the raw string, so a query string
 * survives (presigned / tokenized sources — zarrita's `FetchStore` copies the
 * base search params onto every key it fetches, so the store does the same)
 * and a fragment is dropped instead of swallowing the appended path. Falls
 * back to plain concatenation for anything `URL` cannot parse.
 *
 * Trailing slashes are trimmed off the PATHNAME only. Trimming them off the
 * raw string instead would eat the last character of a credential that
 * happens to end in `/` (`?token=abc/`), sending the probe somewhere the
 * store never goes.
 */
function buildAttrsUrl(datasetUrl: string, doc: string): string {
  try {
    const url = new URL(datasetUrl);
    url.pathname = `${url.pathname.replace(/\/+$/, '')}/${doc}`;
    url.hash = '';
    return url.toString();
  } catch {
    return `${datasetUrl.replace(/\/+$/, '')}/${doc}`;
  }
}

/**
 * Whether a dataset URL is served from this machine, and therefore subject to
 * the port-reuse race {@link CHECK_INTERVAL_MS} exists for.
 *
 * LAN hosts (`192.168.*`, `*.local`, or a `--host 0.0.0.0` server reached from
 * another machine) use the remote cadence because their URL does not identify
 * this browser's loopback interface.
 *
 * Unparseable URLs count as remote: `isWatchable` has already required an
 * `http(s)://` prefix, so anything `URL` still cannot parse is not a local
 * dev server, and the slower cadence is the safe way to be wrong.
 */
function isLocalOrigin(datasetUrl: string): boolean {
  try {
    const { hostname } = new URL(datasetUrl);
    return (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname === '[::1]' ||
      hostname === '0.0.0.0'
    );
  } catch {
    return false;
  }
}

function sortKeysDeep(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortKeysDeep);
  if (value === null || typeof value !== 'object') return value;
  const source = value as Record<string, unknown>;
  const sorted: Record<string, unknown> = {};
  for (const key of Object.keys(source).sort()) sorted[key] = sortKeysDeep(source[key]);
  return sorted;
}

export interface SceneIdentityWatchdogOptions {
  /**
   * Dataset base URL (the `?src=` value). A trailing slash, a query string
   * and a fragment are all tolerated: the probe appends `.zattrs` to the
   * path, keeps the query, and drops the fragment.
   */
  datasetUrl: string;
  /** `content_hash` of the scene actually loaded (null when absent). */
  expectedContentHash: string | null;
  /**
   * {@link canonicalJson} of the root attrs actually loaded — the identity
   * baseline for HASH-LESS scenes (bare nodes), compared against the
   * canonicalized probe body. Baselines on what was loaded, so even a swap
   * before the first probe is caught (a first-probe baseline would adopt
   * the impostor as the identity).
   */
  expectedAttrsJson?: string | null;
  /** Probe cadence override (tests). */
  intervalMs?: number;
  /** Fetch override (tests). */
  fetchImpl?: typeof fetch;
}

/**
 * Outcome of a single identity probe: the address still serves the loaded
 * scene (`ok`), serves something else (`changed`), or did not answer
 * usefully (`unreachable`). The latter two name the banner kinds.
 */
export type SceneIdentityVerdict = 'ok' | 'changed' | 'unreachable';

export class SceneIdentityWatchdog {
  private readonly url: string;
  /** Candidate root-attribute addresses, newest format first (query-preserving). */
  private readonly attrsUrls: readonly string[];
  /**
   * The candidate that last answered, so steady-state polling costs ONE
   * request. Only the first probe of a format-2 dataset pays the extra 404.
   */
  private resolvedAttrsUrl: string | null = null;
  /**
   * ETag of the document at {@link resolvedAttrsUrl}, sent back as
   * `If-None-Match` so an unchanged scene answers `304` with no body.
   *
   * PAIRED with `resolvedAttrsUrl` — an ETag is only meaningful against the
   * URL it came from, so this is cleared whenever the resolved document
   * changes, and only ever sent to that one candidate.
   *
   * Only set from a probe that returned an `ok` verdict (see
   * {@link probe}). That is what makes the 304 short-circuit sound: "unchanged
   * since the ETag" is only equivalent to "still the scene we loaded" if the
   * document that ETag names was itself confirmed to match the baseline.
   */
  private resolvedETag: string | null = null;
  /** ETag seen by the in-flight probe; promoted by {@link probe} on `ok`. */
  private pendingETag: string | null = null;
  private readonly expectedHash: string | null;
  private readonly expectedAttrsJson: string | null;
  private readonly intervalMs: number;
  private readonly fetchImpl: typeof fetch;

  private consecutiveFailures = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastProbeAt = 0;
  private probeInFlight = false;
  /** Abort handle for the probe currently in flight (timeout / disposal). */
  private inFlightAbort: AbortController | null = null;
  private disposed = false;
  /** Terminal once the address demonstrably serves something else. */
  private changed = false;

  private readonly onWake = (): void => {
    if (document.visibilityState === 'hidden') return;
    void this.probe();
  };

  constructor(opts: SceneIdentityWatchdogOptions) {
    // Kept verbatim: the caller hands us the same normalized URL the zarr
    // store was opened with, and the probe must address exactly what the
    // store reads. `buildAttrsUrl` trims trailing slashes off the pathname,
    // which is the only place they mean "directory" — a raw-string trim here
    // would instead truncate a credential ending in `/`.
    this.url = opts.datasetUrl;
    this.attrsUrls = ROOT_ATTR_DOCS.map((doc) => buildAttrsUrl(this.url, doc));
    this.expectedHash = opts.expectedContentHash;
    this.expectedAttrsJson = opts.expectedAttrsJson ?? null;
    this.intervalMs =
      opts.intervalMs ?? (isLocalOrigin(this.url) ? CHECK_INTERVAL_MS : REMOTE_CHECK_INTERVAL_MS);
    // Bind: an unbound window.fetch reference throws "Illegal invocation".
    this.fetchImpl = opts.fetchImpl ?? fetch.bind(globalThis);
  }

  /** Whether this dataset URL is one the watchdog can meaningfully watch. */
  static isWatchable(datasetUrl: string): boolean {
    if (!/^https?:\/\//i.test(datasetUrl)) return false;
    // A zipped store keeps its root attrs INSIDE the archive, so the probe
    // URLs this class builds (`archive.zip/zarr.json`, `archive.zip/.zattrs`)
    // 404 unconditionally. 404 is deliberately not an inconclusive verdict, so
    // watching an archive would report `changed` on the first probe and raise a
    // permanent "scene changed" banner on every zipped scene. Re-probing an
    // archive's identity means a HEAD/ETag on the archive itself. That probe now
    // exists (`ZipChunkSource.probeIdentityToken`), and it stays where it is: it
    // runs once at load, inside the cache's own validation, and its verdict
    // clears cache tiers. This watchdog is a different job — a periodic poll
    // whose verdict is a user-facing banner — and pointing it at the same probe
    // would double the request rate for a second opinion on the same fact.
    return !isZippedStoreUrl(datasetUrl);
  }

  /**
   * Begin watching. No immediate probe: the caller baselines identity from
   * the attrs it actually loaded, and an instant re-fetch would only race
   * the load it just finished.
   */
  start(): void {
    if (this.disposed || this.timer !== null) return;
    this.timer = setInterval(() => void this.probe(), this.intervalMs);
    // Returning to a backgrounded tab is the highest-risk moment for
    // staleness (interval timers are throttled while hidden) — check now.
    window.addEventListener('focus', this.onWake);
    document.addEventListener('visibilitychange', this.onWake);
  }

  dispose(): void {
    this.disposed = true;
    if (this.timer !== null) {
      clearInterval(this.timer);
      this.timer = null;
    }
    window.removeEventListener('focus', this.onWake);
    document.removeEventListener('visibilitychange', this.onWake);
    // Don't leave a request hanging on a dataset nobody watches anymore.
    this.inFlightAbort?.abort();
    // A dataset switch must not leave the previous dataset's verdict up.
    notifier.hideSceneIdentityBanner();
  }

  /** One identity probe; exposed for tests (never rejects). */
  async probe(): Promise<void> {
    const now = Date.now();
    if (
      this.disposed ||
      this.changed ||
      this.probeInFlight ||
      now - this.lastProbeAt < MIN_PROBE_SPACING_MS
    ) {
      return;
    }
    this.probeInFlight = true;
    this.lastProbeAt = now;
    try {
      const verdict = await this.fetchVerdict();
      // Promote the ETag ONLY for a document we just confirmed still matches
      // the baseline. A `changed` or unparseable body must never become the
      // token a later 304 is trusted against — that would let one bad answer
      // freeze the watchdog into agreeing with itself forever.
      if (verdict === 'ok' && this.pendingETag !== null) this.resolvedETag = this.pendingETag;
      this.pendingETag = null;
      this.apply(verdict);
    } finally {
      this.probeInFlight = false;
    }
  }

  /**
   * The document name the last successful probe was served from.
   *
   * `undefined` only before any probe has succeeded, which cannot happen on the
   * paths that call this (they run after a body was read); `rootAttributes`
   * falls back to sniffing the content in that case, which is the same answer
   * it gave before the name was threaded through.
   */
  private servedDocName(): string | undefined {
    return this.resolvedAttrsUrl === null ? undefined : rootAttrDocOf(this.resolvedAttrsUrl);
  }

  private async fetchVerdict(): Promise<SceneIdentityVerdict> {
    let body: string;
    // Bounded: an abort (timeout or disposal) surfaces as a thrown fetch,
    // i.e. an ordinary reachability failure.
    const abort = new AbortController();
    this.inFlightAbort = abort;
    const timeout = setTimeout(() => abort.abort(), PROBE_TIMEOUT_MS);
    try {
      // Try the format we last saw, then the other. A dataset only has ONE of
      // these documents, so a 404 on the first candidate means "wrong format",
      // not "gone" — and 404 is deliberately absent from the inconclusive set,
      // so treating it as an answer would report every poll of a format-3
      // store as `changed` and drive a reload loop.
      const candidates =
        this.resolvedAttrsUrl !== null
          ? [this.resolvedAttrsUrl, ...this.attrsUrls.filter((u) => u !== this.resolvedAttrsUrl)]
          : this.attrsUrls;

      let text: string | null = null;
      let lastStatus = 404;
      for (const candidate of candidates) {
        // Conditional ONLY for the document the stored ETag came from: an ETag
        // is meaningless against any other URL, and sending it to the fallback
        // candidate could turn its 404 into a bogus 304.
        const conditional = candidate === this.resolvedAttrsUrl && this.resolvedETag !== null;
        const res = await this.fetchImpl(candidate, {
          cache: 'no-store',
          signal: abort.signal,
          ...(conditional ? { headers: { 'If-None-Match': this.resolvedETag as string } } : {}),
        });
        // The whole point of the conditional request: the document is
        // byte-identical to the one whose ETag we stored, and we only store an
        // ETag after an `ok` verdict, so identity still holds and there is no
        // body to compare.
        //
        // This MUST be handled before the paths below. `res.ok` is false for
        // 304 and 304 is deliberately absent from the inconclusive set, so
        // falling through would run `isInconclusiveStatus(304) ? … : 'changed'`
        // and report a perfectly healthy scene as CHANGED — latching the
        // terminal banner and stopping the watchdog for good.
        if (res.status === 304) return 'ok';
        if (res.ok) {
          // Keep the ETag paired with its document: a fallback to the other
          // format must not leave the previous document's token behind.
          if (candidate !== this.resolvedAttrsUrl) this.resolvedETag = null;
          this.resolvedAttrsUrl = candidate;
          this.pendingETag = res.headers.get('etag');
          text = await res.text();
          break;
        }
        lastStatus = res.status;
        // Anything other than "not here" is a real answer about THIS address
        // and must not be masked by trying the other document.
        if (res.status !== 404) break;
      }

      if (text === null) {
        return isInconclusiveStatus(lastStatus) ? 'unreachable' : 'changed';
      }
      body = text;
    } catch {
      return 'unreachable';
    } finally {
      clearTimeout(timeout);
      this.inFlightAbort = null;
    }

    if (this.expectedHash !== null) {
      try {
        // `resolvedAttrsUrl` is the document that actually answered, so the
        // format follows from its NAME rather than from the body's content.
        const attrs = rootAttributes(JSON.parse(body), this.servedDocName());
        return attrs.content_hash === this.expectedHash ? 'ok' : 'changed';
      } catch {
        return 'changed';
      }
    }
    // Hash-less scene: compare against the attrs actually LOADED, both sides
    // canonicalized (whitespace and key order are not identity). Baselining
    // on the loaded attrs — never on a probe — means even a swap before the
    // first probe is caught.
    if (this.expectedAttrsJson !== null) {
      try {
        return canonicalJson(rootAttributes(JSON.parse(body), this.servedDocName())) ===
          this.expectedAttrsJson
          ? 'ok'
          : 'changed';
      } catch {
        return 'changed';
      }
    }
    // No identity to compare against (loader passed neither hash nor attrs):
    // only reachability is watchable.
    return 'ok';
  }

  private apply(verdict: SceneIdentityVerdict): void {
    if (this.disposed) return;
    switch (verdict) {
      case 'ok':
        this.consecutiveFailures = 0;
        notifier.hideSceneIdentityBanner('unreachable');
        return;
      case 'unreachable':
        this.consecutiveFailures += 1;
        if (this.consecutiveFailures >= UNREACHABLE_THRESHOLD) {
          notifier.showSceneIdentityBanner('unreachable');
        }
        return;
      case 'changed':
        this.changed = true;
        log.warning(
          Modules.SCENE_LOADER,
          `Scene identity changed at ${this.url} — the loaded scene is stale.`
        );
        notifier.showSceneIdentityBanner('changed');
        // Terminal: the verdict cannot un-happen; stop burning requests.
        if (this.timer !== null) {
          clearInterval(this.timer);
          this.timer = null;
        }
        window.removeEventListener('focus', this.onWake);
        document.removeEventListener('visibilitychange', this.onWake);
        return;
    }
  }
}
