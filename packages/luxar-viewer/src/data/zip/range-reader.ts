/**
 * HTTP range reader for zipped Zarr stores (`.zarr.zip`).
 *
 * `ZipFileStore` reads an archive by asking a `Reader` for byte ranges: first
 * the tail (to find the central directory), then one range per member. This
 * module supplies that reader.
 *
 * It exists rather than reusing `@zarrita/storage`'s own `HTTPRangeReader`
 * because that one only asserts `response.ok`. A server that ignores
 * `Range` answers `200` with the WHOLE file, `ok` is true, and the full body
 * is handed back as if it were the requested slice — the zip parser then reads
 * the wrong bytes at every offset. The failure is silent and looks like a
 * corrupt archive rather than a misconfigured server, and it is not
 * hypothetical: Python's plain `http.server` has no `Range` support at all.
 * The Playwright fixture server and `luxar serve <dir>` both provide the
 * required strict range responses.
 *
 * So every ranged read here REQUIRES `206 Partial Content`, verifies the
 * returned window, and cross-checks the `Content-Range` total against the
 * length learned up front. Range-protocol failures raise
 * {@link RangeUnsupportedError}; missing and access-controlled archives retain
 * status-specific errors instead of suggesting the wrong fix.
 *
 * @module data/zarr/zip-range-reader
 */

/** Thrown when the server cannot (or will not) serve HTTP range requests. */
import { ArchiveFaultError } from '../../cache/chunk-source';
import { fetchWithRetry } from '../../cache/multi-level-caching-store/fetch-retry';

export class RangeUnsupportedError extends ArchiveFaultError {
  constructor(url: string, detail: string) {
    super(
      `Cannot read the zipped store at ${url}: ${detail}. ` +
        'Reading a .zarr.zip requires a server that honours HTTP Range requests ' +
        '(responding 206 Partial Content). Serve the directory containing it with ' +
        '`luxar serve <dir>`, which does, or unpack the archive into a .zarr directory.',
      url
    );
    this.name = 'RangeUnsupportedError';
  }
}

function archiveHttpError(url: string, response: Response, context = ''): Error {
  const status = `HTTP ${response.status} ${response.statusText}`.trimEnd();
  if (response.status === 404 || response.status === 410) {
    return new ArchiveFaultError(
      `Cannot read the zipped store at ${url}: the archive was not found (${status}). ` +
        'Check the `?src=` path.',
      url
    );
  }
  if (response.status === 401 || response.status === 403) {
    return new ArchiveFaultError(
      `Cannot read the zipped store at ${url}: access to the archive was denied (${status}).`,
      url
    );
  }
  return new RangeUnsupportedError(url, `${status}${context}`);
}

/** Merge a caller signal with a probe-timeout signal, without a dependency. */
function mergeProbeSignals(a: AbortSignal | undefined, b: AbortSignal): AbortSignal {
  if (!a) return b;
  const controller = new AbortController();
  const forward = (): void => controller.abort();
  if (a.aborted || b.aborted) controller.abort();
  else {
    a.addEventListener('abort', forward, { once: true });
    b.addEventListener('abort', forward, { once: true });
  }
  return controller.signal;
}

/** Cancel a probe body we never read; otherwise the server keeps streaming it. */
function releaseProbeBody(response: Response | undefined): void {
  const body = response?.body;
  if (!body || body.locked) return;
  void body.cancel().catch(() => {
    /* already torn down */
  });
}

/**
 * Parse the total resource size out of a `Content-Range: bytes a-b/total`
 * header. Returns `null` for a missing header or an unknown (`*`) total.
 */
export function parseContentRangeTotal(header: string | null): number | null {
  if (!header) return null;
  const match = /\/\s*(\d+)\s*$/.exec(header);
  if (!match) return null;
  const total = Number(match[1]);
  return Number.isFinite(total) ? total : null;
}

/**
 * A `unzipit` `Reader` over HTTP, structurally typed so neither `unzipit` nor
 * its types need to become a direct dependency.
 */
export class LuxarHttpRangeReader {
  #length: number | undefined;

  /**
   * Byte ranges kept so a later read fully inside one is served from memory.
   *
   * This exists for a specific, guaranteed duplication: `unzipit` reads a FIXED
   * 65,557-byte tail to locate the end-of-central-directory record, and then
   * re-reads the whole central directory — which sits immediately before that
   * tail and is therefore mostly inside the buffer it just fetched. Without
   * this, every open transfers that overlap twice.
   *
   * DELIBERATELY NOT a general read cache. Retention is only armed around the
   * directory-open phase ({@link retainReads}), because member payloads are
   * already cached a layer up — by chunk key, decoded, in L1/L2 — and keeping
   * them here too would double the memory for the same bytes while churning out
   * the small structural reads this is for.
   */
  #retained: { offset: number; bytes: Uint8Array }[] = [];
  #retaining = false;
  #retainedBytes = 0;

  /**
   * Backstop so a pathological archive (a huge comment, a colossal directory)
   * cannot balloon the heap. The preamble is ~141 kB at a thousand members and
   * scales at ~70 B/member, so this holds a very large directory and still
   * refuses to hold a whole archive.
   */
  static readonly MAX_RETAINED_BYTES = 4 * 1024 * 1024;

  constructor(
    private readonly url: string,
    /** Aborted when the owning store is disposed; cancels reads in flight. */
    private readonly lifetime?: AbortSignal
  ) {}

  /**
   * Arm or disarm retention of fetched ranges.
   *
   * Armed by `LuxarZipStore` around the directory read and disarmed
   * afterwards, so what survives is the archive's structure — not its data.
   */
  retainReads(on: boolean): void {
    this.#retaining = on;
  }

  /** Serve `[offset, offset+size)` entirely from a retained buffer, if one covers it. */
  #fromRetained(offset: number, size: number): Uint8Array<ArrayBuffer> | undefined {
    for (const buffer of this.#retained) {
      const start = offset - buffer.offset;
      if (start >= 0 && start + size <= buffer.bytes.length) {
        // `slice` copies, so a caller cannot mutate the retained buffer.
        return buffer.bytes.slice(start, start + size) as Uint8Array<ArrayBuffer>;
      }
    }
    return undefined;
  }

  /**
   * Find a retained buffer covering the TAIL of `[offset, offset+size)`, so only
   * the missing prefix has to be fetched.
   *
   * Containment alone is not enough for the case this class exists for, which
   * measurement made embarrassingly clear: `unzipit` reads a fixed 65,557-byte
   * tail, then re-reads a central directory that is LARGER than that window
   * (74 kB at ~1000 members, and growing at ~70 B/member). The directory
   * therefore starts *before* the retained tail, full containment never
   * matches, and a containment-only memo saves exactly nothing above ~950
   * members. The overlap is real — most of the directory IS in that buffer — it
   * just has to be stitched rather than looked up.
   */
  #retainedSuffix(
    offset: number,
    size: number
  ): { prefixLength: number; suffix: Uint8Array } | undefined {
    const end = offset + size;
    for (const buffer of this.#retained) {
      const bufferEnd = buffer.offset + buffer.bytes.length;
      // Must start after our offset (else containment would have caught it) and
      // reach at least as far as we need.
      if (buffer.offset > offset && buffer.offset < end && bufferEnd >= end) {
        return {
          prefixLength: buffer.offset - offset,
          suffix: buffer.bytes.subarray(0, end - buffer.offset),
        };
      }
    }
    return undefined;
  }

  #retain(offset: number, bytes: Uint8Array): void {
    if (!this.#retaining) return;
    if (this.#retainedBytes + bytes.length > LuxarHttpRangeReader.MAX_RETAINED_BYTES) return;
    this.#retained.push({ offset, bytes });
    this.#retainedBytes += bytes.length;
  }

  /**
   * Record what a fresh `HEAD` already told us, so {@link getLength} does not
   * repeat it. The identity probe and the length lookup ask the same question
   * of the same URL; paying for it twice adds a serialised round trip to every
   * load.
   */
  seedLength(length: number): void {
    if (Number.isFinite(length) && length > 0) this.#length = length;
  }

  /**
   * Fresh identity of the archive, bypassing every cache — `ETag`, else
   * `Last-Modified` + `Content-Length`. `null` means "cannot tell", which is
   * never evidence that the archive changed.
   *
   * Lives here rather than in the caller because it is a `HEAD` on this exact
   * URL: doing it here lets the length it reports seed {@link getLength}.
   */
  async probeIdentity(signal?: AbortSignal, timeoutMs?: number): Promise<string | null> {
    // Split the caller's budget across both attempts. `doValidateCache` passes
    // `validationTimeoutMs` precisely so a hanging server cannot block the first
    // paint — this probe sits in front of it, via `init()` → `validateCache`.
    const perAttempt = timeoutMs === undefined ? undefined : Math.max(1, Math.ceil(timeoutMs / 2));

    const fromHead = await this.#probeVia({ method: 'HEAD' }, false, signal, perAttempt);
    if (fromHead) return fromHead;

    // Fall back to a one-byte ranged GET, for the same reason `getLength` does:
    // some static hosts and signed-URL schemes allow only `GET`. Without this a
    // HEAD-hostile host yields no token at all, which lands in the no-token
    // branch → validation mode `none` → the archive is never re-checked and a
    // replaced one keeps serving stale chunks. A 206 carries `ETag` /
    // `Last-Modified` just as a HEAD does.
    return this.#probeVia({ headers: { Range: 'bytes=0-0' } }, true, signal, perAttempt);
  }

  /**
   * One identity probe, cache-bypassing; `null` means "cannot tell".
   *
   * `ranged` matters for correctness, not tidiness: on a 206 `Content-Length` is
   * the length of the RANGE (one byte here), so trusting it would overwrite a
   * correct archive length with 1 — after which every read fails the
   * size-change cross-check and blames the server for a change it never made.
   * A total may only come from `Content-Range`, which is not CORS-safelisted and
   * so is often absent; absent means "unknown", not "one".
   */
  async #probeVia(
    init: RequestInit,
    ranged: boolean,
    signal?: AbortSignal,
    timeoutMs?: number
  ): Promise<string | null> {
    const timeout = timeoutMs === undefined ? undefined : new AbortController();
    const timer = timeout === undefined ? undefined : setTimeout(() => timeout.abort(), timeoutMs);
    const merged = timeout === undefined ? signal : mergeProbeSignals(signal, timeout.signal);
    let response: Response | undefined;
    try {
      response = await fetch(this.url, { ...init, cache: 'no-store', signal: merged });
      if (!response.ok) return null;

      const total = ranged
        ? parseContentRangeTotal(response.headers.get('content-range'))
        : Number(response.headers.get('content-length'));
      if (total !== null && Number.isFinite(total) && total > 0) this.seedLength(total);

      const etag = response.headers.get('etag');
      if (etag) return `etag:${etag}`;

      const modified = response.headers.get('last-modified');
      if (modified && total !== null && Number.isFinite(total) && total > 0) {
        return `mtime:${modified}:${total}`;
      }
      return null;
    } catch {
      return null;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
      // On a host that ignores `Range`, this body is the WHOLE archive and we
      // read none of it. Left alone, the server keeps streaming it.
      releaseProbeBody(response);
    }
  }

  /**
   * Total archive size.
   *
   * Prefers `HEAD`, but falls back to a one-byte ranged `GET` when `HEAD` is
   * rejected or answers without a usable `Content-Length` — some static hosts
   * and signed-URL schemes allow only `GET`. The fallback doubles as an early
   * probe: a host that answers it with `200` is telling us it ignores `Range`,
   * and we would rather say so here than at the first chunk.
   */
  async getLength(): Promise<number> {
    if (this.#length !== undefined) return this.#length;

    let head: Response | undefined;
    try {
      head = await fetch(this.url, { method: 'HEAD' });
    } catch {
      // Network/CORS refusal of HEAD specifically — fall through to the probe.
    }
    if (head?.ok) {
      const declared = Number(head.headers.get('content-length'));
      if (Number.isFinite(declared) && declared > 0) {
        this.#length = declared;
        return declared;
      }
    }

    const probe = await fetch(this.url, {
      headers: { Range: 'bytes=0-0' },
    });
    // Released on EVERY exit below — this is the worse of the two probe sites:
    // on a host that ignores `Range`, the 200 body is the whole archive and each
    // of these branches throws without reading a byte of it.
    try {
      this.#length = this.#lengthFromProbe(probe);
      return this.#length;
    } finally {
      releaseProbeBody(probe);
    }
  }

  /** Extract the archive length from a one-byte ranged probe response. */
  #lengthFromProbe(probe: Response): number {
    if (!probe.ok) {
      throw archiveHttpError(this.url, probe);
    }
    if (probe.status !== 206) {
      throw new RangeUnsupportedError(
        this.url,
        `the server answered a Range request with ${probe.status} instead of 206`
      );
    }
    const total = parseContentRangeTotal(probe.headers.get('content-range'));
    if (total === null) {
      throw new RangeUnsupportedError(
        this.url,
        'the 206 response carried no usable Content-Range; cross-origin servers must include ' +
          '`Content-Range` in `Access-Control-Expose-Headers`'
      );
    }
    return total;
  }

  /** Read `size` bytes at `offset`, insisting on a real partial response. */
  async read(offset: number, size: number, signal?: AbortSignal): Promise<Uint8Array<ArrayBuffer>> {
    if (size === 0) return new Uint8Array(0);

    const retained = this.#fromRetained(offset, size);
    if (retained) return retained;

    const stitch = this.#retainedSuffix(offset, size);
    if (stitch) {
      // Fetch only the part we are missing and splice the retained tail on.
      const prefix = await this.#fetchRange(offset, stitch.prefixLength, signal);
      const joined = new Uint8Array(size);
      joined.set(prefix, 0);
      joined.set(stitch.suffix, stitch.prefixLength);
      this.#retain(offset, joined);
      return joined as Uint8Array<ArrayBuffer>;
    }

    const bytes = await this.#fetchRange(offset, size, signal);
    this.#retain(offset, bytes);
    return bytes;
  }

  /**
   * One ranged GET, validated as a real partial response.
   *
   * Routed through `fetchWithRetry` rather than bare `fetch` for two reasons
   * that were both review findings: it applies the SAME retry budget the
   * directory path gets (without it, one transient 5xx on a range GET became a
   * fill-valued chunk), and it holds the bounded-concurrency gate. That gate is
   * the reason nothing wraps a zipped store in `boundedConcurrencyStore` any
   * more — gating at two levels would let a gated outer `get` await a gated
   * inner `read` and deadlock the pool.
   */
  async #fetchRange(
    offset: number,
    size: number,
    signal?: AbortSignal
  ): Promise<Uint8Array<ArrayBuffer>> {
    const end = offset + size - 1;
    const bytes = await fetchWithRetry(
      this.url,
      {
        signal: signal ?? this.lifetime,
        headers: { Range: `bytes=${offset}-${end}` },
      },
      async ({ response, readBody }) => {
        if (!response.ok) {
          throw archiveHttpError(this.url, response, ` for bytes ${offset}-${end}`);
        }
        if (response.status !== 206) {
          // The decisive check: 200 here means the body is the whole archive,
          // not the requested window.
          throw new RangeUnsupportedError(
            this.url,
            `the server answered a Range request with ${response.status} instead of 206, ` +
              'so the response body is the whole file rather than the requested bytes'
          );
        }

        const contentRange = response.headers.get('content-range');
        if (contentRange !== null) {
          const match = /^\s*bytes\s+(\d+)-(\d+)\/(?:\d+|\*)\s*$/i.exec(contentRange);
          const rangeStart = match ? Number(match[1]) : undefined;
          const rangeEnd = match ? Number(match[2]) : undefined;
          if (
            rangeStart !== undefined &&
            rangeEnd !== undefined &&
            (rangeStart !== offset || rangeEnd !== end)
          ) {
            throw new RangeUnsupportedError(
              this.url,
              `the requested window was bytes ${offset}-${end}, but Content-Range reported bytes ${rangeStart}-${rangeEnd}`
            );
          }
        }

        const total = parseContentRangeTotal(contentRange);
        if (total !== null && this.#length !== undefined && total !== this.#length) {
          // Offsets already parsed from the central directory belong to the
          // previous archive size and cannot safely address the new body.
          throw new RangeUnsupportedError(
            this.url,
            `the archive changed size during the read (${this.#length} → ${total} bytes)`
          );
        }

        const data = await readBody();
        if (data.length !== size) {
          throw new RangeUnsupportedError(
            this.url,
            `the requested window was ${size} bytes, but the server returned ${data.length}`
          );
        }
        return data;
      }
    );
    if (!bytes) {
      // The live signal is the lifetime one unless a caller supplied its own —
      // and none do today, so testing `signal` alone never fired.
      if ((signal ?? this.lifetime)?.aborted) {
        throw new DOMException('Archive read aborted', 'AbortError');
      }
      // Retry exhaustion during the DIRECTORY read really is a container fault:
      // without the index nothing in the archive is readable. Exhaustion on a
      // member read is not — it is one flaky chunk, and reporting it as an
      // archive fault would fail the whole load and tell the user to run
      // `luxar serve` about a transient 5xx. `#retaining` is true exactly for
      // the directory phase, which makes it the discriminator.
      const detail = `the request for bytes ${offset}-${end} exhausted its retries`;
      // Fatal during the directory read, because without the index nothing in
      // the archive is readable — but NOT a `RangeUnsupportedError`: the server
      // answered 206s and then started failing, so telling the user to serve it
      // differently or unpack it is advice for a problem they do not have.
      throw this.#retaining
        ? new ArchiveFaultError(
            `Cannot read the zipped store at ${this.url}: ${detail} while reading the ` +
              'archive index, so no part of it can be read. This looks transient — retry, ' +
              'or check the server.',
            this.url
          )
        : new Error(`Cannot read the zipped store at ${this.url}: ${detail}`);
    }
    return bytes;
  }
}
