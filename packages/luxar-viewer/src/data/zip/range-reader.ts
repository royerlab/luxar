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
 * hypothetical: Python's `http.server` (which the Playwright fixture server
 * uses) has no `Range` support at all. Serve the directory containing an
 * archive with `luxar serve <dir>` instead.
 *
 * So every ranged read here REQUIRES `206 Partial Content` and cross-checks the
 * `Content-Range` total against the length learned up front. Range-protocol
 * failures raise {@link RangeUnsupportedError}; missing and access-controlled
 * archives retain status-specific errors instead of suggesting the wrong fix.
 *
 * @module data/zarr/zip-range-reader
 */

/** Thrown when the server cannot (or will not) serve HTTP range requests. */
export class RangeUnsupportedError extends Error {
  constructor(
    readonly url: string,
    detail: string
  ) {
    super(
      `Cannot read the zipped store at ${url}: ${detail}. ` +
        'Reading a .zarr.zip requires a server that honours HTTP Range requests ' +
        '(responding 206 Partial Content). Serve the directory containing it with ' +
        '`luxar serve <dir>`, which does, or unpack the archive into a .zarr directory.'
    );
    this.name = 'RangeUnsupportedError';
  }
}

function archiveHttpError(url: string, response: Response, context = ''): Error {
  const status = `HTTP ${response.status} ${response.statusText}`.trimEnd();
  if (response.status === 404 || response.status === 410) {
    return new Error(
      `Cannot read the zipped store at ${url}: the archive was not found (${status})`
    );
  }
  if (response.status === 401 || response.status === 403) {
    return new Error(
      `Cannot read the zipped store at ${url}: access to the archive was denied (${status})`
    );
  }
  return new RangeUnsupportedError(url, `${status}${context}`);
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

  constructor(private readonly url: string) {}

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
  async probeIdentity(signal?: AbortSignal): Promise<string | null> {
    try {
      const response = await fetch(this.url, { method: 'HEAD', cache: 'no-store', signal });
      if (!response.ok) return null;

      const length = Number(response.headers.get('content-length'));
      if (Number.isFinite(length) && length > 0) this.seedLength(length);

      const etag = response.headers.get('etag');
      if (etag) return `etag:${etag}`;

      const modified = response.headers.get('last-modified');
      if (modified && Number.isFinite(length) && length > 0) {
        return `mtime:${modified}:${length}`;
      }
      return null;
    } catch {
      return null;
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
    this.#length = total;
    return total;
  }

  /** Read `size` bytes at `offset`, insisting on a real partial response. */
  async read(offset: number, size: number): Promise<Uint8Array<ArrayBuffer>> {
    if (size === 0) return new Uint8Array(0);

    const retained = this.#fromRetained(offset, size);
    if (retained) return retained;

    const stitch = this.#retainedSuffix(offset, size);
    if (stitch) {
      // Fetch only the part we are missing and splice the retained tail on.
      const prefix = await this.#fetchRange(offset, stitch.prefixLength);
      const joined = new Uint8Array(size);
      joined.set(prefix, 0);
      joined.set(stitch.suffix, stitch.prefixLength);
      this.#retain(offset, joined);
      return joined as Uint8Array<ArrayBuffer>;
    }

    const bytes = await this.#fetchRange(offset, size);
    this.#retain(offset, bytes);
    return bytes;
  }

  /** One ranged GET, validated as a real partial response. */
  async #fetchRange(offset: number, size: number): Promise<Uint8Array<ArrayBuffer>> {
    const end = offset + size - 1;
    const response = await fetch(this.url, {
      headers: { Range: `bytes=${offset}-${end}` },
    });

    if (!response.ok) {
      throw archiveHttpError(this.url, response, ` for bytes ${offset}-${end}`);
    }
    if (response.status !== 206) {
      // The decisive check. 200 here means the body is the whole archive, not
      // the requested window, and every downstream offset would be wrong.
      throw new RangeUnsupportedError(
        this.url,
        `the server answered a Range request with ${response.status} instead of 206, ` +
          'so the response body is the whole file rather than the requested bytes'
      );
    }

    const total = parseContentRangeTotal(response.headers.get('content-range'));
    if (total !== null && this.#length !== undefined && total !== this.#length) {
      // The archive changed under us mid-read; offsets from the central
      // directory we already parsed no longer describe this file.
      throw new RangeUnsupportedError(
        this.url,
        `the archive changed size during the read (${this.#length} → ${total} bytes)`
      );
    }

    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
  }
}
