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
 * uses) has no `Range` support at all.
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
        '(responding 206 Partial Content). Serve the dataset with `luxar serve`, ' +
        'which does, or unpack the archive into a .zarr directory.'
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

  constructor(private readonly url: string) {}

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
