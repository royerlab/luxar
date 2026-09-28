/**
 * Build-INDEPENDENT counters for the render gate's suites.
 *
 * `installExtCounters` is handed to Playwright's `context.addInitScript`, so it
 * runs in every page BEFORE the viewer's own scripts and wraps browser APIs
 * rather than viewer code. The same numbers therefore exist on any baseline
 * build, however old, which the viewer's own `getPerf().counters` do not.
 *
 * It installs:
 * - `window.__gateExt`: plain counters (read with `window.__gateExtSnapshot()`,
 *   zeroed with `window.__gateExtReset()`);
 * - GPU uploads: `WebGL2RenderingContext` / `WebGLRenderingContext`
 *   `bufferData`, `bufferSubData`, `texImage2D`, `texImage3D`,
 *   `texSubImage2D`, `texSubImage3D`, and `GPUQueue` `writeBuffer`,
 *   `writeTexture` (bytes + calls, split buffer/texture). Bytes come from the
 *   typed-array argument (minus a `srcOffset`, or its `length`); an image
 *   source (bitmap, canvas, video) counts `width * height * 4`; an upload
 *   from a bound PIXEL_UNPACK buffer (a numeric offset) counts its call only;
 * - `fetch` calls (main thread only: a worker's fetches are not seen);
 * - a `PerformanceObserver('resource')` summing `transferSize` and
 *   `encodedBodySize`, and the maximum number of resources in flight at once
 *   (computed from each entry's `startTime`..`responseEnd`). Resource timing
 *   is per realm, so again main-thread requests only;
 * - `console.{log,info,warn,error,debug}` call counts;
 * - `Worker.prototype.postMessage` counts (main thread to workers).
 *
 * The suite runner merges the snapshot into its metric object under the
 * prefix `ext.` (e.g. `ext.gpuUploadBytes`).
 *
 * @module scripts/render-gate/ext-counters
 */

/**
 * Init script: install the counters. SELF-CONTAINED (serialized into the page).
 */
export function installExtCounters() {
  if (window.__gateExt) return;
  const zero = () => ({
    gpuUploadBytes: 0,
    gpuUploadCalls: 0,
    gpuUploadBytesBuffer: 0,
    gpuUploadBytesTexture: 0,
    gpuUploadCallsBuffer: 0,
    gpuUploadCallsTexture: 0,
    fetchRequests: 0,
    consoleCalls: 0,
    consoleLog: 0,
    consoleInfo: 0,
    consoleWarn: 0,
    consoleError: 0,
    consoleDebug: 0,
    workerPostMessages: 0,
  });
  const ext = zero();
  window.__gateExt = ext;
  let resources = [];

  const isView = (v) => ArrayBuffer.isView(v);
  const viewBytes = (v, srcOffset, length) => {
    const bpe = v.BYTES_PER_ELEMENT || 1;
    if (typeof length === 'number' && length > 0) return length * bpe;
    const off = typeof srcOffset === 'number' ? srcOffset * bpe : 0;
    return Math.max(0, v.byteLength - off);
  };
  const imageBytes = (v) => {
    const w = v?.videoWidth || v?.displayWidth || v?.width || 0;
    const h = v?.videoHeight || v?.displayHeight || v?.height || 0;
    return w * h * 4;
  };
  const upload = (kind, bytes) => {
    ext.gpuUploadCalls++;
    ext.gpuUploadBytes += bytes;
    if (kind === 'buffer') {
      ext.gpuUploadCallsBuffer++;
      ext.gpuUploadBytesBuffer += bytes;
    } else {
      ext.gpuUploadCallsTexture++;
      ext.gpuUploadBytesTexture += bytes;
    }
  };
  /** Bytes of a tex(Sub)Image call: the first typed-array or image-like argument. */
  const texBytes = (args) => {
    for (let i = 5; i < args.length; i++) {
      const a = args[i];
      if (isView(a)) return viewBytes(a, args[i + 1]);
      if (a && typeof a === 'object' && ('width' in a || 'videoWidth' in a)) return imageBytes(a);
    }
    return 0;
  };
  const wrap = (proto, name, bytesOf, kind) => {
    const original = proto?.[name];
    if (typeof original !== 'function') return;
    proto[name] = function wrapped(...args) {
      try {
        upload(kind, bytesOf(args));
      } catch {
        /* counting must never break the page */
      }
      return original.apply(this, args);
    };
  };
  for (const ctor of [window.WebGL2RenderingContext, window.WebGLRenderingContext]) {
    const proto = ctor?.prototype;
    if (!proto) continue;
    // bufferData(target, size|data, usage, srcOffset?, length?)
    wrap(
      proto,
      'bufferData',
      (a) =>
        typeof a[1] === 'number'
          ? a[1]
          : isView(a[1])
            ? viewBytes(a[1], a[3], a[4])
            : a[1]?.byteLength || 0,
      'buffer'
    );
    // bufferSubData(target, dstByteOffset, data, srcOffset?, length?)
    wrap(
      proto,
      'bufferSubData',
      (a) => (isView(a[2]) ? viewBytes(a[2], a[3], a[4]) : a[2]?.byteLength || 0),
      'buffer'
    );
    for (const n of ['texImage2D', 'texImage3D', 'texSubImage2D', 'texSubImage3D']) {
      wrap(proto, n, texBytes, 'texture');
    }
  }
  const queue = window.GPUQueue?.prototype;
  // writeBuffer(buffer, bufferOffset, data, dataOffset?, size?) — offset/size
  // are in ELEMENTS for a typed array, bytes for an ArrayBuffer.
  wrap(
    queue,
    'writeBuffer',
    (a) =>
      isView(a[2])
        ? viewBytes(a[2], a[3], a[4])
        : Math.max(0, a[4] ?? (a[2]?.byteLength || 0) - (a[3] || 0)),
    'buffer'
  );
  // writeTexture(destination, data, dataLayout, size)
  wrap(
    queue,
    'writeTexture',
    (a) =>
      isView(a[1])
        ? viewBytes(a[1], a[2]?.offset ? a[2].offset / (a[1].BYTES_PER_ELEMENT || 1) : 0)
        : a[1]?.byteLength || 0,
    'texture'
  );

  const fetchOriginal = window.fetch;
  if (typeof fetchOriginal === 'function') {
    window.fetch = function countedFetch(...args) {
      ext.fetchRequests++;
      return fetchOriginal.apply(this, args);
    };
  }

  for (const level of ['log', 'info', 'warn', 'error', 'debug']) {
    const original = console[level];
    if (typeof original !== 'function') continue;
    const key = `console${level[0].toUpperCase()}${level.slice(1)}`;
    console[level] = function countedConsole(...args) {
      ext.consoleCalls++;
      ext[key]++;
      return original.apply(this, args);
    };
  }

  const post = window.Worker?.prototype?.postMessage;
  if (typeof post === 'function') {
    window.Worker.prototype.postMessage = function countedPost(...args) {
      ext.workerPostMessages++;
      return post.apply(this, args);
    };
  }

  try {
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) {
        resources.push({
          start: e.startTime,
          end: e.responseEnd || e.startTime + e.duration,
          transfer: e.transferSize || 0,
          encoded: e.encodedBodySize || 0,
        });
      }
    }).observe({ type: 'resource', buffered: true });
  } catch {
    /* no resource timing */
  }

  window.__gateExtReset = () => {
    Object.assign(ext, zero());
    resources = [];
  };
  window.__gateExtSnapshot = () => {
    let transfer = 0;
    let encoded = 0;
    const edges = [];
    for (const r of resources) {
      transfer += r.transfer;
      encoded += r.encoded;
      edges.push([r.start, 1], [r.end, -1]);
    }
    // Ends sort before starts at the same instant: back-to-back is not overlap.
    edges.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    let inflight = 0;
    let maxInflight = 0;
    for (const [, d] of edges) {
      inflight += d;
      if (inflight > maxInflight) maxInflight = inflight;
    }
    return {
      ...ext,
      resourceCount: resources.length,
      resourceTransferBytes: transfer,
      resourceEncodedBytes: encoded,
      resourceMaxInflight: maxInflight,
    };
  };
}
