/**
 * Global bounded-concurrency gate for chunk network fetches.
 *
 * zarrita's `get` fans out one `fetch()` per chunk of a selection via an
 * internal `Promise.all`. A single visible range over a multi-million-element
 * LOD level spans thousands of chunks, so an unthrottled load fires thousands
 * of simultaneous requests and exhausts the browser
 * (`net::ERR_INSUFFICIENT_RESOURCES`) — the ceiling that capped large
 * substitutive-LOD scenes (~10M+ finest points).
 *
 * Both data-fetch paths funnel through {@link withFetchGate}: the
 * multi-level caching store's network tier (`fetch-retry.ts`, the default) and
 * the no-cache `FetchStore` (`data/zarr.ts`). A single shared counter caps the
 * total in flight, keeping throughput high (HTTP/2 multiplexes happily at this
 * width) while staying within the browser's socket/memory budget.
 */
export const MAX_CONCURRENT_CHUNK_FETCHES = 64;

let activeFetches = 0;
const fetchQueue: (() => void)[] = [];

/** p-limit-style gate: never lets more than the cap run concurrently. */
export function withFetchGate<T>(fn: () => Promise<T>): Promise<T> {
  const acquire =
    activeFetches < MAX_CONCURRENT_CHUNK_FETCHES
      ? ((activeFetches += 1), Promise.resolve())
      : new Promise<void>((resolve) =>
          fetchQueue.push(() => {
            activeFetches += 1;
            resolve();
          })
        );
  return acquire.then(async () => {
    try {
      return await fn();
    } finally {
      activeFetches -= 1;
      const next = fetchQueue.shift(); // hand the freed slot to the next waiter
      if (next) next();
    }
  });
}

/**
 * Wrap a zarr store so its `get` / `getRange` go through {@link withFetchGate}.
 * A `Proxy` keeps the store's full surface intact (consolidated-metadata and
 * caching wrappers, `contents()`, etc.) while throttling only the two
 * data-fetching methods.
 */
export function boundedConcurrencyStore<S extends object>(store: S): S {
  return new Proxy(store, {
    get(target, prop, receiver) {
      if (prop === 'get' || prop === 'getRange') {
        const orig = Reflect.get(target, prop, receiver);
        if (typeof orig === 'function') {
          return (...args: unknown[]) =>
            withFetchGate(() =>
              (orig as (...a: unknown[]) => Promise<unknown>).apply(target, args)
            );
        }
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as S;
}
