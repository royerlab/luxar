#### First paint no longer waits on probes it already knows the answer to, or on 15 workers

Loading a hosted scene looked serial in the network panel. It is not fully serial — the
loader fans out four attribute chunks per LOD rung — but a full trace of the drosophila
demo showed the network idle for a third of the load and three separate pieces of pure
overhead ahead of the first pixel. Measured against the live CDN, medians of three runs:
**time to first paint 2051 ms → 560 ms, time to full detail 20.9 s → 18.1 s.**

**The identity poll re-downloaded the whole store index every 15 s.**
`SceneIdentityWatchdog` re-fetches the dataset's root attrs document to notice a dev
server that swapped scenes under a reused port. Under zarr format 2 that document was a
small `.zattrs`. The format-3 migration (#1621, #1635) silently made it the root
`zarr.json` — which for a Luxar store carries the entire consolidated metadata index, 248
kB raw and 38 kB brotli for a 92-node scene. The watchdog was never touched, so its
payload grew about a hundredfold and every open tab re-downloaded it four times a minute,
forever. The poll is now conditional: the ETag of a probe that confirmed identity is
replayed as `If-None-Match`, so the steady state is a bodyless `304`. The ETag is stored
only after an `ok` verdict — "unchanged since that ETag" only implies "still the scene we
loaded" if the document it names was itself confirmed — and it is dropped whenever the
resolved document changes, so it is never sent to a URL it did not come from. A remote
source also polls every 120 s rather than every 15 s: a hosted store under a dated prefix
is replaced by publishing a new URL, not by swapping bytes under the old one, so the tight
cadence buys nothing there. `localhost` keeps 15 s, which is the case the watchdog exists
for.

The `304` branch is load-bearing in an unobvious way: `res.ok` is false for 304 and 304 is
deliberately absent from the inconclusive-status set, so without an explicit branch the
verdict fell through to `changed` — reporting a healthy scene as swapped, raising the
terminal banner and stopping the watchdog for good. That is pinned by its own test.

**zarrita guessed zarr format 2 first, twice.** `resolveFormats` returns `["v2", "v3"]`
for any store its version counter has not seen, so `withMaybeConsolidatedMetadata` probed
`.zmetadata` before `zarr.json`; then `zarrita.open` re-guessed from scratch on the
consolidated *wrapper* (also unseen) and probed `.zattrs` and `.zgroup` before falling
back to v3. Three 404s, 816 ms of pure round trips, on the critical path before the first
data byte. `openStore` now passes `format: ['v3', 'v2']` — an order, not a pin, so
format-2 stores still load and simply pay the extra probe instead — and the root group
opens through a new `openGroupPreferV3` facade helper that tries `open.v3` and falls back
to the guessing `open` on a missing node. Only the first open ever paid this: a successful
open increments the counter, which is why no later array open repeated it.

**A `.zarr` URL was HEAD-probed to confirm it was a store.** `shouldShowBrowser` fired
three parallel HEAD probes before scene loading could start at all. It already
short-circuits a `.zarr.zip` on its suffix; a `.zarr` path with no trailing slash names a
store just as plainly, and now short-circuits the same way. Directory URLs are unaffected —
`classifyBrowserUrl` sends every trailing-slash URL to the browser before the suffix test
runs. The trade is that a dead or mistyped `…/typo.zarr` now surfaces a load error instead
of quietly opening a file browser, which is the more honest failure for a URL that
explicitly names a store.

**The first LOD rung waited for all 15 web workers.** `WorkerPool.initialize()` resolved
only after every spawn settled, and both hot-path accessors awaited it — so a 1024-splat
rung blocked on workers it had no use for. On the hosted demo the pool was also created
lazily by the first decode, 1.93 s after the scene fetch began and after LOD 0's bytes had
already arrived; the workers then came ready ~160 ms apart, 2.2 s from first to last,
because each compiled its own copy of the same WASM module. Three changes, in increasing
depth: workers now publish into the pool the moment each is ready and the hot path awaits a
separate "at least one usable worker" gate, while `initialize()` keeps its all-settled
meaning for callers that want a full pool; `warmUpDataWorkerPool()` starts the pool at
scene-load time so startup overlaps the metadata fetch; and the main thread compiles the
WASM binary once and shares the `WebAssembly.Module` with every worker, which then only
instantiates.

Incremental publishing breaks one invariant that had been free: `workers.length === 0` no
longer means "this attempt finished and produced nothing", because worker 1 can die while
spawns 2..N are still in flight. `handleWorkerFailure` now treats a non-empty
`pendingWorkers` as the witness that more are coming and leaves the init promise alone —
clearing it would let the next `initialize()` bump the generation and orphan every worker
still on its way.

The shared-module path degrades three independent ways, because a broken optimization must
not be able to take the pool down with it: a `structuredClone` probe before the module is
handed out (a host where modules are not cloneable would otherwise turn a DataCloneError
inside Comlink into a synchronous throw that fails all workers identically), an
`instanceof` re-check inside the worker, and a bare retry if the shim rejects the object
form. It also resolves `null` rather than rejecting on any failure, self-deadlines so a
stalled fetch cannot delay every worker's init, and skips non-http candidates — under
Node/SSR the shim URLs resolve to `file://`, which `fetch` cannot serve and, depending on
the runtime, stalls on rather than rejecting.

Not addressed here: the LOD ladder still streams one rung at a time with a single rung of
lookahead, fired after the pass completes, so the network idles through decode → project →
commit → paint. That owns most of the remaining 18 s tail and is filed as #2491, with the
trace data and a verification recipe.
