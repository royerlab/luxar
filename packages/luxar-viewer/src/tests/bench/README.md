# Zipped-store open benchmark

The instrument that decides whether reading `.zarr.zip` **through the chunk cache** is
worth building — the gate on royerlab/luxar#1716.

```bash
pnpm bench:zip:fixtures     # one scene, packaged three ways (needs hatch)
pnpm bench:zip              # measure
```

Useful env vars: `LUXAR_BENCH_REPEATS` (default 3), `LUXAR_BENCH_VARIANTS`
(`directory,zip (STORED),zip (DEFLATE)`), `LUXAR_BENCH_OUT` (write JSON),
`LUXAR_BENCH_HEADLESS=0`, `LUXAR_BENCH_FIXTURES`.

## Reference run

One headless Chromium repeat on the small generated fixtures produced:

| variant       | ready (ms) | requests | bytes (kB) | long tasks (ms) | central dir (kB) |
| ------------- | ---------: | -------: | ---------: | --------------: | ---------------: |
| directory     |       7398 |       87 |      102.8 |            1374 |                — |
| zip (STORED)  |       7817 |      116 |      203.7 |            1411 |              5.0 |
| zip (DEFLATE) |       7636 |      116 |      140.4 |            1369 |              5.0 |

This run shows the archive path at about 1.3× the requests and up to about 2×
the transferred bytes of the equivalent uncached directory path. Treat these
as reference deltas, not portable absolute timings; regenerate them on the
target machine before making the Phase 2 cache decision.

## Why it has its own Playwright config

Both `playwright.config.ts` and `playwright.perf.config.ts` serve data with
`python3 -m http.server`, which has **no HTTP `Range` support at all** — it ignores the
header and answers `200` with the whole body. A zipped store read that way receives the
entire archive in place of each requested window, so a benchmark run against it would
measure nonsense. (Before the `206` guard in `data/zip/range-reader.ts`, it would have
done so _silently_.) This config boots `tools/range-http-server.py` instead, which is
also what a positive zip E2E will need.

## What it reports, and why the fourth column exists

| column              | meaning                                                             |
| ------------------- | ------------------------------------------------------------------- |
| ready (ms)          | wall time to the viewer's `initialized`                             |
| requests            | data-origin requests only (the bundle is identical across variants) |
| bytes (kB)          | data bytes over the wire, bodies + headers                          |
| **long tasks (ms)** | **total main-thread blocking**                                      |
| central dir (kB)    | the fixed preamble a zip reader downloads before _any_ chunk        |

The long-task column is the one that is easy to leave out and decisive to have.
`unzipit` ships `useWorkers: false` and `ZipFileStore` never calls `setOptions`, so every
DEFLATE member inflates **on the main thread**. That cost surfaces as jank, not
necessarily as wall-clock, so a benchmark measuring only time/requests/bytes could bless
a change that makes the viewer stutter. The shipped demo archives are 100% DEFLATE.

## Reading the numbers honestly

**Compare deltas, not absolutes.** Headless Chromium here typically falls back to
software rendering, so the absolute `ready` and `long tasks` figures are dominated by
rasterization and are not what a user sees. What is valid is the _difference_ between
rows: the scene, the renderer and the machine are identical, and only the store layer
changes. `zip (DEFLATE) − directory` in the long-task column is the inflate cost.

**Every variant runs `?no-cache`.** A zipped store bypasses L1/L2 today regardless, so
this is an uncached-vs-uncached comparison — the right A/B for the store layer, but _not_
the cold open a user gets on a directory store with the cache on. Read the `directory`
row as "the same store under the same conditions", not as today's baseline.

The bench asserts only that it measured something. A regression here is a decision for
#1716, not a red build — and it is not wired into CI.
