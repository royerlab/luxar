# Zipped-store open benchmark

Measures the zipped archive path against the directory store it was built from, both
uncached and on a chunk-cache revisit.

```bash
pnpm bench:zip:fixtures     # one scene, packaged three ways (needs hatch)
pnpm bench:zip              # measure
```

Useful env vars: `LUXAR_BENCH_REPEATS` (default 3), `LUXAR_BENCH_VARIANTS`
(`directory,zip (STORED),zip (DEFLATE)`), `LUXAR_BENCH_OUT` (write JSON),
`LUXAR_BENCH_HEADLESS=0`, `LUXAR_BENCH_FIXTURES`, and `LUXAR_BENCH_REVISIT=1`.

## Reference run

One headless Chromium run (median of three repeats) on the default generated fixtures
(1,081 members, 5.3 MB) produced:

| variant       | ready (ms) | requests | bytes (kB) | long tasks (ms) | central dir (kB) |
| ------------- | ---------: | -------: | ---------: | --------------: | ---------------: |
| directory     |      18803 |     1208 |       5800 |           16606 |                — |
| zip (STORED)  |      20086 |     1684 |       9281 |           18039 |             74.3 |
| zip (DEFLATE) |      19002 |     1684 |       8568 |           17213 |             74.3 |

This run shows the archive path at about 1.4× the requests and up to about 1.6×
the transferred bytes of the equivalent uncached directory path. Treat these
as reference deltas, not portable absolute timings; regenerate them on the
target machine before comparing an archive against its source directory.

The revisit mode on the same fixtures (median of three repeats) produced:

| variant       | requests | bytes (kB) |
| ------------- | -------: | ---------: |
| directory     |        1 |        0.0 |
| zip (STORED)  |        6 |       88.2 |
| zip (DEFLATE) |        4 |       87.5 |

The chunk cache absorbs the per-member traffic for both layouts. The archive's residual
is its central-directory preamble, which is read below the chunk-key cache.

## Why it has its own Playwright config

`playwright.perf.config.ts` serves data with `python3 -m http.server`, which has **no
HTTP `Range` support at all** — it ignores the header and answers `200` with the whole
body. A zipped store read that way receives the entire archive in place of each requested
window, so a benchmark run against it would measure nonsense. (Before the `206` guard in
`data/zip/range-reader.ts`, it would have done so _silently_.) This config and the standard
E2E config boot `tools/range-http-server.py` instead.

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

**The default run gives every variant `?noCache`.** This is an
uncached-vs-uncached comparison — the right A/B for the store layer. Set
`LUXAR_BENCH_REVISIT=1` to measure the second load in the same browser context
with the L1/L2 chunk cache enabled for every variant.

The bench asserts only that it measured something. A regression here is a decision for
#1716, not a red build — and it is not wired into CI.
