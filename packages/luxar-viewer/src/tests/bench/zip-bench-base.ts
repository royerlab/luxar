/**
 * Shared constants for the zipped-store benchmark.
 *
 * Lives under `src/` so the Playwright config and the bench spec import the
 * SAME values and the port the data server boots on cannot drift from the
 * origin the spec fetches from. Mirrors the `perf-data-base.ts` precedent.
 *
 * @module tests/bench/zip-bench-base
 */

import * as path from 'path';
import { fileURLToPath } from 'url';

const packageRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

/** Vite dev-server port for the bench (offset from the default to avoid squatting). */
export const benchViewerPort = Number(process.env.LUXAR_BENCH_PORT ?? 5199);

/** Port for the Range-capable data server (`tools/range-http-server.py`). */
export const benchDataPort = Number(process.env.LUXAR_BENCH_DATA_PORT ?? 9199);

/** Origin the bench fetches datasets from. */
export const benchDataOrigin = `http://127.0.0.1:${benchDataPort}`;

/** Where `tools/make-zip-bench-fixtures.py` writes the three artifacts. */
export const benchFixtureDir =
  process.env.LUXAR_BENCH_FIXTURES ?? path.join(packageRoot, 'bench-fixtures');
