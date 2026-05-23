/**
 * Lightweight integration smoke for worker-pool wiring under Node.
 *
 * The real Worker API is unavailable in Node (jsdom), so the WorkerPool
 * lifecycle + WASM dispatch can only be exercised end-to-end in a browser.
 * That coverage lives in:
 *   - `src/tests/unit/integration/worker-pool/worker-pool.test.ts` (Node
 *     placeholder, `describe.skipIf(!hasWorkerAPI)`)
 *   - `src/tests/e2e/worker-wasm-integration.spec.ts` (Playwright)
 *
 * This file's previous incarnation set up `vi.mock('../../../workers/worker-pool')`
 * but never imported anything from the mocked module — the mock was dead, and
 * every "worker" test asserted on locally-constructed `vi.fn()` objects that
 * were never wired to the production code. That made the file ~170 lines of
 * tests that could not fail under any production mutation
 * (delme/test-audit-luxar-viewer.src/integration.md, C1–C4).
 *
 * Rewrite: only test things that genuinely depend on production code reachable
 * from Node. Anything that needs the browser Worker API stays under
 * `integration/worker-pool/` or the Playwright suite.
 */

import { describe, it, expect } from 'vitest';
import { config } from '../../../config';

describe('Worker integration — Node-side wiring smoke', () => {
  it('worker config defaults route through useWebWorkers=true (production default)', () => {
    // This is the only meaningful contract we can pin in Node: the
    // production config defaults to using web workers. The default came from
    // src/config/sections/data-loading/performance/data.ts; if a future
    // refactor accidentally flips the default, this test catches it. (The
    // *behavior* of workers vs main-thread fallback is verified in the
    // Playwright suite, which runs in a real browser.)
    expect(config.dataLoading.performance.useWebWorkers).toBe(true);
  });

  it('worker pool module imports without throwing under Node (no top-level Worker access)', async () => {
    // Contract: importing the worker-pool module must not crash in
    // environments without a Worker constructor. The pool defers Worker
    // instantiation until `initialize()` is called, so a bare import is
    // safe. If a future refactor moves `new Worker(...)` to module scope,
    // this test catches it.
    await expect(import('../../../workers/worker-pool')).resolves.toBeDefined();
  });
});
