/**
 * Shared defaults for a {@link NodeBuildCtx} in node-loader unit tests.
 *
 * ## Why this exists
 *
 * `NodeBuildCtx` has ~23 members and every one is required. Five node-loader
 * test files each built the whole thing by hand, so **adding one member to the
 * interface broke all five at once** — five identical one-line edits, in five
 * files, none of which cared about the new member. `releaseLazyMesh` did
 * exactly that (#1413).
 *
 * Worse, the breakage is invisible to most of the suite: `eslint` and `vitest`
 * do not typecheck, so those files pass lint and pass their own test runs while
 * the build is red. Only `tsc` catches it.
 *
 * Spreading these defaults first means a new required member is satisfied HERE,
 * in one place, for every test that does not care about it.
 *
 * ## How to use it
 *
 * Spread first, then name what the test actually exercises — the overrides are
 * the point of the file, so they stay explicit and local:
 *
 * ```ts
 * return {
 *   ...nodeBuildCtxDefaults(viewState),
 *   nodeFactory,
 *   processMeshData: vi.fn().mockResolvedValue(staged) as never,
 * };
 * ```
 *
 * The goal is to stop duplicating the ~20 members nobody in a given file cares
 * about — NOT to hide the two they do. A member a test asserts on belongs at
 * the call site, where a reader can see it.
 *
 * `viewState` is a parameter rather than a default because
 * {@link NodeBuildCtx.deriveNodeViewState} closes over it: a test that supplies
 * its own view state needs the derived one to agree, or the loader takes the
 * skip branch against a state the test never set.
 */

import { vi } from 'vitest';
import type { NodeBuildCtx } from '../../data/scene-loader/nodes/build-ctx';
import type { SceneNode } from '../../data/data-loader-types';
import type { ViewState } from '../../types/points';
import { LoaderRegistry } from '../../data/scene-loader/loaders/loader-registry';

/**
 * A fully-populated `NodeBuildCtx` of inert mocks.
 *
 * Every callback is a `vi.fn()` no-op and every process step returns nothing
 * useful, so a test that does not override a member is asserting that the
 * member is irrelevant to it. Override anything you depend on.
 */
export function nodeBuildCtxDefaults(viewState: ViewState): NodeBuildCtx {
  return {
    registry: new LoaderRegistry(),
    // Deliberately empty: every caller that touches the factory supplies its
    // own, and a shared partial one would silently answer for methods the test
    // never set up.
    nodeFactory: {} as unknown as NodeBuildCtx['nodeFactory'],
    viewState,
    getViewVersion: () => 1,
    factoryDeps: {} as never,
    applyEffectiveAttrs: (node: SceneNode) => node.attrs,
    deriveNodeViewState: vi.fn(() => ({ skip: false as const, viewState })) as never,
    connectLoaderToMonitor: vi.fn(),
    kickRefinementIfIdle: vi.fn(),
    isDatasetLive: () => true,
    releaseLazyGSplats: vi.fn(),
    releaseLazyPoints: vi.fn(),
    releaseLazyLines: vi.fn(),
    releaseLazyMesh: vi.fn(),
    processPointsData: vi.fn() as never,
    commitPointsGeometry: vi.fn(),
    processLinesData: vi.fn() as never,
    commitLinesGeometry: vi.fn(),
    processGSplatsData: vi.fn() as never,
    commitGSplatsGeometry: vi.fn(),
    processMeshData: vi.fn() as never,
    commitMeshGeometry: vi.fn(),
  };
}
