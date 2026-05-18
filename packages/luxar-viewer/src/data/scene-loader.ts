/**
 * Re-export stub. The SceneLoader orchestrator class and its
 * supporting modules (view-state queue, per-type handlers, commit
 * helpers, data processors, loader factory, etc.) live in the
 * ./scene-loader/ folder. Consumers import `SceneLoader` plus the
 * public staged-commit type aliases from this path unchanged.
 *
 * Step 9 of the god-object refactor.
 */

export { SceneLoader } from './scene-loader/index';
export type { StagedLinesCommit } from './scene-loader/data-processor-lines';
export type { StagedGSplatsCommit } from './scene-loader/data-processor-gsplats';
