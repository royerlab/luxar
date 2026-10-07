/**
 * The per-pass playback directives a scene-loader pass hands each node handler.
 *
 * @module data/loaders/pass-directives
 */

import type { ViewState } from '../data-loader-types';

export interface PassDirectives {
  /**
   * Per-tick LOD time budget during dimension-animation playback (see
   * `ViewState.frameBudgetMs`). Injected into the DERIVED per-node view state
   * only — a per-pass directive, so refinement/retry passes (which derive
   * independently) stay budget-free.
   */
  frameBudgetMs?: number;
  /**
   * Pinned playback ladder depth (see `ViewState.ladderDepth`). Same per-pass
   * contract as `frameBudgetMs`: injected into the DERIVED view state only.
   */
  ladderDepth?: number | 'auto';
}

/**
 * The node's derived view state carrying this pass's directives — the SAME
 * object when the pass has none (outside animation playback), so the common
 * case allocates nothing.
 */
export function withPassDirectives(viewState: ViewState, directives: PassDirectives): ViewState {
  const { frameBudgetMs, ladderDepth } = directives;
  if (frameBudgetMs === undefined && ladderDepth === undefined) return viewState;
  return { ...viewState, frameBudgetMs, ladderDepth };
}
