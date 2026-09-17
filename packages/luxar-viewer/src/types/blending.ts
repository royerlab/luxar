/**
 * Canonical blending-mode set — the single source of truth.
 *
 * Everything derives from this tuple: the `BlendingMode` union
 * (re-exported through `rendering/material-manager` for its historical
 * import site), `normalizeBlendingMode`'s membership check, the layers
 * panel dropdown (all via `rendering/blending-state`), and the per-node
 * attr types (`PointsAttrs` / `LinesAttrs` / `GSplatsAttrs` /
 * `LODGroupAttrs` / `PartitionGroupAttrs`). Adding a mode here updates
 * them all together.
 *
 * Lives in `types/` (dependency-free layer) so both `data/` and
 * `rendering/` may reference it without layering violations. Per-mode
 * semantics are documented on the `BlendingMode` re-export in
 * `rendering/material-manager/factories.ts`; the THREE pipeline state
 * per mode lives in `rendering/blending-state.ts`.
 *
 * @module types/blending
 */

import {
  BLENDING_MODES as CONTRACT_BLENDING_MODES,
  type BlendingModeName,
} from './format-contract';

/**
 * The canonical Luxar blending modes, in panel-dropdown order.
 *
 * Re-exported from the generated format contract
 * (`format-contract/contract.yaml::blending_modes`), so the list the Python
 * writer validates against and the one the panel offers are one projection.
 * Typed as `readonly BlendingMode[]` (not a tuple) — consumers iterate it and
 * index records by it; none needs positional literal types.
 */
export const BLENDING_MODES: readonly BlendingMode[] = CONTRACT_BLENDING_MODES;

/** Union of the canonical mode strings — the contract's `BlendingModeName`. */
export type BlendingMode = BlendingModeName;
