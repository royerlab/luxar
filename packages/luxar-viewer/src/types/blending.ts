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

/** The six canonical Luxar blending modes, in panel-dropdown order. */
export const BLENDING_MODES = [
  'additive',
  'volumetric',
  'normal',
  'max',
  'opaque',
  'luminous',
] as const;

/** Union of the canonical mode strings — derived from {@link BLENDING_MODES}. */
export type BlendingMode = (typeof BLENDING_MODES)[number];
