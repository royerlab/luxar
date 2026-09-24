# Slider Kit

Shared interaction and presentation helpers for numeric sliders across the
viewer. The kit keeps gesture tiers, readout precision, exact entry, track
grids, and value/position conversion consistent without owning any panel state.

## Files

- `format.ts` — step-derived precision and off-grid readout widening.
- `hints.ts` — canonical tooltip text for the modifier ladder.
- `inline-number-edit.ts` — accessible click-to-edit numeric readouts.
- `interactions.ts` — wheel, arrow-key, and double-click reset bindings.
- `position.ts` — value/fraction/pixel conversion helpers.
- `track.ts` — fine native range steps and min-anchored base-grid snapping.
- `index.ts` — public barrel for the modules above.

Generic numeric clamping remains in `src/utils/clamp.ts`; consumers should
import it directly rather than routing a foundation helper through this UI
module.
