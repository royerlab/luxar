/**
 * Walk the tile-text ladder until every tile's text fits inside its tile.
 *
 * The ladder itself — what gives way, in what order — is pure and lives in
 * `config/control-panel/tile-text-fit.ts`. This half applies a step to the
 * grid and measures it.
 *
 * Measured rather than predicted. Whether a title fits depends on its words,
 * the font, the tile's size and the label's container-relative font size, and
 * only layout knows all four. Each step is a handful of attributes on the GRID
 * (the stylesheet does the rest), so trying one costs one layout, and a phone
 * with twenty tiles settles in a few.
 */

import type { TileTextStep } from '../../config/control-panel/tile-text-fit';

/** Custom property: most lines a sublabel may take (`none` = unlimited). */
export const CONTROL_SUBLABEL_LINES_PROPERTY = '--luxar-control-sublabel-lines';
/** Custom property: multiplier on the label's fitted font size. */
export const CONTROL_LABEL_SCALE_PROPERTY = '--luxar-control-label-scale';

/**
 * Overflow tolerance, in CSS pixels. `scrollHeight` rounds to an integer while
 * `clientHeight` is the floor of a fractional box, so a tile that fits exactly
 * can report one pixel over.
 */
const FIT_TOLERANCE_PX = 1;

/** Put `step` on the grid. Everything is an attribute or a property of it. */
export function applyTileTextStep(grid: HTMLElement, step: TileTextStep): void {
  grid.dataset.labelText = step.label;
  grid.dataset.sublabelText = step.sublabel;
  grid.dataset.tileIndex = step.index ? 'shown' : 'hidden';
  grid.dataset.labelWrap = step.labelWrap;
  grid.style.setProperty(
    CONTROL_SUBLABEL_LINES_PROPERTY,
    step.sublabelLines === null ? 'none' : String(step.sublabelLines)
  );
  grid.style.setProperty(CONTROL_LABEL_SCALE_PROPERTY, String(step.labelScale));
}

/**
 * True when no tile's content spills out of it, either way.
 *
 * Width as well as height: a label is never broken inside a word until the
 * last step, so a word longer than its tile widens the label instead, and only
 * the horizontal overflow shows it.
 */
export function tilesFit(grid: HTMLElement): boolean {
  for (const tile of Array.from(grid.querySelectorAll<HTMLElement>('.luxar-control-tile'))) {
    if (tile.scrollHeight > tile.clientHeight + FIT_TOLERANCE_PX) return false;
    if (tile.scrollWidth > tile.clientWidth + FIT_TOLERANCE_PX) return false;
  }
  return true;
}

/**
 * Apply the first step of `ladder` at which every tile fits, and return its
 * position. If none fits, the last step stays applied — it is the one that
 * loses the least of the title — and the return is `ladder.length - 1`.
 */
export function fitTileText(grid: HTMLElement, ladder: readonly TileTextStep[]): number {
  for (const [position, step] of ladder.entries()) {
    applyTileTextStep(grid, step);
    if (tilesFit(grid)) return position;
  }
  return ladder.length - 1;
}
