/**
 * Choose a column count so a tile grid fills its container without scrolling.
 *
 * A kiosk panel is not a web page. A visitor gets one glance and one touch, so
 * every chapter has to be on screen at once — a grid that runs off the bottom
 * hides stops that nobody will ever scroll to find, and on a plinth tablet the
 * scroll gesture is disabled anyway.
 *
 * So the layout is a *matrix*: pick rows and columns that use the whole
 * container. With the count fixed by the scene, the only free choice is the
 * column count, and the good choice is the one whose cells come out closest to
 * a pleasing shape.
 *
 * Pure and container-size-driven rather than CSS-only, because
 * `repeat(auto-fill, minmax(...))` cannot express "fit exactly N cells in this
 * box": it fills rows greedily from a minimum width and overflows as soon as N
 * outgrows the viewport, which is precisely the failure to avoid.
 */

/**
 * Preferred cell shape, width ÷ height.
 *
 * Just off square, leaning landscape: a chapter tile holds a short line or two
 * of text, which reads better wide than tall.
 *
 * Chosen by looking at what it produces rather than by taste. Across the
 * plausible kiosk screens — 11" and 13" iPads either way up, and a 1080p
 * display — 1.25 is the value that puts ten to twelve chapters in a steady
 * 4x3 on every landscape screen and 3x4 in portrait, which is the familiar
 * app-grid matrix. Raising it to 1.45 flips landscape to 3x4 with cells
 * almost twice as wide as tall; lowering it to 1.1 starts wobbling between
 * 4x3 and 5x3 as the count changes, and a layout that reshuffles when a
 * chapter is added is worse than one that is slightly off-ideal.
 */
export const CONTROL_TILE_TARGET_ASPECT = 1.25;

/**
 * How much a ragged last row counts against a candidate.
 *
 * Expressed in the same units as the aspect cost (a natural-log ratio), so the
 * two are comparable. Low enough that shape wins normally, high enough to
 * break a near-tie towards the tidier matrix — 11 tiles as 4x3 with one gap
 * rather than 5x3 with four.
 */
export const CONTROL_GRID_EMPTY_WEIGHT = 0.6;

/** The chosen layout. `columns * rows >= count`, with `rows` implied. */
export interface GridFit {
  columns: number;
  rows: number;
}

export interface FitGridOptions {
  /** Preferred cell aspect. Defaults to {@link CONTROL_TILE_TARGET_ASPECT}. */
  targetAspect?: number;
  /** Cap on columns, for a scene with a great many chapters. */
  maxColumns?: number;
}

/**
 * Pick the grid that best fills a container of aspect `containerAspect`.
 *
 * @param count Number of tiles. Zero or negative yields a single column.
 * @param containerAspect Container width ÷ height. A non-finite or
 *   non-positive value (a container not yet laid out, which is what a jsdom
 *   test and a first paint both hand over) falls back to a square, so the
 *   result is always usable rather than `NaN` columns.
 */
export function fitGrid(
  count: number,
  containerAspect: number,
  options: FitGridOptions = {}
): GridFit {
  if (!Number.isFinite(count) || count <= 0) return { columns: 1, rows: 1 };
  const tiles = Math.floor(count);
  const aspect = Number.isFinite(containerAspect) && containerAspect > 0 ? containerAspect : 1;
  const target = options.targetAspect ?? CONTROL_TILE_TARGET_ASPECT;
  const maxColumns = Math.max(1, Math.min(tiles, options.maxColumns ?? tiles));

  let best: GridFit = { columns: 1, rows: tiles };
  let bestScore = Number.POSITIVE_INFINITY;

  for (let columns = 1; columns <= maxColumns; columns += 1) {
    const rows = Math.ceil(tiles / columns);
    // Each cell is (width/columns) by (height/rows), so its aspect is the
    // container's scaled by rows/columns.
    const cellAspect = (aspect * rows) / columns;
    // Compared as a LOG ratio, so "twice as wide as wanted" and "half as
    // wide" cost the same. A linear difference would quietly prefer squat
    // cells, since aspect is bounded below by 0 but not above.
    const aspectCost = Math.abs(Math.log(cellAspect / target));
    const emptyCost = (columns * rows - tiles) / tiles;
    const score = aspectCost + CONTROL_GRID_EMPTY_WEIGHT * emptyCost;
    // Strictly less than, so a tie keeps the FEWER columns — larger tiles,
    // which is the better default for a touch target.
    if (score < bestScore) {
      bestScore = score;
      best = { columns, rows };
    }
  }
  return best;
}
