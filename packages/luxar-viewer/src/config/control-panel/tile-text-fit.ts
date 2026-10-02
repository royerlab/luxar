/**
 * How a tile's text gives way when the tile is too small to hold all of it.
 *
 * The grid is fitted first (`fit-grid.ts`): the scene fixes the tile count and
 * the screen fixes the area, so on a phone with twenty chapters each tile is
 * a fingertip-sized cell, and a sentence-long sublabel cannot fit in it at any
 * legible size. Something has to go, and *what* goes is a priority order, not
 * an accident of overflow: the title is what a visitor taps, so it is the last
 * thing to lose anything.
 *
 * The ladder below is that order. Each step keeps everything the step before
 * it took away and gives up one more thing. The renderer walks it from the
 * top and stops at the first step where every tile fits.
 *
 * One step for the WHOLE grid, never one per tile. A matrix in which some
 * tiles are numbered and some are not, or some carry a second line and their
 * neighbours do not, reads as broken. It is also why the measurement is "does
 * every tile fit": the longest title decides for all of them.
 */

/** Which version of a text a step shows. */
export type TileTextVariant = 'full' | 'short';

/** One rung of the ladder: a complete description of what a tile shows. */
export interface TileTextStep {
  /** The label's version. `short` falls back to `full` where none was authored. */
  label: TileTextVariant;
  /** The sublabel's version, or `none` to drop it. */
  sublabel: TileTextVariant | 'none';
  /** Most lines a sublabel may take, ellipsised past that; `null` = unlimited. */
  sublabelLines: number | null;
  /** Whether the tour-position numeral is shown. */
  index: boolean;
  /** Multiplier on the label's fitted font size. */
  labelScale: number;
  /** `anywhere` lets a label break inside a word; the very last resort. */
  labelWrap: 'words' | 'anywhere';
}

/** What a scene has authored, which decides which steps exist at all. */
export interface TileTextContent {
  /** Some chapter has a sublabel. */
  sublabels: boolean;
  /** Some chapter has an authored short sublabel. */
  shortSublabels: boolean;
  /** Some chapter has an authored short label. */
  shortLabels: boolean;
}

/**
 * Sublabel line limits, tried in order once the full and short texts are both
 * too long. One line is still worth having: it reads as the start of a
 * caption, and a visitor can see there is more by opening the stop.
 */
export const TILE_SUBLABEL_LINE_LIMITS: readonly number[] = [3, 2, 1];

/**
 * Label size multipliers, tried once nothing else is left to drop. Bounded at
 * 0.8 because the label's own floor is already small (0.75rem); below 80% of
 * it a title stops being readable at arm's length, and breaking a long word
 * is the better loss.
 */
export const TILE_LABEL_SCALES: readonly number[] = [0.9, 0.8];

/** Everything shown, nothing clamped: the top of the ladder. */
export const FULL_TILE_TEXT: Readonly<TileTextStep> = {
  label: 'full',
  sublabel: 'full',
  sublabelLines: null,
  index: true,
  labelScale: 1,
  labelWrap: 'words',
};

/**
 * The ladder for a scene with `content`, richest first.
 *
 * Order, and why:
 *
 * 1. the full texts;
 * 2. the authored short sublabel: the author's own condensed text, so it
 *    loses nothing they did not choose to lose;
 * 3. the sublabel cut to three, two, then one line;
 * 4. no sublabel;
 * 5. no numeral, which hands its band back to the label;
 * 6. the authored short label: a title the author wrote to fit, which beats
 *    a smaller copy of the long one;
 * 7. the label at 90% then 80% of its fitted size;
 * 8. a label that may break inside a word, so it is clipped nowhere.
 *
 * Steps that would change nothing (a short sublabel when none is authored,
 * any sublabel step when the scene has none) are left out, so every step the
 * renderer measures is a real change.
 */
export function tileTextLadder(content: TileTextContent): TileTextStep[] {
  const steps: TileTextStep[] = [{ ...FULL_TILE_TEXT }];
  const last = (): TileTextStep => steps[steps.length - 1] as TileTextStep;
  const push = (change: Partial<TileTextStep>): void => {
    steps.push({ ...last(), ...change });
  };

  if (content.sublabels) {
    if (content.shortSublabels) push({ sublabel: 'short' });
    for (const lines of TILE_SUBLABEL_LINE_LIMITS) push({ sublabelLines: lines });
    push({ sublabel: 'none', sublabelLines: null });
  }
  push({ index: false });
  if (content.shortLabels) push({ label: 'short' });
  for (const scale of TILE_LABEL_SCALES) push({ labelScale: scale });
  push({ labelWrap: 'anywhere' });
  return steps;
}
