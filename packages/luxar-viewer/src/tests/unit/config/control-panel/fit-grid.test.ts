/**
 * The kiosk panel must be ONE page. These tests pin that as arithmetic.
 *
 * The invariant that matters is not "which layout looks nicest" — it is that
 * `columns * rows >= count` for every count and every screen shape, because a
 * grid that cannot hold its tiles is a grid that scrolls, and a chapter below
 * the fold on a plinth tablet is a chapter no visitor will ever reach.
 */

import { describe, expect, it } from 'vitest';

import {
  CONTROL_GRID_EMPTY_WEIGHT,
  CONTROL_TILE_TARGET_ASPECT,
  fitGrid,
} from '../../../../config/control-panel/fit-grid';

/** Screens a kiosk plausibly runs on, both ways up. */
const SCREENS: Array<[string, number]> = [
  ['iPad 11" landscape', 1194 / 834],
  ['iPad 11" portrait', 834 / 1194],
  ['iPad 13" landscape', 1366 / 1024],
  ['iPad 13" portrait', 1024 / 1366],
  ['1080p display', 1920 / 1080],
  ['ultrawide', 21 / 9],
  ['phone portrait', 390 / 844],
  ['square', 1],
];

describe('fitGrid', () => {
  it('always has room for every tile', () => {
    for (const [name, aspect] of SCREENS) {
      for (let count = 1; count <= 60; count += 1) {
        const { columns, rows } = fitGrid(count, aspect);
        expect(columns * rows, `${name} with ${count}`).toBeGreaterThanOrEqual(count);
        expect(columns).toBeGreaterThan(0);
        expect(rows).toBeGreaterThan(0);
      }
    }
  });

  it('never wastes a whole row', () => {
    // `rows` is `ceil(count / columns)` by construction, so a layout with an
    // entirely empty final row would mean the row count was computed from
    // something other than the column count.
    for (const [, aspect] of SCREENS) {
      for (let count = 1; count <= 60; count += 1) {
        const { columns, rows } = fitGrid(count, aspect);
        expect(columns * (rows - 1)).toBeLessThan(count);
      }
    }
  });

  it('puts a ten-to-twelve chapter tour in a steady landscape matrix', () => {
    // The ESM tours are eleven stops, and a plinth tablet is usually
    // landscape. A layout that reshuffled between ten, eleven and twelve
    // would make adding a chapter visibly rearrange the panel, so the shape
    // is expected to hold across that range.
    for (const count of [10, 11, 12]) {
      expect(fitGrid(count, 1194 / 834), `landscape ${count}`).toEqual({
        columns: 4,
        rows: 3,
      });
    }
  });

  it('takes an exact-fit portrait grid over a tidier-looking one', () => {
    // Portrait is NOT steady across that range, and deliberately so: ten
    // tiles factor exactly into 2x5 with no holes, which beats 3x4's two
    // empty cells even though 3x4's cells are closer to the target shape.
    // Eleven and twelve have no such exact 2-column fit, so they go to 3x4.
    // Documented rather than smoothed over — an exact-fit grid reads as
    // deliberate, and the no-scroll invariant holds either way.
    expect(fitGrid(10, 834 / 1194)).toEqual({ columns: 2, rows: 5 });
    expect(fitGrid(11, 834 / 1194)).toEqual({ columns: 3, rows: 4 });
    expect(fitGrid(12, 834 / 1194)).toEqual({ columns: 3, rows: 4 });
  });

  it('turns the matrix on its side with the screen', () => {
    // The same count on a rotated screen should transpose, not keep a layout
    // that now runs off the bottom.
    const landscape = fitGrid(12, 4 / 3);
    const portrait = fitGrid(12, 3 / 4);
    expect(landscape.columns).toBeGreaterThan(landscape.rows);
    expect(portrait.rows).toBeGreaterThan(portrait.columns);
  });

  it('keeps cells near the target shape', () => {
    // The point of measuring at all. Allow a factor of two either way: with
    // the count fixed and only whole columns available, an exact hit is often
    // unreachable, but a cell four times too wide would mean the search is
    // not working.
    for (const [name, aspect] of SCREENS) {
      for (let count = 2; count <= 40; count += 1) {
        const { columns, rows } = fitGrid(count, aspect);
        const cellAspect = (aspect * rows) / columns;
        const ratio = cellAspect / CONTROL_TILE_TARGET_ASPECT;
        expect(ratio, `${name} with ${count}`).toBeGreaterThan(0.5);
        expect(ratio, `${name} with ${count}`).toBeLessThan(2);
      }
    }
  });

  it('prefers the tidier matrix when two shapes score alike', () => {
    // Four tiles on a square screen is the clean case: 2x2 wastes nothing,
    // while 3x2 leaves two holes for no gain in shape.
    expect(fitGrid(4, 1)).toEqual({ columns: 2, rows: 2 });
    expect(CONTROL_GRID_EMPTY_WEIGHT).toBeGreaterThan(0);
  });

  it('gives a single tile the whole page', () => {
    expect(fitGrid(1, 1.5)).toEqual({ columns: 1, rows: 1 });
  });

  it('honours a column cap', () => {
    const { columns, rows } = fitGrid(24, 21 / 9, { maxColumns: 4 });
    expect(columns).toBeLessThanOrEqual(4);
    expect(columns * rows).toBeGreaterThanOrEqual(24);
  });

  describe('degenerate input', () => {
    it('survives a container with no size', () => {
      // What a first paint and every jsdom test actually hand over: a 0x0 box,
      // whose width/height is NaN. Returning NaN columns would put
      // `repeat(NaN, ...)` into the stylesheet and drop the track list.
      for (const aspect of [Number.NaN, 0, -1, Number.POSITIVE_INFINITY]) {
        const { columns, rows } = fitGrid(11, aspect);
        expect(Number.isInteger(columns)).toBe(true);
        expect(Number.isInteger(rows)).toBe(true);
        expect(columns * rows).toBeGreaterThanOrEqual(11);
      }
    });

    it('survives a nonsense count', () => {
      for (const count of [0, -5, Number.NaN]) {
        expect(fitGrid(count, 1.5)).toEqual({ columns: 1, rows: 1 });
      }
    });

    it('floors a fractional count rather than emitting a fractional grid', () => {
      const { columns, rows } = fitGrid(11.7, 1);
      expect(Number.isInteger(columns)).toBe(true);
      expect(Number.isInteger(rows)).toBe(true);
      expect(columns * rows).toBeGreaterThanOrEqual(11);
    });
  });
});
