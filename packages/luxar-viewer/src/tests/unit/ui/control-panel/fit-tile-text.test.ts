// @vitest-environment jsdom
/**
 * Walking the tile-text ladder against a measured layout.
 *
 * jsdom lays nothing out, so each test gives the tiles a small fake layout: a
 * tile's content size is a function of what the grid's attributes currently
 * say to show. That is exactly the contract between the stylesheet and
 * `fitTileText` — attributes in, overflow out — without a browser.
 */

import { afterEach, describe, expect, it } from 'vitest';

import { tileTextLadder } from '../../../../config/control-panel/tile-text-fit';
import {
  applyTileTextStep,
  CONTROL_LABEL_SCALE_PROPERTY,
  CONTROL_SUBLABEL_LINES_PROPERTY,
  fitTileText,
  tilesFit,
} from '../../../../ui/control-panel/fit-tile-text';

const LADDER = tileTextLadder({ sublabels: true, shortSublabels: true, shortLabels: true });

/** A grid of `count` tiles, each `height` tall and `width` wide. */
function gridOf(count: number, width = 100, height = 80): HTMLElement {
  const grid = document.createElement('div');
  grid.className = 'luxar-control-grid';
  for (let i = 0; i < count; i += 1) {
    const tile = document.createElement('button');
    tile.className = 'luxar-control-tile';
    Object.defineProperty(tile, 'clientHeight', { value: height });
    Object.defineProperty(tile, 'clientWidth', { value: width });
    grid.append(tile);
  }
  document.body.replaceChildren(grid);
  return grid;
}

/** Make every tile's content size follow `layout(grid)`. */
function fakeLayout(
  grid: HTMLElement,
  layout: (grid: HTMLElement, tile: HTMLElement) => { width: number; height: number }
): void {
  for (const tile of Array.from(grid.querySelectorAll<HTMLElement>('.luxar-control-tile'))) {
    Object.defineProperty(tile, 'scrollHeight', { get: () => layout(grid, tile).height });
    Object.defineProperty(tile, 'scrollWidth', { get: () => layout(grid, tile).width });
  }
}

afterEach(() => document.body.replaceChildren());

describe('applyTileTextStep', () => {
  it('writes every field of the step onto the grid', () => {
    const grid = gridOf(1);
    applyTileTextStep(grid, {
      label: 'short',
      sublabel: 'none',
      sublabelLines: 2,
      index: false,
      labelScale: 0.9,
      labelWrap: 'anywhere',
    });
    expect(grid.dataset).toMatchObject({
      labelText: 'short',
      sublabelText: 'none',
      tileIndex: 'hidden',
      labelWrap: 'anywhere',
    });
    expect(grid.style.getPropertyValue(CONTROL_SUBLABEL_LINES_PROPERTY)).toBe('2');
    expect(grid.style.getPropertyValue(CONTROL_LABEL_SCALE_PROPERTY)).toBe('0.9');
  });

  it('spells an unlimited sublabel as `none`, which line-clamp accepts', () => {
    const grid = gridOf(1);
    applyTileTextStep(grid, LADDER[0]);
    expect(grid.style.getPropertyValue(CONTROL_SUBLABEL_LINES_PROPERTY)).toBe('none');
  });
});

describe('fitTileText', () => {
  it('keeps everything when everything fits', () => {
    const grid = gridOf(4);
    fakeLayout(grid, () => ({ width: 100, height: 60 }));
    expect(fitTileText(grid, LADDER)).toBe(0);
    expect(grid.dataset.sublabelText).toBe('full');
    expect(grid.dataset.tileIndex).toBe('shown');
  });

  it('stops at the first step where every tile fits', () => {
    // Only dropping the sublabel makes room.
    const grid = gridOf(3);
    fakeLayout(grid, (g) => ({ width: 100, height: g.dataset.sublabelText === 'none' ? 70 : 120 }));
    const position = fitTileText(grid, LADDER);
    expect(LADDER[position].sublabel).toBe('none');
    expect(LADDER[position - 1].sublabel).not.toBe('none');
    // ... and nothing further down the ladder was taken.
    expect(grid.dataset.tileIndex).toBe('shown');
    expect(grid.dataset.labelText).toBe('full');
  });

  it('lets the longest tile decide for the whole grid', () => {
    // One tile in three needs the short sublabel; all three get the step,
    // because a matrix of mixed presentations reads as broken.
    const grid = gridOf(3);
    const tall = grid.children[1] as HTMLElement;
    fakeLayout(grid, (g, tile) => ({
      width: 100,
      height: tile === tall && g.dataset.sublabelText === 'full' ? 95 : 70,
    }));
    expect(LADDER[fitTileText(grid, LADDER)].sublabel).toBe('short');
    expect(grid.dataset.sublabelText).toBe('short');
  });

  it('counts a word wider than its tile as not fitting', () => {
    // Labels break only between words until the last step, so a long word
    // shows up as WIDTH; it is answered by shrinking the label.
    const grid = gridOf(2);
    fakeLayout(grid, (g) => ({
      width: Number(g.style.getPropertyValue(CONTROL_LABEL_SCALE_PROPERTY)) <= 0.9 ? 100 : 104,
      height: 50,
    }));
    const position = fitTileText(grid, LADDER);
    expect(LADDER[position].labelScale).toBe(0.9);
    expect(grid.dataset.labelWrap).toBe('words');
  });

  it('settles on the last step when nothing fits', () => {
    const grid = gridOf(2);
    fakeLayout(grid, () => ({ width: 100, height: 500 }));
    expect(fitTileText(grid, LADDER)).toBe(LADDER.length - 1);
    expect(grid.dataset.labelWrap).toBe('anywhere');
  });

  it('tolerates the one-pixel rounding of an exact fit', () => {
    const grid = gridOf(1, 100, 80);
    fakeLayout(grid, () => ({ width: 101, height: 81 }));
    expect(tilesFit(grid)).toBe(true);
    fakeLayout(gridOf(1, 100, 80), () => ({ width: 100, height: 82 }));
    expect(tilesFit(document.querySelector<HTMLElement>('.luxar-control-grid')!)).toBe(false);
  });
});
