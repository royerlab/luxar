/**
 * The order in which a tile's text gives way. The title is what a visitor
 * taps, so every test here is a way of saying it is the last thing to lose
 * anything.
 */

import { describe, expect, it } from 'vitest';

import {
  FULL_TILE_TEXT,
  TILE_LABEL_SCALES,
  TILE_SUBLABEL_LINE_LIMITS,
  tileTextLadder,
  type TileTextContent,
  type TileTextStep,
} from '../../../../config/control-panel/tile-text-fit';

const EVERYTHING: TileTextContent = { sublabels: true, shortSublabels: true, shortLabels: true };

/** How much of the title a step keeps, coarsest last; higher is richer. */
function titleRichness(step: TileTextStep): number[] {
  return [step.label === 'full' ? 1 : 0, step.labelScale, step.labelWrap === 'words' ? 1 : 0];
}

describe('tileTextLadder', () => {
  it('starts from everything shown, at full size', () => {
    for (const content of [
      EVERYTHING,
      { sublabels: false, shortSublabels: false, shortLabels: false },
    ]) {
      expect(tileTextLadder(content)[0]).toEqual(FULL_TILE_TEXT);
    }
  });

  it('gives way in the documented order', () => {
    const changes = tileTextLadder(EVERYTHING)
      .slice(1)
      .map((step, i, rest) => {
        const before = i === 0 ? FULL_TILE_TEXT : rest[i - 1];
        return Object.keys(step).filter(
          (key) => step[key as keyof TileTextStep] !== before[key as keyof TileTextStep]
        );
      });
    expect(changes).toEqual([
      ['sublabel'],
      ...TILE_SUBLABEL_LINE_LIMITS.map(() => ['sublabelLines']),
      ['sublabel', 'sublabelLines'],
      ['index'],
      ['label'],
      ...TILE_LABEL_SCALES.map(() => ['labelScale']),
      ['labelWrap'],
    ]);
  });

  it('drops the whole sublabel before touching the title', () => {
    const ladder = tileTextLadder(EVERYTHING);
    const firstTitleLoss = ladder.findIndex((step) => step.label !== 'full' || step.labelScale < 1);
    const sublabelGone = ladder.findIndex((step) => step.sublabel === 'none');
    expect(sublabelGone).toBeGreaterThan(0);
    expect(firstTitleLoss).toBeGreaterThan(sublabelGone);
  });

  it('prefers the authored short label to a smaller copy of the long one', () => {
    const ladder = tileTextLadder(EVERYTHING);
    const short = ladder.findIndex((step) => step.label === 'short');
    const smaller = ladder.findIndex((step) => step.labelScale < 1);
    expect(short).toBeLessThan(smaller);
  });

  it('never gives anything back on the way down', () => {
    // Each step keeps what the one before it took away, so a step that fits
    // is never followed by one that needs MORE room.
    const ladder = tileTextLadder(EVERYTHING);
    for (const [i, step] of ladder.entries()) {
      if (i === 0) continue;
      const before = ladder[i - 1];
      expect(Number(step.index)).toBeLessThanOrEqual(Number(before.index));
      expect(step.labelScale).toBeLessThanOrEqual(before.labelScale);
      const lines = (s: TileTextStep): number =>
        s.sublabel === 'none' ? 0 : (s.sublabelLines ?? Number.POSITIVE_INFINITY);
      expect(lines(step)).toBeLessThanOrEqual(lines(before));
      const [label, scale, wrap] = titleRichness(step);
      const [labelBefore, scaleBefore, wrapBefore] = titleRichness(before);
      expect(label).toBeLessThanOrEqual(labelBefore);
      expect(scale).toBeLessThanOrEqual(scaleBefore);
      expect(wrap).toBeLessThanOrEqual(wrapBefore);
    }
  });

  it('ends with a label that may break inside a word, so nothing is clipped', () => {
    const last = tileTextLadder(EVERYTHING).at(-1);
    expect(last?.labelWrap).toBe('anywhere');
    expect(last?.labelScale).toBe(Math.min(...TILE_LABEL_SCALES));
  });

  it('leaves out steps that would change nothing', () => {
    // Every step the renderer measures costs a layout, so a no-op is waste.
    const bare = tileTextLadder({ sublabels: false, shortSublabels: false, shortLabels: false });
    expect(bare.some((step) => step.sublabel !== 'full' || step.sublabelLines !== null)).toBe(
      false
    );
    expect(bare.some((step) => step.label === 'short')).toBe(false);
    expect(bare).toHaveLength(1 + 1 + TILE_LABEL_SCALES.length + 1);

    const noShort = tileTextLadder({ sublabels: true, shortSublabels: false, shortLabels: false });
    expect(noShort.some((step) => step.sublabel === 'short' || step.label === 'short')).toBe(false);
  });
});
