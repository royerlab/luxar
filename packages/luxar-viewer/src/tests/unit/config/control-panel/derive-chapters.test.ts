import { describe, expect, it } from 'vitest';

import {
  activeChapterIndex,
  deriveChapters,
  findChapterDimension,
  MAX_DERIVED_CHAPTERS,
  type ChapterDimensions,
} from '../../../../config/control-panel/derive-chapters';
import type { DimensionMetadata } from '../../../../types/dims';

/** A dimension with only the fields this module reads. */
function dim(name: string, extra: Partial<DimensionMetadata> = {}): DimensionMetadata {
  return { name, unit: '', scale: 1, ...extra };
}

/**
 * The shape of the two ESM tours: three displayed spatial axes plus a hidden,
 * labelled `story` axis.
 */
function tour(categories: string[]): ChapterDimensions {
  return {
    displayed: [0, 1, 2],
    metadata: [
      dim('x', { spatial: true }),
      dim('y', { spatial: true }),
      dim('z', { spatial: true }),
      dim('story', { discrete: true, step: 1, categories }),
    ],
    ranges: [
      [0, 100],
      [0, 100],
      [0, 100],
      [0, categories.length - 1],
    ],
  };
}

describe('findChapterDimension', () => {
  it('prefers a hidden categorical dimension', () => {
    expect(findChapterDimension(tour(['Overview', 'Haemoglobin']))).toBe(3);
  });

  it('never picks a displayed axis, however it is labelled', () => {
    // A labelled axis that is on screen is the scene, not its chapters.
    const dims: ChapterDimensions = {
      displayed: [0, 1],
      metadata: [dim('x'), dim('channel', { discrete: true, categories: ['DAPI', 'GFP'] })],
      ranges: [
        [0, 10],
        [0, 1],
      ],
    };
    expect(findChapterDimension(dims)).toBeNull();
  });

  it('falls back to a small unlabelled discrete dimension', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('view', { discrete: true, step: 1 })],
      ranges: [
        [0, 10],
        [0, 4],
      ],
    };
    expect(findChapterDimension(dims)).toBe(1);
  });

  it('refuses a long unlabelled axis rather than drawing hundreds of tiles', () => {
    // A 500-frame timelapse is not a tour, and five hundred tiles would be a
    // worse answer than none.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('time', { discrete: true, step: 1 })],
      ranges: [
        [0, 10],
        [0, 499],
      ],
    };
    expect(findChapterDimension(dims)).toBeNull();
  });

  it('takes a long axis when an author names it explicitly', () => {
    // An explicit choice outranks the cap: they have said they mean it.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('time', { discrete: true, step: 1 })],
      ranges: [
        [0, 10],
        [0, 499],
      ],
    };
    expect(findChapterDimension(dims, { dimensionName: 'time' })).toBe(1);
  });

  it('prefers the authored name over a categorical candidate', () => {
    const dims = tour(['Overview', 'A']);
    dims.metadata.push(dim('mode', { discrete: true, step: 1 }));
    dims.ranges.push([0, 2]);
    expect(findChapterDimension(dims, { dimensionName: 'mode' })).toBe(4);
  });

  it('returns null for an authored name the scene does not have', () => {
    // Silently falling back would hide the author's typo behind a menu that
    // drives the wrong axis.
    expect(findChapterDimension(tour(['A', 'B']), { dimensionName: 'stroy' })).toBeNull();
  });

  it('ignores a blank authored name and derives instead', () => {
    expect(findChapterDimension(tour(['A', 'B']), { dimensionName: '' })).toBe(3);
  });

  it('returns null when every dimension is on screen', () => {
    const dims: ChapterDimensions = {
      displayed: [0, 1, 2],
      metadata: [dim('x'), dim('y'), dim('z')],
      ranges: [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
    };
    expect(findChapterDimension(dims)).toBeNull();
  });

  it('refuses a hidden discrete axis with a single step', () => {
    // One tile is not a menu.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('only', { discrete: true, step: 1 })],
      ranges: [
        [0, 1],
        [0, 0],
      ],
    };
    expect(findChapterDimension(dims)).toBeNull();
  });

  it('refuses a hidden CONTINUOUS axis', () => {
    // A continuous axis has no steps to enumerate.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('pressure', { step: 0.5 })],
      ranges: [
        [0, 1],
        [0, 4],
      ],
    };
    expect(findChapterDimension(dims)).toBeNull();
  });
});

describe('deriveChapters', () => {
  it('enumerates a labelled tour, labels and all', () => {
    const source = deriveChapters(tour(['Overview', 'Haemoglobin', 'Hsp70']));
    expect(source).not.toBeNull();
    expect(source?.dimensionIndex).toBe(3);
    expect(source?.dimensionName).toBe('story');
    expect(source?.chapters).toEqual([
      { index: 0, value: 0, label: 'Overview', authored: true },
      { index: 1, value: 1, label: 'Haemoglobin', authored: true },
      { index: 2, value: 2, label: 'Hsp70', authored: true },
    ]);
  });

  it('generates labels for an unlabelled axis, and says they are generated', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('view', { discrete: true, step: 1, unit: 'deg' })],
      ranges: [
        [0, 2],
        [0, 2],
      ],
    };
    const source = deriveChapters(dims);
    expect(source?.chapters.map((c) => c.label)).toEqual([
      'view 0 deg',
      'view 1 deg',
      'view 2 deg',
    ]);
    expect(source?.chapters.every((c) => !c.authored)).toBe(true);
  });

  it('falls back per-step when categories are SHORTER than the range', () => {
    // The two are written independently; a short list must not print
    // "undefined" on an exhibit wall.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('story', { discrete: true, step: 1, categories: ['A', 'B'] })],
      ranges: [
        [0, 1],
        [0, 3],
      ],
    };
    const source = deriveChapters(dims);
    expect(source?.chapters.map((c) => c.label)).toEqual(['A', 'B', 'story 2', 'story 3']);
    expect(source?.chapters.map((c) => c.authored)).toEqual([true, true, false, false]);
  });

  it('treats an empty-string category as unauthored', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('story', { discrete: true, step: 1, categories: ['A', ''] })],
      ranges: [
        [0, 1],
        [0, 1],
      ],
    };
    expect(deriveChapters(dims)?.chapters[1]).toMatchObject({ label: 'story 1', authored: false });
  });

  it('honours a non-unit step and a non-zero minimum', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('t', { discrete: true, step: 5 })],
      ranges: [
        [0, 1],
        [10, 25],
      ],
    };
    expect(deriveChapters(dims)?.chapters.map((c) => c.value)).toEqual([10, 15, 20, 25]);
  });

  it('derives exactly the cap without refusing it', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('n', { discrete: true, step: 1 })],
      ranges: [
        [0, 1],
        [0, MAX_DERIVED_CHAPTERS - 1],
      ],
    };
    expect(deriveChapters(dims)?.chapters).toHaveLength(MAX_DERIVED_CHAPTERS);
  });

  it('returns null when there is no chapter dimension at all', () => {
    const dims: ChapterDimensions = {
      displayed: [0, 1, 2],
      metadata: [dim('x'), dim('y'), dim('z')],
      ranges: [
        [0, 1],
        [0, 1],
        [0, 1],
      ],
    };
    expect(deriveChapters(dims)).toBeNull();
  });

  it('survives metadata and ranges disagreeing in length', () => {
    // Two arrays built by two code paths; a truncated one must not throw.
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('story', { discrete: true, step: 1, categories: ['A', 'B'] })],
      ranges: [[0, 1]],
    };
    expect(deriveChapters(dims)).toBeNull();
  });

  it('refuses a non-finite range rather than looping forever', () => {
    const dims: ChapterDimensions = {
      displayed: [0],
      metadata: [dim('x'), dim('broken', { discrete: true, step: 1 })],
      ranges: [
        [0, 1],
        [0, Number.POSITIVE_INFINITY],
      ],
    };
    expect(deriveChapters(dims)).toBeNull();
  });
});

describe('activeChapterIndex', () => {
  const source = deriveChapters(tour(['Overview', 'A', 'B']))!;

  it('matches an exact position', () => {
    expect(activeChapterIndex(source, 2)).toBe(2);
  });

  it('matches the nearest step, since the viewer quantizes what it is given', () => {
    expect(activeChapterIndex(source, 1.2)).toBe(1);
    expect(activeChapterIndex(source, 0.4)).toBe(0);
  });

  it('reports no active chapter when the position is off the ladder', () => {
    expect(activeChapterIndex(source, 9)).toBe(-1);
    expect(activeChapterIndex(source, -4)).toBe(-1);
  });

  it('handles a single-chapter source without dividing by zero', () => {
    const single = deriveChapters(tour(['Only']))!;
    expect(activeChapterIndex(single, 0)).toBe(0);
    expect(activeChapterIndex(single, 7)).toBe(-1);
  });
});
