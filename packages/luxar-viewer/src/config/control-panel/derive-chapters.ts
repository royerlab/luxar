/**
 * Deriving a tour's chapters from the scene's own dimension metadata.
 *
 * A guided tour in Luxar is a hidden discrete dimension plus one authored
 * `Waypoint` per stop, and a labelled dimension already carries the stop names:
 * `Dimension(categories=[...])` reaches the viewer as
 * `DimensionMetadata.categories` and is handed out by `getDimensions()`. So a
 * control panel needs **no authored block at all** to draw a usable menu — it
 * asks the viewer what dimensions exist and reads the labels that are already
 * there. That is what keeps the panel generic rather than demo furniture.
 *
 * This module is deliberately pure and transport-free: a controller feeds it an
 * `EmbedderDimensions` snapshot and gets back chapters. It lives in `config`
 * rather than beside the panel's DOM because `ui` may not import `core`, and
 * because a future in-viewer picker should share exactly this logic.
 *
 * `src/ui/dimension-sliders.ts` already chooses a control shape from the
 * category count (2 → toggle, 3–9 → dropdown, ≥10 or unlabelled → slider). A
 * tile grid is the fourth arm of that same taxonomy, and if that inline
 * decision is ever extracted, it should land here.
 */

import type { DimensionMetadata } from '../../types/dims';

/** The dimension snapshot this module reads. Mirrors `EmbedderDimensions`. */
export interface ChapterDimensions {
  /** Indices currently shown on screen. */
  displayed: number[];
  /** Per-dimension metadata, including `categories` when authored. */
  metadata: DimensionMetadata[];
  /** `[min, max]` navigable bounds per dimension. */
  ranges: Array<[number, number]>;
}

/** One tile: a value to set on the chapter dimension, and what to call it. */
export interface DerivedChapter {
  /** Position in the list, 0-based. Stable for as long as the scene is. */
  index: number;
  /** The value to hand `setDimensionValue`. */
  value: number;
  /** Human-readable label — an authored category, or a generated fallback. */
  label: string;
  /** Whether `label` came from the scene rather than being generated. */
  authored: boolean;
}

/** The chapter dimension and its stops. */
export interface ChapterSource {
  /** Positional index, for `setDimensionValue`. */
  dimensionIndex: number;
  /** The dimension's name, for display and for a controller to log. */
  dimensionName: string;
  chapters: DerivedChapter[];
}

/**
 * Most stops a menu will derive on its own.
 *
 * A tour has a dozen; a timelapse's `time` axis has five hundred, and drawing
 * five hundred tiles would be a worse answer than drawing none. An *unlabelled*
 * discrete dimension past this count is therefore not treated as a chapter
 * dimension unless an author names it explicitly — at which point they have
 * said they mean it, and the cap no longer applies.
 */
export const MAX_DERIVED_CHAPTERS = 24;

interface DeriveOptions {
  /**
   * Dimension to walk, by name. When given and resolvable it wins outright,
   * including past {@link MAX_DERIVED_CHAPTERS} — an explicit choice is an
   * explicit choice. Comes from the authored control-panel block.
   */
  dimensionName?: string | null;
}

function isCategorical(meta: DimensionMetadata | undefined): boolean {
  return Array.isArray(meta?.categories) && meta.categories.length > 0;
}

/** Step count for a dimension, from its navigable range and step size. */
function stepCount(dims: ChapterDimensions, index: number): number {
  const range = dims.ranges[index];
  if (!range) return 0;
  const [min, max] = range;
  const step = dims.metadata[index]?.step ?? 1;
  if (!Number.isFinite(min) || !Number.isFinite(max) || !(step > 0)) return 0;
  return Math.floor((max - min) / step + 1e-9) + 1;
}

/**
 * Which dimension the chapters walk.
 *
 * In order: the authored name; the first non-displayed **categorical**
 * dimension (a labelled axis is an author saying "these are named states");
 * then the first non-displayed discrete one small enough to be a menu. Spatial
 * and displayed axes are never candidates — those are the scene, not its
 * chapters.
 */
export function findChapterDimension(
  dims: ChapterDimensions,
  options: DeriveOptions = {}
): number | null {
  const named = options.dimensionName;
  if (named !== undefined && named !== null && named.length > 0) {
    const index = dims.metadata.findIndex((meta) => meta.name === named);
    return index === -1 ? null : index;
  }
  const displayed = new Set(dims.displayed);
  const candidates = dims.metadata
    .map((_meta, index) => index)
    .filter((index) => !displayed.has(index));

  const categorical = candidates.find((index) => isCategorical(dims.metadata[index]));
  if (categorical !== undefined) return categorical;

  const discrete = candidates.find((index) => {
    const count = stepCount(dims, index);
    return dims.metadata[index]?.discrete === true && count > 1 && count <= MAX_DERIVED_CHAPTERS;
  });
  return discrete ?? null;
}

/** The label for one step: an authored category, else `"<name> <n>"`. */
function labelFor(
  meta: DimensionMetadata,
  value: number,
  ordinal: number
): { label: string; authored: boolean } {
  const categories = meta.categories;
  if (Array.isArray(categories)) {
    // Categorical ranges start at 0, so the value IS the category index — but
    // `categories` can be shorter than the range the scene declares, and
    // indexing past the end must not produce "undefined" on a kiosk wall.
    const authored = categories[Math.round(value)];
    if (typeof authored === 'string' && authored.length > 0) {
      return { label: authored, authored: true };
    }
  }
  const unit = meta.unit ? ` ${meta.unit}` : '';
  return { label: `${meta.name} ${ordinal}${unit}`, authored: false };
}

/**
 * Derive the chapter menu, or `null` when this scene has no chapter dimension.
 *
 * `null` is a legitimate, common answer — most scenes are not tours — and the
 * caller's job is to say so plainly rather than draw an empty grid.
 */
export function deriveChapters(
  dims: ChapterDimensions,
  options: DeriveOptions = {}
): ChapterSource | null {
  const dimensionIndex = findChapterDimension(dims, options);
  if (dimensionIndex === null) return null;
  const meta = dims.metadata[dimensionIndex];
  const range = dims.ranges[dimensionIndex];
  if (!meta || !range) return null;

  const [min] = range;
  const step = meta.step ?? 1;
  const count = stepCount(dims, dimensionIndex);
  if (count < 1) return null;

  const chapters: DerivedChapter[] = [];
  for (let ordinal = 0; ordinal < count; ordinal += 1) {
    const value = min + ordinal * step;
    const { label, authored } = labelFor(meta, value, ordinal);
    chapters.push({ index: ordinal, value, label, authored });
  }
  return { dimensionIndex, dimensionName: meta.name, chapters };
}

/**
 * Which chapter a live dimension position corresponds to, or `-1`.
 *
 * Used to mark the active tile from a `dimensions-changed` event. Matches on
 * the nearest step rather than equality: the viewer quantizes and clamps what
 * it is given, so the value that comes back is not always the one sent.
 */
export function activeChapterIndex(source: ChapterSource, position: number): number {
  let best = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const chapter of source.chapters) {
    const distance = Math.abs(chapter.value - position);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = chapter.index;
    }
  }
  // Half a step is the widest a quantized answer can legitimately land from
  // the value that produced it; further than that is a different position.
  const step = Math.abs(source.chapters[1]?.value - source.chapters[0]?.value) || 1;
  return bestDistance <= step / 2 + 1e-9 ? best : -1;
}
