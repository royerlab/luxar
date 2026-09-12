/**
 * The authored `viewer_config.control_panel` block, read off a store.
 *
 * Everything here is optional and everything unset means "derive it". That is
 * the design, not laziness: a scene with no block at all still gets a working
 * touch panel, because the chapters come from a discrete dimension's
 * `categories` (`config/control-panel/derive-chapters.ts`). This block is
 * enrichment layered on top.
 *
 * Written by `luxar.core.viewer_config.ControlPanelConfig`, whose field names
 * are snake_case in the store and camelCase here — the same convention the
 * rest of the zarr bridge follows.
 *
 * Every value arriving here is untrusted input. A store is a file someone
 * downloaded, so each field is shape-checked and out-of-range values are
 * dropped rather than clamped: a dropped field falls back to the derived
 * default, which is always sane, whereas a clamped one silently pretends the
 * author asked for something they did not.
 */

/** Cap on the authored stylesheet, mirroring the Python writer's. */
export const MAX_CONTROL_STYLESHEET_CHARS = 64 * 1024;

/** Widest grid an author may pin. Past this the cells are sub-fingertip. */
export const MAX_CONTROL_COLUMNS = 12;

/** Per-chapter overrides, keyed by the chapter's position in the tour. */
export interface ControlChapterOverride {
  label?: string;
  sublabel?: string;
}

export interface ControlPanelSettings {
  title?: string;
  subtitle?: string;
  /** Dimension the tiles walk, by NAME. Resolved by `deriveChapters`. */
  chapterDimension?: string;
  columns?: number;
  /** Seconds of no touch before returning to the first chapter; 0 disables. */
  idleResetS?: number;
  /** Author CSS, already stripped of the constructs listed below. */
  stylesheet?: string;
  chapters?: Record<number, ControlChapterOverride>;
}

function trimmedString(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  const trimmed = value.trim();
  return trimmed === '' ? undefined : trimmed;
}

function boundedInteger(value: unknown, min: number, max: number): number | undefined {
  if (typeof value !== 'number' || !Number.isInteger(value)) return undefined;
  return value >= min && value <= max ? value : undefined;
}

function boundedNumber(value: unknown, min: number, max: number): number | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined;
  return value >= min && value <= max ? value : undefined;
}

/**
 * Strip the constructs an authored stylesheet may not use.
 *
 * Author CSS is store-supplied, which puts it on exactly the same footing as
 * `overlay_html`: a file a visitor downloaded, applied to a page that holds a
 * live control socket. Two things are removed, and neither is about taste:
 *
 * - `@import` — it fetches a second stylesheet from anywhere, so the cap on
 *   this one would mean nothing and the panel would make a request the author
 *   never declared.
 * - remote `url(...)` — same reasoning, plus every such URL is a beacon that
 *   reports the kiosk's existence and IP to whoever hosts it. `data:` URIs
 *   survive, so an author can still inline a background.
 *
 * Stripped rather than rejected: a panel that refuses to style itself over one
 * bad rule is worse on an exhibit floor than one that drops the rule. The
 * result is injected through a `<style>` element's `textContent`, never
 * `innerHTML`, so it cannot smuggle markup either way.
 */
export function sanitizeAuthorStylesheet(css: string): string {
  return (
    css
      // Remote url() FIRST, and the order is load-bearing: it neutralises the
      // URL inside any `@import` the next pass somehow fails to remove, so a
      // survivor becomes `@import none` and fetches nothing. Quoted or bare;
      // `data:` survives so an author can still inline a background.
      .replace(/url\(\s*(['"]?)(?!data:)[^)'"]*\1\s*\)/gi, 'none')
      // `@import url(...)`, `@import "..."`, with or without media queries.
      //
      // Bounded by `;`, `{` AND a newline. `[^;]*` alone was wrong: an
      // `@import` written without its semicolon ran on to the next `;` and
      // swallowed the following rule whole, so an author lost styling to a
      // sanitiser meant to leave their CSS alone.
      .replace(/@import\b[^;{\r\n]*;?/gi, '')
  );
}

/** True for a JSON object — not an array, not null. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** One chapter's overrides, or `null` when it carries nothing usable. */
function chapterOverride(value: unknown): ControlChapterOverride | null {
  if (!isPlainObject(value)) return null;
  const label = trimmedString(value.label);
  const sublabel = trimmedString(value.sublabel);
  if (label === undefined && sublabel === undefined) return null;
  const override: ControlChapterOverride = {};
  if (label !== undefined) override.label = label;
  if (sublabel !== undefined) override.sublabel = sublabel;
  return override;
}

function extractChapters(raw: unknown): Record<number, ControlChapterOverride> | undefined {
  if (!isPlainObject(raw)) return undefined;
  const out: Record<number, ControlChapterOverride> = {};
  for (const [key, value] of Object.entries(raw)) {
    // JSON has no integer keys, so the writer emits strings. A key that is not
    // a non-negative integer is not a chapter position.
    if (!/^\d+$/.test(key)) continue;
    const override = chapterOverride(value);
    if (override !== null) out[Number(key)] = override;
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/** The three free-text fields, assigned only when present and non-blank. */
function readTextFields(source: Record<string, unknown>, settings: ControlPanelSettings): void {
  const title = trimmedString(source.title);
  if (title !== undefined) settings.title = title;
  const subtitle = trimmedString(source.subtitle);
  if (subtitle !== undefined) settings.subtitle = subtitle;
  const dimension = trimmedString(source.chapter_dimension);
  if (dimension !== undefined) settings.chapterDimension = dimension;
}

/** The two numeric fields, dropped rather than clamped when out of range. */
function readNumericFields(source: Record<string, unknown>, settings: ControlPanelSettings): void {
  const columns = boundedInteger(source.columns, 1, MAX_CONTROL_COLUMNS);
  if (columns !== undefined) settings.columns = columns;
  const idle = boundedNumber(source.idle_reset_s, 0, 86_400);
  if (idle !== undefined) settings.idleResetS = idle;
}

/**
 * The authored stylesheet, sanitised, or `undefined`.
 *
 * Over the cap it is dropped whole rather than truncated: truncation would
 * leave a half-written rule, and the writer already enforces the same cap.
 */
function readStylesheet(source: Record<string, unknown>): string | undefined {
  const css = trimmedString(source.stylesheet);
  if (css === undefined || css.length > MAX_CONTROL_STYLESHEET_CHARS) return undefined;
  const sanitized = sanitizeAuthorStylesheet(css);
  return sanitized.trim() === '' ? undefined : sanitized;
}

/**
 * Read the authored block, or `null` when the scene has none.
 *
 * `null` and an empty block are deliberately the same answer: both mean
 * "derive everything", and a caller should not have to tell them apart.
 */
export function extractControlPanelConfig(raw: unknown): ControlPanelSettings | null {
  if (!isPlainObject(raw)) return null;
  const settings: ControlPanelSettings = {};
  readTextFields(raw, settings);
  readNumericFields(raw, settings);
  const stylesheet = readStylesheet(raw);
  if (stylesheet !== undefined) settings.stylesheet = stylesheet;
  const chapters = extractChapters(raw.chapters);
  if (chapters !== undefined) settings.chapters = chapters;
  return Object.keys(settings).length > 0 ? settings : null;
}
