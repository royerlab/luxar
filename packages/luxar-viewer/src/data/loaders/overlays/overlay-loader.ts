/**
 * Overlay Loader - Reads overlay configurations from zarr store.
 *
 * Enumerates the `overlays/` group in the zarr scene and extracts
 * overlay metadata from each child group's .zattrs.
 */

import * as zarr from '../../zarr';
import { hasContentsMethod } from '../../../types/zarr';
import { log, Modules } from '../../../utils/log';

/**
 * Maximum allowed length, in CHARACTERS, of an overlay's raw `html` value.
 *
 * The browser HTML parser (`template.innerHTML = ...` in
 * `OverlayManager.sanitizeHtml`) runs synchronously on the main thread and is
 * superlinear in nesting depth, so a large deeply-nested `html` string can hang
 * the tab (measured ~1.5 s at 220 KB, ~6 s at 440 KB). Every tag can be an
 * allowlisted one, so sanitizer hardening cannot help — the ONLY effective bound
 * is a cap on the raw input size BEFORE it reaches the parser. The largest
 * legitimate overlay is a few hundred bytes, so 64 KiB is a very generous ceiling.
 * Oversized values are rejected outright (never truncated — a mid-markup cut can
 * yield its own broken, deeply-nested fragment).
 */
export const MAX_OVERLAY_HTML_CHARS = 64 * 1024;

/** Configuration for a single overlay, as stored in zarr .zattrs */
export interface OverlayConfig {
  /** Overlay name (zarr group name) */
  name: string;
  /** Overlay type: 'overlay_text', 'overlay_image', 'overlay_video', 'overlay_html' */
  type: 'overlay_text' | 'overlay_image' | 'overlay_video' | 'overlay_html';
  /** Position in normalized screen coords [0, 1], top-left origin */
  position: [number, number];
  /** Opacity 0-1 */
  opacity: number;
  /** Anchor point for positioning */
  anchor: string;
  /** Dimension-based visibility filter */
  visible_range?: Record<string, number | [number, number]>;
  /** Transition type: 'none' or 'fade' */
  transition: string;
  /** Transition duration in seconds */
  transition_duration: number;
  /** Whether overlay captures pointer events */
  interactive: boolean;
  /** Z-ordering index (insertion order) */
  z_index: number;

  /** Whether this overlay acts as a hover tooltip (content updated by GPU picking) */
  hover?: boolean;
  /** Size of hover image thumbnails as [width, height] in viewport fractions */
  hover_image_size?: [number, number];

  // --- Text-specific ---
  text?: string;
  font_size?: number;
  font?: string;
  color?: string;
  width?: number;
  text_align?: string;
  line_height?: number;
  background?: string;
  padding?: number;
  stroke_color?: string;
  stroke_width?: number;

  /** CSS mix-blend-mode (e.g. 'difference', 'screen', 'multiply') */
  blend_mode?: string;

  // --- Image-specific ---
  image_file?: string;
  /** [width, height] as viewport fractions; a null height keeps the media's own aspect. */
  size?: [number, number | null];

  // --- Video-specific ---
  /** Opaque file beside the group: `video.webm` (VP9, optionally with alpha) or `video.mp4`. */
  video_file?: string;
  /** Optional still (PNG/JPEG/WebP) shown before play and where the video cannot decode. */
  poster_file?: string;
  loop?: boolean;
  autoplay?: boolean;
  muted?: boolean;
  playback_rate?: number;

  // --- HTML-specific ---
  html?: string;
}

/**
 * Load overlay configurations from the zarr store.
 *
 * Looks for an `overlays/` group in the root, enumerates its children,
 * and returns their .zattrs as OverlayConfig objects sorted by z_index.
 *
 * @param store - The zarr readable store
 * @param rootLoc - Root location in the zarr store
 * @returns Array of overlay configs, sorted by z_index (ascending)
 */
export async function loadOverlayConfigs(
  store: zarr.Readable,
  rootLoc: zarr.Location<zarr.Readable>
): Promise<OverlayConfig[]> {
  const configs: OverlayConfig[] = [];

  try {
    // Try to open the overlays group (validates it exists)
    const overlaysLoc = rootLoc.resolve('overlays');
    await zarr.open(overlaysLoc, { kind: 'group' });

    // Enumerate children of the overlays group
    // We need to list the store to find overlay subgroups
    const listing = await listGroupChildren(store, 'overlays');

    for (const childName of listing) {
      try {
        const childLoc = rootLoc.resolve(`overlays/${childName}`);
        const childGroup = await zarr.open(childLoc, { kind: 'group' });
        const attrs = childGroup.attrs as Record<string, unknown>;

        if (typeof attrs?.type !== 'string' || !attrs.type.startsWith('overlay_')) {
          continue;
        }

        const config: OverlayConfig = {
          name: childName,
          type: attrs.type as OverlayConfig['type'],
          position: (attrs.position as [number, number]) ?? [0, 0],
          opacity: (attrs.opacity as number) ?? 1.0,
          anchor: (attrs.anchor as string) ?? 'top-left',
          visible_range: attrs.visible_range as OverlayConfig['visible_range'],
          transition: (attrs.transition as string) ?? 'none',
          transition_duration: (attrs.transition_duration as number) ?? 0.3,
          interactive: (attrs.interactive as boolean) ?? false,
          z_index: (attrs.z_index as number) ?? 0,
          hover: (attrs.hover as boolean) ?? false,
          hover_image_size: attrs.hover_image_size as [number, number] | undefined,

          // Type-specific (only present for matching types)
          text: attrs.text as string | undefined,
          font_size: attrs.font_size as number | undefined,
          font: attrs.font as string | undefined,
          color: attrs.color as string | undefined,
          width: attrs.width as number | undefined,
          text_align: attrs.text_align as string | undefined,
          line_height: attrs.line_height as number | undefined,
          background: attrs.background as string | undefined,
          padding: attrs.padding as number | undefined,
          stroke_color: attrs.stroke_color as string | undefined,
          stroke_width: attrs.stroke_width as number | undefined,
          image_file: attrs.image_file as string | undefined,
          size: attrs.size as [number, number | null] | undefined,
          blend_mode: attrs.blend_mode as string | undefined,
          html: attrs.html as string | undefined,
          video_file: attrs.video_file as string | undefined,
          poster_file: attrs.poster_file as string | undefined,
          loop: attrs.loop as boolean | undefined,
          autoplay: attrs.autoplay as boolean | undefined,
          muted: attrs.muted as boolean | undefined,
          playback_rate: attrs.playback_rate as number | undefined,
        };

        // `.zattrs` is untrusted JSON and the assignments above are only type
        // CASTS, so at runtime `html`/`text` can be any JSON value. A non-string
        // must be dropped, not just size-checked: an array like ["<huge...>"]
        // has .length 1 (passing the caps below) yet `innerHTML = value` coerces
        // it to the full payload string — bypassing the cap entirely.
        if (config.html !== undefined && typeof config.html !== 'string') {
          log.warning(
            Modules.SCENE_LOADER,
            `Overlay "${childName}" html is not a string (${typeof config.html}) — dropping it`
          );
          config.html = undefined;
        }
        if (config.text !== undefined && typeof config.text !== 'string') {
          log.warning(
            Modules.SCENE_LOADER,
            `Overlay "${childName}" text is not a string (${typeof config.text}) — dropping it`
          );
          config.text = undefined;
        }

        // Reject an oversized `html` value before it can reach the DOM parser.
        // Parse cost is superlinear in nesting depth and this runs on the main
        // thread; the only effective bound is on the raw input size. See
        // MAX_OVERLAY_HTML_CHARS.
        if (config.html !== undefined && config.html.length > MAX_OVERLAY_HTML_CHARS) {
          log.warning(
            Modules.SCENE_LOADER,
            `Overlay "${childName}" html is ${config.html.length} chars, exceeding the ` +
              `${MAX_OVERLAY_HTML_CHARS}-char limit — dropping html to protect the main thread`
          );
          config.html = undefined;
        }

        // Apply the same cap to `text` — but only for overlay_html: there `text`
        // is consumed as the hover template (`config.html ?? config.text`) and
        // reaches the DOM via innerHTML through sanitizeHtml, so it carries the
        // identical parse-cost threat. Other overlay types render `text` via
        // textContent (linear, never parsed), so their content is left alone.
        if (
          config.type === 'overlay_html' &&
          config.text !== undefined &&
          config.text.length > MAX_OVERLAY_HTML_CHARS
        ) {
          log.warning(
            Modules.SCENE_LOADER,
            `Overlay "${childName}" text is ${config.text.length} chars, exceeding the ` +
              `${MAX_OVERLAY_HTML_CHARS}-char limit — dropping text to protect the main thread`
          );
          config.text = undefined;
        }

        configs.push(config);
      } catch {
        log.warning(Modules.SCENE_LOADER, `Failed to load overlay: ${childName}`);
      }
    }
  } catch {
    // No overlays group — that's fine, scenes don't have to have overlays
    return [];
  }

  // Sort by z_index (insertion order)
  configs.sort((a, b) => a.z_index - b.z_index);

  if (configs.length > 0) {
    log.info(Modules.UI, `Loaded ${configs.length} overlay(s) from zarr`);
  }

  return configs;
}

/**
 * Zarr's own metadata documents, which are store KEYS but not child nodes.
 *
 * Only the `list()` fallback below needs these: it enumerates raw keys, so a
 * document sitting beside the real children would be reported as an overlay
 * named after it. Both formats are covered — naming only the format-2 pair
 * (the previous filter) lets a format-3 store's `zarr.json` through as a
 * phantom overlay. The `contents()` path lists nodes, not keys, and is unaffected.
 */
const METADATA_DOCS = new Set(['zarr.json', '.zgroup', '.zattrs', '.zarray', '.zmetadata']);

/**
 * List immediate child group names under a given path in the store.
 *
 * Uses the consolidated metadata `contents()` method — the same approach used
 * by SceneLoader.enumerateStore(). This reliably finds all overlay names
 * including custom-named overlays (not just auto-generated ones).
 */
async function listGroupChildren(store: zarr.Readable, parentPath: string): Promise<string[]> {
  const prefix = parentPath + '/';

  // Primary method: consolidated metadata (same pattern as SceneLoader.enumerateStore)
  if (hasContentsMethod(store)) {
    const contents = await store.contents();
    const seen = new Set<string>();
    const children: string[] = [];

    for (const entry of contents) {
      const path = typeof entry === 'string' ? entry : entry.path;
      if (path.startsWith('/' + prefix) || path.startsWith(prefix)) {
        // Normalize: strip leading '/' if present, then strip the prefix
        const normalized = path.startsWith('/') ? path.slice(1) : path;
        const relative = normalized.slice(prefix.length);
        const childName = relative.split('/')[0];
        if (childName && !seen.has(childName)) {
          seen.add(childName);
          children.push(childName);
        }
      }
    }
    return children;
  }

  // Fallback: try the store's list method if available
  const storeWithList = store as { list?: (prefix: string) => Promise<unknown[]> };
  if (typeof storeWithList.list === 'function') {
    try {
      const listing = await storeWithList.list(parentPath + '/');
      const seen = new Set<string>();
      const children: string[] = [];
      for (const item of listing) {
        const key =
          typeof item === 'string'
            ? item
            : ((item as { key?: string; path?: string }).key ??
              (item as { key?: string; path?: string }).path ??
              '');
        const relative = key.startsWith(prefix) ? key.slice(prefix.length) : key;
        const childName = relative.split('/')[0];
        if (childName && !METADATA_DOCS.has(childName) && !seen.has(childName)) {
          seen.add(childName);
          children.push(childName);
        }
      }
      return children;
    } catch {
      // Fall through
    }
  }

  log.warning(
    Modules.UI,
    'Cannot enumerate overlay children: store has no contents() or list() method'
  );
  return [];
}
