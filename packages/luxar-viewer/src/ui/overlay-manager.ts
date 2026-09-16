/**
 * Overlay Manager - Renders screen-space overlays (text, image, HTML, video) over the canvas.
 *
 * Creates an HTML overlay layer between the WebGL canvas and UI controls.
 * Supports dimension-aware visibility (overlays show/hide based on slider positions),
 * CSS transitions, blend modes, and configurable interaction.
 */

import { isZippedStoreUrl } from '../data/zip/entries';
import { sceneDimsManager } from '../scene/scene-dims-manager';
import { log, Modules } from '../utils/log';
import { getViewerContainer } from '../utils/viewer-container';
import { escapeHtml } from '../utils/escape-html';
import { substituteHoverTemplate } from '../utils/hover-template';
import { detectMimeType } from '../utils/image-mime';
import { createVideoMatteCompositor, type VideoMatteCompositor } from './video-matte';
import { MAX_OVERLAY_HTML_CHARS, type OverlayConfig } from '../data/loaders';

/** Read one opaque file from the active scene store. */
export type OverlayFileReader = (path: string) => Promise<Uint8Array | undefined>;

/** Archived binary payloads used to construct an overlay. */
export interface ArchivedOverlayMedia {
  primary?: Uint8Array;
  poster?: Uint8Array;
}

function isOverlayInteractive(config: OverlayConfig): boolean {
  if (config.interactive) return true;
  return config.type === 'overlay_video' && config.autoplay === false;
}

/** Font preset mappings to CSS font-family stacks */
export const FONT_PRESETS: Record<string, string> = {
  sans: 'system-ui, -apple-system, sans-serif',
  serif: 'Georgia, Times, serif',
  mono: 'ui-monospace, monospace',
};

/** Blend mode mapping from Luxar names to CSS mix-blend-mode values */
const BLEND_MODE_MAP: Record<string, string> = {
  normal: 'normal',
  multiply: 'multiply',
  screen: 'screen',
  overlay: 'overlay',
  additive: 'plus-lighter',
  difference: 'difference',
};

/** Anchor to CSS transform mapping for positioning offset */
const ANCHOR_TRANSFORM: Record<string, string> = {
  'top-left': 'translate(0, 0)',
  'top-center': 'translate(-50%, 0)',
  'top-right': 'translate(-100%, 0)',
  'center-left': 'translate(0, -50%)',
  center: 'translate(-50%, -50%)',
  'center-right': 'translate(-100%, -50%)',
  'bottom-left': 'translate(0, -100%)',
  'bottom-center': 'translate(-50%, -100%)',
  'bottom-right': 'translate(-100%, -100%)',
};

const CENTER_ANCHORS = new Set(['top-center', 'center', 'bottom-center']);

/**
 * Transform for RIGHT-side anchors when the element is positioned from the
 * container's RIGHT edge (via `right:` instead of `left:`).
 *
 * The horizontal `-100%` of {@link ANCHOR_TRANSFORM} is dropped — anchoring
 * from the right edge already places the box's right side, so the element's
 * shrink-to-fit width is measured against the container's LEFT edge and can
 * grow leftward. The vertical component is preserved. See issue #773: with
 * `left: 98%` + `translate(-100%, 0)` the browser computes the available
 * width from the PRE-transform position (~2% of the viewport), collapsing a
 * wrapping overlay (e.g. the hover tooltip) to one word per line.
 */
const RIGHT_ANCHOR_TRANSFORM: Record<string, string> = {
  'top-right': 'translate(0, 0)',
  'center-right': 'translate(0, -50%)',
  'bottom-right': 'translate(0, -100%)',
};

/**
 * Allowed HTML tags for client-side sanitization (see `sanitizeHtml`).
 *
 * Note: foreign-content tags (`svg`, `math`) and raw-text tags (`style`,
 * `xmp`, `noscript`, `iframe`) are intentionally kept OUT. Their contents
 * serialize as raw text rather than escaped markup, which sidesteps the
 * escaping the unwrap pass relies on and reopens a mutation-XSS (mXSS) vector;
 * allowlisting them would reintroduce it.
 */
const ALLOWED_TAGS = new Set([
  'b',
  'i',
  'em',
  'strong',
  'a',
  'span',
  'div',
  'br',
  'img',
  'ul',
  'ol',
  'li',
  'p',
  'h1',
  'h2',
  'h3',
  'h4',
  'h5',
  'h6',
  'sub',
  'sup',
  'code',
  'pre',
  'table',
  'tr',
  'td',
  'th',
  'thead',
  'tbody',
]);

/**
 * Allowed HTML attributes for client-side sanitization (see `sanitizeHtml`).
 *
 * This is an ALLOWLIST: any attribute whose lowercased name is not in this set
 * is removed. That drops `on*` event handlers, DOM-clobbering `id`/`name`,
 * `ping`/`srcset`/`download`, `data-*`, etc. for free — only these survive
 * (and `href`/`src`/`style` values still pass a scheme/content guard, while a
 * `rel` carrying the `opener` token is dropped to preserve the `noopener`
 * default `target="_blank"` implies). `colspan`/`rowspan`/`width`/`height`
 * are inert presentational values (no URL or script capability) kept so the
 * advertised table/image authoring keeps working.
 */
const ALLOWED_ATTRS = new Set([
  'style',
  'href',
  'src',
  'alt',
  'class',
  'target',
  'title',
  'rel',
  'colspan',
  'rowspan',
  'width',
  'height',
]);

/** ASCII whitespace + C0 controls — see {@link normalizeUrlForScheme}. */
// eslint-disable-next-line no-control-regex
const URL_NOISE_RE = /[\u0000-\u0020]/g;

/**
 * Strip the characters a URL parser discards, for scheme comparison only.
 *
 * `trim()` is not enough: the parser removes ASCII tab/LF/CR from ANYWHERE in
 * a URL and strips leading C0 controls, so `javascript&Tab;:`,
 * `java&NewLine;script:` and `&#1;javascript:` all resolve to the
 * `javascript:` scheme and fire. Python's `sanitize_html` does not catch these
 * either — it matches the literal `javascript:` and never HTML-decodes — so
 * this guard covers the ordinary `scene.add_html(...)` authoring path as well
 * as hand-crafted zarrs.
 *
 * Deliberately stricter than the parser (an interior plain space is not
 * actually stripped by it), which errs toward blocking. Compares only; the
 * stored attribute value is never rewritten.
 */
function normalizeUrlForScheme(value: string): string {
  return value.replace(URL_NOISE_RE, '').toLowerCase();
}

/** Hover overlay tracking entry. */
interface HoverOverlayEntry {
  el: HTMLDivElement;
  template: string;
  config: OverlayConfig;
  /**
   * The exact string last written to the element's DOM (innerHTML for
   * HTML overlays, textContent otherwise). Used to skip redundant DOM
   * rewrites: the hover UX fades the tooltip out on every mousemove and
   * re-shows it after the settle, which resets the manager-level
   * `_lastHover*` dedup to null. Without this per-entry guard, every
   * re-show rebuilds innerHTML and recreates the `<img>` element — a
   * fresh `<img>` re-decodes its (cached) blob URL asynchronously, so
   * under cursor jitter the image is perpetually recreated and often
   * never paints. Preserving the rendered DOM across the
   * fade-out/fade-in churn keeps the decoded image stable.
   */
  lastRendered?: string;
}

/**
 * Manages the screen-space overlay layer (text, image, and HTML overlays) drawn
 * over the WebGL canvas but beneath the UI controls.
 *
 * Keeps a keyed set of overlay `div`s and their {@link OverlayConfig}s, applies
 * anchor/blend/font styling, and updates visibility as scene dimensions change
 * (subscribing to `sceneDimsManager`) so overlays appear only on the slice
 * positions they belong to, with CSS-transition fades.
 */
export class OverlayManager {
  private overlayElements = new Map<string, HTMLDivElement>();
  private configs = new Map<string, OverlayConfig>();
  private objectUrls = new Set<string>();
  /** Video overlays by name, so visibility can start/stop playback. */
  private videoElements = new Map<string, HTMLVideoElement>();
  /** Stacked-alpha-matte compositors by overlay name (`alpha_matte: 'stacked'`). */
  private matteCompositors = new Map<string, VideoMatteCompositor>();
  private baseUrl = '';
  private readFile?: OverlayFileReader;
  private boundDimChangeHandler: () => void;
  /** Whether overlays are globally hidden by the user toggle (U key) */
  private globallyHidden = false;
  /**
   * Arrival gate for story flights (`Waypoint.reveal = "on_arrival"`): while
   * it returns true, a dimension-bound overlay that is not already showing
   * stays hidden. `null` = no gate.
   */
  private transitGate: (() => boolean) | null = null;
  /** Non-hover overlays currently shown. */
  private shown = new Set<string>();
  /**
   * `shown` as it was BEFORE the current dimension position was reached —
   * what the arrival gate keeps as is. Snapshotted when `updateVisibility()`
   * first sees a new position, so a second pass at the same position (the app
   * re-runs it once the driver has decided the gate) still knows which
   * overlays were already on screen and which only just appeared.
   */
  private settled = new Set<string>();
  private settledAt: string | null = null;
  /** Hover overlays that update from GPU picking results. */
  private hoverOverlays = new Map<string, HoverOverlayEntry>();
  /** Cache last hover result to skip redundant DOM updates. */
  private _lastHoverLabel: string | null = null;
  private _lastHoverKey: string | null = null;
  private _lastHoverImageUrl: string | null = null;
  private _lastHoverIndex: number = -1;
  private _lastHoverNode: string | null = null;

  constructor() {
    this.boundDimChangeHandler = () => this.updateVisibility();
  }

  /**
   * Load overlays from parsed configs and render them.
   *
   * @param overlayConfigs - Array of overlay configurations from zarr
   * @param baseUrl - Base URL of the zarr store (for directory image fetching)
   * @param readFile - Reader for opaque files held inside the active store
   */
  async loadOverlays(
    overlayConfigs: OverlayConfig[],
    baseUrl: string,
    readFile?: OverlayFileReader
  ): Promise<void> {
    this.baseUrl = baseUrl;
    this.readFile = readFile;

    const mediaReads = await Promise.allSettled(
      overlayConfigs.map((config) => this.readArchivedMedia(config))
    );

    for (const [index, config] of overlayConfigs.entries()) {
      try {
        const mediaRead = mediaReads[index];
        if (mediaRead.status === 'rejected') throw mediaRead.reason;
        const el = this.createOverlayElement(config, mediaRead.value);
        getViewerContainer().appendChild(el);
        this.overlayElements.set(config.name, el);
        this.configs.set(config.name, config);

        // Track hover overlays for GPU picking template substitution
        if (config.hover) {
          const template = config.html ?? config.text ?? '{hover_label}';
          this.hoverOverlays.set(config.name, { el, template, config });
          // Start hidden — shown when a pick resolves to a labeled element
          el.style.opacity = '0';
        }
      } catch (e) {
        log.warning(Modules.UI, `Failed to create overlay '${config.name}': ${e}`);
      }
    }

    // Register dimension change listener for visibility updates
    sceneDimsManager.addListener(this.boundDimChangeHandler);

    // Initial visibility check
    this.updateVisibility();

    log.info(Modules.UI, `[Overlay] ${overlayConfigs.length} overlay(s) rendered`);
  }

  /** Toggle global overlay visibility. */
  toggle(): void {
    this.globallyHidden = !this.globallyHidden;
    this.updateVisibility();
  }

  /**
   * True if there is at least one hover overlay that could currently
   * display a pick result. PickingSystem consults this via its
   * `shouldPick` predicate to avoid paying picking cost when no
   * tooltip would be displayed.
   */
  hasVisibleHoverOverlay(): boolean {
    return !this.globallyHidden && this.hoverOverlays.size > 0;
  }

  /** Show all overlays (subject to dimension filtering). */
  show(): void {
    this.globallyHidden = false;
    this.updateVisibility();
  }

  /** Hide all overlays globally. */
  hide(): void {
    this.globallyHidden = true;
    this.updateVisibility();
  }

  /**
   * Install (or clear, with `null`) the arrival gate. While `gate()` is true —
   * the waypoint driver is flying to a waypoint authored `reveal: "on_arrival"`
   * — `updateVisibility()` keeps a dimension-bound overlay that would NEWLY
   * appear hidden, hides one that stops matching at once (leaving is instant),
   * and leaves overlays without a `visible_range` alone. The app re-runs
   * `updateVisibility()` when the flight resolves so the held overlays fade in
   * together, on arrival.
   */
  setTransitGate(gate: (() => boolean) | null): void {
    this.transitGate = gate;
  }

  /**
   * Whether `config` shows now: the dimension rule, the global toggle, and the
   * arrival gate (which only ever withholds a dimension-bound overlay that is
   * not already on screen). Keeps `shown` in step with the answer.
   */
  private resolveVisible(name: string, config: OverlayConfig, inTransit: boolean): boolean {
    let visible = !this.globallyHidden && this.isOverlayVisible(config);
    if (visible && inTransit && config.visible_range && !this.settled.has(name)) {
      visible = false;
    }
    if (visible) this.shown.add(name);
    else this.shown.delete(name);
    return visible;
  }

  /**
   * Re-snapshot `settled` when the dimension position changed since the last
   * pass. The overlay manager and the waypoint driver both listen to the dims
   * manager in unspecified order: if this pass runs first it may show the new
   * story's captions before the driver closes the gate; the app then re-runs
   * it at the SAME position, and the snapshot lets that second pass withhold
   * exactly the overlays the first one had just revealed.
   */
  private refreshSettled(): void {
    const dims = sceneDimsManager.getDims();
    const key = dims ? dims.currentStep.join(',') : '';
    if (key === this.settledAt) return;
    this.settled = new Set(this.shown);
    this.settledAt = key;
  }

  /** Update overlay visibility based on current dimension state. */
  updateVisibility(): void {
    this.refreshSettled();
    const inTransit = this.transitGate?.() === true;
    for (const [name, config] of this.configs) {
      const el = this.overlayElements.get(name);
      if (!el) continue;

      // Hover overlay visibility is driven by updateHoverContent() (GPU picking),
      // but the global hide toggle (U key) still applies.
      if (config.hover) {
        if (this.globallyHidden) {
          el.style.display = 'none';
        } else {
          el.style.display = '';
          // Don't touch opacity — updateHoverContent manages fade in/out
        }
        continue;
      }

      const visible = this.resolveVisible(name, config, inTransit);
      const isFade = config.transition === 'fade';
      if (config.type === 'overlay_video') this.syncVideoPlayback(name, visible);

      if (isFade) {
        // Fade overlays use CSS opacity transition — never display:none
        if (visible) {
          el.classList.remove('luxar-overlay--hidden');
          el.style.opacity = String(config.opacity);
        } else {
          el.classList.add('luxar-overlay--hidden');
        }
      } else {
        if (visible) {
          el.style.display = '';
          el.style.opacity = String(config.opacity);
        } else {
          el.style.display = 'none';
        }
      }
    }
  }

  /**
   * Update hover overlay content from a GPU picking result.
   *
   * Substitutes template variables ({hover_label}, {hover_key},
   * {hover_image_label}, {hover_node}, {hover_index}) in all hover overlays.
   * Fades out if result is null or has no content.
   *
   * @param result - Pick result with label text, or null to clear
   */
  updateHoverContent(
    result: {
      label?: string | null;
      key?: string | null;
      imageUrl?: string | null;
      nodeName: string;
      elementIndex: number;
    } | null
  ): void {
    // Skip redundant DOM updates when hovering over the same element
    const newLabel = result?.label ?? null;
    const newKey = result?.key ?? null;
    const newImageUrl = result?.imageUrl ?? null;
    const newIndex = result?.elementIndex ?? -1;
    const newNode = result?.nodeName ?? null;
    if (
      newLabel === this._lastHoverLabel &&
      newKey === this._lastHoverKey &&
      newImageUrl === this._lastHoverImageUrl &&
      newIndex === this._lastHoverIndex &&
      newNode === this._lastHoverNode
    )
      return;
    this._lastHoverLabel = newLabel;
    this._lastHoverKey = newKey;
    this._lastHoverImageUrl = newImageUrl;
    this._lastHoverIndex = newIndex;
    this._lastHoverNode = newNode;

    for (const hover of this.hoverOverlays.values()) {
      const hasContent = result && (result.label || result.key || result.imageUrl);
      if (!hasContent) {
        // Fade out
        hover.el.style.opacity = '0';
      } else {
        const isHtml = hover.config.type === 'overlay_html';

        // Substitute the shared hover vocabulary. The escaping mode is the
        // whole difference between this consumer and the `link` / `copy`
        // ones: HTML overlays escape (the result reaches innerHTML), text
        // overlays don't (textContent is inert). See utils/hover-template.ts.
        //
        // `hadEmptySubstitution` is deliberately ignored here: a tooltip with
        // a gap in it is fine and visible, and this branch already only runs
        // when the element has some content. It exists for `link`, where an
        // empty segment silently produces a valid-looking wrong URL.
        let text = substituteHoverTemplate(
          hover.template,
          {
            label: result.label,
            key: result.key,
            nodeName: result.nodeName,
            elementIndex: result.elementIndex,
          },
          isHtml ? 'html' : 'text'
        ).text;

        // Image label: render as <img> tag (only meaningful in HTML overlays).
        // When hover_image_size is set, wrap in a fixed-size container so the
        // image scales up to fill it. Otherwise use default max constraints.
        const imgSize = hover.config.hover_image_size;
        let imgHtml = '';
        if (result.imageUrl) {
          const src = escapeHtml(result.imageUrl);
          if (imgSize) {
            const w = `${imgSize[0] * 100}vw`;
            const h = `${imgSize[1] * 100}vh`;
            imgHtml = `<div style="width:${w};height:${h}"><img src="${src}" style="width:100%;height:100%;object-fit:contain;display:block;border-radius:4px" /></div>`;
          } else {
            imgHtml = `<img src="${src}" style="max-width:20vh;max-height:20vh;display:block;border-radius:4px" />`;
          }
        }
        text = text.replace(/\{hover_image_label\}/g, imgHtml);

        // Only touch the DOM when the rendered content actually changed.
        // The hover loop fades out (opacity 0) on every mousemove and
        // re-shows after the settle; rewriting innerHTML here recreates
        // the `<img>` element, forcing an async blob re-decode that
        // flickers the thumbnail. Reusing the existing DOM across an
        // identical re-show keeps the decoded image visible. See
        // HoverOverlayEntry.lastRendered.
        const rendered = isHtml ? this.sanitizeHtml(text) : text;
        if (rendered === '') {
          hover.el.style.opacity = '0';
          continue;
        }
        if (rendered !== hover.lastRendered) {
          if (isHtml) {
            hover.el.innerHTML = rendered;
          } else {
            hover.el.textContent = rendered;
          }
          hover.lastRendered = rendered;
        }
        hover.el.style.opacity = String(hover.config.opacity);
      }
    }
  }

  /**
   * Return all currently visible overlay elements and their configs.
   * Used by the recording panel to composite overlays onto the capture canvas.
   */
  getVisibleOverlays(): { el: HTMLDivElement; config: OverlayConfig }[] {
    const result: { el: HTMLDivElement; config: OverlayConfig }[] = [];
    for (const [name, el] of this.overlayElements) {
      const config = this.configs.get(name);
      if (!config) continue;
      if (this.globallyHidden) continue;
      // Skip hover overlays — their content is transient (GPU picking tooltips)
      if (config.hover) continue;
      if (el.style.display === 'none') continue;
      if (el.classList.contains('luxar-overlay--hidden')) continue;
      if (parseFloat(el.style.opacity) === 0) continue;
      result.push({ el, config });
    }
    return result;
  }

  /** Dispose all overlays and clean up listeners. */
  dispose(): void {
    sceneDimsManager.removeListener(this.boundDimChangeHandler);

    for (const matte of this.matteCompositors.values()) matte.dispose();
    this.matteCompositors.clear();
    for (const video of this.videoElements.values()) {
      // Stop decoding and release the media resource before the element goes.
      video.pause();
      video.removeAttribute('src');
      video.load();
    }
    this.videoElements.clear();
    for (const el of this.overlayElements.values()) {
      el.remove();
    }
    for (const url of this.objectUrls) {
      URL.revokeObjectURL(url);
    }
    this.objectUrls.clear();
    this.readFile = undefined;
    this.overlayElements.clear();
    this.configs.clear();
    this.hoverOverlays.clear();
    this.shown.clear();
    this.settled.clear();
    this.settledAt = null;
    this.transitGate = null;
  }

  // ---------------------------------------------------------------- private

  /**
   * For a zipped store, read the overlay's media payload (image or video) so
   * it can be served from a blob URL. Video posters are loaded alongside the
   * video; plain HTTP stores stream both resources by URL instead.
   */
  private async readArchivedMedia(config: OverlayConfig): Promise<ArchivedOverlayMedia> {
    if (!isZippedStoreUrl(this.baseUrl) || !this.readFile) return {};
    const path = (filename: string): string => `/overlays/${config.name}/${filename}`;
    if (config.type === 'overlay_image' && config.image_file) {
      return { primary: await this.readFile(path(config.image_file)) };
    }
    if (config.type === 'overlay_video' && config.video_file) {
      const [primary, poster] = await Promise.all([
        this.readFile(path(config.video_file)),
        config.poster_file ? this.readFile(path(config.poster_file)) : undefined,
      ]);
      return { primary, poster };
    }
    return {};
  }

  /** Create a DOM element for a single overlay. */
  private createOverlayElement(
    config: OverlayConfig,
    archivedMedia: ArchivedOverlayMedia
  ): HTMLDivElement {
    const el = document.createElement('div');
    el.className = 'luxar-overlay';
    el.dataset.overlayName = config.name;

    // Fade transition support
    if (config.transition === 'fade') {
      el.classList.add('luxar-overlay--fade');
      el.style.setProperty('--luxar-overlay-transition-duration', `${config.transition_duration}s`);
    }

    // Start hidden — updateVisibility() will show the right ones.
    // Hover overlays are managed by updateHoverContent (inline opacity);
    // they must NOT receive --hidden, whose !important opacity:0 would
    // override updateHoverContent's inline writes and trap the tooltip
    // permanently invisible. visible_range is therefore ignored for hover
    // overlays today — see updateVisibility's hover branch.
    if (config.visible_range && !config.hover) {
      if (config.transition === 'fade') {
        el.classList.add('luxar-overlay--hidden');
      } else {
        el.style.display = 'none';
      }
    }

    // Make non-interactive overlays completely inert (no focus, no click, no events)
    if (!isOverlayInteractive(config)) {
      el.inert = true;
    }

    // Apply common positioning and style
    this.applyPositionAndStyle(el, config);

    // Create type-specific content
    switch (config.type) {
      case 'overlay_text':
        this.createTextContent(el, config);
        break;
      case 'overlay_image':
        this.createImageContent(el, config, archivedMedia.primary);
        break;
      case 'overlay_video':
        this.createVideoContent(el, config, archivedMedia.primary, archivedMedia.poster);
        break;
      case 'overlay_html':
        this.createHtmlContent(el, config);
        break;
    }

    return el;
  }

  /**
   * Create video overlay content: a muted, looping `<video>` (the only autoplay
   * a browser permits without a gesture), served from a blob URL for zipped
   * stores or by URL otherwise. Playback is tied to visibility in
   * `updateVisibility` so a hidden turntable does not keep decoding.
   */
  private createVideoContent(
    el: HTMLDivElement,
    config: OverlayConfig,
    archivedVideo?: Uint8Array,
    archivedPoster?: Uint8Array
  ): void {
    el.classList.add('luxar-overlay--video');
    if (!config.video_file) return;

    const video = document.createElement('video');
    // A stacked-matte clip is read by WebGL, which refuses a tainted video:
    // ask for CORS up front (before `src`) when the store lives on another
    // origin — the dev server against `luxar serve`. Same-origin clips (the
    // exported app, zipped stores' blob URLs) are readable as they are, and
    // WKWebView's CORS media path is flaky (a failed CORS load taints the
    // element and every upload then throws), so do not ask when not needed.
    if (this.needsVideoCors(config)) {
      video.crossOrigin = 'anonymous';
    }
    video.muted = config.muted !== false;
    video.loop = config.loop !== false;
    // Deliberately NOT the DOM `autoplay` attribute: the browser honours that
    // once the media has loaded, which is after the initial visibility pass, so
    // every hidden clip would start decoding at load (all ten did). Autoplay is
    // a wish that `syncVideoPlayback` grants only while the overlay is visible.
    video.dataset.autoplay = config.autoplay !== false ? '1' : '0';
    video.controls = config.autoplay === false;
    video.playsInline = true;
    video.preload = 'metadata';
    video.disablePictureInPicture = true;
    if (typeof config.playback_rate === 'number' && config.playback_rate > 0) {
      video.defaultPlaybackRate = config.playback_rate;
      video.playbackRate = config.playback_rate;
    }
    if (!this.setVideoMedia(video, config, archivedVideo, archivedPoster)) return;
    video.onerror = () => {
      log.warning(
        Modules.UI,
        `Video overlay "${config.name}" failed to load ${config.video_file} — showing poster only`
      );
    };

    const shown = this.attachMatteCanvas(el, video, config) ?? video;
    if (config.size) {
      shown.style.width = `${config.size[0] * 100}vw`;
      // A null height keeps the media's own aspect ratio.
      shown.style.height = config.size[1] == null ? 'auto' : `${config.size[1] * 100}vh`;
    }
    shown.style.display = 'block';
    el.appendChild(video);
    this.videoElements.set(config.name, video);
  }

  /** Whether the store's base URL is on another origin than the page (blob URLs are not). */
  private isCrossOriginStore(): boolean {
    if (!this.baseUrl || isZippedStoreUrl(this.baseUrl)) return false;
    try {
      return new URL(this.baseUrl, window.location.href).origin !== window.location.origin;
    } catch {
      return false;
    }
  }

  private needsVideoCors(config: OverlayConfig): boolean {
    return config.alpha_matte === 'stacked' && this.isCrossOriginStore();
  }

  /**
   * For an `alpha_matte: 'stacked'` clip, show a compositor canvas instead of
   * the `<video>` (which stays in the DOM, decoding, but visually hidden — it is
   * the compositor's frame source). The poster sits behind the canvas until the
   * first frame is drawn. Returns the element to size, or `null` when the clip
   * is not a stacked matte. If WebGL is unavailable, the first `start()` reports
   * failure and the manager falls back to the plain video.
   */
  private attachMatteCanvas(
    el: HTMLDivElement,
    video: HTMLVideoElement,
    config: OverlayConfig
  ): HTMLCanvasElement | null {
    if (config.alpha_matte !== 'stacked') return null;
    const posterBackground = video.poster ? `url("${video.poster}")` : '';
    const matte = createVideoMatteCompositor(video, {
      onFailure: (error) => this.abandonMatte(config.name, video, error),
      onFirstFrame: () => {
        const current = this.matteCompositors.get(config.name);
        if (current) current.canvas.style.backgroundImage = 'none';
      },
      // Releasing the context blanks the canvas, so the poster goes back
      // underneath it until the next visible frame is drawn.
      onRelease: () => {
        const current = this.matteCompositors.get(config.name);
        if (current) current.canvas.style.backgroundImage = posterBackground || 'none';
      },
    });
    video.classList.add('luxar-overlay__matte-source');
    if (video.poster) {
      // Longhands, not the `background` shorthand: the shorthand also resets
      // the colour, and the page's `canvas { background-color: #000 }` rule
      // (index.html, meant for the WebGL canvas) must never show through — the
      // stylesheet pins the matte canvas transparent, the poster is image-only.
      const style = matte.canvas.style;
      style.backgroundImage = `url("${video.poster}")`;
      style.backgroundPosition = 'center';
      style.backgroundSize = 'contain';
      style.backgroundRepeat = 'no-repeat';
    }
    el.appendChild(matte.canvas);
    this.matteCompositors.set(config.name, matte);
    return matte.canvas;
  }

  /**
   * The compositor could not read the video (a tainted cross-origin clip):
   * drop the canvas and show the raw clip — colour over matte, visible rather
   * than a blank square — and say why, once.
   */
  private abandonMatte(name: string, video: HTMLVideoElement, error: unknown): void {
    const matte = this.matteCompositors.get(name);
    if (!matte) return;
    this.matteCompositors.delete(name);
    const sizing = { width: matte.canvas.style.width, height: matte.canvas.style.height };
    matte.dispose();
    matte.canvas.remove();
    video.classList.remove('luxar-overlay__matte-source');
    video.style.width = sizing.width;
    video.style.height = sizing.height;
    video.style.display = 'block';
    log.warning(
      Modules.UI,
      `Video overlay "${name}": the alpha matte could not read the clip (${String(error)}) — showing the raw clip`
    );
  }

  private setVideoMedia(
    video: HTMLVideoElement,
    config: OverlayConfig,
    archivedVideo?: Uint8Array,
    archivedPoster?: Uint8Array
  ): boolean {
    const base = `${this.baseUrl}overlays/${config.name}/`;
    if (!isZippedStoreUrl(this.baseUrl)) {
      video.src = `${base}${config.video_file}`;
      if (config.poster_file) video.poster = `${base}${config.poster_file}`;
      return true;
    }
    if (!archivedVideo) {
      log.warning(
        Modules.UI,
        `Video overlay "${config.name}" is missing /overlays/${config.name}/${config.video_file} — skipping`
      );
      return false;
    }
    const blobBytes = Uint8Array.from(archivedVideo);
    const objectUrl = URL.createObjectURL(
      new Blob([blobBytes], { type: detectMimeType(blobBytes) })
    );
    this.objectUrls.add(objectUrl);
    video.src = objectUrl;
    if (archivedPoster) {
      const posterBytes = Uint8Array.from(archivedPoster);
      const posterUrl = URL.createObjectURL(
        new Blob([posterBytes], { type: detectMimeType(posterBytes) })
      );
      this.objectUrls.add(posterUrl);
      video.poster = posterUrl;
    }
    return true;
  }

  /** Start or stop a video overlay with its visibility (muted play needs no gesture). */
  private syncVideoPlayback(name: string, visible: boolean): void {
    const video = this.videoElements.get(name);
    if (!video) return;
    const matte = this.matteCompositors.get(name);
    // RELEASE, not stop, when the clip goes away: a WebGL context per hidden
    // clip is a per-session leak, and a browser caps how many may be live —
    // Chrome then evicts the OLDEST, which is the scene's own renderer. A tour
    // with nineteen stacked turntables lost the renderer partway through
    // before this; only the visible clip holds a context now, whatever the
    // tour's length. The cost is one context acquisition per story step.
    if (visible) matte?.start();
    else matte?.release();
    if (visible) {
      video.preload = 'auto';
      if (video.paused && video.dataset.autoplay === '1') {
        // jsdom returns undefined and gesture-gated browsers reject; a rejected
        // promise must not surface as an unhandled rejection from a
        // visibility update.
        const p: unknown = video.play();
        if (p instanceof Promise) p.catch(() => {});
      }
    } else if (!video.paused) {
      video.pause();
    }
  }

  /** Apply common positioning, transitions, and interaction styles. */
  private applyPositionAndStyle(el: HTMLDivElement, config: OverlayConfig): void {
    const [x, y] = config.position;

    // Position using percentages (normalized coords → CSS %). RIGHT-side
    // anchors are pinned to the container's RIGHT edge (`right:` instead of
    // `left:`) so the element's shrink-to-fit width is measured against the
    // LEFT edge and can grow leftward — see RIGHT_ANCHOR_TRANSFORM / #773.
    // When an element flips between left- and right-anchoring across
    // re-applies, the unused property is cleared so a stale value can't linger.
    const isRightAnchor =
      config.anchor === 'top-right' ||
      config.anchor === 'center-right' ||
      config.anchor === 'bottom-right';
    const isCenterAnchor = CENTER_ANCHORS.has(config.anchor);

    el.classList.toggle('luxar-overlay--center-anchored', isCenterAnchor);

    if (isRightAnchor) {
      el.classList.add('luxar-overlay--right-anchored');
      el.style.right = `${(1 - x) * 100}%`;
      el.style.removeProperty('--luxar-overlay-x');
      el.style.left = '';
      // Transform without the horizontal -100% (the right edge already
      // places the box); vertical component preserved.
      el.style.transform = RIGHT_ANCHOR_TRANSFORM[config.anchor] ?? 'translate(0, 0)';
    } else {
      el.classList.remove('luxar-overlay--right-anchored');
      el.style.setProperty('--luxar-overlay-x', `${x * 100}%`);
      el.style.left = '';
      el.style.right = '';
      el.style.transform = ANCHOR_TRANSFORM[config.anchor] ?? 'translate(0, 0)';
    }
    el.style.top = `${y * 100}%`;

    // Opacity
    el.style.opacity = String(config.opacity);

    // Blend mode (works for all overlay types: text, image, html)
    if (config.blend_mode && config.blend_mode !== 'normal') {
      const cssBlend = BLEND_MODE_MAP[config.blend_mode] ?? config.blend_mode;
      el.style.mixBlendMode = cssBlend;
    }

    // Interaction
    if (isOverlayInteractive(config)) {
      el.classList.add('luxar-overlay--interactive');
    }
  }

  /** Create text overlay content. */
  private createTextContent(el: HTMLDivElement, config: OverlayConfig): void {
    el.classList.add('luxar-overlay--text');

    const text = config.text ?? '';
    el.textContent = text;

    // Font
    const fontFamily = FONT_PRESETS[config.font ?? 'sans'] ?? (config.font || FONT_PRESETS.sans);
    el.style.fontFamily = fontFamily;

    // Font size (viewport-relative → vh units)
    if (config.font_size) {
      el.style.fontSize = `${config.font_size * 100}vh`;
    }

    // Color
    if (config.color) {
      el.style.color = config.color;
    }

    // Width (enables word wrapping). Like the anchor positioning above,
    // each branch clears what the others set so no stale sizing survives a
    // re-apply with a different config.
    if (config.width) {
      // Explicit width is the primary sizing; honor it verbatim.
      el.classList.add('luxar-overlay--explicit-width');
      el.style.width = `${config.width * 100}vw`;
      el.style.maxWidth = '';
      // Hover overlays use pre-line so \n in labels creates line breaks;
      // regular overlays use normal for standard word wrapping.
      el.style.whiteSpace = config.hover ? 'pre-line' : 'normal';
      el.style.wordWrap = 'break-word';
    } else if (config.hover) {
      // Hover tooltips wrap (pre-line). Clamp to a readable measure so a long
      // label wraps to a couple of lines instead of collapsing to one word per
      // line against the right-anchored container edge (see issue #773), rather
      // than the ~2vw the pre-transform container position would otherwise impose.
      el.classList.remove('luxar-overlay--explicit-width');
      el.style.width = '';
      el.style.maxWidth = 'min(30vw, 40ch)';
      el.style.whiteSpace = 'pre-line';
      el.style.wordWrap = 'break-word';
    } else {
      // No explicit width and not a wrapping overlay: single line, sized to its
      // content (unchanged behavior — a max-width here would only clip the box
      // while nowrap text overflows off-screen).
      el.classList.remove('luxar-overlay--explicit-width');
      el.style.width = '';
      el.style.maxWidth = '';
      el.style.whiteSpace = 'nowrap';
      el.style.wordWrap = '';
    }

    // Text alignment
    if (config.text_align) {
      el.style.textAlign = config.text_align;
    }

    // Line height
    if (config.line_height) {
      el.style.lineHeight = String(config.line_height);
    }

    // Background
    if (config.background) {
      el.style.backgroundColor = config.background;
      // Padding (viewport-relative → vh units)
      const padding = config.padding ?? 0.005;
      el.style.padding = `${padding * 100}vh`;
    }

    // Text stroke via text-shadow (4-directional for consistent outline)
    if (config.stroke_color) {
      const sw = (config.stroke_width ?? 0.002) * 100; // vh units
      el.style.textShadow = [
        `${sw}vh 0 ${config.stroke_color}`,
        `-${sw}vh 0 ${config.stroke_color}`,
        `0 ${sw}vh ${config.stroke_color}`,
        `0 -${sw}vh ${config.stroke_color}`,
      ].join(', ');
    }
  }

  /** Create image overlay content. */
  private createImageContent(
    el: HTMLDivElement,
    config: OverlayConfig,
    archivedImage?: Uint8Array
  ): void {
    el.classList.add('luxar-overlay--image');

    const img = document.createElement('img');

    // Construct image URL from base zarr URL
    if (config.image_file) {
      if (isZippedStoreUrl(this.baseUrl)) {
        const path = `/overlays/${config.name}/${config.image_file}`;
        if (!this.readFile) {
          log.warning(
            Modules.UI,
            `Image overlay "${config.name}" has no store file reader for ${path} — skipping`
          );
          return;
        }
        if (!archivedImage) {
          log.warning(Modules.UI, `Image overlay "${config.name}" is missing ${path} — skipping`);
          return;
        }
        // BlobPart requires an ArrayBuffer-backed view, while store reads may expose ArrayBufferLike.
        const blobBytes = Uint8Array.from(archivedImage);
        const objectUrl = URL.createObjectURL(
          new Blob([blobBytes], { type: detectMimeType(blobBytes) })
        );
        this.objectUrls.add(objectUrl);
        img.src = objectUrl;
        img.alt = config.name;
      } else {
        const imageUrl = `${this.baseUrl}overlays/${config.name}/${config.image_file}`;
        img.onerror = () => {
          log.warning(
            Modules.UI,
            `Image overlay "${config.name}" is missing ${imageUrl} — skipping`
          );
          img.remove();
        };
        img.src = imageUrl;
        img.alt = config.name;
      }
    }

    // Size (a null height keeps the image's own aspect ratio)
    if (config.size) {
      img.style.width = `${config.size[0] * 100}vw`;
      img.style.height = config.size[1] == null ? 'auto' : `${config.size[1] * 100}vh`;
    }

    img.style.display = 'block';

    el.appendChild(img);
  }

  /** Create HTML overlay content. */
  private createHtmlContent(el: HTMLDivElement, config: OverlayConfig): void {
    el.classList.add('luxar-overlay--html');

    // Sanitize before injecting — see sanitizeHtml: for remote scene data
    // this is the only control, not a second layer behind Python.
    const sanitized = this.sanitizeHtml(config.html ?? '');
    el.innerHTML = sanitized;

    // Font (shared with text overlays; needed for auto-injected hover HTML overlays)
    if (config.font) {
      const fontFamily = FONT_PRESETS[config.font] ?? (config.font || FONT_PRESETS.sans);
      el.style.fontFamily = fontFamily;
    }
    if (config.font_size) {
      el.style.fontSize = `${config.font_size * 100}vh`;
    }
    if (config.color) {
      el.style.color = config.color;
    }
    if (config.background) {
      el.style.backgroundColor = config.background;
      const padding = config.padding ?? 0.005;
      el.style.padding = `${padding * 100}vh`;
    }

    // Width
    if (config.width) {
      el.style.width = `${config.width * 100}vw`;
    }

    // Force pointer-events: none on all descendants (belt-and-suspenders with CSS rule)
    if (!config.interactive) {
      el.querySelectorAll('*').forEach((child) => {
        (child as HTMLElement).style.pointerEvents = 'none';
      });
    }
  }

  /**
   * Client-side HTML sanitization against DOM tag and attribute allowlists.
   *
   * Pass 1 is an attribute ALLOWLIST (see ALLOWED_ATTRS): every attribute not
   * in the set is removed — this is what drops `on*` handlers, DOM-clobbering
   * `id`/`name`, `ping`/`srcset`/`download`, `data-*`, etc. The allowlisted
   * value-bearing attributes then pass a per-attribute guard (see below):
   * `href`/`src` block `javascript:`/`vbscript:`/`data:` schemes, `style`
   * blocks `javascript:`/`vbscript:`/`expression(` plus the CSS escape and
   * comment syntax (`\`, `/*`) that could smuggle those tokens past a
   * substring check, `rel` drops an `opener` token, and `target` is
   * restricted to `_blank`/`_self` — the last two neutralize reverse
   * tabnabbing. Pass 2 is a tag allowlist (see ALLOWED_TAGS).
   *
   * A disallowed tag is *unwrapped*, not dropped — its children are lifted
   * into its parent — so every element has to be scrubbed whether or not its
   * own tag survives. Skipping the descendants of a disallowed tag hoists
   * them into the output verbatim; that was issue #720.
   *
   * Invariant: pass 1 scrubs attributes on every element unconditionally, and
   * pass 2 only moves existing nodes and drops the elements it unwraps — it
   * never creates, clones or re-parses one, so nothing can reach the output
   * unscrubbed. An unwrapped element is removed only after its children have
   * been lifted, so nothing still awaiting pass 2 is ever detached: every
   * element left in the snapshot is still connected, and still scrubbed, when
   * pass 2 reaches it. Pass 2 runs outermost-first so each node moves exactly
   * once; bottom-up would re-lift the same payload once per enclosing wrapper.
   *
   * Note: a nested `<template>` keeps its payload in a separate `.content`
   * fragment that `querySelectorAll` never sees. It is discarded because
   * `template` is not allowlisted — allowlisting it would ship that subtree
   * unsanitized.
   */
  private sanitizeHtml(html: string): string {
    // Defensive per-render guard: reject an oversized value BEFORE parsing.
    // `template.innerHTML = html` is synchronous, main-thread, and superlinear
    // in nesting depth, so a large deeply-nested string can hang the tab. The
    // loader already caps `html` once per scene load, but this path also runs on
    // every hover-content change (`updateHoverContent`), so we re-check here.
    // See MAX_OVERLAY_HTML_CHARS.
    //
    // The type check backs up the compile-time signature: config values
    // originate as untrusted zarr .zattrs JSON, and a non-string (e.g. an
    // array wrapping a huge payload) would pass the .length cap yet be
    // coerced to its full string form by the innerHTML assignment below.
    if (typeof (html as unknown) !== 'string') {
      log.warning(
        Modules.UI,
        `[Overlay] Refusing to sanitize non-string html (${typeof (html as unknown)}) — ` +
          'returning empty'
      );
      return '';
    }
    if (html.length > MAX_OVERLAY_HTML_CHARS) {
      log.warning(
        Modules.UI,
        `[Overlay] Refusing to sanitize ${html.length}-char html ` +
          `(limit ${MAX_OVERLAY_HTML_CHARS}) — returning empty to protect the main thread`
      );
      return '';
    }

    const template = document.createElement('template');
    template.innerHTML = html;

    // One snapshot of every element at every depth, in document order
    // (ancestors before descendants) — both passes iterate it.
    const elements = Array.from(template.content.querySelectorAll('*'));

    // Pass 1 — attribute allowlist scrub, applied to allowed and disallowed
    // alike. Anything not in ALLOWED_ATTRS is dropped, which covers `on*`
    // event handlers and DOM-clobbering `id`/`name` (plus `ping`, `srcset`,
    // `download`, `data-*`, …) for free. The value-bearing allowlisted
    // attributes then face a per-attribute guard: scheme checks on
    // `href`/`src`, content checks on `style`, an `opener`-token check on
    // `rel`, and a `_blank`/`_self` restriction on `target`. Over-blocking is
    // the deliberate, documented preference here.
    for (const el of elements) {
      for (const attr of Array.from(el.attributes)) {
        const attrName = attr.name.toLowerCase();
        if (!ALLOWED_ATTRS.has(attrName)) {
          el.removeAttribute(attr.name);
        } else if (attrName === 'href' || attrName === 'src') {
          // Block script-bearing and data-URI schemes.
          const scheme = normalizeUrlForScheme(attr.value);
          if (
            scheme.startsWith('javascript:') ||
            scheme.startsWith('vbscript:') ||
            scheme.startsWith('data:')
          ) {
            el.removeAttribute(attr.name);
          }
        } else if (attrName === 'style') {
          // A `url(javascript:…)` collapses to contain `javascript:` after
          // normalization, so this single check also covers the CSS-url vector.
          // The substring checks alone are escapable, though: CSS decodes
          // `\6a ` to `j` and comments can split a token (`expr/**/ession`),
          // so any backslash or comment-opener drops the whole value — a
          // substring check cannot see through CSS tokenization, and a benign
          // inline overlay style needs neither.
          const normalized = normalizeUrlForScheme(attr.value);
          if (
            normalized.includes('\\') ||
            normalized.includes('/*') ||
            normalized.includes('javascript:') ||
            normalized.includes('vbscript:') ||
            normalized.includes('expression(')
          ) {
            el.removeAttribute(attr.name);
          }
        } else if (attrName === 'rel') {
          // `target="_blank"` implies `noopener` by default; a hostile
          // `rel="opener"` opts back OUT of it, handing the opened page a live
          // `window.opener` to cross-origin-navigate the viewer tab (reverse
          // tabnabbing). Drop the whole attribute if the exact `opener` token
          // is present. Token equality, not substring — `noopener` must NOT
          // match. A legitimate author never writes a bare `opener`.
          const tokens = attr.value.toLowerCase().split(/\s+/);
          if (tokens.includes('opener')) {
            el.removeAttribute(attr.name);
          }
        } else if (attrName === 'target') {
          // Implicit `noopener` is granted ONLY to an EXACT `_blank`/`_self`
          // target (ASCII case-insensitive, no trimming — matching HTML's
          // keyword rule). A near-miss the browser treats as a NAMED target
          // ("_blank " with a stray space, a Kelvin-sign homoglyph, …) opens a
          // top-level window with a live `window.opener` (reverse tabnabbing),
          // so compare the RAW value and drop anything that is not
          // letter-for-letter `_blank`/`_self`. The `i` flag folds ASCII case
          // only; do NOT add `u`, which would wrongly accept `_blanK` (Kelvin).
          if (!/^_(blank|self)$/i.test(attr.value)) {
            el.removeAttribute(attr.name);
          }
        }
      }
    }

    // Pass 2 — unwrap disallowed tags, outermost first (one move per node).
    for (const el of elements) {
      if (ALLOWED_TAGS.has(el.tagName.toLowerCase())) continue;
      const parent = el.parentNode;
      // Unreachable: children are lifted before their element is removed, so
      // nothing still awaiting pass 2 has been detached.
      if (!parent) continue;
      while (el.firstChild) {
        parent.insertBefore(el.firstChild, el);
      }
      el.remove();
    }

    return template.innerHTML;
  }

  /** Check if an overlay should be visible given current dimension state. */
  private isOverlayVisible(config: OverlayConfig): boolean {
    if (!config.visible_range) return true;

    const dims = sceneDimsManager.getDims();
    if (!dims) return false; // dims not ready — hide dimension-filtered overlays

    const metadata = dims.metadata;
    if (!metadata) return false;

    for (const [dimName, constraint] of Object.entries(config.visible_range)) {
      // Find dimension index by name
      const dimIndex = metadata.findIndex((m) => m.name === dimName);
      if (dimIndex < 0) continue;

      const currentValue = dims.currentStep[dimIndex];
      if (currentValue === undefined) continue;

      if (typeof constraint === 'number') {
        // Exact match (with small tolerance for floating point)
        if (Math.abs(currentValue - constraint) > 0.5) return false;
      } else if (Array.isArray(constraint) && constraint.length === 2) {
        // Range check [min, max]
        if (currentValue < constraint[0] || currentValue > constraint[1]) return false;
      }
    }

    return true;
  }
}
