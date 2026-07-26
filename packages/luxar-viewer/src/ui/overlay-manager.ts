/**
 * Overlay Manager - Renders screen-space overlays (text, image, HTML) over the canvas.
 *
 * Creates an HTML overlay layer between the WebGL canvas and UI controls.
 * Supports dimension-aware visibility (overlays show/hide based on slider positions),
 * CSS transitions, blend modes, and configurable interaction.
 */

import { sceneDimsManager } from '../scene/scene-dims-manager';
import { log, Modules } from '../utils/log';
import { getViewerContainer } from '../utils/viewer-container';
import type { OverlayConfig } from '../data/loaders';

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

/** Allowed HTML tags for client-side sanitization (see `sanitizeHtml`) */
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

/** Escape HTML entities to prevent XSS in template substitution. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
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

export class OverlayManager {
  private overlayElements = new Map<string, HTMLDivElement>();
  private configs = new Map<string, OverlayConfig>();
  private baseUrl = '';
  private boundDimChangeHandler: () => void;
  /** Whether overlays are globally hidden by the user toggle (U key) */
  private globallyHidden = false;
  /** Hover overlays that update from GPU picking results. */
  private hoverOverlays = new Map<string, HoverOverlayEntry>();
  /** Cache last hover result to skip redundant DOM updates. */
  private _lastHoverLabel: string | null = null;
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
   * @param baseUrl - Base URL of the zarr store (for image fetching)
   */
  async loadOverlays(overlayConfigs: OverlayConfig[], baseUrl: string): Promise<void> {
    this.baseUrl = baseUrl;

    for (const config of overlayConfigs) {
      try {
        const el = this.createOverlayElement(config);
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

  /** Update overlay visibility based on current dimension state. */
  updateVisibility(): void {
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

      const visible = !this.globallyHidden && this.isOverlayVisible(config);
      const isFade = config.transition === 'fade';

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
   * Substitutes template variables ({hover_label}, {hover_image_label},
   * {hover_node}, {hover_index}) in all hover overlays.
   * Fades out if result is null or has no content.
   *
   * @param result - Pick result with label text, or null to clear
   */
  updateHoverContent(
    result: {
      label?: string | null;
      imageUrl?: string | null;
      nodeName: string;
      elementIndex: number;
    } | null
  ): void {
    // Skip redundant DOM updates when hovering over the same element
    const newLabel = result?.label ?? null;
    const newImageUrl = result?.imageUrl ?? null;
    const newIndex = result?.elementIndex ?? -1;
    const newNode = result?.nodeName ?? null;
    if (
      newLabel === this._lastHoverLabel &&
      newImageUrl === this._lastHoverImageUrl &&
      newIndex === this._lastHoverIndex &&
      newNode === this._lastHoverNode
    )
      return;
    this._lastHoverLabel = newLabel;
    this._lastHoverImageUrl = newImageUrl;
    this._lastHoverIndex = newIndex;
    this._lastHoverNode = newNode;

    for (const hover of this.hoverOverlays.values()) {
      const hasContent = result && (result.label || result.imageUrl);
      if (!hasContent) {
        // Fade out
        hover.el.style.opacity = '0';
      } else {
        const isHtml = hover.config.type === 'overlay_html';

        // Substitute template variables
        // HTML overlays: escape values to prevent XSS in innerHTML
        // Text overlays: no escaping needed since textContent is XSS-safe
        const esc = isHtml ? escapeHtml : (s: string) => s;
        let text = hover.template;
        text = text.replace(/\{hover_label\}/g, result.label ? esc(result.label) : '');
        text = text.replace(/\{hover_node\}/g, esc(result.nodeName));
        text = text.replace(/\{hover_index\}/g, String(result.elementIndex));

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

    for (const el of this.overlayElements.values()) {
      el.remove();
    }
    this.overlayElements.clear();
    this.configs.clear();
    this.hoverOverlays.clear();
  }

  // ---------------------------------------------------------------- private

  /** Create a DOM element for a single overlay. */
  private createOverlayElement(config: OverlayConfig): HTMLDivElement {
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
    if (!config.interactive) {
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
        this.createImageContent(el, config);
        break;
      case 'overlay_html':
        this.createHtmlContent(el, config);
        break;
    }

    return el;
  }

  /** Apply common positioning, transitions, and interaction styles. */
  private applyPositionAndStyle(el: HTMLDivElement, config: OverlayConfig): void {
    const [x, y] = config.position;

    // Position using percentages (normalized coords → CSS %)
    el.style.left = `${x * 100}%`;
    el.style.top = `${y * 100}%`;

    // Anchor offset via transform
    const transform = ANCHOR_TRANSFORM[config.anchor] ?? 'translate(0, 0)';
    el.style.transform = transform;

    // Opacity
    el.style.opacity = String(config.opacity);

    // Blend mode (works for all overlay types: text, image, html)
    if (config.blend_mode && config.blend_mode !== 'normal') {
      const cssBlend = BLEND_MODE_MAP[config.blend_mode] ?? config.blend_mode;
      el.style.mixBlendMode = cssBlend;
    }

    // Interaction
    if (config.interactive) {
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

    // Width (enables word wrapping)
    if (config.width) {
      el.style.width = `${config.width * 100}vw`;
      // Hover overlays use pre-line so \n in labels creates line breaks;
      // regular overlays use normal for standard word wrapping.
      el.style.whiteSpace = config.hover ? 'pre-line' : 'normal';
      el.style.wordWrap = 'break-word';
    } else {
      el.style.whiteSpace = config.hover ? 'pre-line' : 'nowrap';
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
  private createImageContent(el: HTMLDivElement, config: OverlayConfig): void {
    el.classList.add('luxar-overlay--image');

    const img = document.createElement('img');

    // Construct image URL from base zarr URL
    if (config.image_file) {
      const imageUrl = `${this.baseUrl}overlays/${config.name}/${config.image_file}`;
      img.src = imageUrl;
      img.alt = config.name;
    }

    // Size
    if (config.size) {
      img.style.width = `${config.size[0] * 100}vw`;
      img.style.height = `${config.size[1] * 100}vh`;
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
   * Client-side HTML sanitization against a DOM tag allowlist.
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
    const template = document.createElement('template');
    template.innerHTML = html;

    // One snapshot of every element at every depth, in document order
    // (ancestors before descendants) — both passes iterate it.
    const elements = Array.from(template.content.querySelectorAll('*'));

    // Pass 1 — attribute scrub, applied to allowed and disallowed alike.
    for (const el of elements) {
      for (const attr of Array.from(el.attributes)) {
        const attrName = attr.name.toLowerCase();
        // Tags are allowlisted; attributes are not. Anything not matched
        // below is kept — blocked here are `on*` event handlers and
        // `javascript:` URLs in `href`/`src`.
        if (attrName.startsWith('on')) {
          el.removeAttribute(attr.name);
        } else if (
          (attrName === 'href' || attrName === 'src') &&
          normalizeUrlForScheme(attr.value).startsWith('javascript:')
        ) {
          el.removeAttribute(attr.name);
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
