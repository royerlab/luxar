/**
 * Overlay Manager - Renders screen-space overlays (text, image, HTML) over the canvas.
 *
 * Creates an HTML overlay layer between the WebGL canvas and UI controls.
 * Supports dimension-aware visibility (overlays show/hide based on slider positions),
 * CSS transitions, blend modes, and configurable interaction.
 */

import { sceneDimsManager } from '../scene/scene-dims-manager';
import { log, Modules } from '../utils/log';
import type { OverlayConfig } from './overlay-loader';

/** Font preset mappings to CSS font-family stacks */
const FONT_PRESETS: Record<string, string> = {
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

/** Allowed HTML tags for client-side sanitization (defense-in-depth) */
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
        document.body.appendChild(el);
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
   * Substitutes template variables ({hover_label}, {hover_node}, {hover_index})
   * in all hover overlays. Fades out if result is null or label is empty.
   *
   * @param result - Pick result with label text, or null to clear
   */
  updateHoverContent(
    result: { label: string; nodeName: string; elementIndex: number } | null
  ): void {
    for (const hover of this.hoverOverlays.values()) {
      if (!result || !result.label) {
        // Fade out
        hover.el.style.opacity = '0';
      } else {
        const isHtml = hover.config.type === 'overlay_html';

        // Substitute template variables
        // HTML overlays: escape values to prevent XSS in innerHTML
        // Text overlays: no escaping needed since textContent is XSS-safe
        const esc = isHtml ? escapeHtml : (s: string) => s;
        let text = hover.template;
        text = text.replace(/\{hover_label\}/g, esc(result.label));
        text = text.replace(/\{hover_node\}/g, esc(result.nodeName));
        text = text.replace(/\{hover_index\}/g, String(result.elementIndex));

        if (isHtml) {
          hover.el.innerHTML = this.sanitizeHtml(text);
        } else {
          hover.el.textContent = text;
        }
        hover.el.style.opacity = String(hover.config.opacity);
      }
    }
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
      el.style.setProperty(
        '--luxar-overlay-transition-duration',
        `${config.transition_duration}s`,
      );
    }

    // Start hidden — updateVisibility() will show the right ones
    if (config.visible_range) {
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
      el.style.whiteSpace = 'normal';
      el.style.wordWrap = 'break-word';
    } else {
      el.style.whiteSpace = 'nowrap';
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

    // Blend mode
    if (config.blend_mode && config.blend_mode !== 'normal') {
      const cssBlend = BLEND_MODE_MAP[config.blend_mode] ?? config.blend_mode;
      el.style.mixBlendMode = cssBlend;
    }

    el.appendChild(img);
  }

  /** Create HTML overlay content. */
  private createHtmlContent(el: HTMLDivElement, config: OverlayConfig): void {
    el.classList.add('luxar-overlay--html');

    // Client-side sanitization (defense-in-depth, Python already sanitizes)
    const sanitized = this.sanitizeHtml(config.html ?? '');
    el.innerHTML = sanitized;

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
   * Client-side HTML sanitization using DOM allowlist.
   * Defense-in-depth — the Python side sanitizes first.
   */
  private sanitizeHtml(html: string): string {
    const template = document.createElement('template');
    template.innerHTML = html;

    const walk = (parent: Element | DocumentFragment) => {
      const toRemove: Element[] = [];

      for (const child of Array.from(parent.children)) {
        const tagName = child.tagName.toLowerCase();

        if (!ALLOWED_TAGS.has(tagName)) {
          toRemove.push(child);
          continue;
        }

        // Remove disallowed attributes
        for (const attr of Array.from(child.attributes)) {
          const attrName = attr.name.toLowerCase();
          // Allow: style, href, src, alt, class, target
          // Block: on* event handlers, javascript: URLs
          if (attrName.startsWith('on')) {
            child.removeAttribute(attr.name);
          } else if (
            (attrName === 'href' || attrName === 'src') &&
            attr.value.trim().toLowerCase().startsWith('javascript:')
          ) {
            child.removeAttribute(attr.name);
          }
        }

        // Recurse into children
        walk(child);
      }

      for (const el of toRemove) {
        // Move children up before removing the disallowed tag
        while (el.firstChild) {
          parent.insertBefore(el.firstChild, el);
        }
        el.remove();
      }
    };

    walk(template.content);
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
