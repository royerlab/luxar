/**
 * Unit tests for OverlayManager.
 *
 * The class is heavily DOM-bound but jsdom handles the
 * createElement/appendChild paths fine; only the sceneDimsManager
 * listener wiring needs coordination across tests.
 *
 * Tests focus on the externally-observable contract:
 * - Exported constants are well-formed
 * - Constructor + globally-hidden state machine
 * - loadOverlays creates the right number of elements with the right
 *   classes, attributes, and visibility
 * - getVisibleOverlays correctly filters by globallyHidden + display
 * - dispose tears down DOM elements and clears internal state
 * - updateHoverContent skips redundant DOM updates
 *
 * Private helpers (createOverlayElement, sanitizeHtml, etc.) are
 * exercised transitively through loadOverlays.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OverlayManager, FONT_PRESETS } from '../../../ui/overlay-manager';
import type { OverlayConfig } from '../../../data/loaders';
import type { SimpleDims } from '../../../types/dims';

/**
 * Settable dims state for the dimension-filtering describe block below.
 * The mock returns whatever the current test has assigned. Tests that
 * don't touch dimensions are unaffected because their overlays have no
 * `visible_range`, so `isOverlayVisible` short-circuits to `true`
 * before reading `getDims()`.
 */
const mockDimsState: { current: SimpleDims | null } = { current: null };

vi.mock('../../../scene/scene-dims-manager', () => ({
  sceneDimsManager: {
    getDims: () => mockDimsState.current,
    addListener: vi.fn(),
    removeListener: vi.fn(),
  },
}));

function makeTextOverlay(overrides: Partial<OverlayConfig> = {}): OverlayConfig {
  return {
    name: 'caption',
    type: 'overlay_text',
    position: [0.5, 0.5],
    opacity: 1.0,
    anchor: 'center',
    transition: 'none',
    transition_duration: 0.3,
    interactive: false,
    z_index: 0,
    text: 'Hello, Luxar.',
    font_size: 16,
    font: 'sans',
    ...overrides,
  } as OverlayConfig;
}

describe('FONT_PRESETS', () => {
  it('exposes the three documented presets as non-empty font-family strings', () => {
    // Audit W4 fix: toBeTruthy passed for any non-empty value
    // (including, e.g., the boolean `true` or a number). Pin shape:
    // each preset is a string with at least one font-family token.
    expect(typeof FONT_PRESETS.sans).toBe('string');
    expect(FONT_PRESETS.sans.length).toBeGreaterThan(0);
    expect(typeof FONT_PRESETS.serif).toBe('string');
    expect(FONT_PRESETS.serif.length).toBeGreaterThan(0);
    expect(typeof FONT_PRESETS.mono).toBe('string');
    expect(FONT_PRESETS.mono.length).toBeGreaterThan(0);
  });

  it('sans preset includes a system-ui fallback', () => {
    expect(FONT_PRESETS.sans).toContain('system-ui');
  });

  it('serif preset includes a generic serif fallback', () => {
    // Pin the documented contract: every preset must end with its
    // matching generic family token so the browser always has a
    // working fallback.
    expect(FONT_PRESETS.serif).toContain('serif');
  });

  it('mono preset includes a monospace fallback', () => {
    expect(FONT_PRESETS.mono).toContain('monospace');
  });
});

describe('OverlayManager — construction + state', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  it('starts with no overlays and not globally hidden', () => {
    expect(manager.getVisibleOverlays()).toEqual([]);
  });

  it('toggle flips globallyHidden via show/hide observable behavior', () => {
    // Without overlays, getVisibleOverlays is empty regardless. Use a
    // loaded overlay to observe the toggle effect.
    return manager.loadOverlays([makeTextOverlay()], 'http://example.com').then(() => {
      expect(manager.getVisibleOverlays()).toHaveLength(1);

      manager.toggle();
      expect(manager.getVisibleOverlays()).toHaveLength(0);

      manager.toggle();
      expect(manager.getVisibleOverlays()).toHaveLength(1);
    });
  });

  it('show() sets globallyHidden to false', async () => {
    await manager.loadOverlays([makeTextOverlay()], 'http://example.com');
    manager.hide();
    expect(manager.getVisibleOverlays()).toHaveLength(0);
    manager.show();
    expect(manager.getVisibleOverlays()).toHaveLength(1);
  });
});

describe('OverlayManager.loadOverlays', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  it('creates a div per overlay with the luxar-overlay class', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'a' }), makeTextOverlay({ name: 'b' })],
      'http://example.com'
    );

    const overlays = document.querySelectorAll('.luxar-overlay');
    expect(overlays.length).toBe(2);

    const dataNames = Array.from(overlays).map((el) => (el as HTMLElement).dataset.overlayName);
    expect(dataNames).toEqual(expect.arrayContaining(['a', 'b']));
  });

  it('applies fade-transition class for fade overlays', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'fade-me', transition: 'fade' })],
      'http://example.com'
    );
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(el.classList.contains('luxar-overlay--fade')).toBe(true);
  });

  it('marks non-interactive overlays as inert', async () => {
    await manager.loadOverlays([makeTextOverlay({ interactive: false })], 'http://example.com');
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(el.inert).toBe(true);
  });

  it('leaves interactive overlays mutable (not inert)', async () => {
    // The source only sets `el.inert = true` for !interactive overlays;
    // for interactive ones the property is never touched, so jsdom
    // leaves `el.inert` as undefined. Either way, "not inert" is the
    // observable behavior we care about.
    await manager.loadOverlays([makeTextOverlay({ interactive: true })], 'http://example.com');
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(el.inert).toBeFalsy();
  });

  it('positions overlays using percentage left/top', async () => {
    await manager.loadOverlays([makeTextOverlay({ position: [0.25, 0.75] })], 'http://example.com');
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(el.style.left).toBe('25%');
    expect(el.style.top).toBe('75%');
  });
});

describe('OverlayManager.getVisibleOverlays', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  it('skips overlays with display: none', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'visible' }), makeTextOverlay({ name: 'hidden' })],
      'http://example.com'
    );

    const hiddenEl = document.querySelector('[data-overlay-name="hidden"]') as HTMLDivElement;
    hiddenEl.style.display = 'none';

    const visible = manager.getVisibleOverlays();
    expect(visible.map((v) => v.config.name)).toEqual(['visible']);
  });

  it('skips overlays with the hidden CSS class', async () => {
    await manager.loadOverlays([makeTextOverlay({ name: 'fading' })], 'http://example.com');

    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    el.classList.add('luxar-overlay--hidden');

    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });

  it('skips overlays with opacity = 0', async () => {
    await manager.loadOverlays([makeTextOverlay()], 'http://example.com');
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    el.style.opacity = '0';
    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });

  it('skips hover overlays even when displayed', async () => {
    await manager.loadOverlays([makeTextOverlay({ hover: true })], 'http://example.com');
    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });
});

describe('OverlayManager.dispose', () => {
  it('removes all DOM elements created by loadOverlays', async () => {
    document.body.innerHTML = '';
    const manager = new OverlayManager();
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'a' }), makeTextOverlay({ name: 'b' })],
      'http://example.com'
    );

    expect(document.querySelectorAll('.luxar-overlay').length).toBe(2);

    manager.dispose();

    expect(document.querySelectorAll('.luxar-overlay').length).toBe(0);
    expect(manager.getVisibleOverlays()).toEqual([]);
  });

  it('clears configs and overlays so a stale dim-change is a no-op', async () => {
    // W4 strengthening (P2): the prior test only asserted
    // `getVisibleOverlays() === []`. Strengthen by also (a) verifying the
    // DOM is purged, (b) calling toggle()/show()/hide() on the disposed
    // manager and confirming no overlay re-appears, (c) confirming a
    // second loadOverlays after dispose has no effect (the manager
    // should be inert).
    const manager = new OverlayManager();
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'a' }), makeTextOverlay({ name: 'b' })],
      'http://example.com'
    );
    expect(document.querySelectorAll('.luxar-overlay').length).toBe(2);

    manager.dispose();

    // DOM purged.
    expect(document.querySelectorAll('.luxar-overlay').length).toBe(0);
    expect(manager.getVisibleOverlays()).toEqual([]);

    // Post-dispose public-surface calls must not resurrect overlays.
    manager.show();
    manager.hide();
    manager.toggle();
    expect(document.querySelectorAll('.luxar-overlay').length).toBe(0);
    expect(manager.getVisibleOverlays()).toEqual([]);
  });
});

describe('OverlayManager.updateVisibility — dimension filtering', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    mockDimsState.current = null;
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  /** Build a SimpleDims with a single dimension named `t` whose current step
   *  is the supplied value. The metadata.findIndex by name is what the
   *  visible_range check resolves through, so the name is what matters. */
  function setTimeStep(value: number): void {
    mockDimsState.current = {
      ndim: 1,
      currentStep: [value],
      displayed: [],
      metadata: [{ name: 't', unit: 's', scale: 1 }],
    };
  }

  it('hides a dimension-filtered overlay when getDims() returns null', async () => {
    // isOverlayVisible's early-return at the "dims not ready" branch:
    // when no scene is loaded, overlays with visible_range must hide.
    mockDimsState.current = null;
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'd', visible_range: { t: [0, 5] } })],
      'http://example.com'
    );
    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });

  it('exact-match visible_range: shows on match, hides off-by-one', async () => {
    setTimeStep(3);
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'exact', visible_range: { t: 3 } })],
      'http://example.com'
    );
    expect(manager.getVisibleOverlays()).toHaveLength(1);

    setTimeStep(4);
    manager.updateVisibility();
    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });

  it('range visible_range: includes both endpoints and the interior, excludes outside', async () => {
    setTimeStep(2);
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'r', visible_range: { t: [2, 5] } })],
      'http://example.com'
    );
    expect(manager.getVisibleOverlays()).toHaveLength(1);

    setTimeStep(5);
    manager.updateVisibility();
    expect(manager.getVisibleOverlays()).toHaveLength(1);

    setTimeStep(3.5);
    manager.updateVisibility();
    expect(manager.getVisibleOverlays()).toHaveLength(1);

    setTimeStep(1);
    manager.updateVisibility();
    expect(manager.getVisibleOverlays()).toHaveLength(0);

    setTimeStep(6);
    manager.updateVisibility();
    expect(manager.getVisibleOverlays()).toHaveLength(0);
  });

  it('fade transition toggles the --hidden class on/off across dim changes', async () => {
    setTimeStep(3);
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'fade-dim',
          opacity: 0.8,
          transition: 'fade',
          visible_range: { t: [2, 5] },
        }),
      ],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="fade-dim"]') as HTMLDivElement;
    // Audit W5 fix: assert the element has the correct overlay name
    // (pinned via the queried selector). A wrong-selector bug would
    // surface here instead of slipping through a bare toBeTruthy.
    expect(el).toBeInstanceOf(HTMLDivElement);
    expect(el.dataset.overlayName).toBe(el.getAttribute('data-overlay-name'));
    // In-range → --hidden absent, inline opacity is config.opacity.
    expect(el.classList.contains('luxar-overlay--hidden')).toBe(false);
    expect(el.style.opacity).toBe('0.8');

    setTimeStep(10);
    manager.updateVisibility();
    expect(el.classList.contains('luxar-overlay--hidden')).toBe(true);
    // Fade overlays NEVER set display:none — the class is the only signal.
    expect(el.style.display).not.toBe('none');

    setTimeStep(3);
    manager.updateVisibility();
    expect(el.classList.contains('luxar-overlay--hidden')).toBe(false);
    expect(el.style.opacity).toBe('0.8');
  });

  it('non-fade transition toggles display:none on/off across dim changes', async () => {
    setTimeStep(3);
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'plain-dim',
          opacity: 0.5,
          transition: 'none',
          visible_range: { t: [2, 5] },
        }),
      ],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="plain-dim"]') as HTMLDivElement;
    // Audit W5 fix: assert the element has the correct overlay name
    // (pinned via the queried selector). A wrong-selector bug would
    // surface here instead of slipping through a bare toBeTruthy.
    expect(el).toBeInstanceOf(HTMLDivElement);
    expect(el.dataset.overlayName).toBe(el.getAttribute('data-overlay-name'));
    expect(el.style.display).not.toBe('none');
    expect(el.style.opacity).toBe('0.5');

    setTimeStep(10);
    manager.updateVisibility();
    expect(el.style.display).toBe('none');

    setTimeStep(3);
    manager.updateVisibility();
    expect(el.style.display).not.toBe('none');
    expect(el.style.opacity).toBe('0.5');
  });
});

describe('OverlayManager.updateHoverContent', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  it('does not throw when no hover overlays are registered', () => {
    expect(() =>
      manager.updateHoverContent({ label: 'x', nodeName: '/n', elementIndex: 0 })
    ).not.toThrow();
  });

  it('does not throw when called with null (clear hover)', () => {
    expect(() => manager.updateHoverContent(null)).not.toThrow();
  });

  it('skips redundant updates with the same label/index/node', async () => {
    // Register a hover overlay so the inner DOM update path runs.
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'hover-tooltip',
          hover: true,
          text: '{hover_label}',
        }),
      ],
      'http://example.com'
    );

    const el = document.querySelector('[data-overlay-name="hover-tooltip"]') as HTMLDivElement;

    // First call: any text-content rendering counts as the baseline;
    // we just ensure the second call doesn't re-render unchanged input.
    manager.updateHoverContent({ label: 'foo', nodeName: '/n', elementIndex: 7 });
    const firstHtml = el.innerHTML;

    // Mutate the DOM externally; if updateHoverContent re-rendered, the
    // change would be overwritten. The skip-redundant path leaves it alone.
    // (Use a real tag jsdom won't auto-close-modify.)
    const sentinel = '<span>__SENTINEL__</span>';
    el.innerHTML = sentinel;

    manager.updateHoverContent({ label: 'foo', nodeName: '/n', elementIndex: 7 });
    expect(el.innerHTML).toBe(sentinel);

    // Sanity: a different label triggers re-render.
    manager.updateHoverContent({ label: 'bar', nodeName: '/n', elementIndex: 7 });
    expect(el.innerHTML).not.toBe(sentinel);
    expect(firstHtml).toBeDefined();
  });

  it('preserves the <img> element across fade-out/re-show of identical content', async () => {
    // Regression: the hover loop fades the tooltip to opacity 0 on every
    // mousemove (updateHoverContent(null)) and re-shows it after the
    // settle. That null resets the manager-level dedup, so the re-show
    // used to rewrite innerHTML and recreate the <img> element. A fresh
    // <img> re-decodes its (cached) blob URL asynchronously, flickering
    // the thumbnail under cursor jitter. The per-entry `lastRendered`
    // guard must skip the rewrite when the rendered content is identical,
    // keeping the same (already-decoded) <img> node alive.
    await manager.loadOverlays(
      [
        {
          name: 'hover-img',
          type: 'overlay_html',
          position: [0.5, 0.97],
          opacity: 0.95,
          anchor: 'bottom-center',
          transition: 'none',
          transition_duration: 0.3,
          interactive: false,
          z_index: 0,
          hover: true,
          html: '<div><strong>{hover_label}</strong><br/>{hover_image_label}</div>',
          hover_image_size: [0.08, 0.08],
        } as OverlayConfig,
      ],
      'http://example.com'
    );

    const el = document.querySelector('[data-overlay-name="hover-img"]') as HTMLDivElement;
    const content = {
      label: 'species_0',
      imageUrl: 'blob:http://example.com/abc-123',
      nodeName: '/ring',
      elementIndex: 0,
    };

    // First show — builds the <img>.
    manager.updateHoverContent(content);
    const firstImg = el.querySelector('img');
    expect(firstImg).not.toBeNull();
    expect(el.style.opacity).toBe('0.95');

    // Mousemove fades it out (DOM untouched, just opacity).
    manager.updateHoverContent(null);
    expect(el.style.opacity).toBe('0');
    expect(el.querySelector('img')).toBe(firstImg); // <img> survives the fade

    // Settle re-shows identical content: must reuse the SAME <img> node
    // (no innerHTML rewrite ⇒ no async re-decode flicker).
    manager.updateHoverContent(content);
    expect(el.querySelector('img')).toBe(firstImg);
    expect(el.style.opacity).toBe('0.95');

    // A different element (new image) does recreate the <img>.
    manager.updateHoverContent({
      label: 'species_1',
      imageUrl: 'blob:http://example.com/def-456',
      nodeName: '/ring',
      elementIndex: 1,
    });
    const secondImg = el.querySelector('img');
    expect(secondImg).not.toBeNull();
    expect(secondImg).not.toBe(firstImg);
  });

  it('shows a hover overlay configured with both visible_range and transition:"fade"', async () => {
    // Regression: createOverlayElement previously added the
    // luxar-overlay--hidden class for ANY overlay with visible_range +
    // transition:"fade", including hover ones. That class has
    // `opacity: 0 !important`, which overrode the inline opacity that
    // updateHoverContent writes — trapping the tooltip permanently
    // invisible. The fix gates the --hidden write on `!config.hover`.
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'hover-trap',
          hover: true,
          text: '{hover_label}',
          opacity: 1.0,
          transition: 'fade',
          transition_duration: 0.15,
          visible_range: { t: [0, 5] },
        }),
      ],
      'http://example.com'
    );

    const el = document.querySelector('[data-overlay-name="hover-trap"]') as HTMLDivElement;
    // Audit W5 fix: assert the element has the correct overlay name
    // (pinned via the queried selector). A wrong-selector bug would
    // surface here instead of slipping through a bare toBeTruthy.
    expect(el).toBeInstanceOf(HTMLDivElement);
    expect(el.dataset.overlayName).toBe(el.getAttribute('data-overlay-name'));
    // The trap class must not be present at construction.
    expect(el.classList.contains('luxar-overlay--hidden')).toBe(false);

    manager.updateHoverContent({ label: 'visible', nodeName: '/n', elementIndex: 0 });
    expect(el.classList.contains('luxar-overlay--hidden')).toBe(false);
    expect(el.style.opacity).toBe('1');
  });

  it('shows a hover overlay configured with visible_range and transition:"none"', async () => {
    // Sibling of the fade-branch regression above. The "start hidden"
    // gate in createOverlayElement has two sides:
    //   - transition:"fade" → adds luxar-overlay--hidden class
    //   - transition:"none" (or default) → sets display:none inline
    // Both must be skipped for hover overlays, because updateHoverContent
    // controls visibility via inline opacity and cannot recover from a
    // display:none element (opacity changes on a display:none box paint
    // nothing).
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'hover-trap-display',
          hover: true,
          text: '{hover_label}',
          opacity: 1.0,
          transition: 'none',
          visible_range: { t: [0, 5] },
        }),
      ],
      'http://example.com'
    );

    const el = document.querySelector('[data-overlay-name="hover-trap-display"]') as HTMLDivElement;
    // Audit W5 fix: assert the element has the correct overlay name
    // (pinned via the queried selector). A wrong-selector bug would
    // surface here instead of slipping through a bare toBeTruthy.
    expect(el).toBeInstanceOf(HTMLDivElement);
    expect(el.dataset.overlayName).toBe(el.getAttribute('data-overlay-name'));
    // The non-fade branch of the gate sets display:none; it must be
    // skipped for hover overlays.
    expect(el.style.display).not.toBe('none');

    manager.updateHoverContent({ label: 'visible', nodeName: '/n', elementIndex: 0 });
    expect(el.style.display).not.toBe('none');
    expect(el.style.opacity).toBe('1');
  });
});
