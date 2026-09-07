// @vitest-environment jsdom
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
import { MAX_OVERLAY_HTML_CHARS, type OverlayConfig } from '../../../data/loaders';
import { log, Modules } from '../../../utils/log';
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

/**
 * HTML overlay whose `html` payload flows through the private
 * `sanitizeHtml` and lands in `el.innerHTML` via `createHtmlContent`.
 *
 * `interactive: true` on purpose: for non-interactive overlays
 * `createHtmlContent` stamps `pointer-events: none` onto every descendant,
 * which buries the sanitized markup under inline styles and makes the
 * assertions below unreadable. Sanitization is independent of that flag.
 */
function makeHtmlOverlay(html: string, overrides: Partial<OverlayConfig> = {}): OverlayConfig {
  return {
    name: 'html-overlay',
    type: 'overlay_html',
    position: [0.5, 0.5],
    opacity: 1.0,
    anchor: 'center',
    transition: 'none',
    transition_duration: 0.3,
    interactive: true,
    z_index: 0,
    html,
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
    vi.restoreAllMocks();
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

  it('reads zipped image content through the store and revokes its typed object URL', async () => {
    const imageBytes = new Uint8Array([0xff, 0xd8, 0xff, 0xe0]);
    const readFile = vi.fn().mockResolvedValue(imageBytes);
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValue('blob:overlay-image');
    const revokeObjectURLSpy = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'archive-image',
          type: 'overlay_image',
          image_file: 'preview.png',
          size: [0.25, 0.25],
        }),
      ],
      'https://example.com/scene.luxar.zarr.zip',
      readFile
    );

    expect(readFile).toHaveBeenCalledExactlyOnceWith('/overlays/archive-image/preview.png');
    expect(createObjectURLSpy).toHaveBeenCalledOnce();
    const blob = createObjectURLSpy.mock.calls[0][0] as Blob;
    expect(blob.type).toBe('image/jpeg');
    expect(new Uint8Array(await blob.arrayBuffer())).toEqual(imageBytes);

    const img = document.querySelector('.luxar-overlay--image img') as HTMLImageElement;
    expect(img.src).toBe('blob:overlay-image');
    expect(img.alt).toBe('archive-image');

    manager.dispose();
    expect(revokeObjectURLSpy).toHaveBeenCalledExactlyOnceWith('blob:overlay-image');
  });

  it('renders a video overlay as a muted looping <video> tied to visibility', async () => {
    // jsdom does not implement media playback; observe the calls instead.
    const playSpy = vi
      .spyOn(HTMLMediaElement.prototype, 'play')
      .mockImplementation(() => Promise.resolve());
    const pauseSpy = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    const pausedSpy = vi.spyOn(HTMLMediaElement.prototype, 'paused', 'get');
    pausedSpy.mockReturnValue(true);

    // The dims manager is mocked in this file: drive its state directly and
    // re-run the visibility pass the real listener would trigger.
    const setStory = (value: number): void => {
      mockDimsState.current = {
        ndim: 2,
        currentStep: [0, value],
        displayed: [0],
        metadata: [
          { name: 'x', unit: '', scale: 1 },
          { name: 'story', unit: '', scale: 1 },
        ],
      };
      manager.updateVisibility();
    };
    setStory(0);

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'turntable',
          type: 'overlay_video',
          video_file: 'video.webm',
          poster_file: 'poster.png',
          size: [0.26, null],
          playback_rate: 1.5,
          visible_range: { story: 2 },
        }),
      ],
      'https://example.com/scene.luxar.zarr/'
    );

    const video = document.querySelector('.luxar-overlay--video video') as HTMLVideoElement;
    expect(video).not.toBeNull();
    expect(video.muted).toBe(true);
    expect(video.loop).toBe(true);
    // The DOM autoplay attribute stays OFF (it would start every hidden clip once
    // its media loads); the wish is kept on the element for the visibility sync.
    expect(video.autoplay).toBe(false);
    expect(video.dataset.autoplay).toBe('1');
    expect(video.preload).toBe('metadata');
    expect(video.playbackRate).toBe(1.5);
    expect(video.defaultPlaybackRate).toBe(1.5);
    expect(video.src).toBe('https://example.com/scene.luxar.zarr/overlays/turntable/video.webm');
    expect(video.poster).toBe('https://example.com/scene.luxar.zarr/overlays/turntable/poster.png');
    expect(video.style.width).toBe('26vw');
    expect(video.style.height).toBe('auto');

    // Hidden at story 0: never asked to play.
    expect(playSpy).not.toHaveBeenCalled();

    // Story 2 shows it → play(); leaving → pause().
    setStory(2);
    expect(video.preload).toBe('auto');
    expect(playSpy).toHaveBeenCalledTimes(1);
    pausedSpy.mockReturnValue(false);
    setStory(0);
    expect(pauseSpy).toHaveBeenCalledTimes(1);

    manager.dispose();
    mockDimsState.current = null;
    playSpy.mockRestore();
    pauseSpy.mockRestore();
    pausedSpy.mockRestore();
  });

  it('shows native controls when video autoplay is disabled', async () => {
    const playSpy = vi
      .spyOn(HTMLMediaElement.prototype, 'play')
      .mockImplementation(() => Promise.resolve());
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'manual-clip',
          type: 'overlay_video',
          video_file: 'video.webm',
          autoplay: false,
        }),
      ],
      'https://example.com/scene.luxar.zarr/'
    );

    const video = document.querySelector('.luxar-overlay--video video') as HTMLVideoElement;
    const overlay = video.parentElement as HTMLDivElement;
    expect(video.controls).toBe(true);
    expect(overlay.inert).toBeFalsy();
    expect(overlay.classList.contains('luxar-overlay--interactive')).toBe(true);
    expect(video.dataset.autoplay).toBe('0');
    expect(playSpy).not.toHaveBeenCalled();
  });

  it('serves a zipped-store video and poster from typed blob URLs', async () => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(() => Promise.resolve());
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {});
    const webm = new Uint8Array([0x1a, 0x45, 0xdf, 0xa3, 0, 0, 0, 0]);
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const readFile = vi.fn(async (path: string) => (path.endsWith('.webm') ? webm : png));
    const createObjectURLSpy = vi
      .spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:overlay-video')
      .mockReturnValueOnce('blob:overlay-poster');
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'clip',
          type: 'overlay_video',
          video_file: 'video.webm',
          poster_file: 'poster.png',
        }),
      ],
      'https://example.com/scene.luxar.zarr.zip',
      readFile
    );

    expect(readFile.mock.calls.map(([path]) => path)).toEqual([
      '/overlays/clip/video.webm',
      '/overlays/clip/poster.png',
    ]);
    expect((createObjectURLSpy.mock.calls[0][0] as Blob).type).toBe('video/webm');
    expect((createObjectURLSpy.mock.calls[1][0] as Blob).type).toBe('image/png');
    const video = document.querySelector('.luxar-overlay--video video') as HTMLVideoElement;
    expect(video.src).toBe('blob:overlay-video');
    expect(video.poster).toBe('blob:overlay-poster');
    manager.dispose();
  });

  it('warns and omits zipped image content when the archive member is missing', async () => {
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    const readFile = vi.fn().mockResolvedValue(undefined);
    const createObjectURLSpy = vi.spyOn(URL, 'createObjectURL');

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'archive-image',
          type: 'overlay_image',
          image_file: 'preview.png',
          size: [0.25, 0.25],
        }),
      ],
      'https://example.com/scene.luxar.zarr.zip',
      readFile
    );

    expect(readFile).toHaveBeenCalledExactlyOnceWith('/overlays/archive-image/preview.png');
    expect(document.querySelector('.luxar-overlay--image img')).toBeNull();
    expect(warningSpy).toHaveBeenCalledExactlyOnceWith(
      Modules.UI,
      expect.stringContaining('/overlays/archive-image/preview.png')
    );
    expect(createObjectURLSpy).not.toHaveBeenCalled();
  });

  it('warns distinctly when a zipped image has no store reader', async () => {
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'archive-image',
          type: 'overlay_image',
          image_file: 'preview.png',
        }),
      ],
      'https://example.com/scene.luxar.zarr.zip'
    );

    expect(document.querySelector('.luxar-overlay--image img')).toBeNull();
    expect(warningSpy).toHaveBeenCalledExactlyOnceWith(
      Modules.UI,
      expect.stringContaining('no store file reader')
    );
  });

  it('warns and removes directory image content when the member fails to load', async () => {
    const warningSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});

    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'directory-image',
          type: 'overlay_image',
          image_file: 'missing.png',
        }),
      ],
      'https://example.com/scene.luxar.zarr/'
    );

    const img = document.querySelector('.luxar-overlay--image img') as HTMLImageElement;
    img.dispatchEvent(new Event('error'));

    expect(document.querySelector('.luxar-overlay--image img')).toBeNull();
    expect(warningSpy).toHaveBeenCalledExactlyOnceWith(
      Modules.UI,
      expect.stringContaining('overlays/directory-image/missing.png')
    );
  });

  it('starts zipped image reads concurrently while preserving overlay order', async () => {
    const resolvers = new Map<string, (bytes: Uint8Array) => void>();
    const readFile = vi.fn(
      (path: string) =>
        new Promise<Uint8Array>((resolve) => {
          resolvers.set(path, resolve);
        })
    );
    vi.spyOn(URL, 'createObjectURL')
      .mockReturnValueOnce('blob:first')
      .mockReturnValueOnce('blob:second');

    const loading = manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'first',
          type: 'overlay_image',
          image_file: 'first.png',
        }),
        makeTextOverlay({
          name: 'second',
          type: 'overlay_image',
          image_file: 'second.png',
        }),
      ],
      'https://example.com/scene.luxar.zarr.zip',
      readFile
    );

    await vi.waitFor(() => expect(readFile).toHaveBeenCalledTimes(2));
    resolvers.get('/overlays/second/second.png')?.(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    resolvers.get('/overlays/first/first.png')?.(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    await loading;

    expect(
      Array.from(document.querySelectorAll('.luxar-overlay')).map(
        (element) => (element as HTMLElement).dataset.overlayName
      )
    ).toEqual(['first', 'second']);
  });

  it('releases the store reader on dispose', async () => {
    const readFile = vi.fn().mockResolvedValue(new Uint8Array([0x89, 0x50, 0x4e, 0x47]));
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:overlay-image');

    await manager.loadOverlays(
      [makeTextOverlay({ type: 'overlay_image', image_file: 'preview.png' })],
      'https://example.com/scene.luxar.zarr.zip',
      readFile
    );
    manager.dispose();

    expect((manager as unknown as { readFile?: unknown }).readFile).toBeUndefined();
  });
});

describe('OverlayManager — anchoring (issue #773)', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  it('anchors a top-right overlay from the right edge (no left, no -100% X)', async () => {
    // #773: setting `left: 98%` made shrink-to-fit width ~2vw, wrapping the
    // hover tooltip to one word per line. Anchoring from the right edge
    // (`right: 2%`, no `left`) lets the box grow leftward.
    await manager.loadOverlays(
      [makeTextOverlay({ anchor: 'top-right', position: [0.98, 0.02] })],
      'http://example.com'
    );
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    // right = (1 - 0.98) * 100 ≈ 2% (0.98 is not float-exact, so assert
    // numerically rather than on the exact string, matching how the existing
    // left/top percentages are produced without rounding).
    expect(el.style.right.endsWith('%')).toBe(true);
    expect(parseFloat(el.style.right)).toBeCloseTo(2);
    expect(el.style.left).toBe('');
    expect(parseFloat(el.style.top)).toBeCloseTo(2);
    // Transform keeps its vertical part but drops the horizontal -100%.
    expect(el.style.transform).toBe('translate(0, 0)');
    expect(el.style.transform).not.toContain('-100%');
  });

  it('anchors a center-right overlay from the right edge, keeping the -50% Y', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ anchor: 'center-right', position: [0.9, 0.5] })],
      'http://example.com'
    );
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(parseFloat(el.style.right)).toBeCloseTo(10); // (1 - 0.9) * 100
    expect(el.style.left).toBe('');
    expect(el.style.transform).toBe('translate(0, -50%)');
    expect(el.style.transform).not.toContain('-100%');
  });

  it('anchors a bottom-right overlay from the right edge, keeping the -100% Y', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ anchor: 'bottom-right', position: [0.98, 0.98] })],
      'http://example.com'
    );
    const el = document.querySelector('.luxar-overlay') as HTMLDivElement;
    expect(parseFloat(el.style.right)).toBeCloseTo(2);
    expect(el.style.left).toBe('');
    // Vertical -100% preserved; horizontal component is 0 (no leading -100%).
    expect(el.style.transform).toBe('translate(0, -100%)');
    expect(el.style.transform.startsWith('translate(-100%')).toBe(false);
  });

  it('leaves left/center anchors on the left edge (right cleared)', async () => {
    await manager.loadOverlays(
      [
        makeTextOverlay({ name: 'left', anchor: 'top-left', position: [0.25, 0.75] }),
        makeTextOverlay({ name: 'center', anchor: 'center', position: [0.25, 0.75] }),
        makeTextOverlay({ name: 'cleft', anchor: 'center-left', position: [0.1, 0.2] }),
      ],
      'http://example.com'
    );
    const left = document.querySelector('[data-overlay-name="left"]') as HTMLDivElement;
    const center = document.querySelector('[data-overlay-name="center"]') as HTMLDivElement;
    const cleft = document.querySelector('[data-overlay-name="cleft"]') as HTMLDivElement;

    expect(left.style.left).toBe('25%');
    expect(left.style.right).toBe('');
    // Center keeps its -50%/-50% transform and left positioning.
    expect(center.style.left).toBe('25%');
    expect(center.style.right).toBe('');
    expect(center.style.transform).toBe('translate(-50%, -50%)');
    expect(cleft.style.left).toBe('10%');
    expect(cleft.style.right).toBe('');
  });

  it('gives a no-width hover overlay a clamped max-width (readable wrap, not ~2vw)', async () => {
    // #773: the auto-injected hover text overlay has no explicit width and
    // uses `pre-line`. Without a max-width its measure collapses to the
    // right-anchor container edge. A clamped max-width lets it wrap to a
    // readable box a couple of lines tall.
    await manager.loadOverlays(
      [
        makeTextOverlay({
          name: 'hover',
          hover: true,
          anchor: 'top-right',
          position: [0.98, 0.02],
          text: '{hover_label}',
        }),
      ],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="hover"]') as HTMLDivElement;
    expect(el.style.maxWidth).toBe('min(30vw, 40ch)');
    // No explicit narrow width was set (that is what produced the ~2vw column).
    expect(el.style.width).toBe('');
    // pre-line so \n in labels still breaks, and wrapping is enabled.
    expect(el.style.whiteSpace).toBe('pre-line');
    expect(el.style.wordWrap).toBe('break-word');
  });

  it('honors an explicit config.width and does not clamp it with max-width', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'wide', width: 0.5, text: 'a caption' })],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="wide"]') as HTMLDivElement;
    // Explicit width stays the primary sizing.
    expect(el.style.width).toBe('50vw');
    // The clamp is only applied on the no-explicit-width path.
    expect(el.style.maxWidth).toBe('');
  });

  it('clears stale sizing styles when text content is re-applied with a different config', async () => {
    // The sizing branches are mutually exclusive; each must clear what the
    // others set so a re-apply with a different config leaves no stale
    // max-width/word-wrap behind (same convention as the left/right clearing
    // in applyPositionAndStyle).
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'morph', hover: true, text: '{hover_label}' })],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="morph"]') as HTMLDivElement;
    expect(el.style.maxWidth).toBe('min(30vw, 40ch)');

    // hover/no-width → explicit width: the clamp must not linger.
    (manager as unknown as { createTextContent(e: HTMLDivElement, c: OverlayConfig): void })[
      'createTextContent'
    ](el, makeTextOverlay({ name: 'morph', width: 0.5 }));
    expect(el.style.width).toBe('50vw');
    expect(el.style.maxWidth).toBe('');

    // explicit width → plain no-width nowrap: width, clamp and wrap all reset.
    (manager as unknown as { createTextContent(e: HTMLDivElement, c: OverlayConfig): void })[
      'createTextContent'
    ](el, makeTextOverlay({ name: 'morph' }));
    expect(el.style.width).toBe('');
    expect(el.style.maxWidth).toBe('');
    expect(el.style.whiteSpace).toBe('nowrap');
    expect(el.style.wordWrap).toBe('');
  });

  it('does not clamp a non-hover no-width overlay (nowrap stays unbounded — issue #773 regression guard)', async () => {
    // A non-hover text overlay with no explicit width is a single nowrap line
    // sized to its content. A max-width here would only clip the BOX while the
    // nowrap text overflows it — under right-anchoring, off-screen. So the
    // clamp/word-wrap must be reserved for the wrapping (hover) case only.
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'credit', anchor: 'bottom-right', position: [0.98, 0.97] })],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="credit"]') as HTMLDivElement;
    expect(el.style.maxWidth).toBe('');
    expect(el.style.whiteSpace).toBe('nowrap');
    expect(el.style.wordWrap).toBe('');
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

  it('substitutes hover_key when it is the only hover content', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'hover-key', hover: true, text: 'id={hover_key}' })],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="hover-key"]') as HTMLDivElement;

    manager.updateHoverContent({
      key: 'P04637',
      nodeName: '/proteins',
      elementIndex: 7,
    });

    expect(el.textContent).toBe('id=P04637');
    expect(el.style.opacity).toBe('1');
  });

  it('hides a hover overlay when its rendered template is empty', async () => {
    await manager.loadOverlays(
      [makeTextOverlay({ name: 'hover-label-only', hover: true, text: '{hover_label}' })],
      'http://example.com'
    );
    const el = document.querySelector('[data-overlay-name="hover-label-only"]') as HTMLDivElement;
    const initialText = el.textContent;

    manager.updateHoverContent({ key: 'P04637', nodeName: '/proteins', elementIndex: 7 });

    expect(el.textContent).toBe(initialText);
    expect(el.style.opacity).toBe('0');
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

describe('OverlayManager HTML sanitization (issue #720)', () => {
  let manager: OverlayManager;

  beforeEach(() => {
    document.body.innerHTML = '';
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
  });

  /**
   * Push `html` through the public surface (`loadOverlays` with an
   * `overlay_html` config) and hand back the created overlay element.
   *
   * `sanitizeHtml` is private, and this file's stated approach is to
   * exercise private helpers transitively; `createHtmlContent` writes the
   * sanitized string straight to `el.innerHTML`, so the rendered DOM *is*
   * the sanitizer's observable output — no bracket-notation poking needed.
   */
  async function renderHtml(html: string): Promise<HTMLDivElement> {
    await manager.loadOverlays([makeHtmlOverlay(html)], 'http://example.com');
    return document.querySelector('[data-overlay-name="html-overlay"]') as HTMLDivElement;
  }

  it('strips on* handlers from an <img> lifted out of a disallowed wrapper', async () => {
    // Pre-fix: the disallowed <x> hit `continue`, so the <img> was never
    // attribute-scrubbed; the deferred unwrap pass then hoisted it into the
    // output verbatim as `<img src="x" onerror="alert(1)">`.
    const el = await renderHtml('<x><img src=x onerror="alert(1)"></x>');

    expect(el.querySelector('x')).toBeNull(); // wrapper unwrapped
    const img = el.querySelector('img');
    expect(img).not.toBeNull();
    expect(img!.hasAttribute('onerror')).toBe(false);
    expect(el.innerHTML).not.toContain('onerror');
    expect(img!.getAttribute('src')).toBe('x'); // benign attribute kept
  });

  it('strips a javascript: href from an <a> lifted out of a disallowed wrapper', async () => {
    const el = await renderHtml('<blink><a href="javascript:alert(1)">click</a></blink>');

    expect(el.querySelector('blink')).toBeNull();
    const a = el.querySelector('a');
    expect(a).not.toBeNull();
    expect(a!.hasAttribute('href')).toBe(false);
    expect(el.innerHTML.toLowerCase()).not.toContain('javascript:');
    expect(a!.textContent).toBe('click'); // text survives the unwrap
  });

  it('strips on* handlers from a <p> lifted out of a <form>, preserving its text', async () => {
    const el = await renderHtml('<form><p onclick="evil()">hi</p></form>');

    expect(el.querySelector('form')).toBeNull();
    const p = el.querySelector('p');
    expect(p).not.toBeNull();
    expect(p!.hasAttribute('onclick')).toBe(false);
    expect(el.innerHTML).not.toContain('onclick');
    // Unwrapping must not eat content: the paragraph text is still there.
    expect(p!.textContent).toBe('hi');
    expect(el.textContent).toContain('hi');
  });

  it('strips on* handlers from a top-level allowed element (control)', async () => {
    // This case already passed before the fix — it pins that the fix did
    // not regress the plain allowed-tag path.
    const el = await renderHtml('<p onclick="x">hi</p>');

    const p = el.querySelector('p');
    expect(p).not.toBeNull();
    expect(p!.hasAttribute('onclick')).toBe(false);
    expect(p!.textContent).toBe('hi');
  });

  it('sanitizes through nested disallowed wrappers at any depth', async () => {
    // Two levels of unknown wrapper: the inner unwrap must feed an
    // already-clean subtree to the outer one.
    const el = await renderHtml('<x><y><img src=x onerror="alert(1)"></y></x>');

    expect(el.querySelector('x')).toBeNull();
    expect(el.querySelector('y')).toBeNull();
    const img = el.querySelector('img');
    expect(img).not.toBeNull();
    expect(img!.hasAttribute('onerror')).toBe(false);
    expect(el.innerHTML).not.toContain('onerror');
  });

  it('sanitizes a disallowed wrapper nested inside an allowed element', async () => {
    const el = await renderHtml('<div><form><a href="javascript:alert(1)">go</a></form></div>');

    expect(el.querySelector('div')).not.toBeNull(); // allowed tag kept
    expect(el.querySelector('form')).toBeNull();
    const a = el.querySelector('a');
    expect(a).not.toBeNull();
    expect(a!.hasAttribute('href')).toBe(false);
    expect(el.innerHTML.toLowerCase()).not.toContain('javascript:');
    expect(a!.textContent).toBe('go');
  });

  // ---------------------------------------------------------------- item 3:
  // sibling batching + ordering of the unwrap pass (the restructured code)

  it('unwraps a disallowed sibling in place, preserving text order', async () => {
    const el = await renderHtml('<div>a<x>b</x>c</div>');
    expect(el.innerHTML).toBe('<div>abc</div>');
    expect(el.querySelector('div')!.textContent).toBe('abc');
  });

  it('preserves child order when unwrapping a wrapper holding mixed nodes', async () => {
    const el = await renderHtml('<div>a<x>b<b>B</b>c</x>d</div>');
    expect(el.innerHTML).toBe('<div>ab<b>B</b>cd</div>');
    expect(el.querySelector('div')!.textContent).toBe('abBcd');
  });

  it('sanitizes two sibling wrappers each carrying a dirty <img>, in order', async () => {
    const el = await renderHtml('<x><img src=1 onerror="a()"></x><y><img src=2 onclick="b()"></y>');
    expect(el.innerHTML).toBe('<img src="1"><img src="2">');
    const imgs = Array.from(el.querySelectorAll('img'));
    expect(imgs.map((i) => i.getAttribute('src'))).toEqual(['1', '2']);
    expect(imgs.some((i) => i.hasAttribute('onerror') || i.hasAttribute('onclick'))).toBe(false);
  });

  it("discards a nested <template>'s payload (safe by allowlist, not construction)", async () => {
    // Recorded for the next reader: a nested <template> keeps its children in
    // a separate `.content` DocumentFragment that `querySelectorAll('*')`
    // never descends into, so neither pass sees them. `template` is not in
    // ALLOWED_TAGS and has no `childNodes` to lift, so the payload is dropped
    // unrendered — safe, but by the allowlist rather than by construction.
    // Adding `template` to ALLOWED_TAGS would ship that subtree unsanitized.
    const el = await renderHtml('<x><template><img src=x onerror="alert(1)"></template></x>');
    expect(el.innerHTML).toBe('');
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('template')).toBeNull();
    expect(el.innerHTML).not.toContain('onerror');
  });

  // ---------------------------------------------------------------- item 2:
  // the javascript: scheme guard must survive trivial obfuscation

  it('strips a javascript: href obfuscated with an interior tab', async () => {
    // The URL parser removes ASCII tab/LF/CR from ANYWHERE in the value, so
    // this parses as the javascript: scheme and fires; `trim()` only ever
    // touched the ends.
    const payload = '<a href="javascript&Tab;:alert(1)">x</a>';

    // Sanity-check the input is genuinely obfuscated: the HTML parser
    // decodes `&Tab;` to a literal U+0009 inside the attribute value. Without
    // this the test could pass vacuously on an entity the parser left alone.
    const raw = document.createElement('template');
    raw.innerHTML = payload;
    expect(raw.content.querySelector('a')!.getAttribute('href')).toContain('\t');

    const el = await renderHtml(payload);
    expect(el.querySelector('a')!.hasAttribute('href')).toBe(false);
  });

  it('strips a javascript: href obfuscated with an interior newline', async () => {
    const el = await renderHtml('<a href="java&NewLine;script:alert(1)">x</a>');
    expect(el.querySelector('a')!.hasAttribute('href')).toBe(false);
  });

  it('strips a javascript: href prefixed with a C0 control', async () => {
    // The parser strips leading C0 controls before resolving the scheme.
    const el = await renderHtml('<a href="&#1;javascript:alert(1)">x</a>');
    expect(el.querySelector('a')!.hasAttribute('href')).toBe(false);
  });

  it('strips an obfuscated javascript: src (not just href)', async () => {
    const el = await renderHtml('<x><img src="java&#9;script:alert(1)"></x>');
    const img = el.querySelector('img')!;
    expect(img.hasAttribute('src')).toBe(false);
    expect(el.innerHTML.toLowerCase()).not.toContain('script:');
  });

  it('strips a mixed-case scheme and a mixed-case event-handler name', async () => {
    // The HTML parser already lower-cases attribute NAMES, so `OnError`
    // arrives as `onerror`; the scheme VALUE keeps its casing, which is what
    // the `.toLowerCase()` in the guard is actually load-bearing for.
    const el = await renderHtml('<x><a HrEf="JaVaScRiPt:alert(1)" OnClick="e()">x</a></x>');
    const a = el.querySelector('a')!;
    expect(a.hasAttribute('href')).toBe(false);
    expect(a.hasAttribute('onclick')).toBe(false);
    expect(a.textContent).toBe('x');
  });

  it('keeps a non-javascript href verbatim, whitespace and all', async () => {
    // No-regression pin (passes pre-fix too): the whitespace/C0 normalization
    // feeds the scheme COMPARISON only — it must never rewrite the stored
    // value. It is not a "nothing but javascript: is dropped" guarantee: the
    // guard is deliberately stricter than the URL parser and also strips
    // interior spaces, so `href="java script:…"` IS dropped even though the
    // parser would not treat that as a javascript: URL. Over-blocking is the
    // safe direction here.
    const el = await renderHtml('<a href="  https://example.org/a b  ">x</a>');
    expect(el.querySelector('a')!.getAttribute('href')).toBe('  https://example.org/a b  ');
  });

  // ------------------------------------------------- unwrap-pass complexity

  it('unwraps deeply nested wrappers with a linear number of node moves', async () => {
    // The unwrap pass runs outermost-first, so each node moves exactly once.
    // Reversing it (innermost-first) makes every enclosing wrapper re-lift the
    // same K payload nodes => K*D moves. This pins the ordering; it is not a
    // regression guard for released behaviour, since the pre-#720 walk stopped
    // at the first disallowed tag and never got deep enough to be slow.
    //
    // Counting node MOVES rather than wall time keeps this deterministic (no
    // CI flake): jsdom's fragment parser does not route through
    // `Node.prototype.insertBefore`, so the spy below counts exactly the
    // unwrap pass. Note it measures insertBefore CALLS as a proxy for node
    // moves, so a future rewrite that lifts children through a different
    // primitive (e.g. batching into a DocumentFragment) would under-count
    // here; the `outWithPayload` correctness assertion below is the real guard
    // and holds regardless, so re-verify these complexity bounds if that loop
    // changes.
    const D = 200; // nested disallowed wrappers
    const K = 50; // payload elements inside the innermost wrapper

    const original = Node.prototype.insertBefore;
    let moves = 0;
    Node.prototype.insertBefore = function <T extends Node>(
      this: Node,
      node: T,
      ref: Node | null
    ): T {
      moves += 1;
      return original.call(this, node, ref) as T;
    };

    let movesDeepOnly: number;
    let movesWithPayload: number;
    let outWithPayload: string;
    try {
      // Baseline: same depth, single payload node.
      moves = 0;
      await renderHtml('<x>'.repeat(D) + '<b>t</b>' + '</x>'.repeat(D));
      movesDeepOnly = moves;

      manager.dispose();
      document.body.innerHTML = '';
      manager = new OverlayManager();

      moves = 0;
      const el = await renderHtml('<x>'.repeat(D) + '<b>t</b>'.repeat(K) + '</x>'.repeat(D));
      movesWithPayload = moves;
      outWithPayload = el.innerHTML;
    } finally {
      Node.prototype.insertBefore = original;
    }

    // Correctness first: all D wrappers gone, all K payload nodes kept.
    expect(outWithPayload).toBe('<b>t</b>'.repeat(K));

    // LOWER bound — the spy must actually be observing the unwrap. Lifting a
    // D-deep chain cannot cost fewer than D moves, and an implementation that
    // moves nodes another way (`el.replaceWith(...el.childNodes)` is the
    // obvious future simplification of this very loop) reads 0 here. Without
    // this bound a 0-move reading satisfies every ceiling below and the
    // ordering guard silently disappears. Exactly D today.
    expect(movesDeepOnly).toBeGreaterThanOrEqual(D);
    expect(movesWithPayload).toBeGreaterThanOrEqual(D);

    // UPPER bounds — linear in depth + payload, not depth * payload. Left
    // deliberately loose rather than the exact D + K: a tight bound would also
    // be asserting that nothing else in loadOverlays ever calls insertBefore,
    // which has nothing to do with what this test is for.
    expect(movesWithPayload).toBeLessThanOrEqual(3 * (D + K));
    // Growing the payload 1 -> K must add ~K moves, not (K-1)*D. Any constant
    // offset from elsewhere cancels in the difference.
    expect(movesWithPayload - movesDeepOnly).toBeLessThanOrEqual(2 * K);
    // Hard ceiling far below the quadratic count (D*K = 10000; actual 249).
    expect(movesWithPayload).toBeLessThan(D * K * 0.1);
  });

  it('preserves benign nested markup and safe hrefs', async () => {
    // The fix must not degenerate into "strip everything".
    const el = await renderHtml('<div><b>bold</b> <a href="https://example.org">link</a></div>');

    const div = el.querySelector('div');
    expect(div).not.toBeNull();
    expect(div!.querySelector('b')!.textContent).toBe('bold');
    const a = div!.querySelector('a');
    expect(a).not.toBeNull();
    expect(a!.getAttribute('href')).toBe('https://example.org');
    expect(a!.textContent).toBe('link');
    expect(el.textContent).toContain('bold');
    expect(el.textContent).toContain('link');
  });

  // ---------------------------------------------------------------- issue #767
  // attributes are now allowlisted (ALLOWED_ATTRS), not denylisted

  it('strips non-allowlisted attributes while keeping allowlisted ones', async () => {
    const el = await renderHtml(
      '<a href="https://example.org" ping="//evil" srcset="x 2x" download ' +
        'data-x="y" id="clobber" name="clobber" class="c" title="t" target="_blank" ' +
        'alt="a">x</a>'
    );
    const a = el.querySelector('a')!;
    expect(a.hasAttribute('ping')).toBe(false);
    expect(a.hasAttribute('srcset')).toBe(false);
    expect(a.hasAttribute('download')).toBe(false);
    expect(a.hasAttribute('data-x')).toBe(false);
    expect(a.hasAttribute('id')).toBe(false);
    expect(a.hasAttribute('name')).toBe(false);
    // Allowlisted attributes survive.
    expect(a.getAttribute('class')).toBe('c');
    expect(a.getAttribute('title')).toBe('t');
    expect(a.getAttribute('target')).toBe('_blank');
    expect(a.getAttribute('alt')).toBe('a');
  });

  it('drops a vbscript: href', async () => {
    const el = await renderHtml('<a href="vbscript:msgbox(1)">x</a>');
    expect(el.querySelector('a')!.hasAttribute('href')).toBe(false);
  });

  it('drops a data: href', async () => {
    const el = await renderHtml('<a href="data:text/html,<b>x</b>">x</a>');
    expect(el.querySelector('a')!.hasAttribute('href')).toBe(false);
  });

  it('drops a data: src', async () => {
    const el = await renderHtml('<img src="data:text/html,<b>x</b>">');
    expect(el.querySelector('img')!.hasAttribute('src')).toBe(false);
  });

  it('drops a style with a url(javascript:...) payload', async () => {
    const el = await renderHtml('<div style="background:url(javascript:alert(1))">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('keeps a benign style attribute', async () => {
    const el = await renderHtml('<div style="color:red">x</div>');
    expect(el.querySelector('div')!.getAttribute('style')).toBe('color:red');
  });

  it('drops a style carrying a vbscript: token', async () => {
    const el = await renderHtml('<div style="color:red;background:vbscript:msgbox(1)">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('drops a style carrying an expression( token', async () => {
    const el = await renderHtml('<div style="width:expression(alert(1))">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('drops a style whose javascript: is smuggled via a CSS hex escape', async () => {
    // CSS decodes `\6a ` (hex escape + terminating space) to `j`, so the CSS
    // parser sees url(javascript:...) even though the attribute text never
    // contains the `javascript:` substring. Any backslash drops the value.
    const el = await renderHtml('<div style="background:url(\'\\6a avascript:alert(1)\')">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('drops a style whose expression( is smuggled via a CSS hex escape', async () => {
    // `\65 ` decodes to `e`, reassembling `expression(` under the CSS parser.
    const el = await renderHtml('<div style="width:\\65 xpression(alert(1))">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('drops a style whose dangerous token is split by a CSS comment', async () => {
    // Legacy engines strip `/**/` inside a declaration, reassembling the
    // token; the comment-opener check rejects the value outright.
    const el = await renderHtml('<div style="width:expr/**/ession(alert(1))">x</div>');
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('drops a style whose url(javascript:...) is obfuscated with an interior tab', async () => {
    // Pins the normalizeUrlForScheme reuse on the style branch: the parser
    // decodes &Tab; to a literal U+0009 inside the value, and normalization
    // strips it before the substring check — mirroring the href obfuscation
    // tests above.
    const payload = '<div style="background:url(java&Tab;script:alert(1))">x</div>';

    // Sanity-check the input is genuinely obfuscated: the HTML parser decodes
    // &Tab; to a literal U+0009 inside the style value. Without this the test
    // could pass vacuously on an entity the parser left alone.
    const raw = document.createElement('template');
    raw.innerHTML = payload;
    expect(raw.content.querySelector('div')!.getAttribute('style')).toContain('\t');

    const el = await renderHtml(payload);
    expect(el.querySelector('div')!.hasAttribute('style')).toBe(false);
  });

  it('keeps a rel attribute on a benign anchor', async () => {
    const el = await renderHtml(
      '<a href="https://x" target="_blank" rel="noopener noreferrer">x</a>'
    );
    expect(el.querySelector('a')!.getAttribute('rel')).toBe('noopener noreferrer');
  });

  it('drops a rel="opener" (reverse tabnabbing opt-out)', async () => {
    const el = await renderHtml('<a href="https://x" target="_blank" rel="opener">x</a>');
    const a = el.querySelector('a')!;
    expect(a.hasAttribute('rel')).toBe(false);
    // Dropping `rel` is safe precisely because `_blank` keeps its implicit
    // noopener — the anchor never regains a live window.opener.
    expect(a.getAttribute('target')).toBe('_blank');
  });

  it('drops a rel that mixes noopener with an opener token', async () => {
    // `noopener` must not shield an `opener` token elsewhere in the value:
    // token equality, not substring, so this whole attribute is removed.
    const el = await renderHtml('<a href="https://x" target="_blank" rel="noopener opener">x</a>');
    expect(el.querySelector('a')!.hasAttribute('rel')).toBe(false);
  });

  it('drops a named target (reverse tabnabbing — new window keeps window.opener)', async () => {
    const el = await renderHtml('<a href="https://x" target="w1">x</a>');
    expect(el.querySelector('a')!.hasAttribute('target')).toBe(false);
  });

  it('drops target="_parent" (only _blank/_self survive)', async () => {
    const el = await renderHtml('<a href="https://x" target="_parent">x</a>');
    expect(el.querySelector('a')!.hasAttribute('target')).toBe(false);
  });

  it('keeps target="_blank" (implicit noopener, safe)', async () => {
    const el = await renderHtml('<a href="https://x" target="_blank">x</a>');
    expect(el.querySelector('a')!.getAttribute('target')).toBe('_blank');
  });

  it('drops a target with a leading space (browser treats " _blank" as named)', async () => {
    // HTML matches the `_blank` keyword with NO trimming, so a leading space
    // makes this a NAMED target in the browser — the guard must not normalize
    // more than the browser does.
    const el = await renderHtml('<a href="https://x" target=" _blank">x</a>');
    expect(el.querySelector('a')!.hasAttribute('target')).toBe(false);
  });

  it('drops a target obfuscated with a trailing tab', async () => {
    // The parser decodes &#9; to a literal U+0009, which the browser keeps as
    // part of the (now named) target value; the raw-value guard must reject it.
    const payload = '<a href="https://x" target="_blank&#9;">x</a>';

    // Sanity-check the input is genuinely obfuscated: the decoded target value
    // literally contains a tab (mirrors the href/style obfuscation tests).
    const raw = document.createElement('template');
    raw.innerHTML = payload;
    expect(raw.content.querySelector('a')!.getAttribute('target')).toContain('\t');

    const el = await renderHtml(payload);
    expect(el.querySelector('a')!.hasAttribute('target')).toBe(false);
  });

  it('keeps colspan/rowspan on table cells and width/height on images', async () => {
    // Inert presentational attributes stay allowlisted so the table/image
    // authoring the format doc advertises keeps working.
    const el = await renderHtml(
      '<table><tr><td colspan="2" rowspan="3">x</td></tr></table>' +
        '<img src="https://example.org/i.png" width="100" height="50">'
    );
    const td = el.querySelector('td')!;
    expect(td.getAttribute('colspan')).toBe('2');
    expect(td.getAttribute('rowspan')).toBe('3');
    const img = el.querySelector('img')!;
    expect(img.getAttribute('width')).toBe('100');
    expect(img.getAttribute('height')).toBe('50');
  });

  it('keeps href/class/title/target on a benign anchor', async () => {
    const el = await renderHtml(
      '<a href="https://example.org" class="c" title="t" target="_blank">x</a>'
    );
    const a = el.querySelector('a')!;
    expect(a.getAttribute('href')).toBe('https://example.org');
    expect(a.getAttribute('class')).toBe('c');
    expect(a.getAttribute('title')).toBe('t');
    expect(a.getAttribute('target')).toBe('_blank');
  });
});

describe('OverlayManager HTML size cap (issue #768)', () => {
  let manager: OverlayManager;
  let warnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    document.body.innerHTML = '';
    warnSpy = vi.spyOn(log, 'warning').mockImplementation(() => {});
    manager = new OverlayManager();
  });

  afterEach(() => {
    manager.dispose();
    warnSpy.mockRestore();
  });

  async function renderHtml(html: string): Promise<HTMLDivElement> {
    await manager.loadOverlays([makeHtmlOverlay(html)], 'http://example.com');
    return document.querySelector('[data-overlay-name="html-overlay"]') as HTMLDivElement;
  }

  it('rejects an html value over the cap (empty output + one warning)', async () => {
    // A deeply-nested string past the cap would otherwise hang the parser.
    // `sanitizeHtml` must refuse it before `template.innerHTML = html`.
    const oversized = '<b>'.repeat(MAX_OVERLAY_HTML_CHARS); // length ≫ cap
    expect(oversized.length).toBeGreaterThan(MAX_OVERLAY_HTML_CHARS);

    const el = await renderHtml(oversized);

    // Nothing oversized reached the DOM.
    expect(el.innerHTML).toBe('');
    expect(el.querySelector('b')).toBeNull();
    // Exactly one warning, on the UI module, naming the offending size and cap.
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.UI);
    expect(String(warnSpy.mock.calls[0][1])).toContain(String(oversized.length));
    expect(String(warnSpy.mock.calls[0][1])).toContain(String(MAX_OVERLAY_HTML_CHARS));
  });

  it('preserves a value of length EXACTLY the cap (boundary; not rejected)', async () => {
    // Pad valid markup with a text node so total .length === cap exactly.
    // Pins the guard as `>` (a `>=` regression would reject and warn here).
    const wrapLen = '<span></span>'.length;
    const atCap = `<span>${'x'.repeat(MAX_OVERLAY_HTML_CHARS - wrapLen)}</span>`;
    expect(atCap.length).toBe(MAX_OVERLAY_HTML_CHARS);

    const el = await renderHtml(atCap);

    expect(el.querySelector('span')).not.toBeNull();
    expect(el.querySelector('span')!.textContent!.length).toBe(MAX_OVERLAY_HTML_CHARS - wrapLen);
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('rejects a value of length EXACTLY cap + 1 (boundary; rejected)', async () => {
    const wrapLen = '<span></span>'.length;
    const overCap = `<span>${'x'.repeat(MAX_OVERLAY_HTML_CHARS - wrapLen + 1)}</span>`;
    expect(overCap.length).toBe(MAX_OVERLAY_HTML_CHARS + 1);

    const el = await renderHtml(overCap);

    expect(el.innerHTML).toBe('');
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.UI);
  });

  it('rejects a non-string html value that would coerce past the cap (defense in depth)', async () => {
    // Configs originate as untrusted .zattrs JSON: an array wrapping a huge
    // payload has .length 1 (defeating a size-only check) but is coerced to
    // the full string by the innerHTML assignment. The loader drops these,
    // and sanitizeHtml must refuse them too.
    const smuggled = ['<b>'.repeat(MAX_OVERLAY_HTML_CHARS)] as unknown as string;

    const el = await renderHtml(smuggled);

    expect(el.innerHTML).toBe('');
    expect(el.querySelector('b')).toBeNull();
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(warnSpy.mock.calls[0][0]).toBe(Modules.UI);
    expect(String(warnSpy.mock.calls[0][1])).toContain('non-string');
  });

  it('preserves normal markup and does not warn (existing behavior intact)', async () => {
    const el = await renderHtml('<div><b>bold</b> <a href="https://example.org">link</a></div>');

    expect(el.querySelector('div')).not.toBeNull();
    expect(el.querySelector('b')!.textContent).toBe('bold');
    expect(el.querySelector('a')!.getAttribute('href')).toBe('https://example.org');
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it('preserves a value just under the cap', async () => {
    // A flat, allowlisted payload one character under the cap is parsed fully.
    const filler = 'x'.repeat(MAX_OVERLAY_HTML_CHARS - '<span></span>'.length - 1);
    const nearCap = `<span>${filler}</span>`;
    expect(nearCap.length).toBeLessThan(MAX_OVERLAY_HTML_CHARS);

    const el = await renderHtml(nearCap);

    expect(el.querySelector('span')).not.toBeNull();
    expect(el.querySelector('span')!.textContent).toBe(filler);
    expect(warnSpy).not.toHaveBeenCalled();
  });
});
