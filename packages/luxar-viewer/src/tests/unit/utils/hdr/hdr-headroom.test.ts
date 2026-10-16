// @vitest-environment jsdom
/**
 * Unit tests for utils/hdr/hdr-headroom.ts — headroom resolution order and
 * per-display persistence (PROTOTYPE, `?hdr`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DisplaySignals } from '../../../../utils/hdr/display-identity';
import { DEFAULT_HDR_HEADROOM } from '../../../../types/hdr-output';
import {
  HdrHeadroomController,
  loadManualHeadroom,
  resolveHdrHeadroomFor,
  storeManualHeadroom,
  type HdrHeadroomResolution,
} from '../../../../utils/hdr/hdr-headroom';

const XDR: DisplaySignals = {
  platform: 'macOS',
  gpuRenderer: 'ANGLE (Apple, ANGLE Metal Renderer: Apple M4 Max, Unspecified Version)',
  width: 1512,
  height: 982,
  dpr: 2,
  label: null,
  isInternal: null,
};
const UNKNOWN: DisplaySignals = { ...XDR, platform: 'Windows', gpuRenderer: null };

function stubHdrDisplay(hdr: boolean): void {
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (q: string) =>
      ({ matches: hdr && q.includes('dynamic-range: high'), media: q }) as MediaQueryList
  );
}

describe('resolveHdrHeadroomFor', () => {
  beforeEach(() => {
    localStorage.clear();
    stubHdrDisplay(true);
  });
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('goes url > manual > measured > model > default', () => {
    storeManualHeadroom('screen:1512x982', 5);
    expect(resolveHdrHeadroomFor(XDR, 6.35, 8)).toMatchObject({ value: 8, source: 'url' });
    expect(resolveHdrHeadroomFor(XDR, 6.35, null)).toMatchObject({ value: 5, source: 'manual' });
    storeManualHeadroom('screen:1512x982', null);
    expect(resolveHdrHeadroomFor(XDR, 6.35, null)).toMatchObject({
      value: 6.35,
      source: 'measured',
    });
    expect(resolveHdrHeadroomFor(XDR, null, null)).toMatchObject({ value: 3, source: 'model' });
    expect(resolveHdrHeadroomFor(UNKNOWN, null, null)).toMatchObject({
      value: DEFAULT_HDR_HEADROOM,
      source: 'default',
    });
  });

  it('drops to 1x on a display in SDR mode, unless the URL pins it', () => {
    stubHdrDisplay(false);
    expect(resolveHdrHeadroomFor(XDR, 6.35, null)).toMatchObject({ value: 1, source: 'sdr' });
    expect(resolveHdrHeadroomFor(XDR, 6.35, 4)).toMatchObject({ value: 4, source: 'url' });
  });

  it('clamps a measurement into the slider range', () => {
    expect(resolveHdrHeadroomFor(XDR, 40, null).value).toBe(16);
  });
});

describe('manual headroom persistence', () => {
  beforeEach(() => {
    localStorage.clear();
  });
  afterEach(() => vi.unstubAllGlobals());

  it('remembers one value per display', () => {
    storeManualHeadroom('label:M28U', 1.4);
    storeManualHeadroom('label:Built-in Retina Display', 7);
    expect(loadManualHeadroom('label:M28U')).toBe(1.4);
    expect(loadManualHeadroom('label:Built-in Retina Display')).toBe(7);
    storeManualHeadroom('label:M28U', null);
    expect(loadManualHeadroom('label:M28U')).toBeNull();
  });

  it('ignores a corrupt stored document', () => {
    localStorage.setItem('luxar.hdrHeadroom', '4');
    expect(loadManualHeadroom('label:M28U')).toBeNull();
    localStorage.setItem('luxar.hdrHeadroom', '{"a": "x", "b": 3}');
    expect(loadManualHeadroom('a')).toBeNull();
    expect(loadManualHeadroom('b')).toBe(3);
  });

  it('works without localStorage', () => {
    vi.stubGlobal('localStorage', undefined);
    expect(loadManualHeadroom('screen:1512x982')).toBeNull();
    expect(() => storeManualHeadroom('screen:1512x982', 4)).not.toThrow();
  });

  it('treats a storage read error as an empty map', () => {
    const setItem = vi.fn();
    vi.stubGlobal('localStorage', {
      getItem: () => {
        throw new Error('storage blocked');
      },
      setItem,
    });
    expect(loadManualHeadroom('screen:1512x982')).toBeNull();
    expect(setItem).not.toHaveBeenCalled();
  });

  it('does not throw when a storage write is blocked', () => {
    const setItem = vi.fn(() => {
      throw new Error('storage blocked');
    });
    vi.stubGlobal('localStorage', { getItem: () => null, setItem });
    expect(() => storeManualHeadroom('screen:1512x982', 4)).not.toThrow();
    expect(setItem).toHaveBeenCalledWith('luxar.hdrHeadroom', '{"screen:1512x982":4}');
  });
});

/** A matchMedia stub whose `(dynamic-range: high)` answer the test controls. */
function stubMatchMedia(state: { hdr: boolean }): EventTarget {
  const target = new EventTarget();
  vi.spyOn(window, 'matchMedia').mockImplementation(
    (q: string) =>
      Object.assign(target, {
        matches: state.hdr && q.includes('dynamic-range: high'),
        media: q,
      }) as unknown as MediaQueryList
  );
  return target;
}

/** A Window Management API stub: one screen, optional measured headroom. */
function stubScreenDetails(screen: { label: string; hdrHeadroom?: number }, granted: boolean) {
  const currentScreen = Object.assign(new EventTarget(), { isInternal: true, ...screen });
  const details = Object.assign(new EventTarget(), { currentScreen });
  const getScreenDetails = vi.fn(() =>
    granted ? Promise.resolve(details) : Promise.reject(new Error('denied'))
  );
  vi.stubGlobal('getScreenDetails', getScreenDetails);
  Object.assign(window, { getScreenDetails });
  return { currentScreen, details, getScreenDetails };
}

describe('HdrHeadroomController', () => {
  let applied: HdrHeadroomResolution[];
  const apply = (r: HdrHeadroomResolution): void => {
    applied.push(r);
  };

  beforeEach(() => {
    localStorage.clear();
    applied = [];
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    delete (window as { getScreenDetails?: unknown }).getScreenDetails;
  });

  it('a URL pin wins and the slider cannot override or persist past it', async () => {
    stubMatchMedia({ hdr: true });
    const c = new HdrHeadroomController(apply, { pinned: 4, gpuRenderer: null });
    await c.start();
    c.setManual(9);
    expect(applied.map((r) => [r.value, r.source])).toEqual([[4, 'url']]);
    expect(localStorage.getItem('luxar.hdrHeadroom')).toBeNull();
    c.dispose();
  });

  it('drops to 1x when the window lands on an SDR display, and refuses edits there', async () => {
    const state = { hdr: true };
    const query = stubMatchMedia(state);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    state.hdr = false;
    query.dispatchEvent(new Event('change'));
    c.setManual(5);
    expect(applied.map((r) => r.source)).toEqual(['default', 'sdr']);
    expect(c.current()?.value).toBe(1);
    c.dispose();
  });

  it('Detect with the permission denied keeps the hand-set value', async () => {
    stubMatchMedia({ hdr: true });
    stubScreenDetails({ label: 'Built-in Retina Display', hdrHeadroom: 2 }, false);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    c.setManual(5);
    expect(await c.detect()).toBe('denied');
    expect(c.current()).toMatchObject({ value: 5, source: 'manual' });
    c.dispose();
  });

  it('Detect without the screen-details API leaves headroom unchanged', async () => {
    stubMatchMedia({ hdr: true });
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    expect(c.canMeasure()).toBe(false);
    expect(await c.detect()).toBe('denied');
    expect(c.current()).toMatchObject({ value: 2, source: 'default' });
    expect(localStorage.getItem('luxar.hdrHeadroom')).toBeNull();
    c.dispose();
  });

  it('Detect measures, migrates the geometry-keyed value, then lets the measurement win', async () => {
    stubMatchMedia({ hdr: true });
    const { currentScreen } = stubScreenDetails(
      { label: 'Built-in Retina Display', hdrHeadroom: Math.log2(6) },
      true
    );
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start(); // permission not granted yet: geometry key
    c.setManual(5);
    expect(await c.detect()).toBe('measured');
    expect(c.current()).toMatchObject({ source: 'measured' });
    expect(c.current()?.value).toBeCloseTo(6);
    expect(loadManualHeadroom('label:Built-in Retina Display')).toBeNull();

    // Brightness change: the live headroom follows.
    Object.assign(currentScreen, { hdrHeadroom: Math.log2(3) });
    currentScreen.dispatchEvent(new Event('hdrheadroomchange'));
    expect(c.current()?.value).toBeCloseTo(3);
    c.dispose();
  });

  it('a double-clicked Detect shares one permission request', async () => {
    stubMatchMedia({ hdr: true });
    const { getScreenDetails } = stubScreenDetails(
      { label: 'Built-in Retina Display', hdrHeadroom: 1 },
      true
    );
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    const outcomes = await Promise.all([c.detect(), c.detect()]);
    expect(outcomes).toEqual(['measured', 'measured']);
    expect(getScreenDetails).toHaveBeenCalledTimes(1);
    c.dispose();
  });

  it('starts watching an EventTarget screen and detaches it on dispose', async () => {
    stubMatchMedia({ hdr: true });
    const screen = Object.assign(new EventTarget(), { width: 1512, height: 982 });
    vi.stubGlobal('screen', screen);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    vi.spyOn(window, 'matchMedia').mockImplementation(() => ({ matches: false }) as MediaQueryList);
    screen.dispatchEvent(new Event('change'));
    expect(c.current()).toMatchObject({ value: 1, source: 'sdr' });
    c.dispose();
    const count = applied.length;
    screen.dispatchEvent(new Event('change'));
    expect(applied).toHaveLength(count);
  });

  it('attaches screen details at startup only with an already granted permission', async () => {
    stubMatchMedia({ hdr: true });
    const { getScreenDetails } = stubScreenDetails(
      { label: 'Built-in Retina Display', hdrHeadroom: 2 },
      true
    );
    const query = vi.fn().mockResolvedValue({ state: 'granted' });
    vi.stubGlobal('navigator', { platform: 'MacIntel', permissions: { query } });
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    expect(query).toHaveBeenCalledWith({ name: 'window-management' });
    expect(getScreenDetails).toHaveBeenCalledTimes(1);
    expect(c.current()).toMatchObject({ value: 4, source: 'measured' });
    c.dispose();
  });

  it('does not request screen details at startup without a grant', async () => {
    stubMatchMedia({ hdr: true });
    const { getScreenDetails } = stubScreenDetails(
      { label: 'Built-in Retina Display', hdrHeadroom: 2 },
      true
    );
    vi.stubGlobal('navigator', {
      platform: 'MacIntel',
      permissions: { query: vi.fn().mockResolvedValue({ state: 'prompt' }) },
    });
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    expect(getScreenDetails).not.toHaveBeenCalled();
    expect(c.current()).toMatchObject({ value: 2, source: 'default' });
    c.dispose();
  });

  it('keeps a geometry-keyed value when the detailed screen has no label', async () => {
    stubMatchMedia({ hdr: true });
    stubScreenDetails({ label: '' }, true);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    c.setManual(5);
    const setItem = vi.spyOn(localStorage, 'setItem');
    expect(await c.detect()).toBe('unmeasurable');
    expect(c.current()).toMatchObject({ value: 5, source: 'manual' });
    expect(setItem).not.toHaveBeenCalled();
    c.dispose();
  });

  it('does not overwrite a hand-set value already stored under the display label', async () => {
    stubMatchMedia({ hdr: true });
    stubScreenDetails({ label: 'Built-in Retina Display' }, true);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    c.setManual(5);
    storeManualHeadroom('label:Built-in Retina Display', 6);
    const setItem = vi.spyOn(localStorage, 'setItem');
    expect(await c.detect()).toBe('unmeasurable');
    expect(c.current()).toMatchObject({ value: 6, source: 'manual' });
    expect(loadManualHeadroom('label:Built-in Retina Display')).toBe(6);
    expect(setItem).not.toHaveBeenCalled();
    c.dispose();
  });

  it('stops watching the previous screen when the current screen disappears', async () => {
    stubMatchMedia({ hdr: true });
    const { currentScreen, details } = stubScreenDetails(
      { label: 'Built-in Retina Display', hdrHeadroom: 2 },
      true
    );
    vi.stubGlobal('navigator', {
      platform: 'MacIntel',
      permissions: { query: vi.fn().mockResolvedValue({ state: 'granted' }) },
    });
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    Object.assign(details, { currentScreen: null });
    details.dispatchEvent(new Event('currentscreenchange'));
    expect(c.current()).toMatchObject({ value: 2, source: 'default' });
    const count = applied.length;
    Object.assign(currentScreen, { hdrHeadroom: 3 });
    currentScreen.dispatchEvent(new Event('hdrheadroomchange'));
    expect(applied).toHaveLength(count);
    c.dispose();
  });

  it('ignores a screen-details request completed after disposal', async () => {
    stubMatchMedia({ hdr: true });
    const currentScreen = Object.assign(new EventTarget(), {
      label: 'Built-in Retina Display',
      hdrHeadroom: 2,
    });
    const details = Object.assign(new EventTarget(), { currentScreen });
    const onDetails = vi.spyOn(details, 'addEventListener');
    const onScreen = vi.spyOn(currentScreen, 'addEventListener');
    let resolveDetails!: (value: typeof details) => void;
    const pending = new Promise<typeof details>((resolve) => {
      resolveDetails = resolve;
    });
    Object.assign(window, { getScreenDetails: () => pending });
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    const detecting = c.detect();
    c.dispose();
    resolveDetails(details);
    expect(await detecting).toBe('denied');
    expect(c.current()).toMatchObject({ value: 2, source: 'default' });
    expect(onDetails).not.toHaveBeenCalled();
    expect(onScreen).not.toHaveBeenCalled();
    expect(localStorage.getItem('luxar.hdrHeadroom')).toBeNull();
  });

  it('stores a hand-set value under the geometry key too (startup reads it)', async () => {
    stubMatchMedia({ hdr: true });
    stubScreenDetails({ label: 'Built-in Retina Display' }, true);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    await c.detect(); // 'unmeasurable', but the label is now known
    c.setManual(4);
    const map = JSON.parse(localStorage.getItem('luxar.hdrHeadroom') ?? '{}');
    expect(Object.values(map)).toEqual([4, 4]);
    expect(Object.keys(map).some((k) => k.startsWith('label:'))).toBe(true);
    expect(Object.keys(map).some((k) => k.startsWith('screen:'))).toBe(true);
    c.dispose();
  });

  it('dispose detaches every listener (and is idempotent)', async () => {
    const state = { hdr: true };
    const query = stubMatchMedia(state);
    const c = new HdrHeadroomController(apply, { pinned: null, gpuRenderer: null });
    await c.start();
    c.dispose();
    c.dispose();
    state.hdr = false;
    query.dispatchEvent(new Event('change'));
    expect(applied).toHaveLength(1);
  });
});
