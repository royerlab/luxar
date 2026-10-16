// @vitest-environment jsdom
/**
 * Unit tests for utils/hdr/hdr-headroom.ts — headroom resolution order and
 * per-display persistence (PROTOTYPE, `?hdr`).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DisplaySignals } from '../../../../utils/hdr/display-identity';
import {
  DEFAULT_HDR_HEADROOM,
  loadManualHeadroom,
  resolveHdrHeadroomFor,
  storeManualHeadroom,
  urlHdrHeadroom,
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
    storeManualHeadroom('screen:1512x982@2', 5);
    expect(resolveHdrHeadroomFor(XDR, 6.35, 8)).toMatchObject({ value: 8, source: 'url' });
    expect(resolveHdrHeadroomFor(XDR, 6.35, null)).toMatchObject({ value: 5, source: 'manual' });
    storeManualHeadroom('screen:1512x982@2', null);
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
});

describe('urlHdrHeadroom', () => {
  it('reads an explicit ?hdr=<n>, null for a bare flag', () => {
    expect(urlHdrHeadroom('?hdr=4')).toBe(4);
    expect(urlHdrHeadroom('?hdr')).toBeNull();
    expect(urlHdrHeadroom('?renderer=webgpu')).toBeNull();
    expect(urlHdrHeadroom('?hdr=99')).toBe(16);
  });
});
