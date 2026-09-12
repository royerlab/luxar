import { describe, expect, it } from 'vitest';

import { normalizeControlSocketUrl, readUrlParams } from '../../../config/url-params';

/** The page's own origin, as the control-socket validator sees it. */
const INSECURE_PAGE = { protocol: 'http:', host: 'kiosk.local:5173' };
const SECURE_PAGE = { protocol: 'https:', host: 'kiosk.local' };

/**
 * A same-origin socket URL with a control character smuggled into its path.
 * Built rather than written out: a literal control byte in a source file is
 * invisible in review and mangled by tooling.
 */
const CONTROL_CHAR_URL = `ws://kiosk.local:5173/con${String.fromCharCode(1)}trol`;

describe('normalizeControlSocketUrl', () => {
  it('is off when the flag is absent', () => {
    expect(normalizeControlSocketUrl(null, INSECURE_PAGE)).toBeNull();
  });

  it('derives the same-origin socket from a bare flag', () => {
    // `?control` with no value: the hub rides on the app that served the page,
    // so there is no address to type and nothing to validate.
    expect(normalizeControlSocketUrl('', INSECURE_PAGE)).toBe('ws://kiosk.local:5173/control');
  });

  it('derives a secure socket for a secure page', () => {
    expect(normalizeControlSocketUrl('', SECURE_PAGE)).toBe('wss://kiosk.local/control');
  });

  it('resolves a bare path against the page origin', () => {
    expect(normalizeControlSocketUrl('/hub', INSECURE_PAGE)).toBe('ws://kiosk.local:5173/hub');
  });

  it('accepts an explicit same-origin socket URL', () => {
    expect(normalizeControlSocketUrl('ws://kiosk.local:5173/control', INSECURE_PAGE)).toBe(
      'ws://kiosk.local:5173/control'
    );
  });

  it('keeps a token in the query, since that is where the hub reads it', () => {
    expect(
      normalizeControlSocketUrl('ws://kiosk.local:5173/control?token=abc', INSECURE_PAGE)
    ).toBe('ws://kiosk.local:5173/control?token=abc');
  });

  it('refuses a cross-origin socket by default', () => {
    // The attack this exists for: a crafted link that keeps the real ?src but
    // hands the display to someone else's hub.
    expect(normalizeControlSocketUrl('ws://attacker.example/control', INSECURE_PAGE)).toBeNull();
  });

  it('allows a cross-origin socket only when explicitly opted in', () => {
    expect(normalizeControlSocketUrl('ws://booth.local/control', INSECURE_PAGE, true)).toBe(
      'ws://booth.local/control'
    );
  });

  describe('rejections', () => {
    it.each([
      ['a protocol-relative URL', '//attacker.example/control', INSECURE_PAGE],
      ['an http scheme', 'http://kiosk.local:5173/control', INSECURE_PAGE],
      ['a javascript scheme', 'javascript:alert(1)', INSECURE_PAGE],
      ['a file scheme', 'file:///control', INSECURE_PAGE],
      ['credentials in the URL', 'ws://user:pw@kiosk.local:5173/control', INSECURE_PAGE],
      ['a bare username', 'ws://user@kiosk.local:5173/control', INSECURE_PAGE],
      ['a fragment', 'ws://kiosk.local:5173/control#frag', INSECURE_PAGE],
      ['an unexpected query key', 'ws://kiosk.local:5173/control?role=viewer', INSECURE_PAGE],
      ['an embedded control character', CONTROL_CHAR_URL, INSECURE_PAGE],
      ['an angle bracket', 'ws://kiosk.local:5173/<script>', INSECURE_PAGE],
      ['an over-long value', `ws://kiosk.local:5173/${'x'.repeat(3000)}`, INSECURE_PAGE],
      ['a scheme with no host', 'ws://', INSECURE_PAGE],
      // Mixed content: the browser blocks it anyway; failing here says why.
      ['an insecure socket from a secure page', 'ws://kiosk.local/control', SECURE_PAGE],
    ])('refuses %s', (_label, raw, page) => {
      expect(normalizeControlSocketUrl(raw, page)).toBeNull();
    });

    it('refuses a cross-origin socket even when it carries a plausible token', () => {
      expect(
        normalizeControlSocketUrl('wss://attacker.example/control?token=abc', SECURE_PAGE)
      ).toBeNull();
    });
  });
});

describe('readUrlParams control wiring', () => {
  it('turns the bare flag into a resolved socket URL', () => {
    const params = readUrlParams('?src=http://host:8000&control', INSECURE_PAGE);
    expect(params.control).toBe('ws://kiosk.local:5173/control');
    expect(params.controlToken).toBeNull();
    expect(params.controlAllowCrossOrigin).toBe(false);
  });

  it('reads the token separately from the socket URL', () => {
    expect(readUrlParams('?control&controlToken=hunter2', INSECURE_PAGE).controlToken).toBe(
      'hunter2'
    );
  });

  it('treats a blank token as absent', () => {
    expect(readUrlParams('?control&controlToken=%20%20', INSECURE_PAGE).controlToken).toBeNull();
  });

  it('reports control as off when the supplied URL is refused', () => {
    // A refused value must NOT fall back to the derived socket: the author
    // asked for a specific hub, and silently driving a different one is worse
    // than not connecting.
    expect(
      readUrlParams('?control=ws://attacker.example/control', INSECURE_PAGE).control
    ).toBeNull();
  });

  it('honours the cross-origin opt-in through the parser', () => {
    const params = readUrlParams(
      '?control=ws://booth.local/control&controlAllowCrossOrigin',
      INSECURE_PAGE
    );
    expect(params.control).toBe('ws://booth.local/control');
    expect(params.controlAllowCrossOrigin).toBe(true);
  });

  it('leaves control off when the flag is absent, whatever else is present', () => {
    expect(readUrlParams('?controlToken=hunter2', INSECURE_PAGE).control).toBeNull();
  });
});
