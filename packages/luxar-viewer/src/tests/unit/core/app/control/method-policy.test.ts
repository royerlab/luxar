import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vitest';

import {
  CONTROL_ALLOWED_METHODS,
  CONTROL_EXCLUDED_METHODS,
  CONTROL_EXCLUSION_REASONS,
  CONTROL_WIRE_ONLY_METHODS,
  controlRefusalReason,
  isControlMethodAllowed,
} from '../../../../../core/app/control/method-policy';
import { CONTROL_FORWARDED_EVENTS } from '../../../../../core/app/control/control-client';

const APP_SOURCE = resolve(dirname(fileURLToPath(import.meta.url)), '../../../../../core/app.ts');
const EVENTS_SOURCE = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../../../core/app/embedder/events.ts'
);

/**
 * Public members declared on `LuxarApp`, read out of the source.
 *
 * Scanning the text rather than reflecting over the class is deliberate: the
 * point is to catch a method the moment it is *written*, without constructing
 * an app (which needs a canvas, a WebGL context and a dataset).
 *
 * The shape matched is a member declaration at class-body indentation — two
 * spaces, then an optional `public` / `async` / `get`, then a name and an open
 * paren or, for a getter, a name. `private` and `protected` members are
 * skipped: they are not a controller's business and are not part of the
 * embedder contract.
 */
function publicMembersOfLuxarApp(): string[] {
  const source = readFileSync(APP_SOURCE, 'utf8');
  const classBody = source.slice(source.indexOf('export class LuxarApp'));
  const found = new Set<string>();
  const declaration =
    /^ {2}(?:(public|private|protected)\s+)?(?:(async|get|set)\s+)?([A-Za-z_]\w*)\s*(?:<[^>]*>)?\(/gm;
  for (const match of classBody.matchAll(declaration)) {
    const [, visibility, , name] = match;
    if (visibility === 'private' || visibility === 'protected') continue;
    // Reserved words that can start a line at this indentation but are not
    // member declarations.
    if (['if', 'for', 'while', 'switch', 'catch', 'return', 'constructor'].includes(name)) continue;
    found.add(name);
  }
  return [...found].sort();
}

function embedderEventNames(): string[] {
  const source = readFileSync(EVENTS_SOURCE, 'utf8');
  const start = source.indexOf('export interface LuxarEmbedderEventMap');
  const body = source.slice(start, source.indexOf('\n}', start));
  const found = new Set<string>();
  const declaration = /^ {2}(?:'([^']+)'|([A-Za-z_]\w*)):/gm;
  for (const match of body.matchAll(declaration)) found.add(match[1] ?? match[2]);
  return [...found].sort();
}

describe('control method policy', () => {
  it('finds the embedder API it is meant to be locking', () => {
    // A regex that matched nothing would make every assertion below vacuous.
    const members = publicMembersOfLuxarApp();
    expect(members.length).toBeGreaterThan(20);
    for (const anchor of ['getViewerState', 'setDimensionValue', 'dispose', 'flyTo']) {
      expect(members).toContain(anchor);
    }
  });

  it('classifies every public member as allowed or excluded', () => {
    const classified = new Set<string>([...CONTROL_ALLOWED_METHODS, ...CONTROL_EXCLUDED_METHODS]);
    const unclassified = publicMembersOfLuxarApp().filter((name) => !classified.has(name));
    expect(
      unclassified,
      'A new public LuxarApp member must be added to CONTROL_ALLOWED_METHODS or ' +
        'CONTROL_EXCLUDED_METHODS (with a reason) in core/app/control/method-policy.ts. ' +
        'Left unclassified it is silently unreachable from a controller.'
    ).toEqual([]);
  });

  it('lists nothing that LuxarApp does not actually have', () => {
    // Guards the other direction: a renamed method leaving a stale entry.
    const members = new Set(publicMembersOfLuxarApp());
    const stale = [...CONTROL_ALLOWED_METHODS, ...CONTROL_EXCLUDED_METHODS].filter(
      (name) => !members.has(name)
    );
    expect(stale, 'stale policy entries — the method no longer exists on LuxarApp').toEqual([]);
  });

  it('never classifies a member as both allowed and excluded', () => {
    const excluded = new Set(CONTROL_EXCLUDED_METHODS);
    expect(CONTROL_ALLOWED_METHODS.filter((name) => excluded.has(name))).toEqual([]);
  });

  it('records a reason for every exclusion', () => {
    const missing = CONTROL_EXCLUDED_METHODS.filter(
      (name) => (CONTROL_EXCLUSION_REASONS[name] ?? '').length === 0
    );
    expect(missing, 'every exclusion must say why, in CONTROL_EXCLUSION_REASONS').toEqual([]);
  });

  it('keeps the wire-only methods out of the LuxarApp lists', () => {
    // subscribe/unsubscribe exist only on the wire; if one ever became a real
    // LuxarApp method, it should move to the allow-list instead.
    const members = new Set(publicMembersOfLuxarApp());
    for (const name of CONTROL_WIRE_ONLY_METHODS) {
      expect(members.has(name)).toBe(false);
      expect(CONTROL_ALLOWED_METHODS).not.toContain(name);
    }
  });

  it('sorts both lists, so a diff shows the change and not a reshuffle', () => {
    expect([...CONTROL_ALLOWED_METHODS]).toEqual([...CONTROL_ALLOWED_METHODS].sort());
    expect([...CONTROL_EXCLUDED_METHODS]).toEqual([...CONTROL_EXCLUDED_METHODS].sort());
  });

  it('forwards every public embedder event', () => {
    const events = embedderEventNames();
    expect(events.length).toBeGreaterThan(10);
    expect([...CONTROL_FORWARDED_EVENTS].sort()).toEqual(events);
  });
});

describe('isControlMethodAllowed', () => {
  it('allows the ordinary drive verbs', () => {
    for (const method of [
      'setDimensionValue',
      'flyTo',
      'getViewerState',
      'setLayer',
      'screenshot',
    ]) {
      expect(isControlMethodAllowed(method)).toBe(true);
    }
  });

  it('allows the wire-only subscription verbs', () => {
    expect(isControlMethodAllowed('subscribe')).toBe(true);
    expect(isControlMethodAllowed('unsubscribe')).toBe(true);
  });

  it('refuses dispose, the one verb with no way back', () => {
    expect(isControlMethodAllowed('dispose')).toBe(false);
  });

  it('refuses the unmarshallable input-context methods', () => {
    for (const method of ['on', 'registerBinding', 'registerContext', 'pushContext']) {
      expect(isControlMethodAllowed(method)).toBe(false);
    }
  });

  it('refuses anything it has never heard of', () => {
    expect(isControlMethodAllowed('rm -rf')).toBe(false);
    expect(isControlMethodAllowed('')).toBe(false);
    expect(isControlMethodAllowed('constructor')).toBe(false);
    // Prototype keys must not leak through the Set lookup.
    expect(isControlMethodAllowed('toString')).toBe(false);
    expect(isControlMethodAllowed('__proto__')).toBe(false);
  });
});

describe('controlRefusalReason', () => {
  it('explains a deliberate exclusion', () => {
    const reason = controlRefusalReason('dispose');
    expect(reason).toContain('not exposed');
    expect(reason).toContain('no recovery');
  });

  it('distinguishes an unknown method from an excluded one', () => {
    // The two send an integrator looking in completely different places.
    expect(controlRefusalReason('nonesuch')).toContain('unknown method');
    expect(controlRefusalReason('on')).toContain('not exposed');
  });
});
