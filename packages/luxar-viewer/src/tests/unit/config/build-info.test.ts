/**
 * Tests for the build stamp — `src/config/build-info.ts` and its build-side
 * producer `tools/build-identity.ts`.
 *
 * The stamp exists so a bug report can be tied to a revision, which means the
 * failure that matters is not "wrong value" but "no value, silently". Each test
 * below pins one of the ways that could happen.
 */

import { afterEach, describe, it, expect, vi } from 'vitest';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { fileURLToPath } from 'url';
import type { Plugin } from 'vite';

import { buildInfo, buildInfoLine, UNKNOWN } from '../../../config/build-info';
import {
  buildDefine,
  buildIdentity,
  buildIdentityHtmlPlugin,
  buildMetaTag,
  packageVersion,
  BUILD_DEFINE,
  UNKNOWN as PRODUCER_UNKNOWN,
} from '../../../../tools/build-identity';

const VIEWER_ROOT = fileURLToPath(new URL('../../../../', import.meta.url));

function transformIndexHtml(plugin: Plugin, html: string): string {
  const transform = plugin.transformIndexHtml;
  if (!transform) throw new Error('plugin has no transformIndexHtml hook');
  const handler = typeof transform === 'function' ? transform : transform.handler;
  const result = handler.call({} as never, html, {} as never);
  if (typeof result !== 'string') throw new Error('transformIndexHtml did not return HTML');
  return result;
}

describe('build-info (runtime side)', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  it('reports unstamped rather than throwing when no define was injected', () => {
    // vitest uses vitest.config.ts, which defines `__LUXAR_VIEWER_VERSION__`
    // but NOT `__LUXAR_BUILD__` — so this run IS the unstamped case. A bare `__LUXAR_BUILD__` reference here would be a
    // ReferenceError, and the guard in build-info.ts is the only thing
    // preventing that from taking the whole viewer down in every embedder.
    const info = buildInfo();
    expect(info.stamped).toBe(false);
    expect(info.version).toBe(UNKNOWN);
    expect(info.commit).toBe(UNKNOWN);
    expect(info.buildTime).toBe(UNKNOWN);
  });

  it('says "development build" rather than presenting unknown as a version', () => {
    expect(buildInfoLine()).toBe('development build (unstamped)');
  });

  it('renders a stamped identity as one greppable line', () => {
    expect(
      buildInfoLine({
        version: '2026.9.15',
        commit: 'abc1234',
        buildTime: '2026-09-15T10:00:00Z',
        stamped: true,
      })
    ).toBe('2026.9.15 (abc1234, built 2026-09-15T10:00:00Z)');
  });

  it('parses the injected build stamp through the real runtime module', async () => {
    const identity = {
      version: '2026.9.15',
      commit: 'abc1234',
      buildTime: '2026-09-15T10:00:00Z',
    };
    vi.stubGlobal('__LUXAR_BUILD__', JSON.stringify(identity));
    vi.resetModules();

    const stamped = await import('../../../config/build-info');

    expect(stamped.buildInfo()).toEqual({ ...identity, stamped: true });
    expect(stamped.buildInfoLine()).toBe('2026.9.15 (abc1234, built 2026-09-15T10:00:00Z)');
  });

  it('falls back to unstamped when the injected build stamp is malformed', async () => {
    vi.stubGlobal('__LUXAR_BUILD__', '{not-json');
    vi.resetModules();

    const malformed = await import('../../../config/build-info');

    expect(malformed.buildInfo()).toEqual({
      version: UNKNOWN,
      commit: UNKNOWN,
      buildTime: UNKNOWN,
      stamped: false,
    });
    expect(malformed.buildInfoLine()).toBe('development build (unstamped)');
  });

  it('agrees with the producer on the placeholder spelling', () => {
    // build-info.ts cannot import tools/build-identity.ts (child_process, fs),
    // so the constant is duplicated. This is the seam that keeps the copies
    // honest — without it a producer emitting 'unknown' and a consumer testing
    // for 'UNKNOWN' would both pass their own tests.
    expect(UNKNOWN).toBe(PRODUCER_UNKNOWN);
  });
});

describe('build-identity (build side)', () => {
  it('resolves the version from package.json, the file set_version.py stamps', () => {
    const pkg: unknown = JSON.parse(readFileSync(`${VIEWER_ROOT}package.json`, 'utf8'));
    expect(buildIdentity().version).toBe((pkg as { version: string }).version);
  });

  it('reads package.json when the checkout path contains URL metacharacters', () => {
    const parent = mkdtempSync(join(tmpdir(), 'luxar-build-identity-'));
    const root = join(parent, 'viewer#copy');
    mkdirSync(root);
    writeFileSync(join(root, 'package.json'), JSON.stringify({ version: '9.9.9' }));
    try {
      expect(packageVersion(root)).toBe('9.9.9');
    } finally {
      rmSync(parent, { recursive: true, force: true });
    }
  });

  it('resolves a commit, or says unknown — never an empty string', () => {
    const commit = buildIdentity().commit;
    expect(commit.length).toBeGreaterThan(0);
    expect(commit === UNKNOWN || /^[0-9a-f]{7,}(-dirty)?$/.test(commit)).toBe(true);
  });

  it('stamps a second-precision UTC instant', () => {
    const at = new Date('2026-09-15T10:11:12.345Z');
    expect(buildIdentity(at).buildTime).toBe('2026-09-15T10:11:12Z');
  });

  it('double-encodes the define so it is a well-formed expression, not a bare object', () => {
    // Vite splices a define's value in as RAW SOURCE. A bare `{...}` would be a
    // syntax hazard in statement position; a quoted JSON string never is.
    const value = buildDefine({
      version: '1.2.3',
      commit: 'abc1234',
      buildTime: '2026-09-15T10:00:00Z',
    })[BUILD_DEFINE];
    expect(value.startsWith('"')).toBe(true);
    expect(JSON.parse(JSON.parse(value) as string)).toEqual({
      version: '1.2.3',
      commit: 'abc1234',
      buildTime: '2026-09-15T10:00:00Z',
    });
  });

  it('round-trips the define through the runtime parser', () => {
    // The producer and the consumer are in different modules and different
    // runtimes; this is the only test that runs the actual wire format through
    // both. `read()` is module-private, so re-implement its one parse step.
    const identity = { version: '9.9.9', commit: 'deadbee', buildTime: '2026-01-01T00:00:00Z' };
    const injected = JSON.parse(buildDefine(identity)[BUILD_DEFINE]) as string;
    expect(JSON.parse(injected)).toEqual(identity);
  });

  it('injects a meta tag into the head, before </head>', () => {
    const identity = { version: '1.2.3', commit: 'abc1234', buildTime: '2026-09-15T10:00:00Z' };
    const html = transformIndexHtml(
      buildIdentityHtmlPlugin(identity),
      '<html><head><title>x</title></head><body></body></html>'
    );
    expect(html).toContain(buildMetaTag(identity));
    expect(html.indexOf('luxar-build')).toBeLessThan(html.indexOf('</head>'));
  });

  it('escapes the build identity before placing it in an HTML attribute', () => {
    expect(buildMetaTag({ version: '1&2', commit: 'a"b', buildTime: '<now>' })).toContain(
      'content="1&amp;2 a&quot;b &lt;now&gt;"'
    );
  });

  it('leaves html without a head untouched rather than corrupting it', () => {
    const fragment = '<div>no head here</div>';
    expect(transformIndexHtml(buildIdentityHtmlPlugin(), fragment)).toBe(fragment);
  });
});
