/**
 * `src/version.ts` — the public `VIEWER_VERSION` constant.
 *
 * The failure that matters is silent: a dropped or misspelled `define` leaves
 * the constant on its development fallback while every other test passes. So
 * the value is compared against `package.json` itself, read from disk, never
 * against a literal.
 */

import { describe, expect, it } from 'vitest';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';

import { DEV_VIEWER_VERSION, resolveViewerVersion, VIEWER_VERSION } from '../../version';
import { UNKNOWN } from '../../config/build-info';

const PACKAGE_JSON = fileURLToPath(new URL('../../../package.json', import.meta.url));

function packageVersion(): string {
  const parsed: unknown = JSON.parse(readFileSync(PACKAGE_JSON, 'utf8'));
  const version = (parsed as { version?: unknown }).version;
  if (typeof version !== 'string') throw new Error('package.json has no version');
  return version;
}

describe('VIEWER_VERSION', () => {
  it('equals package.json version (vitest.config.ts applies the same define as the builds)', () => {
    expect(VIEWER_VERSION).toBe(packageVersion());
  });

  it('is a semver-normalized CalVer, not the development fallback', () => {
    expect(VIEWER_VERSION).toMatch(/^\d{4}\.\d{1,2}\.\d{1,2}$/);
    expect(VIEWER_VERSION).not.toBe(DEV_VIEWER_VERSION);
  });

  it('is exported from the public barrel under the same name', async () => {
    const mod = await import('../../index');
    expect(mod.VIEWER_VERSION).toBe(VIEWER_VERSION);
  });
});

describe('resolveViewerVersion', () => {
  it('passes a real version through untouched', () => {
    expect(resolveViewerVersion('2026.6.5')).toBe('2026.6.5');
  });

  it.each([
    ['absent define', undefined],
    ['empty string', ''],
    ['build-stamp placeholder', UNKNOWN],
    ['non-string', 42],
  ])('falls back to the development version for %s', (_label, raw) => {
    expect(resolveViewerVersion(raw)).toBe(DEV_VIEWER_VERSION);
  });
});
