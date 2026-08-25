import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const VIEWER_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../../..');

function stylesheetHrefs(source: string): string[] {
  return [...source.matchAll(/<link\b[^>]*>/gi)]
    .map(([tag]) => tag)
    .filter((tag) => /\brel=["']stylesheet["']/i.test(tag))
    .flatMap((tag) => [...tag.matchAll(/\bhref=["']([^"']+)["']/gi)].map((match) => match[1]));
}

function normalizeSourceHref(href: string): string {
  return href.replace(/^(?:\.\/|\/)/, '');
}

describe('viewer HTML stylesheet contract', () => {
  const htmlFiles = readdirSync(VIEWER_ROOT)
    .filter((file) => file.endsWith('.html'))
    .sort();

  it.each(htmlFiles)('%s uses only the aggregate source stylesheet', (file) => {
    const source = readFileSync(path.join(VIEWER_ROOT, file), 'utf8');
    const sourceStylesheets = stylesheetHrefs(source)
      .map(normalizeSourceHref)
      .filter((href) => href.startsWith('src/styles/') || href.startsWith('src/ui/gui/styles/'));

    expect(sourceStylesheets).toEqual(
      sourceStylesheets.filter((href) => href === 'src/styles/index.css')
    );

    if (source.includes('/src/ui/gui/gui.ts')) {
      expect(sourceStylesheets).toContain('src/styles/index.css');
    }
  });
});
