/**
 * The authored control-panel block is store-supplied, i.e. untrusted input.
 *
 * Two properties matter here and they pull in opposite directions: an author's
 * intent must survive the trip, and anything malformed must fall back to the
 * DERIVED default rather than reaching the page. The derived default is always
 * sane, so dropping a bad field is strictly better than clamping it — a clamp
 * silently pretends the author asked for something they did not.
 */

import { describe, expect, it } from 'vitest';

import {
  MAX_CONTROL_COLUMNS,
  MAX_CONTROL_STYLESHEET_CHARS,
  extractControlPanelConfig,
  sanitizeAuthorStylesheet,
} from '../../../../config/zarr-bridge/control-panel';

describe('extractControlPanelConfig', () => {
  it('reads a fully authored block', () => {
    expect(
      extractControlPanelConfig({
        title: 'Eleven stories in the protein universe',
        subtitle: 'Touch a tile',
        chapter_dimension: 'story',
        columns: 4,
        idle_reset_s: 90,
        chapters: { '1': { sublabel: 'the molecule of breath' } },
      })
    ).toEqual({
      title: 'Eleven stories in the protein universe',
      subtitle: 'Touch a tile',
      chapterDimension: 'story',
      columns: 4,
      idleResetS: 90,
      chapters: { 1: { sublabel: 'the molecule of breath' } },
    });
  });

  it('treats no block and an empty block as the same answer', () => {
    // Both mean "derive everything", and a caller should not have to tell them
    // apart to do the right thing.
    for (const raw of [null, undefined, {}, [], 'nope', 42]) {
      expect(extractControlPanelConfig(raw)).toBeNull();
    }
  });

  it('keys chapters by integer, from the string keys JSON forces', () => {
    const settings = extractControlPanelConfig({
      chapters: { '0': { label: 'Start' }, '7': { sublabel: 'seventh' } },
    });
    expect(settings?.chapters).toEqual({ 0: { label: 'Start' }, 7: { sublabel: 'seventh' } });
  });

  it('drops chapter entries that carry nothing', () => {
    // An all-default `Chapter()` is not an override; emitting a key for it
    // would make the panel think position 3 was authored.
    expect(
      extractControlPanelConfig({ chapters: { '3': {}, '4': { label: 'Real' } } })?.chapters
    ).toEqual({ 4: { label: 'Real' } });
  });

  it('ignores chapter keys that are not positions', () => {
    expect(
      extractControlPanelConfig({
        chapters: { story: { label: 'x' }, '-1': { label: 'y' }, '1.5': { label: 'z' } },
      })
    ).toBeNull();
  });

  describe('out-of-range and malformed fields fall back rather than clamp', () => {
    it.each([
      ['columns 0', { columns: 0 }],
      ['columns past the cap', { columns: MAX_CONTROL_COLUMNS + 1 }],
      ['columns fractional', { columns: 3.5 }],
      ['columns as a string', { columns: '4' }],
      ['idle_reset_s negative', { idle_reset_s: -1 }],
      ['idle_reset_s infinite', { idle_reset_s: Number.POSITIVE_INFINITY }],
      ['idle_reset_s NaN', { idle_reset_s: Number.NaN }],
      ['blank title', { title: '   ' }],
      ['title as a number', { title: 7 }],
      ['chapters as an array', { chapters: [{ label: 'x' }] }],
    ])('%s', (_label, raw) => {
      expect(extractControlPanelConfig(raw)).toBeNull();
    });

    it('zero is a real idle_reset_s, not a missing one', () => {
      // It means "never reset" — an exhibit that should hold its last stop —
      // so a falsy check here would silently re-enable the reset.
      expect(extractControlPanelConfig({ idle_reset_s: 0 })).toEqual({ idleResetS: 0 });
    });
  });

  it('refuses a stylesheet over the cap instead of truncating it', () => {
    // Truncating would leave a half-written rule, and a cap the writer already
    // enforces should not need a second, different behaviour here.
    const tooLong = `.luxar-control-tile{color:red}${'/*pad*/'.repeat(20000)}`;
    expect(tooLong.length).toBeGreaterThan(MAX_CONTROL_STYLESHEET_CHARS);
    expect(extractControlPanelConfig({ stylesheet: tooLong })).toBeNull();
  });

  it('keeps a stylesheet that survives sanitising', () => {
    const settings = extractControlPanelConfig({
      stylesheet: '.luxar-control-tile { border-radius: 4px; }',
    });
    expect(settings?.stylesheet).toContain('border-radius: 4px');
  });

  it('drops a stylesheet that is nothing BUT stripped constructs', () => {
    // Otherwise the panel injects an empty `<style>` element and reports an
    // authored stylesheet that does nothing.
    expect(extractControlPanelConfig({ stylesheet: '@import url(http://x/a.css);' })).toBeNull();
  });
});

describe('sanitizeAuthorStylesheet', () => {
  it('strips @import in every spelling', () => {
    for (const rule of [
      '@import url(http://evil.example/a.css);',
      "@import 'http://evil.example/a.css';",
      '@IMPORT   url("//evil.example/a.css") screen;',
      '@import url(a.css)',
    ]) {
      const out = sanitizeAuthorStylesheet(`${rule}\n.tile { color: red; }`);
      expect(out.toLowerCase(), rule).not.toContain('@import');
      // The legitimate rule after it must survive.
      expect(out).toContain('color: red');
    }
  });

  it('strips a remote url() but keeps a data: URI', () => {
    // Every remote URL in author CSS is also a beacon reporting the kiosk's
    // IP to whoever hosts it, which is why this is not merely about the cap.
    const out = sanitizeAuthorStylesheet(
      '.a{background:url(http://evil.example/p.png)}' +
        ".b{background:url('//evil.example/q.png')}" +
        '.c{background:url(/local/r.png)}' +
        '.d{background:url(data:image/gif;base64,R0lGOD)}'
    );
    expect(out).not.toContain('evil.example');
    expect(out).not.toContain('/local/r.png');
    expect(out).toContain('data:image/gif;base64,R0lGOD');
  });

  it('leaves ordinary CSS untouched', () => {
    const css = '.luxar-control-tile { border-radius: 18px; color: #fff; }';
    expect(sanitizeAuthorStylesheet(css)).toBe(css);
  });
});
