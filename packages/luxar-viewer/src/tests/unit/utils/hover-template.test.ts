/**
 * Unit tests for the shared hover placeholder vocabulary
 * (`utils/hover-template.ts`, issue #1917).
 *
 * The escaping table is the reason this module exists: three consumers share
 * the vocabulary and must NOT share the escaping. Each mode gets its own
 * adversarial case, because getting `url` wrong is a live security bug rather
 * than a cosmetic one.
 *
 * Pure string function — no jsdom needed.
 */

import { describe, it, expect } from 'vitest';
import { substituteHoverTemplate } from '../../../utils/hover-template';

const VALUES = { label: 'P04637', key: 'P04637', nodeName: '/proteins', elementIndex: 42 };

describe('substituteHoverTemplate — vocabulary', () => {
  it('substitutes {hover_key} independently of {hover_label}', () => {
    // The whole reason `keys` exists: the label is composite prose a reader
    // sees, the key is the bare id a URL needs. A template must be able to
    // take one without the other.
    const { text } = substituteHoverTemplate(
      '{hover_key} | {hover_label}',
      { ...VALUES, label: 'P04637 · DNA-binding cluster', key: 'P04637' },
      'text'
    );
    expect(text).toBe('P04637 | P04637 · DNA-binding cluster');
  });

  it('flags a missing key, so a {hover_key} link is suppressed not truncated', () => {
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      'https://uniprot.org/{hover_key}',
      { ...VALUES, key: null },
      'url'
    );
    expect(text).toBe('https://uniprot.org/');
    expect(hadEmptySubstitution).toBe(true);
  });

  it('percent-encodes a key in url mode', () => {
    const { text } = substituteHoverTemplate(
      'https://e.org/{hover_key}',
      { ...VALUES, key: 'a/b?c' },
      'url'
    );
    expect(new URL(text).pathname).toBe('/a%2Fb%3Fc');
  });

  it('substitutes all three placeholders', () => {
    const { text } = substituteHoverTemplate(
      '{hover_label} in {hover_node} at #{hover_index}',
      VALUES,
      'text'
    );
    expect(text).toBe('P04637 in /proteins at #42');
  });

  it('substitutes every occurrence, not just the first', () => {
    const { text } = substituteHoverTemplate('{hover_label}/{hover_label}', VALUES, 'text');
    expect(text).toBe('P04637/P04637');
  });

  it('leaves an unrecognised placeholder verbatim', () => {
    // A typo should be visible to the author, not silently blanked.
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      '{hover_labl}-{hover_label}',
      VALUES,
      'text'
    );
    expect(text).toBe('{hover_labl}-P04637');
    // An unknown name is not a value we failed to supply.
    expect(hadEmptySubstitution).toBe(false);
  });

  it('leaves {hover_image_label} alone — it is not in the shared vocabulary', () => {
    // It expands to an <img> and needs overlay config, so OverlayManager owns
    // it. Meaningless in a URL or on the clipboard.
    const { text } = substituteHoverTemplate('{hover_image_label}', VALUES, 'html');
    expect(text).toBe('{hover_image_label}');
  });
});

describe('substituteHoverTemplate — escaping per mode', () => {
  it('text mode does not escape (textContent is inert)', () => {
    const { text } = substituteHoverTemplate(
      '{hover_label}',
      { ...VALUES, label: 'a<b>&c' },
      'text'
    );
    expect(text).toBe('a<b>&c');
  });

  it('html mode escapes markup', () => {
    const { text } = substituteHoverTemplate(
      '{hover_label}',
      { ...VALUES, label: '<img src=x onerror=alert(1)>' },
      'html'
    );
    expect(text).not.toContain('<img');
    expect(text).toContain('&lt;');
  });

  it('url mode percent-encodes, so a label cannot restructure the URL', () => {
    // The core safety property. Each of these characters would otherwise
    // change what the URL MEANS rather than what it says.
    const { text } = substituteHoverTemplate(
      'https://ex.org/{hover_label}',
      { ...VALUES, label: 'a/b?c#d&e' },
      'url'
    );
    expect(text).toBe('https://ex.org/a%2Fb%3Fc%23d%26e');
    expect(new URL(text).pathname).toBe('/a%2Fb%3Fc%23d%26e');
    expect(new URL(text).search).toBe('');
    expect(new URL(text).hash).toBe('');
  });

  it('url mode neutralises a path-traversal label', () => {
    const { text } = substituteHoverTemplate(
      'https://ex.org/entry/{hover_label}',
      { ...VALUES, label: '../../admin' },
      'url'
    );
    expect(new URL(text).pathname).toBe('/entry/..%2F..%2Fadmin');
  });

  it('url mode encodes the node path, whose slashes are structural', () => {
    const { text } = substituteHoverTemplate(
      'https://ex.org/?n={hover_node}',
      { ...VALUES, nodeName: '/a/b' },
      'url'
    );
    expect(new URL(text).searchParams.get('n')).toBe('/a/b');
  });
});

describe('substituteHoverTemplate — empty-substitution reporting', () => {
  it('flags an empty label the template references', () => {
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      'https://ex.org/{hover_label}',
      { ...VALUES, label: null },
      'url'
    );
    expect(text).toBe('https://ex.org/');
    expect(hadEmptySubstitution).toBe(true);
  });

  it('flags an empty-string label too', () => {
    const { hadEmptySubstitution } = substituteHoverTemplate(
      '{hover_label}',
      { ...VALUES, label: '' },
      'url'
    );
    expect(hadEmptySubstitution).toBe(true);
  });

  it('does NOT flag when the template never references the missing value', () => {
    // A link built purely from the index is perfectly usable on a label-less
    // element; suppressing it would be wrong.
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      'https://ex.org/{hover_index}',
      { ...VALUES, label: null },
      'url'
    );
    expect(text).toBe('https://ex.org/42');
    expect(hadEmptySubstitution).toBe(false);
  });

  it('does NOT flag element index 0', () => {
    // "0" is falsy in JS; a naive truthiness check here would kill every link
    // on the first element of every layer.
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      'https://ex.org/{hover_index}',
      { ...VALUES, elementIndex: 0 },
      'url'
    );
    expect(text).toBe('https://ex.org/0');
    expect(hadEmptySubstitution).toBe(false);
  });

  it.each([
    ['NaN', Number.NaN],
    ['Infinity', Number.POSITIVE_INFINITY],
    ['-1', -1],
    ['a fraction', 1.5],
  ])('flags a corrupt element index (%s) instead of stringifying it', (_name, elementIndex) => {
    // An index is a non-negative integer by construction; anything else did
    // not survive the pick pipeline. Stringifying it would render "NaN" in a
    // tooltip and — far worse — navigate a `link` to https://site/NaN.
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      '{hover_index}',
      { ...VALUES, elementIndex },
      'url'
    );
    expect(text).toBe('');
    expect(hadEmptySubstitution).toBe(true);
  });

  it('flags an empty node name', () => {
    const { hadEmptySubstitution } = substituteHoverTemplate(
      '{hover_node}',
      { ...VALUES, nodeName: '' },
      'url'
    );
    expect(hadEmptySubstitution).toBe(true);
  });

  it('reports false for a template with no placeholders at all', () => {
    const { text, hadEmptySubstitution } = substituteHoverTemplate(
      'https://ex.org/static',
      { ...VALUES, label: null },
      'url'
    );
    expect(text).toBe('https://ex.org/static');
    expect(hadEmptySubstitution).toBe(false);
  });
});
