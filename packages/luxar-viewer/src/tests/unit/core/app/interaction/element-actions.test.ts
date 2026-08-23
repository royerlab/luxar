/**
 * Unit tests for element link/copy resolution
 * (`core/app/interaction/element-actions.ts`, issue #1917).
 *
 * This module decides what URL the browser NAVIGATES to on a left-click, from
 * two pieces of untrusted input: an author-supplied template and a
 * data-supplied per-element value. So the bulk of these tests are adversarial
 * — each one is a way the feature could become an XSS or a phishing vector if
 * the guard were dropped.
 *
 * Pure functions; no jsdom, no PickingSystem.
 */

import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  buildElementUrl,
  buildElementCopyText,
  explainLinkRejection,
  readInteractionTemplates,
  resolveLinkTarget,
  resolveElementActions,
  DEFAULT_LINK_TARGET,
  MAX_LINK_CHARS,
  MAX_COPY_CHARS,
} from '../../../../../core/app/interaction/element-actions';

const VALUES = { label: 'P04637', nodeName: '/proteins', elementIndex: 42 };

/** A node carrying the given `.zattrs`, as the loader would have written them. */
function nodeWithAttrs(attrs: Record<string, unknown>): THREE.Object3D {
  const o = new THREE.Object3D();
  o.userData = { attrs };
  return o;
}

describe('buildElementUrl — the happy path', () => {
  it('substitutes and returns an absolute https URL', () => {
    const url = buildElementUrl('https://www.uniprot.org/uniprotkb/{hover_label}/entry', VALUES);
    expect(url).toBe('https://www.uniprot.org/uniprotkb/P04637/entry');
  });

  it('accepts http as well as https', () => {
    expect(buildElementUrl('http://example.org/{hover_index}', VALUES)).toBe(
      'http://example.org/42'
    );
  });

  it('supports a query-parameter shape', () => {
    const url = buildElementUrl('https://ex.org/search?q={hover_label}', VALUES);
    expect(new URL(url!).searchParams.get('q')).toBe('P04637');
  });

  it('works with no placeholders at all', () => {
    expect(buildElementUrl('https://example.org/docs', VALUES)).toBe('https://example.org/docs');
  });
});

describe('buildElementUrl — scheme allowlist', () => {
  // Each of these is a live vector if the allowlist is dropped: the first two
  // execute script in the viewer's origin, the rest reach outside the web.
  it.each([
    ['javascript:', 'javascript:alert(document.domain)'],
    ['data:', 'data:text/html,<script>alert(1)</script>'],
    ['blob:', 'blob:https://example.org/f0e1'],
    ['file:', 'file:///etc/passwd'],
    ['vbscript:', 'vbscript:msgbox(1)'],
    ['about:', 'about:blank'],
    ['mailto:', 'mailto:someone@example.org'],
    ['ftp:', 'ftp://example.org/x'],
  ])('rejects %s', (_name, template) => {
    expect(buildElementUrl(template, VALUES)).toBeNull();
  });

  it('rejects a scheme smuggled in via mixed case', () => {
    expect(buildElementUrl('JaVaScRiPt:alert(1)', VALUES)).toBeNull();
  });
});

describe('buildElementUrl — relative templates', () => {
  // A relative URL would resolve against whatever origin the VIEWER is served
  // from, letting a third-party dataset aim a click at the embedder's own site.
  it.each(['/admin/{hover_label}', '//evil.example/{hover_label}', 'foo/bar', '?q={hover_label}'])(
    'rejects %s',
    (template) => {
      expect(buildElementUrl(template, VALUES)).toBeNull();
    }
  );

  /**
   * Documents a deliberate divergence from the Python writer's check.
   *
   * For a *special* scheme (http/https) the WHATWG parser collapses the extra
   * slash and reads `nowhere` as the HOST, so this is a well-formed URL to the
   * browser and the empty-host branch is unreachable here. Python's
   * `urlsplit`, by contrast, reports an empty netloc and `validate_link`
   * refuses it at authoring time.
   *
   * Python being stricter is the right way round: `https:///x` is a typo far
   * more often than an intent, and refusing to WRITE it costs nothing, while
   * the viewer must faithfully model what the browser will actually do with a
   * store it did not author. The host check stays in the viewer as a cheap
   * guard that stays correct if the scheme allowlist ever grows to a
   * non-special scheme, where an empty host IS reachable.
   */
  it('treats https:///host the way the browser does — host, not path', () => {
    expect(buildElementUrl('https:///nowhere', VALUES)).toBe('https://nowhere/');
  });
});

describe('buildElementUrl — a value cannot restructure the URL', () => {
  it('a label containing a query separator stays in the path', () => {
    const url = buildElementUrl('https://ex.org/{hover_label}', {
      ...VALUES,
      label: 'x?admin=1',
    });
    expect(new URL(url!).search).toBe('');
    expect(new URL(url!).pathname).toBe('/x%3Fadmin%3D1');
  });

  it('a label containing a fragment separator stays in the path', () => {
    const url = buildElementUrl('https://ex.org/{hover_label}', { ...VALUES, label: 'x#frag' });
    expect(new URL(url!).hash).toBe('');
  });

  it('a label cannot escape the intended host', () => {
    // The classic: if the value were interpolated raw, `//evil.example` in a
    // scheme-relative position would change the authority.
    const url = buildElementUrl('https://good.example/{hover_label}', {
      ...VALUES,
      label: '..//evil.example',
    });
    expect(new URL(url!).host).toBe('good.example');
  });

  it('a label cannot traverse the target path', () => {
    const url = buildElementUrl('https://ex.org/entry/{hover_label}', {
      ...VALUES,
      label: '../../admin',
    });
    expect(new URL(url!).pathname).toBe('/entry/..%2F..%2Fadmin');
  });

  it('a label cannot inject an extra query parameter', () => {
    const url = buildElementUrl('https://ex.org/?q={hover_label}', {
      ...VALUES,
      label: 'a&admin=1',
    });
    expect(new URL(url!).searchParams.get('admin')).toBeNull();
    expect(new URL(url!).searchParams.get('q')).toBe('a&admin=1');
  });
});

describe('buildElementUrl — empty substitution is suppressed', () => {
  it('returns null when a referenced label is missing', () => {
    // Not hypothetical: coarse substitutive-LOD levels inherit the attrs but
    // carry no labels, so this is the normal case at coarse LOD. Emitting
    // `https://uniprot.org/` would be a valid URL to the wrong place.
    expect(
      buildElementUrl('https://uniprot.org/{hover_label}', { ...VALUES, label: null })
    ).toBeNull();
  });

  it('still resolves a link that does not reference the missing value', () => {
    expect(buildElementUrl('https://ex.org/{hover_index}', { ...VALUES, label: null })).toBe(
      'https://ex.org/42'
    );
  });

  it('resolves for element index 0', () => {
    // "0" is falsy; a truthiness check would kill the first element of a layer.
    expect(buildElementUrl('https://ex.org/{hover_index}', { ...VALUES, elementIndex: 0 })).toBe(
      'https://ex.org/0'
    );
  });
});

describe('buildElementUrl — length cap', () => {
  it('rejects a resolved URL over the cap', () => {
    expect(buildElementUrl(`https://ex.org/${'x'.repeat(MAX_LINK_CHARS)}`, VALUES)).toBeNull();
  });

  it('rejects when the ELEMENT VALUE is what pushes it over', () => {
    // The cap has to be applied after substitution: the template alone is
    // short, and the store controls the label.
    const url = buildElementUrl('https://ex.org/{hover_label}', {
      ...VALUES,
      label: 'y'.repeat(MAX_LINK_CHARS),
    });
    expect(url).toBeNull();
  });
});

describe('explainLinkRejection', () => {
  it('returns null for a usable template', () => {
    expect(explainLinkRejection('https://ex.org/{hover_label}')).toBeNull();
  });

  it('names a bad scheme', () => {
    expect(explainLinkRejection('javascript:alert(1)')).toMatch(/scheme/);
  });

  it('names a relative template', () => {
    expect(explainLinkRejection('/admin')).toMatch(/absolute/);
  });

  it('does NOT complain about a template that is only element-dependent', () => {
    // An empty label is a per-element condition, normal and frequent. Logging
    // it at scene load would be noise blaming a template that is fine.
    expect(explainLinkRejection('https://ex.org/{hover_label}')).toBeNull();
  });
});

describe('resolveLinkTarget', () => {
  it('defaults to _blank', () => {
    expect(resolveLinkTarget(undefined)).toBe(DEFAULT_LINK_TARGET);
    expect(DEFAULT_LINK_TARGET).toBe('_blank');
  });

  it('accepts _self', () => {
    expect(resolveLinkTarget('_self')).toBe('_self');
  });

  it.each(['_parent', '_top', 'myframe', '_blank ', '_BLANK'])(
    'falls back to _blank for %p',
    (raw) => {
      // Anything else is a NAMED target, which the browser opens with a live
      // window.opener the destination can use to navigate the viewer tab
      // (reverse tabnabbing). Note `_BLANK` is rejected too: HTML's keyword
      // match is case-insensitive, but we compare exactly and fall back to the
      // safe value, so there is no path to a named context.
      expect(resolveLinkTarget(raw)).toBe('_blank');
    }
  );
});

describe('buildElementCopyText', () => {
  it('falls back to the bare label when no copy template is authored', () => {
    // Every labelled layer gets a working Copy with zero authoring.
    expect(buildElementCopyText(undefined, VALUES)).toBe('P04637');
  });

  it('returns null with neither a template nor a label', () => {
    expect(buildElementCopyText(undefined, { ...VALUES, label: null })).toBeNull();
  });

  it('uses the authored template when present', () => {
    expect(buildElementCopyText('{hover_label} @ {hover_node}#{hover_index}', VALUES)).toBe(
      'P04637 @ /proteins#42'
    );
  });

  it('does NOT escape — plain text is the point', () => {
    expect(buildElementCopyText('{hover_label}', { ...VALUES, label: 'a<b>&c' })).toBe('a<b>&c');
  });

  it('suppresses when a referenced value is empty', () => {
    expect(buildElementCopyText('{hover_label}', { ...VALUES, label: null })).toBeNull();
  });

  it('truncates rather than rejecting an over-long string', () => {
    // Unlike a URL, a clipped copy string is still useful, so the cap clips.
    const out = buildElementCopyText('{hover_label}', {
      ...VALUES,
      label: 'z'.repeat(MAX_COPY_CHARS * 2),
    });
    expect(out).toHaveLength(MAX_COPY_CHARS);
  });
});

describe('readInteractionTemplates — where the attrs live', () => {
  it('reads from the hit leaf', () => {
    const leaf = nodeWithAttrs({ link: 'https://ex.org/', copy: 'c', link_target: '_self' });
    expect(readInteractionTemplates(leaf)).toEqual({
      link: 'https://ex.org/',
      copy: 'c',
      linkTarget: '_self',
    });
  });

  it('walks up to an ancestor when the leaf carries none', () => {
    // Covers a store that put the attrs on a partition WRAPPER rather than on
    // each part_<i> leaf.
    const wrapper = nodeWithAttrs({ link: 'https://ex.org/{hover_label}' });
    const leaf = nodeWithAttrs({ has_labels: true });
    wrapper.add(leaf);
    expect(readInteractionTemplates(leaf).link).toBe('https://ex.org/{hover_label}');
  });

  it('the nearest level wins outright — templates are not merged', () => {
    // A leaf declaring only `copy` must not silently inherit a grandparent's
    // `link`; that would attach a link the author never put on this layer.
    const wrapper = nodeWithAttrs({ link: 'https://grandparent.example/' });
    const leaf = nodeWithAttrs({ copy: '{hover_label}' });
    wrapper.add(leaf);
    const t = readInteractionTemplates(leaf);
    expect(t.copy).toBe('{hover_label}');
    expect(t.link).toBeUndefined();
  });

  it('ignores non-string attr values (untrusted JSON)', () => {
    const leaf = nodeWithAttrs({ link: 42, copy: ['a'], link_target: {} });
    expect(readInteractionTemplates(leaf)).toEqual({});
  });

  it('returns empty for a node with no attrs, and for null', () => {
    expect(readInteractionTemplates(new THREE.Object3D())).toEqual({});
    expect(readInteractionTemplates(null)).toEqual({});
  });
});

describe('resolveElementActions', () => {
  it('resolves url, target and copy together', () => {
    const node = nodeWithAttrs({
      link: 'https://ex.org/{hover_label}',
      copy: 'id={hover_label}',
      link_target: '_self',
    });
    expect(resolveElementActions(node, VALUES)).toEqual({
      url: 'https://ex.org/P04637',
      target: '_self',
      copyText: 'id=P04637',
    });
  });

  it('offers copy from the label alone when only a link is authored', () => {
    const node = nodeWithAttrs({ link: 'https://ex.org/{hover_label}' });
    expect(resolveElementActions(node, VALUES)).toEqual({
      url: 'https://ex.org/P04637',
      target: '_blank',
      copyText: 'P04637',
    });
  });

  it('offers nothing but the label copy for a node with no templates', () => {
    expect(resolveElementActions(nodeWithAttrs({}), VALUES)).toEqual({
      url: null,
      target: '_blank',
      copyText: 'P04637',
    });
  });

  it('offers nothing at all for an unlabelled node with no templates', () => {
    expect(resolveElementActions(nodeWithAttrs({}), { ...VALUES, label: null })).toEqual({
      url: null,
      target: '_blank',
      copyText: null,
    });
  });
});
