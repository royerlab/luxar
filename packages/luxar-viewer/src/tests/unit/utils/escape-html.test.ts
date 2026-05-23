/**
 * Unit tests for `utils/escape-html.ts`.
 *
 * Strengthens utils.md G3 — pre-existing tests covered the obvious 4
 * substitutions but missed two structurally-important properties:
 *   1. `&` is escaped FIRST so already-escaped entities aren't
 *      double-escaped on a single pass (otherwise `escapeHtml('<')` would
 *      produce `&amp;lt;` instead of `&lt;`).
 *   2. The function is NOT idempotent (running it twice DOES re-escape
 *      the `&` in `&lt;`). This is a load-bearing contract: callers must
 *      escape exactly once.
 *
 * Also pins the apostrophe behavior (currently passes through unescaped —
 * see utils.md OOS3) so any future hardening is intentional.
 */

import { describe, it, expect } from 'vitest';
import { escapeHtml } from '../../../utils/escape-html';

describe('escapeHtml', () => {
  describe('basic substitutions', () => {
    it('escapes the four canonical HTML special characters', () => {
      expect(escapeHtml('<')).toBe('&lt;');
      expect(escapeHtml('>')).toBe('&gt;');
      expect(escapeHtml('&')).toBe('&amp;');
      expect(escapeHtml('"')).toBe('&quot;');
    });

    it('escapes HTML tags with nested attributes', () => {
      expect(escapeHtml('<script>alert("xss")</script>')).toBe(
        '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
      );
    });

    it('passes through safe strings unchanged', () => {
      expect(escapeHtml('hello world 123')).toBe('hello world 123');
    });

    it('handles empty string', () => {
      expect(escapeHtml('')).toBe('');
    });
  });

  describe('ordering invariant — ampersand must be escaped first', () => {
    it('does not double-escape when the input already contains an entity-like substring', () => {
      // If `&` were escaped LAST, this input would become `&lt;` (correct)
      // for the `<` and `&` would already be present; but for input `<`
      // alone the order matters: replace `&` first ⇒ no-op, then `<` ⇒ `&lt;`.
      // Inverted order ⇒ `<` ⇒ `&lt;` ⇒ `&` escapes ⇒ `&amp;lt;` ❌.
      expect(escapeHtml('<')).toBe('&lt;');
      expect(escapeHtml('<&>')).toBe('&lt;&amp;&gt;');
    });

    it('escapes the ampersand in pre-existing entity-looking text (NOT a sanitizer)', () => {
      // The function is dumb on purpose: it does not detect "already
      // an entity" and skip. `&amp;` becomes `&amp;amp;`. This is the
      // intended contract — callers must escape exactly once.
      expect(escapeHtml('&amp;')).toBe('&amp;amp;');
    });
  });

  describe('non-idempotence — must NOT escape twice', () => {
    it('is NOT idempotent (double-escape produces double-encoded output)', () => {
      // Locks in the "exactly once" contract. If a refactor makes
      // escapeHtml idempotent (e.g., regex avoiding `&` adjacent to
      // entity names), this assertion flips and the engineer must
      // re-validate every caller.
      const once = escapeHtml('a & b');
      const twice = escapeHtml(once);
      expect(once).toBe('a &amp; b');
      expect(twice).toBe('a &amp;amp; b');
      expect(twice).not.toBe(once);
    });
  });

  describe('apostrophe behavior (utils.md OOS3 — current implementation pins)', () => {
    it("passes apostrophe ' through unescaped", () => {
      // MED-32 fix: source now escapes `'` to `&#39;` for
      // defense-in-depth against single-quoted-attribute injection.
      // The CURRENT_BEHAVIOR_LEAVES_APOSTROPHE pin from the round-2
      // audit is flipped: the apostrophe IS escaped.
      expect(escapeHtml("can't")).toBe('can&#39;t');
      expect(escapeHtml("it's a 'quoted' value")).toBe(
        'it&#39;s a &#39;quoted&#39; value'
      );
    });
  });

  describe('full payload — all four entities at once preserve count', () => {
    it('escapes every special char exactly once in a mixed payload', () => {
      const input = '<a href="x">&amp;</a>';
      const out = escapeHtml(input);
      // Each special char appears in the input the right number of times;
      // the output substitutes each occurrence with its entity. Count the
      // entity substrings to pin the "exactly once each" invariant.
      const count = (s: string, sub: string) =>
        s.split(sub).length - 1;
      // Input has: 2 `<`, 2 `>`, 2 `"`, 1 `&` ⇒ output entities:
      expect(count(out, '&lt;')).toBe(2);
      expect(count(out, '&gt;')).toBe(2);
      expect(count(out, '&quot;')).toBe(2);
      expect(count(out, '&amp;')).toBe(1);
    });
  });
});
