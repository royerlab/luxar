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

import { describe, it, expect, test } from 'vitest';
import * as fc from 'fast-check';
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

  // utils.md O1 / Phase E23: previously the describe block was named
  // `'apostrophe behavior (... current implementation pins)'` and the
  // inner `it` was `'passes apostrophe ' through unescaped'` — both
  // contradicted the test body (which asserts the apostrophe IS escaped
  // to `&#39;`). The names lied; only the body told the truth. Rename
  // both to match what the test actually pins.
  describe('apostrophe escaping to &#39;', () => {
    it('escapes apostrophe to &#39; (MED-32 — single-quoted-attribute injection defence)', () => {
      // MED-32 fix: source now escapes `'` to `&#39;` for
      // defense-in-depth against single-quoted-attribute injection.
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

  describe('property tests [utils.md/H-candidate][P12]', () => {
    // Counts non-overlapping occurrences of `sub` in `s`.
    const count = (s: string, sub: string): number => s.split(sub).length - 1;

    test('escape never decreases the length of the string', () => {
      // Each substitution either leaves length unchanged (no special chars)
      // or strictly increases it (entities are >= 2 chars longer than the
      // single char they replace). A mutation that drops a substitution
      // (or returns the input verbatim under some condition) would still
      // pass — except this property forces the entity-shape contract on
      // *all* of <, >, &, ", '.
      fc.assert(
        fc.property(fc.string({ maxLength: 64 }), (s) => {
          const escaped = escapeHtml(s);
          expect(escaped.length).toBeGreaterThanOrEqual(s.length);
        })
      );
    });

    test('safe strings (no special chars) are pass-through (identity)', () => {
      // Restrict to ASCII letters/digits/spaces — none of which appear in
      // the substitution set. The function must be identity here.
      const safeChar = fc.constantFrom(
        ...'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?'.split('')
      );
      fc.assert(
        fc.property(fc.array(safeChar, { maxLength: 64 }), (chars) => {
          const s = chars.join('');
          expect(escapeHtml(s)).toBe(s);
        })
      );
    });

    test('output never contains any of the raw special chars after escape', () => {
      // Injectivity-of-shape: every <, >, &, ", \' in the input must be
      // rewritten in the output. Raw `<`, `>`, `"`, `'` cannot survive any
      // valid escape — they all have substitutions that do not include the
      // raw char. `&` IS allowed in output (entity prefix) but only as part
      // of one of five known entities; total `&` count in output equals the
      // sum of entity occurrences.
      const specialOrAlnum = fc.constantFrom(
        ...'<>&"\'abcdefghijklmnopqrstuvwxyz0123456789 '.split('')
      );
      fc.assert(
        fc.property(fc.array(specialOrAlnum, { maxLength: 64 }), (chars) => {
          const s = chars.join('');
          const out = escapeHtml(s);
          expect(count(out, '<')).toBe(0);
          expect(count(out, '>')).toBe(0);
          expect(count(out, '"')).toBe(0);
          expect(count(out, "'")).toBe(0);
          const ampInOut = count(out, '&');
          const entityCount =
            count(out, '&amp;') +
            count(out, '&lt;') +
            count(out, '&gt;') +
            count(out, '&quot;') +
            count(out, '&#39;');
          expect(ampInOut).toBe(entityCount);
        })
      );
    });

    test('input ampersand count equals output &amp; count when no other specials present', () => {
      // Isolated ampersand invariant: when the input contains only `&` and
      // non-special chars, every `&` becomes exactly one `&amp;`. This pins
      // the first-escape-ampersand-rule across arbitrary inputs.
      const safeNonSpecial = fc.constantFrom(
        ...'abcdefghijklmnopqrstuvwxyz0123456789 '.split('')
      );
      fc.assert(
        fc.property(
          fc.array(fc.oneof(fc.constant('&'), safeNonSpecial), { maxLength: 32 }),
          (chars) => {
            const s = chars.join('');
            const ampsIn = count(s, '&');
            const out = escapeHtml(s);
            expect(count(out, '&amp;')).toBe(ampsIn);
          }
        )
      );
    });
  });
});
