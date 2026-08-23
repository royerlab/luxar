// @vitest-environment jsdom
/**
 * Parity guard for the two copies of the "is focus on a typing surface?"
 * predicate.
 *
 * `src/tests/e2e/page-predicates.ts::isTypingSurfaceInPage` has to be
 * self-contained so Playwright can serialize it into the page, so it cannot
 * import the production `input/input-handler/commands/focus-utils.ts::
 * isTypingInInput` — the exact predicate `InputHandler.onKeyDown` guards on
 * (issue #1922). Without this test a case added to one copy could silently
 * diverge from the other, and the E2E panel-toggle assertions
 * (`isFocusOnTypingSurface(page)` must be `false`) would quietly stop meaning
 * what they claim.
 *
 * Table-driven and run against a REAL focused element, so both copies see the
 * same `document.activeElement` a browser would give them.
 */
import { describe, it, expect, afterEach } from 'vitest';
import { isTypingSurfaceInPage } from '../../e2e/page-predicates';
import { isTypingInInput } from '../../../input/input-handler/commands/focus-utils';

interface Case {
  name: string;
  build: () => HTMLElement;
  /** What both predicates must answer. */
  typing: boolean;
}

function input(type: string): () => HTMLElement {
  return () => {
    const el = document.createElement('input');
    el.type = type;
    return el;
  };
}

function tagged(tag: string): () => HTMLElement {
  return () => document.createElement(tag) as HTMLElement;
}

const CASES: Case[] = [
  { name: 'input[type=text]', build: input('text'), typing: true },
  { name: 'input[type=search]', build: input('search'), typing: true },
  { name: 'input[type=number]', build: input('number'), typing: true },
  { name: 'input[type=password]', build: input('password'), typing: true },
  { name: 'input[type=email]', build: input('email'), typing: true },
  { name: 'input[type=range]', build: input('range'), typing: false },
  { name: 'input[type=checkbox]', build: input('checkbox'), typing: false },
  { name: 'input[type=radio]', build: input('radio'), typing: false },
  { name: 'textarea', build: tagged('textarea'), typing: true },
  {
    name: 'select',
    build: () => {
      const el = document.createElement('select');
      const option = document.createElement('option');
      option.textContent = 'a';
      el.appendChild(option);
      return el;
    },
    typing: true,
  },
  {
    name: 'contenteditable div',
    build: () => {
      const el = document.createElement('div');
      el.setAttribute('contenteditable', 'true');
      el.tabIndex = 0;
      return el;
    },
    typing: true,
  },
  {
    name: 'contenteditable="false" div',
    build: () => {
      const el = document.createElement('div');
      el.setAttribute('contenteditable', 'false');
      el.tabIndex = 0;
      return el;
    },
    typing: false,
  },
  {
    name: 'tabindex="-1" panel container (the #1922 resting place)',
    build: () => {
      const el = document.createElement('div');
      el.setAttribute('role', 'dialog');
      el.tabIndex = -1;
      return el;
    },
    typing: false,
  },
  { name: 'button', build: tagged('button'), typing: false },
  {
    name: 'anchor with href',
    build: () => {
      const el = document.createElement('a');
      el.href = '#';
      return el;
    },
    typing: false,
  },
  {
    name: 'listing row (div[role=button])',
    build: () => {
      const el = document.createElement('div');
      el.setAttribute('role', 'button');
      el.tabIndex = 0;
      return el;
    },
    typing: false,
  },
];

describe('E2E typing-surface predicate matches focus-utils::isTypingInInput', () => {
  afterEach(() => {
    document.body.innerHTML = '';
  });

  it.each(CASES)('$name', ({ build, typing }) => {
    const el = build();
    document.body.appendChild(el);
    el.focus();
    expect(document.activeElement).toBe(el);

    expect(isTypingInInput(document.activeElement)).toBe(typing);
    expect(isTypingSurfaceInPage()).toBe(typing);
  });

  it('agrees when nothing is focused', () => {
    // jsdom parks focus on <body> with nothing else focusable.
    expect(isTypingInInput(document.activeElement)).toBe(false);
    expect(isTypingSurfaceInPage()).toBe(false);
  });
});
