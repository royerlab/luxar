// @vitest-environment jsdom
/**
 * Guards for `src/tests/e2e/page-predicates.ts`: content parity with the
 * production predicate, AND the self-containment invariant that makes the
 * copy legal in the first place.
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
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
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

/**
 * Identifiers a page-side predicate may legally reach for: they exist in
 * every browser page, so Playwright's serialized copy still resolves them.
 * Anything else is a closure/module-scope reference and would throw a
 * `ReferenceError` inside `page.evaluate()`.
 */
const PAGE_GLOBALS = new Set([
  'document',
  'window',
  'navigator',
  'Node',
  'Element',
  'HTMLElement',
  'HTMLInputElement',
  'Boolean',
  'String',
  'Number',
  'Array',
  'Object',
  'JSON',
  'Math',
]);

/** Keywords/literals the crude tokenizer below cannot tell from identifiers. */
const RESERVED = new Set([
  'function',
  'const',
  'let',
  'var',
  'return',
  'if',
  'else',
  'true',
  'false',
  'null',
  'undefined',
  'typeof',
  'instanceof',
  'new',
  'void',
  'in',
  'of',
  'for',
  'while',
  'do',
  'switch',
  'case',
  'break',
  'continue',
  'default',
  'delete',
  'this',
  'throw',
  'try',
  'catch',
  'finally',
  'class',
  'extends',
  'async',
  'await',
  'yield',
]);

/**
 * Every identifier the function body reads that it does not itself declare.
 *
 * Deliberately crude — strings, comments, property accesses (`el.tagName`,
 * `el?.type`) and locally-declared names are stripped, and what remains must
 * be a page global. A parser would be more precise, but this operates on the
 * function's OWN `toString()`, which is exactly the text Playwright ships to
 * the page, so it cannot drift from what actually runs there.
 */
function freeIdentifiers(fn: () => unknown): string[] {
  const source = fn.toString();
  const stripped = source
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/(['"`])(?:\\.|(?!\1)[^\\])*\1/g, ' ');

  // Locally bound names: the function's own name, its parameters, and every
  // `const`/`let`/`var` it declares.
  const local = new Set<string>();
  const header = /^\s*(?:async\s+)?function\s*\*?\s*([A-Za-z_$][\w$]*)?\s*\(([^)]*)\)/.exec(
    stripped
  );
  if (header?.[1]) local.add(header[1]);
  for (const param of (header?.[2] ?? '').split(',')) {
    const name = /[A-Za-z_$][\w$]*/.exec(param.trim())?.[0];
    if (name) local.add(name);
  }
  for (const [, name] of stripped.matchAll(/\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)/g)) {
    local.add(name);
  }

  // Property accesses are not free identifiers (`.tagName`, `?.type`).
  const withoutMembers = stripped.replace(/\.\s*[A-Za-z_$][\w$]*/g, ' ');

  const free = new Set<string>();
  for (const [name] of withoutMembers.matchAll(/[A-Za-z_$][\w$]*/g)) {
    if (!RESERVED.has(name) && !local.has(name)) free.add(name);
  }
  return [...free].sort();
}

describe('page-predicates stays serializable into the page', () => {
  // `fileURLToPath` on the raw string, not `new URL(...)`: under the jsdom
  // environment the global `URL` is jsdom's, which `node:fs` rejects.
  const moduleSource = readFileSync(
    resolve(dirname(fileURLToPath(import.meta.url)), '../../e2e/page-predicates.ts'),
    { encoding: 'utf8' }
  );

  it('the module has no imports', () => {
    // Playwright serializes only the FUNCTION BODY, so an import is invisible
    // to TypeScript's checker but fatal in the page.
    const code = moduleSource.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/[^\n]*$/gm, '');
    expect(code).not.toMatch(/^\s*import\b/m);
    expect(code).not.toMatch(/\brequire\s*\(/);
  });

  it('the module declares no module-scope bindings to close over', () => {
    const code = moduleSource.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/[^\n]*$/gm, '');
    // Column 0 = module scope; anything indented is inside a function body.
    expect(code).not.toMatch(/^(?:export\s+)?(?:const|let|var)\s/m);
  });

  it('isTypingSurfaceInPage references nothing but page globals', () => {
    // The real invariant behind the duplication: a free identifier here is a
    // page-side ReferenceError that only a full Playwright run would surface.
    const free = freeIdentifiers(isTypingSurfaceInPage);

    expect(free.length).toBeGreaterThan(0); // sanity: the tokenizer sees something
    for (const name of free) {
      expect(PAGE_GLOBALS.has(name), `free identifier "${name}" is not a page global`).toBe(true);
    }
  });
});
