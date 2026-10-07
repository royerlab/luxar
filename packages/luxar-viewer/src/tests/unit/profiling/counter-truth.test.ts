/**
 * Counter truth: the perf-counter registry, the code that writes the
 * counters, and the tests that prove their values must agree.
 *
 * The render gate judges builds on these numbers (`gate-scenes.json`), so a
 * counter nobody tests can drift from what it claims to count without any
 * test going red — the third viewer review found demand stats counting
 * aborts and warm-ups that way. Three checks keep the registry honest:
 *
 * 1. Every name the source hands `perfCounters.slot` / `.inc` is declared,
 *    and every declared name is still written somewhere (no dead entries).
 * 2. Loading every module that owns a counter registers exactly the declared
 *    fixed names — the snapshot the gate reads has no undeclared key.
 * 3. Every declared counter has an exact-value assertion in the test file
 *    `_conformance/perf-counter-tests.ts` names for it. "Exact" is judged
 *    syntactically (approximate, but it catches the never-tested counter):
 *    the counter's name appears inside an `expect(...)` (directly, or inside a
 *    local getter the expect calls) whose matcher is `toBe` / `toEqual` /
 *    `toStrictEqual` / `toMatchObject`, or inside such a matcher's expected
 *    value — and neither side compares against a `before` / `prev` /
 *    `baseline` / `initial` / `start…` snapshot or reduces the observed
 *    counter to a comparison or difference.
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';
import {
  PERF_COUNTERS,
  PERF_COUNTER_FAMILIES,
  perfCounters,
  perfCounterSpec,
} from '../../../profiling/perf-counters';
import { PERF_COUNTER_TESTS } from '../../_conformance/perf-counter-tests';
import {
  UNIT_TESTS_DIR,
  VIEWER_SRC_DIR,
  forEachNode,
  lineOf,
  parseSourceFile,
  productionSourceFiles,
} from '../../helpers/source-scan';

const FIXED = Object.keys(PERF_COUNTERS);
const FAMILIES = Object.keys(PERF_COUNTER_FAMILIES);

interface CounterUse {
  file: string;
  line: number;
  /** A literal name, or the static head of a template (`fetch.` of `fetch.${lane}.requests`). */
  text: string;
  /** For a template: a regex over the names it can produce. */
  pattern?: RegExp;
}

function escapeRegex(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function templatePattern(node: ts.TemplateExpression): RegExp {
  const parts = [escapeRegex(node.head.text)];
  for (const span of node.templateSpans) parts.push('.+', escapeRegex(span.literal.text));
  return new RegExp(`^${parts.join('')}$`);
}

/** Strip `as` / `satisfies` / `<T>` / parentheses: a cast must not hide the name. */
function unwrapExpression(node: ts.Expression): ts.Expression {
  let current = node;
  while (
    ts.isAsExpression(current) ||
    ts.isSatisfiesExpression(current) ||
    ts.isTypeAssertionExpression(current) ||
    ts.isParenthesizedExpression(current)
  ) {
    current = current.expression;
  }
  return current;
}

/** Every name argument of `perfCounters.slot(...)` / `perfCounters.inc(...)` in production code. */
function counterUses(): CounterUse[] {
  const uses: CounterUse[] = [];
  for (const file of productionSourceFiles()) {
    const sf = parseSourceFile(file);
    if (!sf.text.includes('perfCounters.')) continue;
    forEachNode(sf, (node) => {
      if (!ts.isCallExpression(node) || !ts.isPropertyAccessExpression(node.expression)) return;
      const callee = node.expression;
      if (callee.expression.getText(sf) !== 'perfCounters') return;
      if (callee.name.text !== 'slot' && callee.name.text !== 'inc') return;
      const arg = unwrapExpression(node.arguments[0]);
      const line = lineOf(node);
      if (ts.isStringLiteral(arg) || ts.isNoSubstitutionTemplateLiteral(arg)) {
        uses.push({ file, line, text: arg.text });
      } else if (ts.isTemplateExpression(arg)) {
        uses.push({ file, line, text: arg.head.text, pattern: templatePattern(arg) });
      } else {
        uses.push({ file, line, text: `<non-literal ${arg.getText(sf)}>` });
      }
    });
  }
  return uses;
}

const DELTA_IDENTIFIER = /^(before|prev|previous|baseline|initial|start)/i;
const EXACT_MATCHERS = new Set(['toBe', 'toEqual', 'toStrictEqual', 'toMatchObject']);

/** `expect(<arg>).<matcher>(<expected>)` → the matcher call, when the matcher is exact. */
function exactMatcherOf(expectCall: ts.CallExpression): ts.CallExpression | undefined {
  const access = expectCall.parent;
  if (!access || !ts.isPropertyAccessExpression(access) || access.expression !== expectCall) {
    return undefined;
  }
  const call = access.parent;
  if (!EXACT_MATCHERS.has(access.name.text) || !call || !ts.isCallExpression(call))
    return undefined;
  return call;
}

function readsRelativeValue(node: ts.Node): boolean {
  let relative = false;
  const scan = (child: ts.Node): void => {
    if (ts.isIdentifier(child) && DELTA_IDENTIFIER.test(child.text)) relative = true;
    if (
      ts.isBinaryExpression(child) &&
      [
        ts.SyntaxKind.GreaterThanToken,
        ts.SyntaxKind.GreaterThanEqualsToken,
        ts.SyntaxKind.LessThanToken,
        ts.SyntaxKind.LessThanEqualsToken,
        ts.SyntaxKind.MinusToken,
      ].includes(child.operatorToken.kind)
    ) {
      relative = true;
    }
    ts.forEachChild(child, scan);
  };
  scan(node);
  return relative;
}

function readsDelta(matcher: ts.CallExpression): boolean {
  let delta = false;
  const actual = ts.isPropertyAccessExpression(matcher.expression)
    ? matcher.expression.expression
    : undefined;
  if (actual && isExpectCall(actual)) {
    for (const arg of actual.arguments) {
      if (readsRelativeValue(arg)) delta = true;
    }
  }
  for (const arg of matcher.arguments) {
    if (arg.kind === ts.SyntaxKind.TrueKeyword || arg.kind === ts.SyntaxKind.FalseKeyword)
      delta = true;
    const scan = (node: ts.Node): void => {
      if (ts.isIdentifier(node) && DELTA_IDENTIFIER.test(node.text)) delta = true;
      ts.forEachChild(node, scan);
    };
    scan(arg);
  }
  return delta;
}

function isExpectCall(node: ts.Node): node is ts.CallExpression {
  return (
    ts.isCallExpression(node) &&
    ts.isIdentifier(node.expression) &&
    node.expression.text === 'expect'
  );
}

/** The name a function-valued node is bound to (`const pinned = () => …`, `function pinned()`). */
function boundFunctionName(node: ts.Node): string | undefined {
  if (ts.isFunctionDeclaration(node)) return node.name?.text;
  if (
    (ts.isArrowFunction(node) || ts.isFunctionExpression(node)) &&
    ts.isVariableDeclaration(node.parent) &&
    ts.isIdentifier(node.parent.name)
  ) {
    return node.parent.name.text;
  }
  return undefined;
}

/** Whether the file has `expect(<getter>(…)).<exact>(…)` without a delta. */
function getterIsExactlyAsserted(sf: ts.SourceFile, getter: string): boolean {
  let found = false;
  forEachNode(sf, (node) => {
    if (found || !isExpectCall(node)) return;
    const [arg] = node.arguments;
    if (!arg || !ts.isCallExpression(arg) || arg.expression.getText(sf) !== getter) return;
    const matcher = exactMatcherOf(node);
    if (matcher && !readsDelta(matcher)) found = true;
  });
  return found;
}

/** Whether the literal `node` sits in an exact, non-delta assertion. */
function isExactlyAsserted(sf: ts.SourceFile, node: ts.Node): boolean {
  for (let child = node, parent = node.parent; parent; child = parent, parent = parent.parent) {
    if (ts.isCallExpression(parent)) {
      // Inside a matcher's expected value: `.toEqual({ 'l2.hits': 1 })`.
      if (
        ts.isPropertyAccessExpression(parent.expression) &&
        EXACT_MATCHERS.has(parent.expression.name.text) &&
        isExpectCall(parent.expression.expression) &&
        parent.arguments.includes(child as ts.Expression)
      ) {
        return !readsDelta(parent);
      }
      if (isExpectCall(parent)) {
        const matcher = exactMatcherOf(parent);
        return matcher !== undefined && !readsDelta(matcher);
      }
    }
    const getter = boundFunctionName(parent);
    if (getter !== undefined)
      return !readsRelativeValue(parent) && getterIsExactlyAsserted(sf, getter);
  }
  return false;
}

/** Lines of the exact assertions of `name` (a family: of any member) in `testFile`. */
function exactAssertionLines(testFile: string, name: string, family: boolean): number[] {
  const path = join(UNIT_TESTS_DIR, testFile);
  const sf = ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true);
  const lines: number[] = [];
  forEachNode(sf, (node) => {
    if (!ts.isStringLiteral(node) && !ts.isNoSubstitutionTemplateLiteral(node)) return;
    const matches = family
      ? node.text.startsWith(name) && node.text.length > name.length
      : node.text === name;
    if (matches && isExactlyAsserted(sf, node)) lines.push(lineOf(node));
  });
  return lines;
}

describe('perf counter registry ↔ the code that writes it', () => {
  const uses = counterUses();

  it('every name the source writes is declared', () => {
    const undeclared = uses
      .filter((use) =>
        use.pattern
          ? !FAMILIES.includes(use.text) && !FIXED.some((name) => use.pattern!.test(name))
          : perfCounterSpec(use.text) === undefined
      )
      .map((use) => `src/${use.file}:${use.line} writes '${use.text}'`);
    expect(undeclared, 'declare these in PERF_COUNTERS (profiling/perf-counters.ts)').toEqual([]);
  });

  it('every declared name is still written somewhere (no dead entries)', () => {
    const dead = [
      ...FIXED.filter(
        (name) => !uses.some((use) => (use.pattern ? use.pattern.test(name) : use.text === name))
      ),
      ...FAMILIES.filter((prefix) => !uses.some((use) => use.pattern && use.text === prefix)),
    ];
    expect(dead, 'remove these from PERF_COUNTERS / PERF_COUNTER_FAMILIES').toEqual([]);
  });

  it('loading every counter-owning module registers exactly the declared fixed names', async () => {
    const owners = [...new Set(uses.map((use) => use.file))];
    for (const file of owners) await import(/* @vite-ignore */ join(VIEWER_SRC_DIR, file));
    // The one lazily-registered fixed name: the debug-only getObjectByName
    // counter resolves its slot when the instrument installs.
    const { installDebugPerfInstruments } =
      await import('../../../core/app/debug/perf-instruments');
    installDebugPerfInstruments(() => null);
    const keys = Object.keys(perfCounters.snapshot());
    expect({
      registeredButUndeclared: keys.filter((key) => !FIXED.includes(key)),
      declaredButNeverRegistered: FIXED.filter((name) => !keys.includes(name)),
    }).toEqual({ registeredButUndeclared: [], declaredButNeverRegistered: [] });
  });
});

describe('every perf counter has an exact-value test', () => {
  it.each([
    "expect(perfCounters.get('l2.hits') > 0).toBe(true)",
    "const before = perfCounters.get('l2.hits'); expect(perfCounters.get('l2.hits') - before).toBe(1)",
    "const before = 0; const read = () => perfCounters.get('l2.hits') - before; expect(read()).toBe(1)",
    "const snap = {}; expect(snap).not.toEqual({ 'l2.hits': 1 })",
  ])('rejects non-exact assertion: %s', (assertion) => {
    const sf = ts.createSourceFile('probe.ts', assertion, ts.ScriptTarget.Latest, true);
    let counter: ts.StringLiteral | undefined;
    forEachNode(sf, (node) => {
      if (ts.isStringLiteral(node) && node.text === 'l2.hits') counter = node;
    });
    expect(counter, assertion).toBeDefined();
    expect(isExactlyAsserted(sf, counter!), assertion).toBe(false);
  });

  it('accepts an absolute counter value', () => {
    const sf = ts.createSourceFile(
      'probe.ts',
      "expect(perfCounters.get('l2.hits')).toBe(1)",
      ts.ScriptTarget.Latest,
      true
    );
    let counter: ts.StringLiteral | undefined;
    forEachNode(sf, (node) => {
      if (ts.isStringLiteral(node) && node.text === 'l2.hits') counter = node;
    });
    expect(counter).toBeDefined();
    expect(isExactlyAsserted(sf, counter!)).toBe(true);
  });

  it.each(Object.entries(PERF_COUNTER_TESTS))('%s → %s', (name, testFile) => {
    expect(existsSync(join(UNIT_TESTS_DIR, testFile)), `${testFile} does not exist`).toBe(true);
    const lines = exactAssertionLines(testFile, name, FAMILIES.includes(name));
    expect(
      lines.length,
      `${testFile} has no exact toBe/toEqual assertion of '${name}' (a delta or a > 0 does not count)`
    ).toBeGreaterThan(0);
  });
});
