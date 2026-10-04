/**
 * Keeps `_conformance/async-operations.ts` honest: every cancellable or
 * disposable async operation in `src/` has a row (and so a lifecycle test),
 * or says why not with `// lifecycle-exempt: <reason>`.
 *
 * "Cancellable or disposable" is decided syntactically, on exported top-level
 * declarations only:
 *  - a function (declaration, or a `const` arrow / function expression) with a
 *    parameter typed with `AbortSignal`;
 *  - a class with a non-private method taking an `AbortSignal`, or a class
 *    defining `dispose()` next to at least one `async` method.
 * An options bag carrying a signal (`opts: LoadOptions`) is invisible to this
 * scan — it reads type annotations, not types — so the registry also lists
 * operations the scan cannot see (they are checked to exist all the same).
 */

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';
import { ASYNC_OPERATIONS } from '../../_conformance/async-operations';
import { UNIT_TESTS_DIR, parseSourceFile, productionSourceFiles } from '../../helpers/source-scan';

const EXEMPT = /lifecycle-exempt:(.*)/;

interface Finding {
  /** `<path under src/>#<name>`. */
  key: string;
  /** Why the scan flags it. */
  why: string;
  /** The `lifecycle-exempt` reason on the declaration, if any (may be empty: an error). */
  exempt?: string;
}

const isExported = (node: ts.Node): boolean =>
  ts.canHaveModifiers(node) &&
  (ts.getModifiers(node) ?? []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword);

const hasModifier = (node: ts.Node, kind: ts.SyntaxKind): boolean =>
  ts.canHaveModifiers(node) && (ts.getModifiers(node) ?? []).some((m) => m.kind === kind);

function takesSignal(params: ts.NodeArray<ts.ParameterDeclaration>, sf: ts.SourceFile): boolean {
  return params.some((p) => p.type !== undefined && /\bAbortSignal\b/.test(p.type.getText(sf)));
}

function exemption(statement: ts.Statement, sf: ts.SourceFile): string | undefined {
  const ranges = ts.getLeadingCommentRanges(sf.text, statement.getFullStart()) ?? [];
  for (const range of ranges) {
    const match = EXEMPT.exec(sf.text.slice(range.pos, range.end));
    if (match) return match[1].replace(/\*\/\s*$/, '').trim();
  }
  return undefined;
}

function classFinding(node: ts.ClassDeclaration, sf: ts.SourceFile): string | undefined {
  const methods = node.members.filter(ts.isMethodDeclaration);
  const signalled = methods.filter(
    (m) =>
      takesSignal(m.parameters, sf) &&
      !hasModifier(m, ts.SyntaxKind.PrivateKeyword) &&
      !ts.isPrivateIdentifier(m.name)
  );
  if (signalled.length > 0) {
    return `public method ${signalled.map((m) => m.name.getText(sf)).join(', ')} takes an AbortSignal`;
  }
  const disposes = methods.some((m) => m.name.getText(sf) === 'dispose');
  const asyncMethod = methods.find((m) => hasModifier(m, ts.SyntaxKind.AsyncKeyword));
  if (disposes && asyncMethod) {
    return `defines dispose() and async ${asyncMethod.name.getText(sf)}()`;
  }
  return undefined;
}

/** The cancellable / disposable exports of one parsed file (`rel` is its path under src/). */
export function cancellableExports(sf: ts.SourceFile, rel: string): Finding[] {
  const out: Finding[] = [];
  const add = (statement: ts.Statement, name: string, why: string): void => {
    out.push({ key: `${rel}#${name}`, why, exempt: exemption(statement, sf) });
  };
  for (const statement of sf.statements) {
    if (!isExported(statement)) continue;
    if (ts.isFunctionDeclaration(statement) && statement.name) {
      if (takesSignal(statement.parameters, sf)) {
        add(statement, statement.name.text, 'takes an AbortSignal');
      }
    } else if (ts.isVariableStatement(statement)) {
      for (const decl of statement.declarationList.declarations) {
        const init = decl.initializer;
        if (
          init &&
          (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) &&
          takesSignal(init.parameters, sf)
        ) {
          add(statement, decl.name.getText(sf), 'takes an AbortSignal');
        }
      }
    } else if (ts.isClassDeclaration(statement) && statement.name) {
      const why = classFinding(statement, sf);
      if (why) add(statement, statement.name.text, why);
    }
  }
  return out;
}

/** Every `lifecycle-exempt` comment in a file, with its line. */
function exemptionComments(sf: ts.SourceFile, rel: string): string[] {
  return sf.text
    .split('\n')
    .flatMap((line, i) => (EXEMPT.test(line) ? [`src/${rel}:${i + 1}`] : []));
}

/** Names a file exports (declarations, `export { … }` and `export const`). */
function exportedNames(sf: ts.SourceFile): Set<string> {
  const names = new Set<string>();
  for (const statement of sf.statements) {
    if (ts.isExportDeclaration(statement) && statement.exportClause) {
      if (ts.isNamedExports(statement.exportClause)) {
        for (const el of statement.exportClause.elements) names.add(el.name.text);
      }
    } else if (isExported(statement)) {
      if (ts.isVariableStatement(statement)) {
        for (const decl of statement.declarationList.declarations) {
          names.add(decl.name.getText(sf));
        }
      } else if (
        (ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) &&
        statement.name
      ) {
        names.add(statement.name.text);
      }
    }
  }
  return names;
}

function parse(source: string): ts.SourceFile {
  return ts.createSourceFile('probe.ts', source, ts.ScriptTarget.Latest, true);
}

describe('the cancellable-operation scan itself', () => {
  it('flags signal-taking exports and dispose()+async classes, and reads exemptions', () => {
    const sf = parse(`
      export async function load(url: string, signal?: AbortSignal) {}
      export const fetchIt = (signal: AbortSignal) => signal;
      function internal(signal: AbortSignal) {}
      export function pure(n: number) { return n; }
      export class Owner { async start() {} dispose() {} }
      export class SyncOwner { start() {} dispose() {} }
      export class Reader { get(key: string, signal?: AbortSignal) {} }
      export class Hidden { private get(signal: AbortSignal) {} #read(signal: AbortSignal) {} }
      // lifecycle-exempt: tags a signal, starts no work
      export function tag(signal: AbortSignal) {}
      /** Doc. */
      // lifecycle-exempt:
      export function unexplained(signal: AbortSignal) {}
    `);
    expect(cancellableExports(sf, 'probe.ts')).toEqual([
      { key: 'probe.ts#load', why: 'takes an AbortSignal', exempt: undefined },
      { key: 'probe.ts#fetchIt', why: 'takes an AbortSignal', exempt: undefined },
      { key: 'probe.ts#Owner', why: 'defines dispose() and async start()', exempt: undefined },
      {
        key: 'probe.ts#Reader',
        why: 'public method get takes an AbortSignal',
        exempt: undefined,
      },
      { key: 'probe.ts#tag', why: 'takes an AbortSignal', exempt: 'tags a signal, starts no work' },
      { key: 'probe.ts#unexplained', why: 'takes an AbortSignal', exempt: '' },
    ]);
  });
});

describe('async-operation registry', () => {
  const files = productionSourceFiles();
  const parsed = new Map(files.map((file) => [file, parseSourceFile(file)]));
  const findings = files.flatMap((file) => cancellableExports(parsed.get(file)!, file));
  const registered = new Map<string, string>();
  for (const row of ASYNC_OPERATIONS)
    for (const symbol of row.symbols) registered.set(symbol, row.id);

  it('every cancellable or disposable export has a row or a reasoned exemption', () => {
    const missing = findings
      .filter((f) => f.exempt === undefined && !registered.has(f.key))
      .map((f) => `src/${f.key} (${f.why})`);
    expect(
      missing.join('\n'),
      'add each to _conformance/async-operations.ts with its lifecycle test, ' +
        'or mark it `// lifecycle-exempt: <reason>`'
    ).toBe('');
  });

  it('every exemption gives a reason, sits on a flagged export, and is not also registered', () => {
    const flagged = new Set(findings.filter((f) => f.exempt !== undefined).map((f) => f.key));
    expect(findings.filter((f) => f.exempt === '').map((f) => f.key)).toEqual([]);
    expect(
      findings.filter((f) => f.exempt !== undefined && registered.has(f.key)).map((f) => f.key)
    ).toEqual([]);
    const comments = files.flatMap((file) => exemptionComments(parsed.get(file)!, file));
    expect(comments.length, 'stray lifecycle-exempt comments').toBe(flagged.size);
  });

  it('every row names symbols that exist and are listed once', () => {
    const stale = ASYNC_OPERATIONS.flatMap((row) =>
      row.symbols.filter((symbol) => {
        const [file, name] = symbol.split('#');
        const sf = parsed.get(file);
        return !sf || !exportedNames(sf).has(name);
      })
    );
    expect(stale, 'no such export').toEqual([]);
    const all = ASYNC_OPERATIONS.flatMap((row) => row.symbols);
    expect(all.filter((symbol, i) => all.indexOf(symbol) !== i)).toEqual([]);
  });

  it.each(ASYNC_OPERATIONS.map((row) => [row.id, row] as const))(
    '%s: its tests exist and exercise it',
    (_id, row) => {
      const missing = row.tests.filter((test) => !existsSync(join(UNIT_TESTS_DIR, test)));
      expect(missing, 'no such test file').toEqual([]);
      const sources = row.tests.map((test) => readFileSync(join(UNIT_TESTS_DIR, test), 'utf8'));
      const names = row.symbols.map((symbol) => symbol.split('#')[1]);
      expect(
        sources.some((text) => names.some((name) => new RegExp(`\\b${name}\\b`).test(text))),
        `none of ${row.tests.join(', ')} names ${names.join(' / ')}`
      ).toBe(true);
      if (row.contract) {
        const call = new RegExp(
          `defineLifecycleContract\\(\\s*['"\`]${row.contract.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}['"\`]`
        );
        expect(
          sources.some((text) => call.test(text)),
          `no defineLifecycleContract('${row.contract}', …) in ${row.tests.join(', ')}`
        ).toBe(true);
      }
    }
  );
});
