/**
 * The two source scans that keep per-type code from quietly going asymmetric,
 * written as pure functions over a source string so their own tests can feed
 * them synthetic files.
 *
 * - {@link findGeometrySubsetEaches}: an `it.each` / `describe.each` /
 *   `test.each` over a literal list of geometry names that is a STRICT subset
 *   of the four ("three types, mesh forgotten"). Such a list must say why, in
 *   a `// geometry-subset: <reason>` comment on the line(s) just above it.
 * - {@link exportedNames}: the names a module exports, so parallel per-type
 *   modules (`commit-<t>-geometry.ts`, …) can be diffed.
 *
 * TypeScript-AST rather than regex for the first: the list is often a named
 * const declared elsewhere in the file, or the receiver of `.map(…)`, and a
 * regex over `each(` cannot follow either.
 *
 * @module tests/_conformance/source-scans
 */

import ts from 'typescript';

import { GEOMETRY_TYPES, type GeometryTypeName } from '../../types/format-contract';

/**
 * The spellings a test list uses for each type: the contract names, and the
 * singular material-kind keys (`point`, `line`, `gsplat`) of
 * `PICKING_FACTORIES` / `VISUAL_FACTORIES`.
 */
const SPELLINGS: Readonly<Record<string, GeometryTypeName>> = {
  points: 'points',
  lines: 'lines',
  gsplats: 'gsplats',
  mesh: 'mesh',
  point: 'points',
  line: 'lines',
  gsplat: 'gsplats',
};

/** The marker that excuses a subset list; the reason after the colon is required. */
export const SUBSET_MARKER = /\/\/\s*geometry-subset:\s*\S/;

/** One `*.each` call over a strict geometry subset with no excuse. */
export interface SubsetFinding {
  line: number;
  types: GeometryTypeName[];
  text: string;
}

/** Strip `as const`, `satisfies T`, parentheses and `<T>x` around an expression. */
function unwrap(node: ts.Expression): ts.Expression {
  let e = node;
  for (;;) {
    if (ts.isAsExpression(e) || ts.isSatisfiesExpression(e) || ts.isTypeAssertionExpression(e)) {
      e = e.expression;
    } else if (ts.isParenthesizedExpression(e)) {
      e = e.expression;
    } else {
      return e;
    }
  }
}

/** Every `const x = <init>` in the file, by name (block scoping is ignored; names are rare to reuse). */
function constInitializers(file: ts.SourceFile): Map<string, ts.Expression> {
  const out = new Map<string, ts.Expression>();
  const visit = (node: ts.Node): void => {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.initializer) {
      out.set(node.name.text, node.initializer);
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return out;
}

/**
 * The array literal an `each` argument stands for: the literal itself, a const
 * bound to one, or the receiver of a `.map` / `.filter` chain over one.
 */
function resolveArray(
  arg: ts.Expression,
  consts: Map<string, ts.Expression>,
  depth = 0
): ts.ArrayLiteralExpression | undefined {
  if (depth > 4) return undefined;
  const e = unwrap(arg);
  if (ts.isArrayLiteralExpression(e)) return e;
  if (ts.isIdentifier(e)) {
    const init = consts.get(e.text);
    return init ? resolveArray(init, consts, depth + 1) : undefined;
  }
  if (ts.isCallExpression(e) && ts.isPropertyAccessExpression(e.expression)) {
    return resolveArray(e.expression.expression, consts, depth + 1);
  }
  return undefined;
}

/**
 * The type a label names by its first word: `'line capsule'` and
 * `'gsplat/pick.tsl.ts'` both name a type, so a list of variants is still a
 * list of types.
 */
function typeOfLabel(text: string): GeometryTypeName | undefined {
  const word = text.split(/[\s/]/, 1)[0];
  return Object.hasOwn(SPELLINGS, word) ? SPELLINGS[word] : undefined;
}

/** The geometry types an array literal enumerates: string elements, or each tuple's first. */
function enumeratedTypes(array: ts.ArrayLiteralExpression): Set<GeometryTypeName> {
  const out = new Set<GeometryTypeName>();
  for (const el of array.elements) {
    let head = unwrap(el as ts.Expression);
    if (ts.isArrayLiteralExpression(head) && head.elements.length > 0) {
      head = unwrap(head.elements[0] as ts.Expression);
    }
    const type = ts.isStringLiteralLike(head) ? typeOfLabel(head.text) : undefined;
    if (type) out.add(type);
  }
  return out;
}

/** Whether `callee` is `it.each`, `describe.each`, `test.each` (through `.only` / `.skip` / `.fails`). */
function isEachCall(callee: ts.Expression): boolean {
  if (!ts.isPropertyAccessExpression(callee) || callee.name.text !== 'each') return false;
  let root: ts.Expression = callee.expression;
  while (ts.isPropertyAccessExpression(root)) root = root.expression;
  return ts.isIdentifier(root) && ['it', 'describe', 'test'].includes(root.text);
}

/** The comment lines directly above `line` (1-based), nearest first, until code. */
function commentBlockAbove(lines: string[], line: number): string[] {
  const out: string[] = [];
  for (let i = line - 2; i >= 0; i--) {
    const t = lines[i].trim();
    if (!t.startsWith('//') && !t.startsWith('*') && !t.startsWith('/*')) break;
    out.push(t);
  }
  return out;
}

/**
 * Every `*.each(list)` in `source` whose list names at least two geometry
 * types but not all four, and that carries no `// geometry-subset:` reason in
 * the comment block directly above the call.
 */
export function findGeometrySubsetEaches(source: string, fileName = 'x.ts'): SubsetFinding[] {
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true);
  const consts = constInitializers(file);
  const lines = source.split('\n');
  const findings: SubsetFinding[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node) && isEachCall(node.expression) && node.arguments.length > 0) {
      const array = resolveArray(node.arguments[0], consts);
      const types = array ? enumeratedTypes(array) : new Set<GeometryTypeName>();
      if (types.size >= 2 && types.size < GEOMETRY_TYPES.length) {
        const line = file.getLineAndCharacterOfPosition(node.getStart(file)).line + 1;
        const own = lines[line - 1];
        const excused = [own, ...commentBlockAbove(lines, line)].some((l) => SUBSET_MARKER.test(l));
        if (!excused) {
          findings.push({
            line,
            types: GEOMETRY_TYPES.filter((t) => types.has(t)),
            text: own.trim(),
          });
        }
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(file);
  return findings;
}

/** A class's public member names: no constructor, no `private` / `protected`, no `#private`. */
function publicMembers(cls: ts.ClassDeclaration): string[] {
  const hidden = new Set([ts.SyntaxKind.PrivateKeyword, ts.SyntaxKind.ProtectedKeyword]);
  return cls.members.flatMap((m) => {
    if (ts.isConstructorDeclaration(m) || !m.name || ts.isPrivateIdentifier(m.name)) return [];
    const modifiers = ts.canHaveModifiers(m) ? ts.getModifiers(m) : undefined;
    if (modifiers?.some((mod) => hidden.has(mod.kind))) return [];
    return ts.isIdentifier(m.name) ? [m.name.text] : [];
  });
}

/**
 * The names `source` exports — declarations and `export { … }` lists, not
 * re-exports from other modules — plus each exported class's public members
 * as `Class.member`, since a method added to one sibling class is the same
 * drift as a function added to one sibling module.
 */
export function exportedNames(source: string, fileName = 'x.ts'): string[] {
  const file = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true);
  const names: string[] = [];
  for (const stmt of file.statements) {
    const exported = ts
      .getModifiers(stmt as ts.HasModifiers)
      ?.some((m) => m.kind === ts.SyntaxKind.ExportKeyword);
    if (ts.isExportDeclaration(stmt) && !stmt.moduleSpecifier && stmt.exportClause) {
      if (ts.isNamedExports(stmt.exportClause)) {
        for (const el of stmt.exportClause.elements) names.push(el.name.text);
      }
      continue;
    }
    if (!exported) continue;
    if (ts.isVariableStatement(stmt)) {
      for (const d of stmt.declarationList.declarations) {
        if (ts.isIdentifier(d.name)) names.push(d.name.text);
      }
    } else if (ts.isClassDeclaration(stmt) && stmt.name) {
      names.push(stmt.name.text, ...publicMembers(stmt).map((m) => `${stmt.name!.text}.${m}`));
    } else if (
      (ts.isFunctionDeclaration(stmt) ||
        ts.isInterfaceDeclaration(stmt) ||
        ts.isTypeAliasDeclaration(stmt) ||
        ts.isEnumDeclaration(stmt)) &&
      stmt.name
    ) {
      names.push(stmt.name.text);
    }
  }
  return names;
}

/** The type tokens a per-type module name uses, in both cases, for normalising export names. */
const TYPE_TOKENS: Readonly<Record<GeometryTypeName, readonly string[]>> = {
  points: ['Points', 'points', 'Point', 'point'],
  lines: ['Lines', 'lines', 'Line', 'line'],
  gsplats: ['GSplats', 'gsplats', 'GSplat', 'gsplat'],
  mesh: ['Mesh', 'mesh'],
};

/** `name` with `type`'s token replaced by `<T>` (`commitLinesGeometry` → `commit<T>Geometry`). */
export function normaliseTypeToken(name: string, type: GeometryTypeName): string {
  for (const token of TYPE_TOKENS[type]) {
    if (name.includes(token)) return name.replaceAll(token, '<T>');
  }
  return name;
}
