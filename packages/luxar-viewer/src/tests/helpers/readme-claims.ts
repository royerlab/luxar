/**
 * Extractors for the README claim checks (`src/tests/unit/readme/readme-claims.test.ts`).
 *
 * Two claims a README makes about code, both of which went stale silently:
 *
 * 1. **A symbol exists.** An inline code span that is exactly one code-shaped
 *    identifier (optionally dotted, optionally called — `fooBar`, `FooBar`,
 *    `FOO_BAR`, `Foo.bar()`) names something the source still declares or uses.
 *    The check is deliberately NARROW to keep false positives near zero:
 *    fenced blocks, spans with spaces or operators, file names, plain lowercase
 *    words and snake_case (zarr attrs, Python, Rust) are not read at all.
 * 2. **An enumerated list is complete.** A list wrapped in
 *    `<!-- mirrors: <path>#<target> -->` … `<!-- /mirrors -->` names exactly the
 *    members of a code set: `#exports` (the module's value exports) or
 *    `#<Class>.getters` (a class's public get-accessors). Each item's first code
 *    span is the member it lists.
 */

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import ts from 'typescript';

/** Inline code spans outside fenced code blocks. */
export function codeSpans(markdown: string): string[] {
  const prose = markdown.replace(/^(```|~~~)[\s\S]*?^\1/gm, '');
  return [...prose.matchAll(/`([^`\n]+)`/g)].map((m) => m[1].trim());
}

const FILE_NAME = /\.(?:md|ts|mts|js|mjs|json|py|rs|css|html|glsl|wgsl|ya?ml|toml|zarr|txt|sh)$/;
const DOTTED_CALL = /^[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:\(\))?$/;
/** `loadX`, `createXLoader`, `makeXxxCtx`, `XxxOrThrow`: a documented PATTERN, not a name. */
const PLACEHOLDER = /(?:^|[a-z])(?:X|Xxx)(?=[A-Z]|$)/;

/** Whether one identifier is code-shaped enough to check. */
export function isCheckable(name: string): boolean {
  if (PLACEHOLDER.test(name)) return false;
  return (
    /^[a-z][a-z0-9]*[A-Z][\w$]*$/.test(name) || // camelCase
    /^[A-Z][a-z0-9]+[A-Z][\w$]*$/.test(name) || // PascalCase, two humps or more
    /^[A-Z][A-Z0-9]*_[A-Z0-9_]+$/.test(name) // UPPER_SNAKE
  );
}

/** The identifiers one code span claims exist (empty when the span is not a symbol). */
export function symbolMentions(span: string): string[] {
  if (FILE_NAME.test(span) || !DOTTED_CALL.test(span)) return [];
  const parts = span.replace(/\(\)$/, '').split('.');
  // A snake_case segment marks a Python / zarr-attr path (`ViewerConfig.from_json()`).
  if (parts.some((p) => /^[a-z][a-z0-9]*_[a-z0-9_]+$/.test(p))) return [];
  return parts.filter(isCheckable);
}

/** Every file under `dir` (recursively), skipping build output. */
export function walkFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'target' || name === 'pkg') continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walkFiles(path, out);
    else out.push(path);
  }
  return out;
}

const WORD = /[A-Za-z_$][\w$]*/g;

function addWords(text: string, into: Set<string>): void {
  for (const m of text.matchAll(WORD)) into.add(m[0]);
}

/**
 * Every name a TS/JS source file uses as an identifier, plus every word inside its
 * string and template literals (the GLSL/WGSL shader sources, counter names and
 * storage keys live there). Comments are NOT read: a name that survives only in a
 * comment is exactly the staleness this check exists to catch.
 */
export function sourceNames(source: string, fileName: string, into: Set<string>): void {
  const sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, false);
  const visit = (node: ts.Node): void => {
    if (ts.isIdentifier(node) || ts.isPrivateIdentifier(node)) {
      into.add(node.text.replace(/^#/, ''));
    } else if (ts.isStringLiteralLike(node) || ts.isTemplateLiteralToken(node)) {
      addWords(node.text, into);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
}

/** The names a plain-text source (Rust, a declaration file) contains. */
export function plainNames(source: string, into: Set<string>): void {
  addWords(source, into);
}

// ---------------------------------------------------------------------------
// Enumerated lists
// ---------------------------------------------------------------------------

/** One `<!-- mirrors: … -->` block. */
export interface MirroredList {
  /** The path as written, resolved against the README's directory. */
  file: string;
  target: string;
  items: string[];
}

/** A marker counts only on a line of its own, so prose QUOTING the syntax is not a block. */
const MIRROR_BLOCK = /^<!--\s*mirrors:\s*(\S+?)#(\S+)\s*-->$([\s\S]*?)^<!--\s*\/mirrors\s*-->$/gm;

/** The mirrored lists a README declares; each list item's first code span is its member. */
export function mirroredLists(markdown: string, readmePath: string): MirroredList[] {
  return [...markdown.matchAll(MIRROR_BLOCK)].map(([, file, target, body]) => ({
    file: resolve(dirname(readmePath), file),
    target,
    items: body
      .split('\n')
      .filter((line) => /^\s*[-*]\s/.test(line))
      .map((line) => (/`([^`]+)`/.exec(line)?.[1] ?? line.trim()).replace(/\(\)$/, '')),
  }));
}

function hasExport(node: ts.Node): boolean {
  return (
    ts.canHaveModifiers(node) &&
    (ts.getModifiers(node) ?? []).some((m) => m.kind === ts.SyntaxKind.ExportKeyword)
  );
}

function valueExports(sf: ts.SourceFile): string[] {
  const names: string[] = [];
  for (const st of sf.statements) {
    if ((ts.isFunctionDeclaration(st) || ts.isClassDeclaration(st)) && st.name && hasExport(st)) {
      names.push(st.name.text);
    } else if (ts.isVariableStatement(st) && hasExport(st)) {
      for (const d of st.declarationList.declarations) {
        if (ts.isIdentifier(d.name)) names.push(d.name.text);
      }
    } else if (ts.isExportDeclaration(st) && !st.isTypeOnly && st.exportClause) {
      if (ts.isNamedExports(st.exportClause)) {
        for (const el of st.exportClause.elements) if (!el.isTypeOnly) names.push(el.name.text);
      }
    }
  }
  return names;
}

function classGetters(sf: ts.SourceFile, className: string): string[] {
  const cls = sf.statements.find(
    (st): st is ts.ClassDeclaration => ts.isClassDeclaration(st) && st.name?.text === className
  );
  if (!cls) throw new Error(`no class ${className} in ${sf.fileName}`);
  return cls.members
    .filter((m): m is ts.GetAccessorDeclaration => ts.isGetAccessorDeclaration(m))
    .filter((m) => !(ts.getModifiers(m) ?? []).some((x) => x.kind === ts.SyntaxKind.PrivateKeyword))
    .map((m) => m.name.getText(sf));
}

/** The code set a mirrors target names: `exports`, or `<Class>.getters`. */
export function mirrorTargetMembers(file: string, target: string): string[] {
  const sf = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true);
  if (target === 'exports') return valueExports(sf);
  const getters = /^(\w+)\.getters$/.exec(target);
  if (getters) return classGetters(sf, getters[1]);
  throw new Error(`unknown mirrors target "#${target}" (use #exports or #<Class>.getters)`);
}
