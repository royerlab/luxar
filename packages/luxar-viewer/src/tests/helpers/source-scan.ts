/**
 * Read the viewer's own TypeScript sources for structural tests (registries
 * that must stay in step with the code they describe).
 *
 * Paths are POSIX-style and relative to `src/`, so a failure message names a
 * file the way a contributor would type it.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

/** Absolute path of the viewer's `src/` directory. */
export const VIEWER_SRC_DIR = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

/** Absolute path of `src/tests/unit/`. */
export const UNIT_TESTS_DIR = join(VIEWER_SRC_DIR, 'tests', 'unit');

function toPosix(path: string): string {
  return path.split(sep).join('/');
}

function walk(dir: string, out: string[]): void {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path, out);
    else if (entry.name.endsWith('.ts') && !entry.name.endsWith('.d.ts')) out.push(path);
  }
}

/**
 * Every production `.ts` file under `src/` (no `src/tests/**`, no `*.test.ts`,
 * no declaration files), relative to `src/`, sorted.
 */
export function productionSourceFiles(): string[] {
  const all: string[] = [];
  walk(VIEWER_SRC_DIR, all);
  const testsDir = join(VIEWER_SRC_DIR, 'tests') + sep;
  return all
    .filter((path) => !path.startsWith(testsDir) && !path.endsWith('.test.ts'))
    .map((path) => toPosix(relative(VIEWER_SRC_DIR, path)))
    .sort();
}

/** Parse a file relative to `src/` (with parent pointers set). */
export function parseSourceFile(srcRelative: string): ts.SourceFile {
  const path = join(VIEWER_SRC_DIR, srcRelative);
  return ts.createSourceFile(path, readFileSync(path, 'utf8'), ts.ScriptTarget.Latest, true);
}

/** Visit every node of `sf` depth-first. */
export function forEachNode(sf: ts.SourceFile, visit: (node: ts.Node) => void): void {
  const step = (node: ts.Node): void => {
    visit(node);
    ts.forEachChild(node, step);
  };
  step(sf);
}

/** 1-based line of `node` in its file. */
export function lineOf(node: ts.Node): number {
  const sf = node.getSourceFile();
  return sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
}
