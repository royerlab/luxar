/**
 * Keeps the geometry-behaviour matrix (`tests/_conformance/geometry-behaviours.ts`)
 * honest, and catches the per-type drift no single row can.
 *
 * 1. The table: every row is probed exactly once, in the file it names;
 *    every absent cell names a reason and a spec reference that resolves; the
 *    capability rows agree with `GEOMETRY_CAPABILITIES` cell for cell.
 * 2. Literal subsets in tests: an `it.each` / `describe.each` over two or
 *    three geometry names must say why (`// geometry-subset: <reason>`). This
 *    is the "three types, mesh forgotten" shape every review found.
 * 3. Parallel per-type modules: each family's siblings export the same names
 *    (with the type token normalised), or the asymmetry is declared in
 *    `MODULE_FAMILY_ASYMMETRIES` with its reason — which is how "added to
 *    GSplats only" is caught.
 *
 * The scanners are tested on synthetic sources first, so a scanner that
 * stopped finding anything cannot pass vacuously.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';

import { GEOMETRY_CAPABILITIES } from '../../../types/geometry-capabilities';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../../types/format-contract';
import {
  BEHAVIOUR_IDS,
  behaviourRow,
  isAbsent,
  MODULE_FAMILY_ASYMMETRIES,
  type BehaviourId,
} from '../../_conformance/geometry-behaviours';
import {
  exportedNames,
  findGeometrySubsetEaches,
  normaliseTypeToken,
} from '../../_conformance/source-scans';

const SRC = resolve(__dirname, '../../..');
const UNIT = resolve(SRC, 'tests/unit');
const TESTS = resolve(SRC, 'tests');
const REPO = resolve(SRC, '../../..');

/** Every `.ts` file under `dir`, recursively. */
function tsFiles(dir: string, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      if (entry !== 'node_modules') tsFiles(full, acc);
    } else if (entry.endsWith('.ts')) {
      acc.push(full);
    }
  }
  return acc;
}

describe('the geometry-behaviour matrix', () => {
  it('has a row per behaviour and a cell per geometry type', () => {
    expect(BEHAVIOUR_IDS.length).toBeGreaterThanOrEqual(15);
    for (const id of BEHAVIOUR_IDS) {
      expect(Object.keys(behaviourRow(id).cells).sort(), id).toEqual([...GEOMETRY_TYPES].sort());
    }
  });

  it('probes every row exactly once, in the file the row names', () => {
    const callers = tsFiles(UNIT).filter(
      (f) => f !== __filename && readFileSync(f, 'utf8').includes('defineBehaviourConformance(')
    );
    // Fail closed: a scan that found no callers would pass every check below.
    expect(callers.length).toBeGreaterThanOrEqual(10);
    const probedIn = new Map<BehaviourId, string[]>();
    for (const file of callers) {
      const text = readFileSync(file, 'utf8');
      for (const id of BEHAVIOUR_IDS) {
        if (text.includes(`'${id}'`)) {
          probedIn.set(id, [...(probedIn.get(id) ?? []), relative(UNIT, file)]);
        }
      }
    }
    const problems: string[] = [];
    for (const id of BEHAVIOUR_IDS) {
      const files = probedIn.get(id) ?? [];
      const declared = behaviourRow(id).probedIn;
      if (files.length !== 1 || files[0] !== declared) {
        problems.push(`${id}: declared in ${declared}, found in [${files.join(', ')}]`);
      }
    }
    for (const file of callers) {
      const rel = relative(UNIT, file);
      if (!BEHAVIOUR_IDS.some((id) => behaviourRow(id).probedIn === rel)) {
        problems.push(`${rel} calls defineBehaviourConformance but no row names it`);
      }
    }
    expect(problems, problems.join('\n')).toEqual([]);
  });

  it('gives every absent cell a reason and a spec reference that resolves', () => {
    const problems: string[] = [];
    for (const id of BEHAVIOUR_IDS) {
      for (const type of GEOMETRY_TYPES) {
        const cell = behaviourRow(id).cells[type];
        if (!isAbsent(cell)) continue;
        const where = `${id}.${type}`;
        if (cell.absent.trim().length < 20) problems.push(`${where}: reason too short`);
        const [path, section] = cell.spec.split(' §');
        // `src/…` is the viewer's own tree; anything else is repo-relative.
        const file = path.startsWith('src/') ? resolve(SRC, '..', path) : resolve(REPO, path);
        if (!existsSync(file)) {
          problems.push(`${where}: spec file ${path} does not exist`);
          continue;
        }
        if (section) {
          const heading = new RegExp(`^#+ ${section.replace(/\./g, '\\.')}[ .]`, 'm');
          if (!heading.test(readFileSync(file, 'utf8'))) {
            problems.push(`${where}: ${path} has no section §${section}`);
          }
        }
      }
    }
    expect(problems, problems.join('\n')).toEqual([]);
  });

  it('agrees with GEOMETRY_CAPABILITIES on the rows that restate a capability', () => {
    const yes = (id: BehaviourId, t: GeometryTypeName) => !isAbsent(behaviourRow(id).cells[t]);
    for (const t of GEOMETRY_TYPES) {
      const caps = GEOMETRY_CAPABILITIES[t];
      expect(yes('lodLevel', t), `lodLevel.${t}`).toBe(caps.lod);
      expect(yes('partitionPart', t), `partitionPart.${t}`).toBe(caps.partition);
      expect(yes('depthSortRegistration', t), `depthSortRegistration.${t}`).toBe(
        caps.depthSortable
      );
      // The hold is the instanced element-texture ordering; only pooled types have one.
      expect(yes('sortedAppendHold', t), `sortedAppendHold.${t}`).toBe(caps.pooled);
    }
  });
});

describe('geometry-subset scan', () => {
  it('flags a strict subset list, through a const and a .map, and accepts a marked one', () => {
    const src = [
      "it.each(['points', 'lines', 'gsplats'])('a %s', () => {});",
      "const KINDS = [['point', 1], ['line', 2]] as const;",
      'describe.each(KINDS.map((k) => k))("%s", () => {});',
      '// geometry-subset: mesh has no element texture to bind',
      "it.each(['points', 'lines', 'gsplats'])('b %s', () => {});",
      "it.each(['points', 'lines', 'gsplats', 'mesh'])('all %s', () => {});",
      "it.each(['mesh'])('one %s', () => {});",
      "it.each([['quad', 1], ['capsule', 2]])('%s', () => {});",
    ].join('\n');
    const findings = findGeometrySubsetEaches(src);
    expect(findings.map((f) => [f.line, f.types])).toEqual([
      [1, ['points', 'lines', 'gsplats']],
      [3, ['points', 'lines']],
    ]);
  });

  it('every subset list in src/tests says why it leaves a type out', () => {
    const files = tsFiles(TESTS);
    expect(files.length).toBeGreaterThan(500);
    const problems: string[] = [];
    for (const file of files) {
      for (const f of findGeometrySubsetEaches(readFileSync(file, 'utf8'), file)) {
        problems.push(`${relative(SRC, file)}:${f.line} [${f.types.join(', ')}] ${f.text}`);
      }
    }
    expect(
      problems,
      'these *.each lists name some geometry types but not all four. Add the missing ' +
        'type, drive the list from GEOMETRY_TYPES / the behaviour matrix, or say why with ' +
        '`// geometry-subset: <reason>` directly above the call:\n  ' +
        problems.join('\n  ')
    ).toEqual([]);
  });
});

/** The parallel per-type module families, each `<t>` the type's directory/file token. */
const MODULE_FAMILIES: Record<string, (t: GeometryTypeName) => string> = {
  'commit-<t>-geometry.ts': (t) => `data/scene-loader/commit/commit-${t}-geometry.ts`,
  'load-<t>-node.ts': (t) => `data/scene-loader/nodes/load-${t}-node.ts`,
  '<t>-progressive-loader.ts': (t) => `data/${t}/${t}-progressive-loader.ts`,
};

describe('parallel per-type modules export the same names', () => {
  it('normalises the type token in an export name', () => {
    expect(normaliseTypeToken('commitGSplatsGeometry', 'gsplats')).toBe('commit<T>Geometry');
    expect(normaliseTypeToken('loadMeshNodeCheap', 'mesh')).toBe('load<T>NodeCheap');
    expect(exportedNames('export const a = 1;\nexport function b() {}\nconst c = 2;')).toEqual([
      'a',
      'b',
    ]);
  });

  it.each(Object.entries(MODULE_FAMILIES))('%s', (family, pathOf) => {
    const byType = new Map<GeometryTypeName, Set<string>>();
    for (const t of GEOMETRY_TYPES) {
      const source = readFileSync(resolve(SRC, pathOf(t)), 'utf8');
      byType.set(t, new Set(exportedNames(source).map((n) => normaliseTypeToken(n, t))));
    }
    const union = new Set([...byType.values()].flatMap((s) => [...s]));
    expect(union.size).toBeGreaterThan(0);
    const declared = MODULE_FAMILY_ASYMMETRIES[family] ?? {};
    const problems: string[] = [];
    for (const name of union) {
      const presentIn = GEOMETRY_TYPES.filter((t) => byType.get(t)!.has(name));
      if (presentIn.length === GEOMETRY_TYPES.length) continue;
      const entry = declared[name];
      if (!entry) {
        problems.push(`${name} is exported only by [${presentIn.join(', ')}]`);
      } else if (entry.presentIn.join() !== presentIn.join()) {
        problems.push(
          `${name}: declared in [${entry.presentIn.join(', ')}], exported by [${presentIn.join(', ')}]`
        );
      }
    }
    for (const name of Object.keys(declared)) {
      if (!union.has(name)) problems.push(`${name} is declared but no sibling exports it`);
    }
    expect(
      problems,
      `${family}: add the export to the other siblings, or declare the asymmetry in ` +
        'MODULE_FAMILY_ASYMMETRIES (tests/_conformance/geometry-behaviours.ts):\n  ' +
        problems.join('\n  ')
    ).toEqual([]);
  });
});
