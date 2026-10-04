/**
 * README claims are checked against the code (G7).
 *
 * - Every code-shaped symbol a `src/**\/README.md` names in an inline code span
 *   must still be a name the source uses — so renaming or deleting a helper fails
 *   the PR that leaves its README behind. What counts as code-shaped, and what is
 *   deliberately not read, is in `src/tests/helpers/readme-claims.ts`.
 * - A list fenced by `<!-- mirrors: <path>#<target> -->` must name exactly that
 *   code set, so "the eight factory helpers" cannot quietly become nine.
 *
 * The corpus a name may come from: the viewer's own sources (`src/`, `scripts/`,
 * `tools/`), plus the APIs it documents against — three (types AND sources: the
 * READMEs explain renderer internals such as `RenderObjects`), the DOM and the
 * Playwright test API. Anything else must sit in {@link NOT_IN_CODE} with the reason
 * the mention is right anyway.
 */

import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import {
  codeSpans,
  isCheckable,
  mirroredLists,
  mirrorTargetMembers,
  plainNames,
  sourceNames,
  symbolMentions,
  walkFiles,
} from '../../helpers/readme-claims';

const VIEWER = join(dirname(fileURLToPath(import.meta.url)), '../../../..');
const SRC = join(VIEWER, 'src');

/**
 * Mentions that are right although the corpus lacks the name: it belongs to the
 * Python writer, the native launcher or Node — or the README names it precisely to
 * say it is GONE. Keep each reason; prefer fixing a README over adding a name here.
 */
const NOT_IN_CODE: Readonly<Record<string, string>> = {
  AudioConfig: 'Python: luxar.config dataclass behind viewer_config.audio',
  ArrayEncoder: 'Python: the encoder the decoder must mirror',
  COMPOSITING_ATTRS: 'Python: luxar/core/group/compositing.py',
  DELTA_PROBE_MIN_GAIN: 'Python: the delta-codec encode-time probe threshold',
  LUXAR_CACHE_BUDGET_MB: 'Go launcher environment variable',
  ERR_VM_DYNAMIC_IMPORT_CALLBACK_MISSING: 'Node error code',
  materialCacheMaxSize: 'named as a REMOVED config knob',
  projectLinesTo3DWASM: 'named as the DELETED main-thread lines projection',
  initLinesWASM: 'named as the DELETED main-thread lines projection',
  LUXAR_PEAK_PROJECTION: 'named as a define that went with the deleted primitive',
  effectiveGeometryMode: 'named as the REMOVED additive-state fallback helper',
  addToScene: 'named as a method SceneManager deliberately does not have',
  setDimensions: 'named as an API the dimension sliders deliberately do not have',
};

/** A package's root, resolved the way `via` (a chain of dependents) would import it. */
function packageDir(name: string, ...via: string[]): string {
  let resolver = createRequire(join(VIEWER, 'package.json'));
  for (const dependent of via)
    resolver = createRequire(resolver.resolve(`${dependent}/package.json`));
  return dirname(resolver.resolve(`${name}/package.json`));
}

function buildCorpus(): Set<string> {
  const names = new Set<string>();
  const sources = ['src', 'scripts', 'tools'].flatMap((dir) => walkFiles(join(VIEWER, dir)));
  for (const file of sources) {
    // This check's own files name every allowlisted symbol; they prove nothing.
    if (/readme-claims\.(?:test\.)?ts$/.test(file)) continue;
    if (/\.(?:ts|mts|mjs|js)$/.test(file)) sourceNames(readFileSync(file, 'utf8'), file, names);
    else if (/\.(?:rs|glsl|wgsl)$/.test(file)) plainNames(readFileSync(file, 'utf8'), names);
  }
  // three's `exports` map hides package.json; it is a direct dependency, so link it.
  const three = join(VIEWER, 'node_modules/three');
  for (const file of walkFiles(join(three, 'src'))) {
    if (file.endsWith('.js')) plainNames(readFileSync(file, 'utf8'), names);
  }
  const external = [
    ...walkFiles(packageDir('@types/three')),
    join(packageDir('typescript'), 'lib/lib.dom.d.ts'),
    ...walkFiles(join(packageDir('playwright', '@playwright/test'), 'types')),
    ...walkFiles(join(packageDir('playwright-core', '@playwright/test', 'playwright'), 'types')),
  ];
  for (const file of external) {
    if (file.endsWith('.d.ts')) plainNames(readFileSync(file, 'utf8'), names);
  }
  return names;
}

const READMES = walkFiles(SRC).filter((file) => file.endsWith('README.md'));

describe('README symbol references', () => {
  it('every code-shaped name a README mentions is still in the code', () => {
    const corpus = buildCorpus();
    const stale: string[] = [];
    for (const readme of READMES) {
      for (const span of codeSpans(readFileSync(readme, 'utf8'))) {
        for (const name of symbolMentions(span)) {
          if (!corpus.has(name) && !(name in NOT_IN_CODE)) {
            stale.push(`${relative(VIEWER, readme)}: \`${span}\``);
          }
        }
      }
    }
    expect(stale, 'README names no source declares (renamed or removed?)').toEqual([]);
  });

  it('every NOT_IN_CODE entry is still mentioned and still absent from the corpus', () => {
    const corpus = buildCorpus();
    const mentioned = new Set(
      READMES.flatMap((readme) => codeSpans(readFileSync(readme, 'utf8')).flatMap(symbolMentions))
    );
    expect(Object.keys(NOT_IN_CODE).filter((n) => !mentioned.has(n) || corpus.has(n))).toEqual([]);
  });
});

describe('README enumerated lists (<!-- mirrors: -->)', () => {
  const lists = READMES.flatMap((readme) =>
    mirroredLists(readFileSync(readme, 'utf8'), readme).map((list) => ({ readme, ...list }))
  );

  it('the loaders README mirrors its factory helpers and registry accessors', () => {
    const loaders = lists.filter((l) => l.readme.endsWith('scene-loader/loaders/README.md'));
    expect(loaders.map((l) => l.target).sort()).toEqual(['LoaderRegistry.getters', 'exports']);
  });

  it.each(lists.map((l) => [`${relative(VIEWER, l.readme)} → ${l.target}`, l] as const))(
    '%s lists exactly the code set',
    (_title, list) => {
      expect([...list.items].sort()).toEqual(mirrorTargetMembers(list.file, list.target).sort());
    }
  );
});

describe('readme-claims extractors', () => {
  it('reads inline spans only, outside fenced blocks', () => {
    const md = 'Use `fooBar` here.\n```ts\nconst `notThis` = 1;\n```\nand `BazQux()`.';
    expect(codeSpans(md)).toEqual(['fooBar', 'BazQux()']);
  });

  it('checks camelCase, multi-hump PascalCase and UPPER_SNAKE, nothing else', () => {
    expect(['fooBar', 'FooBar', 'FOO_BAR'].map(isCheckable)).toEqual([true, true, true]);
    expect(['points', 'Points', 'foo_bar', 'FOO', 'loadX', 'XxxOrThrow'].map(isCheckable)).toEqual([
      false,
      false,
      false,
      false,
      false,
      false,
    ]);
  });

  it('splits dotted and called symbols; skips paths, files, prose and Python', () => {
    expect(symbolMentions('LayerRegistry.registerPointsLoader()')).toEqual([
      'LayerRegistry',
      'registerPointsLoader',
    ]);
    expect(symbolMentions('layer.blendingMode')).toEqual(['blendingMode']);
    expect(symbolMentions('layer-state.ts')).toEqual([]);
    expect(symbolMentions('a + bC')).toEqual([]);
    expect(symbolMentions('ViewerConfig.from_json()')).toEqual([]);
  });

  it('collects identifiers and literal words but not comments', () => {
    const names = new Set<string>();
    sourceNames(
      "// staleHelper\nexport function liveHelper() { return `uniform vec3 shaderName;` + 'keyName'; }",
      'x.ts',
      names
    );
    expect(names.has('liveHelper')).toBe(true);
    expect(names.has('shaderName')).toBe(true);
    expect(names.has('keyName')).toBe(true);
    expect(names.has('staleHelper')).toBe(false);
  });

  it('parses a mirrors block into its target and listed members', () => {
    const md = [
      '<!-- mirrors: ./a.ts#exports -->',
      '- `alpha` — first',
      '- `beta()` — second',
      '<!-- /mirrors -->',
    ].join('\n');
    const [list] = mirroredLists(md, '/x/README.md');
    expect(list).toEqual({ file: '/x/a.ts', target: 'exports', items: ['alpha', 'beta'] });
    // Prose that quotes the syntax inline is not a block.
    expect(
      mirroredLists('Use `<!-- mirrors: a.ts#exports -->` … `<!-- /mirrors -->`.', '/x')
    ).toEqual([]);
  });
});
