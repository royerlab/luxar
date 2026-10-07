/**
 * Draw and pick run the same visibility rules — the source-level half of the
 * draw/pick conformance table (`tests/_conformance/pick-visibility-rules.ts`).
 *
 * For every `(type, primitive variant, rule)` cell:
 *
 * - `same`: the rule's shared helper is called in BOTH the GLSL draw and pick
 *   sources, and its emitted block is in BOTH TSL codegen snapshots (every
 *   draw variant's, and the pick's) — or, where the snapshot fixture does not
 *   exercise the rule, in both TSL factory sources.
 * - `deliberate`: both pick sources carry the evidence of the declared footprint.
 * - `drawOnly`: the draw has it and the pick does not.
 * - `neither`: absent from both, so a rule that starts applying shows up.
 *
 * Then the table is held complete: every `PICKING_FACTORIES` kind and every
 * variant of `GEOMETRY_MATERIAL_VARIANTS` has a row, and every pick shader the
 * picking tree exports (GLSL `*_PICK_VERTEX_SHADER`, TSL `*.tsl.ts`) is one
 * this test reads — a new pick variant with no row fails, which is how the
 * capsule line pick slipped past the density guard. Finally the CPU-side rules
 * hold because the pick node reuses the visual geometry, for every type.
 *
 * The codegen snapshots are pinned to the live TSL output by the
 * `tsl-codegen-snapshot` e2e spec, so matching them is matching what WebGPU runs.
 */
import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { PICKING_FACTORIES } from '../../../../rendering/material-manager/factories';
import {
  GSPLAT_FRAGMENT_SHADER,
  GSPLAT_VERTEX_SHADER,
} from '../../../../rendering/materials/gsplat/shader-glsl';
import {
  LINE_FRAGMENT_SHADER,
  LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl';
import {
  CAPSULE_LINE_FRAGMENT_SHADER,
  CAPSULE_LINE_VERTEX_SHADER,
} from '../../../../rendering/materials/line/shader-glsl-capsule';
import {
  MESH_FRAGMENT_SHADER,
  MESH_VERTEX_SHADER,
} from '../../../../rendering/materials/mesh/shader-glsl';
import {
  POINT_FRAGMENT_SHADER,
  POINT_VERTEX_SHADER,
} from '../../../../rendering/materials/point/shader-glsl';
import { NodeFactory } from '../../../../rendering/node-factory';
import * as gsplatPick from '../../../../rendering/picking/gsplat/shaders';
import * as linePick from '../../../../rendering/picking/line/shaders';
import * as capsulePick from '../../../../rendering/picking/line/shaders-capsule';
import * as meshPick from '../../../../rendering/picking/mesh/shaders';
import * as pointPick from '../../../../rendering/picking/point/shaders';
import type { PickingSystem } from '../../../../rendering/picking/picking-system';
import { GEOMETRY_TYPES, type GeometryTypeName } from '../../../../types/format-contract';
import { LINE_PRIMITIVES } from '../../../../types/line-primitive';
import {
  GEOMETRY_MATERIAL_VARIANTS,
  type PrimitiveVariant,
} from '../../../helpers/geometry-materials';
import {
  PICK_VISIBILITY_RULES,
  SHARED_GEOMETRY_RULES,
  VISIBILITY_RULES,
  type RuleSignature,
  type RuleVerdict,
  type VisibilityRuleId,
} from '../../../_conformance/pick-visibility-rules';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RENDERING = path.resolve(HERE, '../../../../rendering');
const CODEGEN = path.resolve(HERE, '../../../__codegen__');
const read = (rel: string) => readFileSync(path.join(RENDERING, rel), 'utf8');

/** One side (draw or pick) of a variant, in every form the rules are checked in. */
interface ShaderSide {
  /** GLSL vertex + fragment source. */
  glsl: string;
  /** TSL factory sources (relative to `src/rendering`). */
  tslSources: string[];
  /** Codegen snapshot stems (`<stem>.{vertex,fragment}.glsl.txt`). */
  codegen: (stem: string) => boolean;
}

interface VariantShaders {
  draw: ShaderSide;
  pick: ShaderSide;
}

const codegenStems = [
  ...new Set(
    readdirSync(CODEGEN)
      .filter((f) => f.endsWith('.glsl.txt'))
      .map((f) => f.replace(/\.(vertex|fragment)\.glsl\.txt$/, ''))
  ),
];

/** The shaders of every `(type, variant)` the table has a row for. */
const SHADERS: Record<GeometryTypeName, Partial<Record<PrimitiveVariant, VariantShaders>>> = {
  points: {
    quad: {
      draw: {
        glsl: POINT_VERTEX_SHADER + POINT_FRAGMENT_SHADER,
        tslSources: ['materials/point/shader-tsl.ts', 'materials/point/material-tsl.ts'],
        codegen: (s) => /^point(-|$)/.test(s) && !s.startsWith('point-pick'),
      },
      pick: {
        glsl: pointPick.POINT_PICK_VERTEX_SHADER + pointPick.POINT_PICK_FRAGMENT_SHADER,
        tslSources: ['picking/point/pick.tsl.ts', 'picking/point/material-tsl.ts'],
        codegen: (s) => s.startsWith('point-pick'),
      },
    },
  },
  lines: {
    quad: {
      draw: {
        glsl: LINE_VERTEX_SHADER + LINE_FRAGMENT_SHADER,
        tslSources: ['materials/line/shader-tsl.ts', 'materials/line/material-tsl.ts'],
        codegen: (s) =>
          /^line(-|$)/.test(s) && !s.startsWith('line-pick') && !s.startsWith('line-capsule'),
      },
      pick: {
        glsl: linePick.LINE_PICK_VERTEX_SHADER + linePick.LINE_PICK_FRAGMENT_SHADER,
        tslSources: ['picking/line/pick.tsl.ts', 'picking/line/material-tsl.ts'],
        codegen: (s) => s.startsWith('line-pick'),
      },
    },
    capsule: {
      draw: {
        glsl: CAPSULE_LINE_VERTEX_SHADER + CAPSULE_LINE_FRAGMENT_SHADER,
        tslSources: ['materials/line/shader-tsl-capsule.ts', 'materials/line/material-tsl.ts'],
        codegen: (s) => s.startsWith('line-capsule') && !s.startsWith('line-capsule-pick'),
      },
      pick: {
        glsl:
          capsulePick.CAPSULE_LINE_PICK_VERTEX_SHADER +
          capsulePick.CAPSULE_LINE_PICK_FRAGMENT_SHADER,
        tslSources: ['picking/line/pick-capsule.tsl.ts', 'picking/line/material-tsl.ts'],
        codegen: (s) => s.startsWith('line-capsule-pick'),
      },
    },
  },
  gsplats: {
    quad: {
      draw: {
        glsl: GSPLAT_VERTEX_SHADER + GSPLAT_FRAGMENT_SHADER,
        tslSources: ['materials/gsplat/shader-tsl.ts', 'materials/gsplat/material-tsl.ts'],
        codegen: (s) => /^gsplat(-|$)/.test(s) && !s.startsWith('gsplat-pick'),
      },
      pick: {
        glsl: gsplatPick.GSPLAT_PICK_VERTEX_SHADER + gsplatPick.GSPLAT_PICK_FRAGMENT_SHADER,
        tslSources: ['picking/gsplat/pick.tsl.ts', 'picking/gsplat/material-tsl.ts'],
        codegen: (s) => s.startsWith('gsplat-pick'),
      },
    },
  },
  mesh: {
    triangle: {
      draw: {
        glsl: MESH_VERTEX_SHADER + MESH_FRAGMENT_SHADER,
        tslSources: ['materials/mesh/shader-tsl.ts', 'materials/mesh/material-tsl.ts'],
        codegen: (s) => /^mesh(-|$)/.test(s) && !s.startsWith('mesh-pick'),
      },
      pick: {
        glsl: meshPick.MESH_PICK_VERTEX_SHADER + meshPick.MESH_PICK_FRAGMENT_SHADER,
        tslSources: ['picking/mesh/pick.tsl.ts', 'picking/mesh/material-tsl.ts'],
        codegen: (s) => s.startsWith('mesh-pick'),
      },
    },
  },
};

/** Each codegen snapshot of `side`, vertex + fragment joined. */
function codegenOf(side: ShaderSide): [string, string][] {
  const stems = codegenStems.filter(side.codegen);
  return stems.map((stem) => [
    stem,
    ['vertex', 'fragment']
      .map((stage) => readFileSync(path.join(CODEGEN, `${stem}.${stage}.glsl.txt`), 'utf8'))
      .join('\n'),
  ]);
}

/** Where `rule` shows up on `side`: per GLSL source and per TSL form, keyed by a label. */
function presence(rule: VisibilityRuleId, side: ShaderSide): Record<string, boolean> {
  const sig: RuleSignature = VISIBILITY_RULES[rule];
  const out: Record<string, boolean> = { glsl: sig.glsl.test(side.glsl) };
  if ('codegen' in sig.tsl) {
    const { codegen, helper, exemptStems } = sig.tsl;
    for (const [stem, text] of codegenOf(side)) {
      if (!exemptStems?.test(stem)) out[`codegen ${stem}`] = codegen.test(text);
    }
    out['tsl source'] = side.tslSources.some((f) => helper.test(read(f)));
  } else {
    const source = sig.tsl.source;
    out['tsl source'] = side.tslSources.some((f) => source.test(read(f)));
  }
  return out;
}

const allTrue = (p: Record<string, boolean>) => Object.values(p).every(Boolean);
const allFalse = (p: Record<string, boolean>) => Object.values(p).every((v) => !v);
const missing = (p: Record<string, boolean>) =>
  Object.entries(p)
    .filter(([, v]) => !v)
    .map(([k]) => k);
const present = (p: Record<string, boolean>) =>
  Object.entries(p)
    .filter(([, v]) => v)
    .map(([k]) => k);

/** Assert one cell; returns the problem, or null. */
function checkCell(
  rule: VisibilityRuleId,
  verdict: RuleVerdict,
  shaders: VariantShaders
): string | null {
  const draw = presence(rule, shaders.draw);
  const pick = presence(rule, shaders.pick);
  if (verdict === 'same') {
    if (allTrue(draw) && allTrue(pick)) return null;
    return `declared 'same' but draw lacks [${missing(draw)}], pick lacks [${missing(pick)}]`;
  }
  if ('deliberate' in verdict) {
    if (!verdict.pickEvidence.test(shaders.pick.glsl)) {
      return `declared deliberate, but the pick GLSL no longer shows ${verdict.pickEvidence}`;
    }
    const missingTSL = verdict.tslEvidence.filter(
      (evidence) => !shaders.pick.tslSources.some((source) => evidence.test(read(source)))
    );
    return missingTSL.length
      ? `declared deliberate, but the pick TSL no longer shows [${missingTSL}]`
      : null;
  }
  if ('drawOnly' in verdict) {
    if (allTrue(draw) && allFalse(pick)) return null;
    return `declared draw-only but draw lacks [${missing(draw)}], pick has [${present(pick)}]`;
  }
  if (allFalse(draw) && allFalse(pick)) return null;
  return `declared neither, but draw has [${present(draw)}], pick has [${present(pick)}]`;
}

const CELLS = GEOMETRY_TYPES.flatMap((type) =>
  GEOMETRY_MATERIAL_VARIANTS[type].map(({ variant }) => [type, variant] as const)
);

describe('every draw/pick visibility rule holds as declared', () => {
  it.each(CELLS)('%s %s', (type, variant) => {
    const rules = PICK_VISIBILITY_RULES[type][variant];
    const shaders = SHADERS[type][variant];
    expect(rules, `no rule row for ${type} ${variant}`).toBeDefined();
    expect(shaders, `no shader sources listed for ${type} ${variant}`).toBeDefined();
    const problems = (Object.keys(VISIBILITY_RULES) as VisibilityRuleId[]).flatMap((rule) => {
      const problem = checkCell(rule, rules![rule], shaders!);
      return problem ? [`${rule}: ${problem}`] : [];
    });
    expect(problems, problems.join('\n')).toEqual([]);
  });

  it('reads at least one codegen snapshot for every draw and pick side', () => {
    for (const [type, variant] of CELLS) {
      const shaders = SHADERS[type][variant]!;
      expect(codegenOf(shaders.draw).length, `${type} ${variant} draw`).toBeGreaterThan(0);
      expect(codegenOf(shaders.pick).length, `${type} ${variant} pick`).toBeGreaterThan(0);
    }
  });
});

describe('the table covers every pick shader there is', () => {
  it('has a row for every PICKING_FACTORIES kind and every material variant', () => {
    expect(Object.keys(PICKING_FACTORIES)).toHaveLength(GEOMETRY_TYPES.length);
    for (const type of GEOMETRY_TYPES) {
      const variants = GEOMETRY_MATERIAL_VARIANTS[type].map((v) => v.variant).sort();
      expect(Object.keys(PICK_VISIBILITY_RULES[type]).sort(), type).toEqual(variants);
      expect(Object.keys(SHADERS[type]).sort(), type).toEqual(variants);
    }
    // Every line primitive the viewer can build is a variant here.
    expect(GEOMETRY_MATERIAL_VARIANTS.lines).toHaveLength(LINE_PRIMITIVES.length);
  });

  it('reads every GLSL pick shader the picking tree exports', () => {
    const exported = [gsplatPick, linePick, capsulePick, meshPick, pointPick].flatMap((mod) =>
      Object.entries(mod).filter(([name]) => /_PICK_(VERTEX|FRAGMENT)_SHADER$/.test(name))
    );
    const read = CELLS.map(([t, v]) => SHADERS[t][v]!.pick.glsl).join('\n');
    for (const [name, source] of exported) {
      expect(read.includes(source as string), `${name} is not in any row`).toBe(true);
    }
    // And no GLSL pick module is missing from the import list above.
    const modules = readdirSync(path.join(RENDERING, 'picking'), { recursive: true })
      .map(String)
      .filter((f) => /\/shaders(-[a-z]+)?\.ts$/.test(f));
    expect(modules.sort()).toEqual([
      'gsplat/shaders.ts',
      'line/shaders-capsule.ts',
      'line/shaders.ts',
      'mesh/shaders.ts',
      'point/shaders.ts',
    ]);
  });

  it('reads every TSL pick factory the picking tree has', () => {
    const factories = readdirSync(path.join(RENDERING, 'picking'), { recursive: true })
      .map(String)
      .filter((f) => /^(point|line|gsplat|mesh)\/.*\.tsl\.ts$/.test(f));
    const listed = new Set(
      CELLS.flatMap(([t, v]) => SHADERS[t][v]!.pick.tslSources).map((f) =>
        f.replace(/^picking\//, '')
      )
    );
    for (const f of factories) expect(listed.has(f), `${f} is not in any row`).toBe(true);
  });
});

describe.each(SHARED_GEOMETRY_RULES)(
  'the CPU-side rule "%s" lives in the shared geometry',
  (_rule) => {
    // Decided before upload, so it holds for picking exactly when the pick
    // node draws the visual geometry — the one check below, run once per
    // type and listed here under each rule's own name.
    it.each(GEOMETRY_TYPES)('%s: the pick node reuses the visual geometry', (type) => {
      const registered: THREE.Mesh[] = [];
      const factory = new NodeFactory();
      factory.setPickingSystem({
        allocatePickId: () => 1,
        registerNode: (main: THREE.Object3D, pick: THREE.Object3D) => {
          main.userData.pickNode = pick;
          registered.push(pick as THREE.Mesh);
        },
        get registeredNodeCount() {
          return registered.length;
        },
      } as unknown as PickingSystem);
      const node = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial());
      node.name = `/${type}`;
      node.userData = { nodeType: type, attrs: {} };
      const root = new THREE.Group();
      root.add(node);

      factory.registerExistingSceneNodes(root);

      expect(registered).toHaveLength(1);
      expect(registered[0].geometry).toBe(node.geometry);
    });
  }
);
