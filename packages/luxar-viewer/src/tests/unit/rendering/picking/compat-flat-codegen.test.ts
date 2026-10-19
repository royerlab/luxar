/** Compatibility-mode WebGPU rejects flat varyings with default first sampling. */
import * as THREE from 'three';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';
import ts from 'typescript';
import { expect, it } from 'vitest';

import {
  forEachNode,
  lineOf,
  parseSourceFile,
  productionSourceFiles,
} from '../../../helpers/source-scan';
import { LineTSLMaterial } from '../../../../rendering/materials/line/material-tsl';
import { LinePickingTSLMaterial } from '../../../../rendering/picking/line/material-tsl';
import { GSplatPickingTSLMaterial } from '../../../../rendering/picking/gsplat/material-tsl';
import { MeshPickingTSLMaterial } from '../../../../rendering/picking/mesh/material-tsl';

function interpolationViolations(file: ts.SourceFile, relative: string): string[] {
  const violations: string[] = [];
  forEachNode(file, (node) => {
    if (!ts.isCallExpression(node) || !ts.isPropertyAccessExpression(node.expression)) return;
    if (node.expression.name.text !== 'setInterpolation') return;
    const mode = node.arguments[0];
    const sampling = node.arguments[1];
    const site = `${relative}:${lineOf(node)}`;
    if (!mode || !ts.isStringLiteralLike(mode)) {
      violations.push(`${site}: interpolation mode must be a literal`);
    } else if (mode.text === 'flat') {
      const meshFirst = relative === 'rendering/picking/mesh/pick.tsl.ts';
      if (
        node.arguments.length !== 2 ||
        !sampling ||
        !(
          (ts.isStringLiteralLike(sampling) &&
            (sampling.text === 'either' || (meshFirst && sampling.text === 'first'))) ||
          (meshFirst && ts.isConditionalExpression(sampling))
        )
      ) {
        violations.push(`${site}: flat interpolation needs compatibility-safe sampling`);
      }
    }
  });
  return violations;
}

it('every rendering flat interpolation specifies compatibility-safe sampling', () => {
  const violations: string[] = [];
  for (const relative of productionSourceFiles().filter((file) => file.startsWith('rendering/'))) {
    violations.push(...interpolationViolations(parseSourceFile(relative), relative));
  }
  expect(violations).toEqual([]);
});

it.each([
  ["varying(x).setInterpolation('flat')", 'rendering/example.ts', 1],
  ["varying(x).setInterpolation(\n'flat',\n)", 'rendering/example.ts', 1],
  ["const FLAT = 'flat'; varying(x).setInterpolation(FLAT, 'either')", 'rendering/example.ts', 1],
  ["varying(x).setInterpolation('flat', 'first')", 'rendering/example.ts', 1],
  [
    "const order = 'first'; varying(x).setInterpolation('flat', 'either')",
    'rendering/example.ts',
    0,
  ],
  ["varying(x).setInterpolation('flat', 'first')", 'rendering/picking/mesh/pick.tsl.ts', 0],
  [
    "varying(x).setInterpolation('flat', core ? 'first' : 'either')",
    'rendering/picking/mesh/pick.tsl.ts',
    0,
  ],
] as const)('source scan checks interpolation calls in %s', (source, relative, count) => {
  const file = ts.createSourceFile(relative, source, ts.ScriptTarget.Latest, true);
  expect(interpolationViolations(file, relative)).toHaveLength(count);
});

function wgsl(
  material: NodeMaterial,
  compatibilityMode: boolean | null = true,
  omitMode = false
): string {
  const renderer = {
    getRenderTarget: () => null,
    getMRT: () => null,
    depth: true,
    reversedDepthBuffer: false,
    logarithmicDepthBuffer: false,
    contextNode: context(),
    library: { fromMaterial: (m: unknown) => m },
    hasFeature: () => false,
    hasCompatibility: () => false,
    backend: {
      ...(!omitMode && { compatibilityMode }),
      capabilities: { getUniformBufferLimit: () => 65536 },
      utils: { getTextureSampleData: () => ({ samples: 1, primarySamples: 1, isMSAA: false }) },
    },
  };
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute([1, 1, 1, 1], 4));
  geometry.setAttribute('aSortedIndex', new THREE.Float32BufferAttribute([0], 1));
  geometry.setAttribute('aSortedIndexB', new THREE.Float32BufferAttribute([0], 1));
  geometry.setAttribute('aQuadCorner', new THREE.Float32BufferAttribute([0, 0], 2));
  const builder = new WGSLNodeBuilder(
    new THREE.Mesh(geometry, material),
    renderer as unknown as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  ) as unknown as {
    camera: THREE.Camera;
    scene: THREE.Scene;
    material: NodeMaterial;
    vertexShader: string;
    fragmentShader: string;
    build(): void;
  };
  builder.camera = new THREE.PerspectiveCamera();
  builder.scene = new THREE.Scene();
  builder.material = material;
  builder.build();
  return builder.vertexShader;
}

// geometry-subset: Points have no flat TSL pick varyings; this covers every shader with one.
it.each([
  ['line draw', () => new LineTSLMaterial({ primitive: 'screen-space' }), 3],
  ['line capsule draw', () => new LineTSLMaterial({ primitive: 'capsule' }), 3],
  ['line pick', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'screen-space' }), 5],
  ['line capsule pick', () => new LinePickingTSLMaterial({ nodeId: 1, primitive: 'capsule' }), 5],
  ['gsplat pick', () => new GSplatPickingTSLMaterial({ nodeId: 1 }), 1],
  ['mesh pick', () => new MeshPickingTSLMaterial({ nodeId: 1 }), 2],
] as const)('%s has only compatibility-valid flat varyings', (_name, make, count) => {
  const source = wgsl(make());
  expect(source.match(/@interpolate\( flat, either \)/g)).toHaveLength(count);
  expect(source).not.toContain('@interpolate( flat )');
});

it('mesh picking keeps first-corner sampling on core WebGPU', () => {
  const material = new MeshPickingTSLMaterial({ nodeId: 1 });
  expect(wgsl(material, true).match(/@interpolate\( flat, either \)/g)).toHaveLength(2);
  const source = wgsl(material, false);
  expect(source.match(/@interpolate\( flat, first \)/g)).toHaveLength(1);
  expect(source.match(/@interpolate\( flat, either \)/g)).toHaveLength(1);
});

it.each([
  ['null', null, false],
  ['undefined', true, true],
] as const)(
  'mesh picking uses either-corner sampling for unknown mode %s',
  (_name, mode, omitMode) => {
    const source = wgsl(new MeshPickingTSLMaterial({ nodeId: 1 }), mode, omitMode);
    expect(source.match(/@interpolate\( flat, either \)/g)).toHaveLength(2);
    expect(source).not.toContain('@interpolate( flat, first )');
  }
);
