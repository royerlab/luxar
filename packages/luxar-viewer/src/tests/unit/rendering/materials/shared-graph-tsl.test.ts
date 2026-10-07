/**
 * Per-object values through a SHARED TSL graph (`shared-graph-tsl.ts`).
 *
 * Materials of one configuration draw with one graph, so a draw can only see
 * its own opacity / colormap / textures through the forwarding leaves: each
 * leaf's object-granularity `updateBefore` (which three runs for every draw,
 * ahead of the draw's binding upload) copies the drawn material's own value.
 * These tests drive that hook exactly as `NodeFrame.updateBeforeNode` does —
 * with a frame whose `material` is the one being drawn.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three';
import { texture } from 'three/tsl';
import { NodeMaterial } from 'three/webgpu';
import { GSplatTSLMaterial } from '../../../../rendering/materials/gsplat/material-tsl';
import {
  getSharedTSLGraph,
  forwardTSLLeaves,
  sharedTSLGraphKey,
  type SharedTSLGraph,
} from '../../../../rendering/materials/_shared/shared-graph-tsl';
import { setDensityAlphaExp } from '../../../../rendering/materials/_shared/density-drop';
import {
  configureRenderObjectEviction,
  invalidateRenderObjectFor,
} from '../../../../data/scene-loader/commit/invalidate-render-object';

type Leaf = { value: unknown; updateBefore: (frame: { material: unknown }) => void };

/** Run every forwarding leaf for one draw of `material`. */
function draw(graph: SharedTSLGraph, material: unknown): Record<string, unknown> {
  const seen: Record<string, unknown> = {};
  for (const [name, leaf] of Object.entries(graph.inputs) as [string, Leaf][]) {
    leaf.updateBefore({ material });
    seen[name] = leaf.value;
  }
  return seen;
}

function graphOf(material: NodeMaterial): SharedTSLGraph {
  const graph = getSharedTSLGraph(material);
  expect(graph).toBeDefined();
  return graph as SharedTSLGraph;
}

function floatTexture(width: number): THREE.DataTexture {
  return new THREE.DataTexture(
    new Float32Array(width * 4),
    width,
    1,
    THREE.RGBAFormat,
    THREE.FloatType
  );
}

function colormap(): THREE.DataTexture {
  return new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1, THREE.RGBAFormat);
}

describe('shared TSL graph forwarding (gsplat)', () => {
  it('two materials share one graph object and each draw sees its own values', () => {
    const a = new GSplatTSLMaterial({ opacity: 0.3, absorption: 0.5 });
    const b = new GSplatTSLMaterial({ opacity: 0.7, absorption: 2 });
    const graph = graphOf(a);
    expect(graphOf(b)).toBe(graph);
    expect(b.colorNode).toBe(a.colorNode);
    expect(b.vertexNode).toBe(a.vertexNode);

    expect(draw(graph, a)).toMatchObject({ uOpacity: 0.3, uAbsorption: 0.5 });
    expect(draw(graph, b)).toMatchObject({ uOpacity: 0.7, uAbsorption: 2 });
    expect(draw(graph, a)).toMatchObject({ uOpacity: 0.3, uAbsorption: 0.5 });
  });

  it('layer edits made AFTER the graph is shared reach the next draw', () => {
    const a = new GSplatTSLMaterial({});
    const b = new GSplatTSLMaterial({});
    const graph = graphOf(a);
    a.updateOpacity(0.25); // also the LOD dissolve's write path
    a.updateAbsorption(3);
    a.updateTruncationRadius(2);
    a.updateLabelStyle(true, 4);
    expect(setDensityAlphaExp(a, 4)).toBe(true);
    const seenA = draw(graph, a);
    expect(seenA).toMatchObject({
      uOpacity: 0.25,
      uAbsorption: 3,
      uTruncate: 2,
      uLabelColorMode: 1,
      uLabelFilterIndex: 4,
      uDensityAlphaExp: 4,
    });
    const seenB = draw(graph, b);
    expect(seenB).toMatchObject({ uOpacity: 1, uAbsorption: 1, uDensityAlphaExp: 1 });
    expect(seenB.uTruncate).not.toBe(2);
  });

  it('forwards each material its own splat and colormap textures and scalar range', () => {
    const texA = floatTexture(4096);
    const texB = floatTexture(4096);
    const cmA = colormap();
    const cmB = colormap();
    const a = new GSplatTSLMaterial({ colormapTexture: cmA, scalarRange: [0, 2] });
    const b = new GSplatTSLMaterial({ colormapTexture: cmB, scalarRange: [1, 5] });
    a.updateSplatTexture(texA);
    b.updateSplatTexture(texB);
    const graph = graphOf(a);
    expect(graphOf(b)).toBe(graph);
    const seenA = draw(graph, a);
    expect(seenA.uSplatTex).toBe(texA);
    expect(seenA.uColormapTex).toBe(cmA);
    const seenB = draw(graph, b);
    expect(seenB.uSplatTex).toBe(texB);
    expect(seenB.uColormapTex).toBe(cmB);
    expect(seenB.uScalarMin).not.toBe(seenA.uScalarMin);
    // A colormap swapped in later is a value too: same graph, new texture.
    const cmC = colormap();
    a.updateColormapTexture(cmC);
    expect(graphOf(a)).toBe(graph);
    expect(draw(graph, a).uColormapTex).toBe(cmC);
  });

  it('a material created after the graph was built forwards its own values', () => {
    const first = new GSplatTSLMaterial({});
    const graph = graphOf(first);
    draw(graph, first);
    const late = new GSplatTSLMaterial({ opacity: 0.42 });
    expect(graphOf(late)).toBe(graph);
    expect(draw(graph, late).uOpacity).toBe(0.42);
  });

  it('does not retain a disposed material texture before or after a draw', () => {
    const firstTexture = floatTexture(3072);
    const first = new GSplatTSLMaterial({});
    first.updateSplatTexture(firstTexture);
    const graph = graphOf(first);
    const standIn = graph.inputs.uSplatTex.value as THREE.DataTexture;
    expect(standIn).not.toBe(firstTexture);
    expect(standIn).toBeInstanceOf(THREE.DataTexture);
    expect(standIn.type).toBe(firstTexture.type);
    expect(standIn.format).toBe(firstTexture.format);
    expect(standIn.image).toMatchObject({ width: 1, height: 1 });
    expect(standIn.image.data).toBeInstanceOf(Float32Array);

    expect(draw(graph, first).uSplatTex).toBe(firstTexture);
    first.dispose();
    firstTexture.dispose();
    expect(graph.inputs.uSplatTex.value).toBe(standIn);

    const secondTexture = floatTexture(3072);
    const second = new GSplatTSLMaterial({});
    second.updateSplatTexture(secondTexture);
    expect(graphOf(second)).toBe(graph);
    expect(draw(graph, second).uSplatTex).toBe(secondTexture);
    second.dispose();
    expect(graph.inputs.uSplatTex.value).toBe(standIn);
  });

  it('releases an undrawn texture and leaves the latest material bound', () => {
    const firstTexture = floatTexture(2048);
    const first = new GSplatTSLMaterial({});
    first.updateSplatTexture(firstTexture);
    const graph = graphOf(first);
    const standIn = graph.inputs.uSplatTex.value;
    first.dispose();
    expect(graph.inputs.uSplatTex.value).toBe(standIn);

    const secondTexture = floatTexture(2048);
    const second = new GSplatTSLMaterial({});
    second.updateSplatTexture(secondTexture);
    expect(draw(graph, second).uSplatTex).toBe(secondTexture);
    const thirdTexture = floatTexture(2048);
    const third = new GSplatTSLMaterial({});
    third.updateSplatTexture(thirdTexture);
    expect(draw(graph, third).uSplatTex).toBe(thirdTexture);
    second.dispose();
    expect(graph.inputs.uSplatTex.value).toBe(thirdTexture);
    third.dispose();
    expect(graph.inputs.uSplatTex.value).toBe(standIn);
  });

  it('keeps a live material registered across WebGPU RenderObject eviction', () => {
    const texA = floatTexture(1024);
    const texB = floatTexture(1024);
    const a = new GSplatTSLMaterial({ opacity: 0.3 });
    const b = new GSplatTSLMaterial({ opacity: 0.7 });
    a.updateSplatTexture(texA);
    b.updateSplatTexture(texB);
    const graph = graphOf(a);
    expect(graphOf(b)).toBe(graph);
    expect(draw(graph, a).uSplatTex).toBe(texA);
    expect(draw(graph, b).uSplatTex).toBe(texB);

    configureRenderObjectEviction(true);
    try {
      invalidateRenderObjectFor(new THREE.Mesh(new THREE.BufferGeometry(), a));
    } finally {
      configureRenderObjectEviction(false);
    }

    const afterEviction = draw(graph, a);
    expect(afterEviction.uSplatTex).toBe(texA);
    expect(afterEviction.uOpacity).toBe(0.3);
    a.dispose();
    expect(graph.inputs.uSplatTex.value).not.toBe(texA);
  });

  it('a drawn material that is not registered leaves the forwarded value alone', () => {
    const a = new GSplatTSLMaterial({ opacity: 0.6 });
    const graph = graphOf(a);
    draw(graph, a);
    expect(draw(graph, new NodeMaterial()).uOpacity).toBe(0.6);
  });

  it('a material that leaves a configuration forwards through its new graph', () => {
    const a = new GSplatTSLMaterial({ opacity: 0.3 });
    const plain = graphOf(a);
    a.updateGamma(2);
    const gamma2 = graphOf(a);
    expect(gamma2).not.toBe(plain);
    a.updateOpacity(0.8);
    expect(draw(gamma2, a).uOpacity).toBe(0.8);
  });

  it('the key separates leaf presence and texture format', () => {
    const f = { uTex: { isTextureNode: true, value: floatTexture(4) } };
    const u = {
      uTex: {
        isTextureNode: true,
        value: new THREE.DataTexture(new Uint8Array(16), 4, 1, THREE.RGBAFormat),
      },
    };
    expect(sharedTSLGraphKey('x', {}, f)).not.toBe(sharedTSLGraphKey('x', {}, u));
    expect(sharedTSLGraphKey('x', {}, f)).not.toBe(sharedTSLGraphKey('x', {}, {}));
    expect(sharedTSLGraphKey('x', {}, f)).not.toBe(sharedTSLGraphKey('y', {}, f));
    expect(sharedTSLGraphKey('x', { a: 1 }, f)).not.toBe(sharedTSLGraphKey('x', { a: 2 }, f));
  });

  it('seeds forwarding textures with sampler state and separates filter and wrap modes', () => {
    const linear = colormap();
    linear.minFilter = THREE.LinearMipmapLinearFilter;
    linear.magFilter = THREE.LinearFilter;
    linear.wrapS = THREE.RepeatWrapping;
    linear.wrapT = THREE.MirroredRepeatWrapping;
    linear.generateMipmaps = true;
    const nearest = colormap();
    nearest.minFilter = THREE.NearestFilter;
    nearest.magFilter = THREE.NearestFilter;

    const linearLeaves = { uTex: texture(linear) };
    const nearestLeaves = { uTex: texture(nearest) };
    const seed = forwardTSLLeaves(linearLeaves).uTex.value as THREE.Texture;
    expect(seed).not.toBe(linear);
    expect(seed.minFilter).toBe(linear.minFilter);
    expect(seed.magFilter).toBe(linear.magFilter);
    expect(seed.wrapS).toBe(linear.wrapS);
    expect(seed.wrapT).toBe(linear.wrapT);
    expect(seed.generateMipmaps).toBe(linear.generateMipmaps);
    expect(sharedTSLGraphKey('filter', {}, linearLeaves)).not.toBe(
      sharedTSLGraphKey('filter', {}, nearestLeaves)
    );
    const magOnly = colormap();
    magOnly.minFilter = linear.minFilter;
    magOnly.magFilter = THREE.NearestFilter;
    expect(sharedTSLGraphKey('filter', {}, linearLeaves)).not.toBe(
      sharedTSLGraphKey('filter', {}, { uTex: texture(magOnly) })
    );
    const wrapped = colormap();
    wrapped.minFilter = nearest.minFilter;
    wrapped.magFilter = nearest.magFilter;
    wrapped.wrapS = THREE.RepeatWrapping;
    expect(sharedTSLGraphKey('filter', {}, nearestLeaves)).not.toBe(
      sharedTSLGraphKey('filter', {}, { uTex: texture(wrapped) })
    );
    wrapped.wrapS = nearest.wrapS;
    wrapped.wrapT = THREE.MirroredRepeatWrapping;
    expect(sharedTSLGraphKey('filter', {}, nearestLeaves)).not.toBe(
      sharedTSLGraphKey('filter', {}, { uTex: texture(wrapped) })
    );

    const depth = new THREE.DepthTexture(1, 1);
    depth.compareFunction = THREE.LessEqualCompare;
    const depthSeed = forwardTSLLeaves({ uTex: texture(depth) }).uTex.value as THREE.DepthTexture;
    expect(depthSeed.compareFunction).toBe(depth.compareFunction);
  });

  it('the key separates texture colour space and depth compare function', () => {
    // Both are copied onto the build's stand-in, so they reach the shader the
    // first material builds: a material differing in either must not share it.
    const srgb = colormap();
    srgb.colorSpace = THREE.SRGBColorSpace;
    const linear = colormap();
    linear.colorSpace = THREE.NoColorSpace;
    expect(sharedTSLGraphKey('cs', {}, { uTex: texture(srgb) })).not.toBe(
      sharedTSLGraphKey('cs', {}, { uTex: texture(linear) })
    );

    const less = new THREE.DepthTexture(1, 1);
    less.compareFunction = THREE.LessEqualCompare;
    const none = new THREE.DepthTexture(1, 1);
    expect(sharedTSLGraphKey('depth', {}, { uTex: texture(less) })).not.toBe(
      sharedTSLGraphKey('depth', {}, { uTex: texture(none) })
    );
  });
});
