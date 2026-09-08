/**
 * Render executors for the TSL ↔ GLSL parity harness. The single-pass
 * helpers compare registered materials and capture generated shaders;
 * the bloom helpers render the production multi-pass pyramid so blend
 * state and destination preservation are observable at frame level.
 *
 * @module tests/e2e/harnesses/tsl-harness/render
 */

import * as THREE from 'three';
import { requireWebGLSources } from '../../../../rendering/materials/_shared/shader-source';
import {
  elementTextureWidthDefines,
  LINE_TEXTURE_LAYOUT,
  POINT_TEXTURE_LAYOUT,
  SPLAT_TEXTURE_LAYOUT,
} from '../../../../rendering/element-texture-layout';
import { buildDefaultCamera } from './shared';
import { BloomChain } from '../../../../rendering/post-processing/bloom/chain';
import { FullscreenPass } from '../../../../rendering/post-processing/fullscreen/pass';
import type { Renderer, RendererCapabilities } from '../../../../rendering/renderer-capabilities';
// Intentional module cycle: `index.ts` assembles SHADER_REGISTRY from the
// family modules and re-exports these executors. The registry binding is
// only dereferenced inside the function bodies (long after the module
// graph has evaluated), so the cycle is benign under ESM live bindings.
import { SHADER_REGISTRY } from './index';
import { loadTslMaterials } from '../../../../rendering/tsl/load';

/** Edge length (px) of the square offscreen render target the parity harness draws into. */
export const HARNESS_SIZE = 64;

export interface BloomChainRenderResult {
  pixels: Uint8Array;
  mipCount: number;
}

function copyReadback(readback: ArrayBufferView): Uint8Array {
  return new Uint8Array(readback.buffer, readback.byteOffset, readback.byteLength).slice();
}

function flipRowsInPlace(pixels: Uint8Array, width: number, height: number): void {
  const rowBytes = width * 4;
  const temporaryRow = new Uint8Array(rowBytes);
  for (let y = 0; y < height >> 1; y++) {
    const top = y * rowBytes;
    const bottom = (height - 1 - y) * rowBytes;
    temporaryRow.set(pixels.subarray(top, top + rowBytes));
    pixels.copyWithin(top, bottom, bottom + rowBytes);
    pixels.set(temporaryRow, bottom);
  }
}

function buildBloomChainTexture(): THREE.DataTexture {
  // A vertical ramp makes row-order and V-range errors visible. The
  // composed parity case is currently fixme while #2584 establishes
  // whether its DataTexture input should emulate a render-target source.
  const data = new Uint8Array(HARNESS_SIZE * HARNESS_SIZE * 4);
  for (let y = 0; y < HARNESS_SIZE; y++) {
    for (let x = 0; x < HARNESS_SIZE; x++) {
      const offset = (y * HARNESS_SIZE + x) * 4;
      const value = 40 + 3 * y;
      data[offset] = value;
      data[offset + 1] = Math.round(value * 0.75);
      data[offset + 2] = Math.round(value * 0.5);
      data[offset + 3] = 255;
    }
  }
  const texture = new THREE.DataTexture(
    data,
    HARNESS_SIZE,
    HARNESS_SIZE,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.wrapS = THREE.ClampToEdgeWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.needsUpdate = true;
  return texture;
}

async function renderBloomChain(
  renderer: Renderer,
  caps: RendererCapabilities
): Promise<BloomChainRenderResult> {
  const input = buildBloomChainTexture();
  const chain = new BloomChain({
    width: HARNESS_SIZE,
    height: HARNESS_SIZE,
    levels: 3,
    threshold: 0.25,
    smoothing: 0.15,
    radius: 1,
    caps,
  });
  const outputSize = chain.outputSize;
  const copyMaterial = new THREE.MeshBasicMaterial({
    map: chain.outputTexture,
    depthTest: false,
    depthWrite: false,
    toneMapped: false,
  });
  const copyPass = new FullscreenPass(copyMaterial, caps);
  const target = new THREE.WebGLRenderTarget(outputSize.width, outputSize.height, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
    depthBuffer: false,
  });

  chain.render(renderer, input);
  renderer.setRenderTarget(target);
  copyPass.render(renderer);
  renderer.setRenderTarget(null);

  let pixels: Uint8Array;
  if (caps.apiSurface === 'webgl2') {
    pixels = new Uint8Array(outputSize.width * outputSize.height * 4);
    (renderer as THREE.WebGLRenderer).readRenderTargetPixels(
      target,
      0,
      0,
      outputSize.width,
      outputSize.height,
      pixels
    );
  } else {
    const readback = await (
      renderer as import('three/webgpu').WebGPURenderer
    ).readRenderTargetPixelsAsync(target, 0, 0, outputSize.width, outputSize.height);
    pixels = copyReadback(readback);
    // At levels=1 the composed WebGPU readback is the exact row reversal of
    // WebGL (0.0000 against its mirror), so normalize before comparison.
    // renderTSL's force-WebGL path needs no flip; #2584 tracks why forcing
    // framebufferYDown alone does not remove the reversal here.
    flipRowsInPlace(pixels, outputSize.width, outputSize.height);
  }
  const mipCount = chain.mipCount;

  target.dispose();
  copyPass.dispose();
  copyMaterial.dispose();
  chain.dispose();
  input.dispose();
  return { pixels, mipCount };
}

/** Render the production bloom pyramid through WebGL ShaderMaterials. */
export async function renderBloomChainGLSL(): Promise<BloomChainRenderResult> {
  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);
  const caps = {
    apiSurface: 'webgl2',
    framebufferYDown: false,
  } as RendererCapabilities;
  const result = await renderBloomChain(renderer, caps);
  renderer.dispose();
  return result;
}

/** Render the production bloom pyramid through NodeMaterials on WebGL fallback. */
export async function renderBloomChainTSL(): Promise<BloomChainRenderResult> {
  await loadTslMaterials();
  const { WebGPURenderer } = await import('three/webgpu');
  const renderer = new WebGPURenderer({ antialias: false, alpha: false, forceWebGL: true });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);
  await renderer.init();
  const caps = {
    apiSurface: 'webgpu',
    framebufferYDown: true,
  } as RendererCapabilities;
  const result = await renderBloomChain(renderer, caps);
  renderer.dispose();
  return result;
}

/**
 * Render the GLSL3 path of a registered shader to an offscreen target
 * and return the readback pixel buffer (RGBA8, length = w*h*4).
 */
export function renderGLSL(shaderName: string): Uint8Array {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);

  // GLSL parity render — the harness can only check parity when the
  // registry entry ships a GLSL fallback. `requireWebGLSources`
  // throws with a clear diagnostic if it doesn't (shouldn't happen
  // for any shader currently in the registry).
  const glsl = requireWebGLSources(entry.source);
  const uniforms = entry.buildUniforms();
  if (!uniforms.uPixelRatio) uniforms.uPixelRatio = { value: 1 };
  // Build the ShaderMaterial. `defines` is always an object (never
  // undefined — Three.js warns "parameter 'defines' has value of
  // undefined"); it is assembled below from the element-texture width
  // defines plus whatever the registry entry supplies.
  const materialParams: THREE.ShaderMaterialParameters = {
    vertexShader: glsl.vertex,
    fragmentShader: glsl.fragment,
    uniforms,
    glslVersion: THREE.GLSL3,
    // Depth stays off for the single-primitive parity cases; entries
    // whose scenario needs fragments to COMPETE through the depth test
    // (surface-pick depth) opt in via `depthCompete`.
    depthTest: entry.depthCompete ?? false,
    depthWrite: entry.depthCompete ?? false,
    transparent: false,
  };
  // Element-texture width defines: production materials stamp their
  // own layout's define at construction and re-stamp it at texture
  // bind; this harness compiles the raw sources, so inject all three
  // unconditionally (an unused define is inert). When the fixture
  // supplies its own element texture, its width is the authority —
  // the multirow/sorted-permuted sentinels deliberately use tiny
  // custom-width textures to pin the row-stride addressing.
  const widthDefines = elementTextureWidthDefines();
  for (const [uniformName, layout] of [
    ['uLineTex', LINE_TEXTURE_LAYOUT],
    ['uPointTex', POINT_TEXTURE_LAYOUT],
    ['uSplatTex', SPLAT_TEXTURE_LAYOUT],
  ] as const) {
    const tex = uniforms[uniformName]?.value as { image?: { width?: number } } | undefined;
    const width = tex?.image?.width;
    if (typeof width === 'number' && width > 0) {
      widthDefines[layout.widthDefine] = String(width);
    }
  }
  materialParams.defines = {
    ...widthDefines,
    ...(entry.buildDefines ? entry.buildDefines() : {}),
  };
  if (entry.vertexColors) {
    materialParams.vertexColors = true;
  }
  const material = new THREE.ShaderMaterial(materialParams);

  const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);

  const target = new THREE.WebGLRenderTarget(HARNESS_SIZE, HARNESS_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });

  const scene = new THREE.Scene();
  // Default is an OrthographicCamera at (0,0,1) so world (0,0,0) projects to
  // NDC (0,0) — viewport centre (point shaders depend on this for centring).
  // Cases override via `buildCamera` (e.g. the perspective behind-camera guard).
  const camera = entry.buildCamera ? entry.buildCamera() : buildDefaultCamera();
  const mesh = entry.buildMesh
    ? entry.buildMesh(material)
    : new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(mesh);

  renderer.setRenderTarget(target);
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  const pixels = new Uint8Array(HARNESS_SIZE * HARNESS_SIZE * 4);
  renderer.readRenderTargetPixels(target, 0, 0, HARNESS_SIZE, HARNESS_SIZE, pixels);

  target.dispose();
  if ('geometry' in mesh) (mesh as THREE.Mesh | THREE.Points).geometry.dispose();
  material.dispose();
  (uniforms.uInput?.value as THREE.Texture | null | undefined)?.dispose();
  renderer.dispose();

  return pixels;
}

/**
 * Render the TSL path of a registered shader via `WebGPURenderer`
 * (WebGL backend by default; real WebGPU with `{ native: true }`) and
 * return both the readback pixels and the generated shader strings.
 *
 * Capturing the GLSL strings requires walking the `WebGLBackend`'s
 * pipeline cache after the render completes — there's no public
 * "give me the source" API, so we read the strings off the
 * `NodeBuilderState` that the backend stashes per-RenderObject.
 */
export async function renderTSL(
  shaderName: string,
  opts: { native?: boolean } = {}
): Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }> {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);
  if (!entry.source.webgpu) {
    throw new Error(`Shader ${shaderName} has no TSL factory yet`);
  }

  // The `ShaderSource.webgpu` closures pull their TSL factory from the lazy
  // registry (`rendering/tsl/load.ts`) rather than importing it, so the WebGL
  // bundle stays free of `three/webgpu` (issue #1679). The harness is a WebGPU
  // consumer by definition, so it loads the registry up front — one await here
  // covers every shader the harness can be asked to build.
  await loadTslMaterials();

  const uniforms = entry.buildUniforms();
  if (!uniforms.uPixelRatio) uniforms.uPixelRatio = { value: 1 };
  const material = entry.buildTSLMaterial
    ? entry.buildTSLMaterial(uniforms)
    : (entry.source.webgpu(uniforms) as THREE.Material);

  // `native: true` runs the graph on REAL WebGPU (WGSL codegen + Dawn/Metal
  // execution) instead of the WebGL backend — the pixel-equivalence probe
  // for browsers with a working adapter (system Chrome). The mode must
  // FAIL CLOSED: `navigator.gpu` exists even in Playwright's bundled
  // Chromium with WebGPU off (its requestAdapter() just yields nothing),
  // and WebGPURenderer does NOT reject there — its `getFallback` swaps in
  // the WebGL backend with only a console warning, which would silently
  // hand back a WebGL image with a spurious row flip on top. So the real
  // guard is the backend-identity assertion AFTER init() below (#1449);
  // the navigator.gpu check just gives a clearer error where the API is
  // absent outright. The raw native readback is Y-FLIPPED relative to the
  // WebGL paths', so the flip is applied before returning — both modes
  // hand back the same bottom-up row convention and compare directly
  // against renderGLSL (verified 2026-08 on Apple Metal 3, where every
  // line-primitive fixture matched its GLSL render EXACTLY — mean-covered
  // diff 0.000).
  if (opts.native && !('gpu' in navigator)) {
    throw new Error('native WebGPU requested but navigator.gpu is unavailable');
  }
  const { WebGPURenderer } = await import('three/webgpu');
  const renderer = new WebGPURenderer({
    antialias: false,
    alpha: false,
    forceWebGL: !opts.native,
  });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);
  await renderer.init();
  if (opts.native) {
    const backend = (renderer as unknown as { backend?: { isWebGPUBackend?: boolean } }).backend;
    if (!backend?.isWebGPUBackend) {
      renderer.dispose();
      throw new Error(
        'native WebGPU requested but the renderer fell back to WebGL (no usable adapter)'
      );
    }
  }

  // Capture the generated GLSL / WGSL strings by patching the renderer's
  // NodeManager. `_createNodeBuilderState(nodeBuilder)` is called by
  // both the sync and async build paths and receives a builder whose
  // `.vertexShader` / `.fragmentShader` strings are fully populated.
  // We hook it once per renderer instance and restore right after.
  // (See node_modules/three/src/renderers/common/nodes/NodeManager.js:469.)

  const nodesInstance = (renderer as unknown as { _nodes: any })._nodes;

  const origCreateState = nodesInstance._createNodeBuilderState.bind(nodesInstance);
  // Ref object so TS doesn't narrow `value` to `null` through the
  // closure mutation below.
  const capturedRef: { value: { vertex: string; fragment: string } | null } = {
    value: null,
  };

  nodesInstance._createNodeBuilderState = function (nodeBuilder: any) {
    if (!capturedRef.value && nodeBuilder?.material === material) {
      capturedRef.value = {
        vertex: typeof nodeBuilder.vertexShader === 'string' ? nodeBuilder.vertexShader : '',
        fragment: typeof nodeBuilder.fragmentShader === 'string' ? nodeBuilder.fragmentShader : '',
      };
    }
    return origCreateState(nodeBuilder);
  };

  const target = new THREE.WebGLRenderTarget(HARNESS_SIZE, HARNESS_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });

  const scene = new THREE.Scene();
  // Mirrors the GLSL path's camera setup (default ortho, or the entry's
  // `buildCamera` override). See renderGLSL for the rationale.
  const camera = entry.buildCamera ? entry.buildCamera() : buildDefaultCamera();
  const mesh = entry.buildMesh
    ? entry.buildMesh(material)
    : new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(mesh);

  renderer.setRenderTarget(target);
  // `renderAsync()` is deprecated in r184 — `init()` is already awaited
  // at renderer creation above, so plain render() is the supported form.
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);

  // Restore the original NodeManager method now that the capture
  // window is over.
  nodesInstance._createNodeBuilderState = origCreateState;

  const readback = await renderer.readRenderTargetPixelsAsync(
    target,
    0,
    0,
    HARNESS_SIZE,
    HARNESS_SIZE
  );
  // The renderer returns its own typed array — copy into Uint8Array so
  // the rest of the harness treats both paths uniformly.
  const pixels = copyReadback(readback);
  if (opts.native) {
    // Normalize the native readback to the WebGL bottom-up row order
    // (see the note above `renderer` construction).
    flipRowsInPlace(pixels, HARNESS_SIZE, HARNESS_SIZE);
  }

  const vertexShader = capturedRef.value?.vertex ?? '';
  const fragmentShader = capturedRef.value?.fragment ?? '';

  target.dispose();
  if ('geometry' in mesh) (mesh as THREE.Mesh | THREE.Points).geometry.dispose();
  material.dispose();
  (uniforms.uInput?.value as THREE.Texture | null | undefined)?.dispose();
  renderer.dispose();

  return { pixels, vertexShader, fragmentShader };
}
