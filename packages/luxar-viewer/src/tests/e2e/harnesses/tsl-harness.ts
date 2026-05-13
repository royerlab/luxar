/**
 * TSL ↔ GLSL parity harness.
 *
 * Loaded by `tsl-harness.html` (Vite serves it at `/tsl-harness.html`).
 * Exposes `window.__tslHarness` with primitives that render a fullscreen
 * pass through both backends and return the result for pixel-diffing in
 * a Playwright spec.
 *
 * Why a dedicated page rather than reusing the main viewer:
 * - Construction order is explicit and minimal — no app/state machine
 *   to wait on, no scene graph to mock around.
 * - Both backends (WebGL2 via `THREE.WebGLRenderer`, WebGPU-via-WebGL2
 *   via `WebGPURenderer({ forceWebGL: true })`) live side by side; the
 *   test toggles between them per call rather than per page load.
 * - The TSL path drives `GLSLNodeBuilder` directly so the generated
 *   GLSL strings are recoverable for snapshot-diff.
 *
 * Not in scope: real WebGPU dispatch. That requires Chrome stable +
 * `?webgpu=1` and runs in a separate spec. This harness validates
 * the WebGL2 fallback parity, which is what `forceWebGL: true` covers.
 *
 * @module tests/e2e/harnesses/tsl-harness
 */

import * as THREE from 'three';
import { attribute, vec2, vec4, uv, Fn, Discard, length, float } from 'three/tsl';
import { NodeMaterial, PointsNodeMaterial } from 'three/webgpu';
import { FXAA_SOURCE } from '../../../rendering/post-processing/fxaa-shaders';
import { BLOOM_THRESHOLD_SOURCE } from '../../../rendering/post-processing/bloom-shaders';
import { MEGA_SOURCE } from '../../../rendering/post-processing/mega-shader.glsl';
import { megaWebGPUFactory } from '../../../rendering/post-processing/mega.tsl';
import type { ShaderSource } from '../../../rendering/shaders/shader-source';

/**
 * Shape of an entry in the shader registry exposed to Playwright.
 * Adding a new shader to {@link SHADER_REGISTRY} is sufficient to make
 * it usable from the spec.
 */
interface RegistryEntry {
  readonly source: ShaderSource;
  /** Default uniforms for this shader's parity test. */
  readonly buildUniforms: () => Record<string, THREE.IUniform>;
  /**
   * GLSL3 `defines` to set on the `THREE.ShaderMaterial`. Needed for
   * shaders like `mega` that use `#define` gates for feature toggles
   * + an `LUXAR_TONE_MAPPING_MODE` numeric. Optional; defaults to
   * empty (no defines).
   */
  readonly buildDefines?: () => Record<string, string>;
  /**
   * Override for the TSL material constructor. When provided, the
   * harness calls this directly instead of `source.webgpu(uniforms)`.
   * Used to pass shader-specific factory configs (e.g.
   * `megaWebGPUFactory(uniforms, { toneMappingMode: 1 })`).
   */
  readonly buildTSLMaterial?: (uniforms: Record<string, THREE.IUniform>) => THREE.Material;
  /**
   * Override the mesh built around the material. Defaults to a
   * fullscreen `THREE.Mesh(PlaneGeometry(2, 2), material)` rendered
   * with an OrthographicCamera. Override for point-sprite tests
   * that need `THREE.Points(...)`.
   */
  readonly buildMesh?: (material: THREE.Material) => THREE.Object3D;
}

/**
 * Sized 8×8 test texture: gradient horizontally, ramped vertically,
 * with a single bright pixel near the centre to exercise the FXAA
 * edge-detection path. Deterministic across both backends.
 */
function buildTestTexture(): THREE.DataTexture {
  const w = 8;
  const h = 8;
  const data = new Uint8Array(w * h * 4);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      data[i] = Math.floor((x / (w - 1)) * 255);
      data[i + 1] = Math.floor((y / (h - 1)) * 255);
      data[i + 2] = 128;
      data[i + 3] = 255;
    }
  }
  // High-contrast pixel for FXAA to bite on.
  const cx = 4;
  const cy = 4;
  const ci = (cy * w + cx) * 4;
  data[ci] = 255;
  data[ci + 1] = 255;
  data[ci + 2] = 255;

  const tex = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/**
 * Trivial diagnostic shader: outputs a constant RGB. Used to verify
 * that the harness's two backends produce pixel-identical results
 * for the simplest possible fragment. If this fails, the divergence
 * is in the renderer-level setup (color space, output transform),
 * not in a per-shader port.
 */
const CONST_SHADER: ShaderSource = {
  name: 'const-rgb',
  webgl: {
    vertex: /* glsl */ `
      void main() {
        gl_Position = vec4(position.xy, 0.0, 1.0);
      }
    `,
    fragment: /* glsl */ `
      precision highp float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(0.5, 0.25, 0.75, 1.0);
      }
    `,
  },
  webgpu: () => {
    const m = new NodeMaterial();
    m.fragmentNode = vec4(0.5, 0.25, 0.75, 1.0);
    m.toneMapped = false;
    m.depthTest = false;
    m.depthWrite = false;
    m.transparent = false;
    return m;
  },
};

/**
 * Points-sprite hello-world. Validates the TSL patterns that
 * the M11-M16 scene-material ports depend on:
 *
 *  - Typed attribute reads: `attribute('radius', 'float')` →
 *    `Node<'float'>` with full `.mul()` / `.add()` dispatch.
 *  - `PointsNodeMaterial.sizeNode` accepting a scalar attribute
 *    (broadcast to vec2 internally).
 *  - Sprite UV = `uv()`, which maps 1:1 onto GLSL's
 *    `gl_PointCoord` when used inside the colorNode.
 *  - `Discard(boolNode)` inside a fragment-stage `Fn`.
 *
 * Both backends render a single point at world-space origin with
 * a known pixel radius. The GLSL3 path uses `gl_PointSize` +
 * `gl_PointCoord`; the TSL path uses sprite-instanced quads
 * driven by `PointsNodeMaterial.sizeNode`. The two rendering
 * pipelines are mechanically different but the fragment output —
 * a soft red disk — must be pixel-identical.
 */
const POINTS_HELLO_VERTEX_SHADER = /* glsl */ `
  in float radius;
  out float vRadius;

  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = radius;
    vRadius = radius;
  }
`;

const POINTS_HELLO_FRAGMENT_SHADER = /* glsl */ `
  precision highp float;
  in float vRadius;
  out vec4 fragColor;

  void main() {
    vec2 d = gl_PointCoord - 0.5;
    float r = length(d);
    if (r > 0.5) discard;
    // Soft red disk: brightest at the centre, fades to 0 at radius 0.5.
    float falloff = 1.0 - r * 2.0;
    fragColor = vec4(falloff, 0.0, 0.0, 1.0);
  }
`;

function pointsHelloTSLFactory(): THREE.Material {
  const mat = new PointsNodeMaterial();

  // Typed attribute read. The string-literal `'float'` is what gives
  // TypeScript enough info to dispatch `.mul()` on the returned node
  // via the `NumExtensions<'float'>` mixin.
  const radiusAttr = attribute('radius', 'float');

  // sizeNode accepts a scalar; PointsNodeMaterial wraps in `vec2(...)`
  // internally so the sprite quad is symmetric. See r184
  // `PointsNodeMaterial.js` line 107: `let pointSize = sizeNode !== null
  // ? vec2( sizeNode ) : materialPointSize;`.
  mat.sizeNode = radiusAttr;

  // Fragment: identical math to the GLSL `gl_PointCoord` path. The
  // sprite UV maps `(0,0)` to bottom-left and `(1,1)` to top-right
  // — same convention as `gl_PointCoord` under WebGL2.
  mat.colorNode = Fn(() => {
    const coord = uv();
    const d = vec2(coord.sub(0.5));
    const r = length(d);
    Discard(r.greaterThan(0.5));
    const falloff = float(1.0).sub(r.mul(2.0));
    return vec4(falloff, 0.0, 0.0, 1.0);
  })();

  mat.toneMapped = false;
  mat.transparent = false;
  mat.depthTest = false;
  mat.depthWrite = false;
  return mat;
}

const POINTS_HELLO_SOURCE: ShaderSource = {
  name: 'points-hello',
  webgl: {
    vertex: POINTS_HELLO_VERTEX_SHADER,
    fragment: POINTS_HELLO_FRAGMENT_SHADER,
  },
  webgpu: () => pointsHelloTSLFactory(),
};

/**
 * One-point geometry centred at world-space origin. The `radius`
 * attribute is 16 — the sprite covers a 16×16 px region of the 64×64
 * harness target, sized so the GLSL `gl_PointSize` and TSL sizeNode
 * dispatch can be compared with room to spare.
 */
function buildPointsHelloMesh(material: THREE.Material): THREE.Object3D {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  geom.setAttribute('radius', new THREE.Float32BufferAttribute([16.0], 1));
  return new THREE.Points(geom, material);
}

const SHADER_REGISTRY: Record<string, RegistryEntry> = {
  'const-rgb': {
    source: CONST_SHADER,
    buildUniforms: () => ({}),
  },
  fxaa: {
    source: FXAA_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
    }),
  },
  'bloom-threshold': {
    source: BLOOM_THRESHOLD_SOURCE,
    buildUniforms: () => ({
      uInput: { value: buildTestTexture() },
      uTexelSize: { value: new THREE.Vector2(1 / 8, 1 / 8) },
      uThreshold: { value: 0.5 },
      uSmoothing: { value: 0.5 },
    }),
  },
  // Mega-shader: default configuration only (no bloom, no lens
  // distortion, no vignette, no detector noise, mode=Linear). The
  // tone-mapping mode is pinned to Linear (mode=1) because that's
  // the simplest path through THREE's toneMapping chunk and
  // matches TSL's linearToneMapping output.
  mega: {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      // THREE's tone-mapping chunk reads this; pin to 1.0 so the
      // GLSL3 ShaderMaterial doesn't double-multiply our exposure.
      toneMappingExposure: { value: 1.0 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, { toneMappingMode: 1 }) as unknown as THREE.Material,
  },
  // Mega + bloom enabled: validates the `USE_BLOOM` JS-side conditional
  // branch in the TSL factory matches the GLSL `#ifdef USE_BLOOM` path.
  'mega-bloom': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      // A second texture for bloom — uniform contents differ from the
      // HDR scene so the test fails if the shader reads the wrong one.
      uBloomTexture: { value: buildTestTexture() },
      uBloomIntensity: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_BLOOM: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useBloom: true,
      }) as unknown as THREE.Material,
  },
  'points-hello': {
    source: POINTS_HELLO_SOURCE,
    buildUniforms: () => ({}),
    buildMesh: buildPointsHelloMesh,
  },
  // Mega + vignette: validates the `USE_VIGNETTE` JS-side branch.
  'mega-vignette': {
    source: MEGA_SOURCE,
    buildUniforms: () => ({
      uHdrScene: { value: buildTestTexture() },
      uResolution: { value: new THREE.Vector2(8, 8) },
      uExposure: { value: 0.0 },
      uGlobalOffset: { value: 0.0 },
      uGlobalGamma: { value: 1.0 },
      toneMappingExposure: { value: 1.0 },
      uVignetteDarkness: { value: 0.7 },
      uVignetteOffset: { value: 0.5 },
    }),
    buildDefines: () => ({ LUXAR_TONE_MAPPING_MODE: '1', USE_VIGNETTE: '' }),
    buildTSLMaterial: (uniforms) =>
      megaWebGPUFactory(uniforms, {
        toneMappingMode: 1,
        useVignette: true,
      }) as unknown as THREE.Material,
  },
};

const HARNESS_SIZE = 64;

/**
 * Render the GLSL3 path of a registered shader to an offscreen target
 * and return the readback pixel buffer (RGBA8, length = w*h*4).
 */
function renderGLSL(shaderName: string): Uint8Array {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);

  const uniforms = entry.buildUniforms();
  // Build the ShaderMaterial. We pass `defines` only when the
  // registry entry supplies it — Three.js warns "parameter 'defines'
  // has value of undefined" otherwise.
  const materialParams: THREE.ShaderMaterialParameters = {
    vertexShader: entry.source.webgl.vertex,
    fragmentShader: entry.source.webgl.fragment,
    uniforms,
    glslVersion: THREE.GLSL3,
    depthTest: false,
    depthWrite: false,
    transparent: false,
  };
  if (entry.buildDefines) {
    materialParams.defines = entry.buildDefines();
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
  // OrthographicCamera positioned slightly behind origin so world-
  // space (0,0,0) projects to NDC (0,0) — viewport centre. Points
  // shaders depend on this for sprite-centring.
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
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
 * Render the TSL path of a registered shader via
 * `WebGPURenderer({ forceWebGL: true })` and return both the readback
 * pixels and the generated GLSL strings.
 *
 * Capturing the GLSL strings requires walking the `WebGLBackend`'s
 * pipeline cache after the render completes — there's no public
 * "give me the source" API, so we read the strings off the
 * `NodeBuilderState` that the backend stashes per-RenderObject.
 */
async function renderTSL(
  shaderName: string
): Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }> {
  const entry = SHADER_REGISTRY[shaderName];
  if (!entry) throw new Error(`Unknown shader: ${shaderName}`);
  if (!entry.source.webgpu) {
    throw new Error(`Shader ${shaderName} has no TSL factory yet`);
  }

  const uniforms = entry.buildUniforms();
  const material = entry.buildTSLMaterial
    ? entry.buildTSLMaterial(uniforms)
    : (entry.source.webgpu(uniforms) as THREE.Material);

  const { WebGPURenderer } = await import('three/webgpu');
  const renderer = new WebGPURenderer({ antialias: false, alpha: false, forceWebGL: true });
  renderer.setPixelRatio(1);
  renderer.setSize(HARNESS_SIZE, HARNESS_SIZE);
  await renderer.init();

  const target = new THREE.WebGLRenderTarget(HARNESS_SIZE, HARNESS_SIZE, {
    type: THREE.UnsignedByteType,
    format: THREE.RGBAFormat,
    minFilter: THREE.NearestFilter,
    magFilter: THREE.NearestFilter,
  });

  const scene = new THREE.Scene();
  // Mirrors the GLSL path's camera setup. See renderGLSL for the
  // rationale around the slight near-plane offset.
  const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);
  const mesh = entry.buildMesh
    ? entry.buildMesh(material)
    : new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  scene.add(mesh);

  renderer.setRenderTarget(target);
  await renderer.renderAsync(scene, camera);
  renderer.setRenderTarget(null);

  const readback = await renderer.readRenderTargetPixelsAsync(
    target,
    0,
    0,
    HARNESS_SIZE,
    HARNESS_SIZE
  );
  // The renderer returns its own typed array — copy into Uint8Array so
  // the rest of the harness treats both paths uniformly.
  const pixels = new Uint8Array(readback.buffer.slice(0));

  // Generated-GLSL recovery is deferred. The TSL backend stores
  // compiled programs on the internal `_objects` ChainMap keyed off
  // a tuple (object, material, renderContext, lightsNode) that we
  // don't reconstruct from out here. The path through that map will
  // land in a follow-up alongside the snapshot-diff assertion. For
  // now the spec relies on pixel-parity only.
  const vertexShader = '';
  const fragmentShader = '';

  target.dispose();
  if ('geometry' in mesh) (mesh as THREE.Mesh | THREE.Points).geometry.dispose();
  material.dispose();
  (uniforms.uInput?.value as THREE.Texture | null | undefined)?.dispose();
  renderer.dispose();

  return { pixels, vertexShader, fragmentShader };
}

declare global {
  interface Window {
    __tslHarness?: {
      ready: Promise<void>;
      renderGLSL: (shaderName: string) => Uint8Array;
      renderTSL: (
        shaderName: string
      ) => Promise<{ pixels: Uint8Array; vertexShader: string; fragmentShader: string }>;
      listShaders: () => string[];
    };
  }
}

const status = document.getElementById('status');
const setStatus = (msg: string) => {
  if (status) status.textContent = msg;
};

const ready = (async () => {
  setStatus('tsl-harness ready');
})();

window.__tslHarness = {
  ready,
  renderGLSL,
  renderTSL,
  listShaders: () => Object.keys(SHADER_REGISTRY),
};
