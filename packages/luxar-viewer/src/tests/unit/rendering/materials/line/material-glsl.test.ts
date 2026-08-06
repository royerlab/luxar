import { describe, it, expect, vi } from 'vitest';
import * as THREE from 'three';
import { LineMaterial } from '../../../../../rendering/materials/line/material-glsl';
import {
  createLineQuadGeometry,
  createInstancedLinesMesh,
  getLineTexture,
} from '../../../../../rendering/line-geometry';
import { LINE_FLOATS_PER_SEGMENT } from '../../../../../rendering/element-texture-layout';
import {
  GLSL_LINE_JOINT_CODE,
  GLSL_LINE_JOIN,
} from '../../../../../rendering/materials/_shared/glsl-lib';
import {
  LINE_PICK_VERTEX_SHADER,
  LINE_PICK_FRAGMENT_SHADER,
} from '../../../../../rendering/picking/line/shaders';

// Mock THREE.ShaderMaterial
vi.mock('three', async () => {
  const actual = await vi.importActual<typeof import('three')>('three');

  const ShaderMaterial = vi.fn(function (this: any, params: any) {
    Object.assign(this, {
      uniforms: params.uniforms,
      vertexShader: params.vertexShader,
      fragmentShader: params.fragmentShader,
      transparent: params.transparent,
      depthWrite: params.depthWrite,
      toneMapped: params.toneMapped,
      blending: params.blending,
      side: params.side,
      userData: {},
      dispose: vi.fn(),
    });
  });

  return {
    ...actual,
    ShaderMaterial: ShaderMaterial as any,
    Vector2: actual.Vector2,
    AdditiveBlending: 'AdditiveBlending',
    NormalBlending: 'NormalBlending',
    CustomBlending: 'CustomBlending',
    AddEquation: 'AddEquation',
    OneFactor: 'OneFactor',
    DoubleSide: 2,
  };
});

describe('LineMaterial', () => {
  describe('constructor', () => {
    it('should create a material with default values', () => {
      const material = new LineMaterial();

      expect(material.uniforms.uResolution.value).toBeInstanceOf(THREE.Vector2);
      expect(material.uniforms.uOpacity.value).toBe(1.0);

      expect(material.transparent).toBe(true);
      expect(material.depthWrite).toBe(false); // Additive blending default
      expect(material.toneMapped).toBe(false);
      // Default 'additive' mode uses classic THREE.AdditiveBlending (SrcAlpha, One)
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.side).toBe(2); // DoubleSide
    });

    it('should default invGamma to 1.0', () => {
      const material = new LineMaterial();

      expect(material.uniforms.uInvGamma.value).toBe(1.0);
      expect(material.userData.gamma).toBe(1.0);
    });

    it('should accept custom gamma', () => {
      const material = new LineMaterial({ gamma: 2.2 });

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.userData.gamma).toBe(2.2);
    });

    it('should accept custom configuration', () => {
      const material = new LineMaterial({
        opacity: 0.5,
        blendingMode: 'normal',
      });

      expect(material.uniforms.uOpacity.value).toBe(0.5);
      expect(material.blending).toBe('NormalBlending');
      // depthWrite is only true for normal blending when opacity >= 0.99
      expect(material.depthWrite).toBe(false); // opacity 0.5 < 0.99, so no depth write
    });

    it('should create additive material without depth write', () => {
      const material = new LineMaterial({
        blendingMode: 'additive',
      });

      // 'additive' uses classic THREE.AdditiveBlending (SrcAlpha, One)
      expect(material.blending).toBe('AdditiveBlending');
      expect(material.depthWrite).toBe(false);
    });
  });

  describe('shaders', () => {
    it('carries no backtick in any line GLSL source (template-literal guard)', () => {
      // The GLSL sources are template literals, so a single backtick anywhere in
      // them -- including inside a comment, where it reads as ordinary prose
      // quoting -- closes the string early and turns the rest of the module into
      // a parse error. It is a genuinely expensive mistake to diagnose: Vite
      // reports it far from the cause, and a stale dev-server cache keeps serving
      // the broken module afterwards. It happened four times while this file was
      // being written, so pin every line GLSL source, including the shared block
      // and the picking pair (whose sources are assembled the same way).
      const material = new LineMaterial();
      const sources: Array<[string, string]> = [
        ['visual vertex', material.vertexShader],
        ['visual fragment', material.fragmentShader],
        ['shared joint-code block', GLSL_LINE_JOINT_CODE],
        ['shared join-geometry block', GLSL_LINE_JOIN],
        ['pick vertex', LINE_PICK_VERTEX_SHADER],
        ['pick fragment', LINE_PICK_FRAGMENT_SHADER],
      ];
      for (const [label, src] of sources) {
        expect(src, `${label} must contain no backtick`).not.toContain('`');
        expect(src.length, `${label} should be non-empty`).toBeGreaterThan(0);
      }
    });

    it('should have correct vertex shader with screen-space expansion', () => {
      const material = new LineMaterial();

      // Texture-backed storage: the only per-instance attributes are
      // aSortedIndex (aQuadCorner is per quad vertex); per-segment values
      // are texelFetch'd from the line texture (6 texels/segment) into
      // locals with the historical names so the downstream math is
      // unchanged.
      expect(material.vertexShader).toContain('in vec2 aQuadCorner');
      expect(material.vertexShader).toContain('in uint aSortedIndex');
      expect(material.vertexShader).toContain('uniform highp sampler2D uLineTex');
      expect(material.vertexShader).toContain('int lineBase = int(luxarSortedIndex()) * 6');
      expect(material.vertexShader).toContain('vec3 aStartPos = lineT0.xyz');
      expect(material.vertexShader).toContain('float aStartWidth = lineT0.w');
      expect(material.vertexShader).toContain('vec3 aEndPos = lineT1.xyz');
      expect(material.vertexShader).toContain('float aEndWidth = lineT1.w');
      expect(material.vertexShader).toContain('float aSegmentLength = lineT4.x');
      expect(material.vertexShader).toContain('float aStartJointCode = lineT4.y');
      expect(material.vertexShader).toContain('float aEndJointCode = lineT4.z');
      // The interleaved era's per-instance attribute declarations are gone.
      expect(material.vertexShader).not.toContain('in vec3 aStartPos;');
      expect(material.vertexShader).not.toContain('in vec3 aStartColor;');
      expect(material.vertexShader).not.toContain('in float aStartWidth;');

      // Check for uniforms
      expect(material.vertexShader).toContain('uniform vec2 uResolution');

      // Check for varyings (GLSL ES 3.0 uses "out" instead of "varying")
      expect(material.vertexShader).toContain('out vec3 vColor');
      expect(material.vertexShader).toContain('out float vSharpness');
      expect(material.vertexShader).toContain('out float vPerpNorm');
      // Cap math lives in the fragment shader; vertex passes
      // vT/vSegmentLength/vWidthAtT/vCapSuppressStart/vCapSuppressEnd.
      expect(material.vertexShader).toContain('out float vT');
      expect(material.vertexShader).toContain('out float vSegmentLength');
      expect(material.vertexShader).toContain('out float vWidthAtT');
      expect(material.vertexShader).toContain('out float vPixelWidth');

      // Check for screen-space expansion with aspect ratio handling
      expect(material.vertexShader).toContain('perpendicular');
      // pixel width is now `clampedPixelWidth`/`rawPixelWidth`/`vPixelWidth`
      // because the vertex shader applies a max-pixel-width clamp.
      expect(material.vertexShader).toContain('clampedPixelWidth');
      // `pixelDir` is computed directly from NDC endpoints — `pixelStart`
      // / `pixelEnd` no longer exist (the +0.5 bias cancels under
      // subtraction).
      expect(material.vertexShader).toContain('pixelDir');
      expect(material.vertexShader).toContain('minPixelWidth');

      // Vertex shader passes segment metadata for fragment-side cap math.
      expect(material.vertexShader).toContain('aSegmentLength');
    });

    it('should have correct fragment shader with shifted-truncated super-Gaussian falloff', () => {
      const material = new LineMaterial();

      // Check for uniforms
      expect(material.fragmentShader).toContain('uniform float uOpacity');
      expect(material.fragmentShader).toContain('uniform float uInvGamma');

      // Check for GOG model (gain-offset-gamma)
      expect(material.fragmentShader).toContain('vColor * uIntensity + uOffset');
      expect(material.fragmentShader).toContain('pow(adjusted, vec3(uInvGamma))');

      // Shifted-truncated super-Gaussian perpendicular cross-section,
      // beta = 2^(6s - 2), K=ln(100), C=exp(-K). NOT the old parabolic
      // (1 - p²)^sharpness kernel, and no LUXAR_SHARPNESS_TWO fast path.
      expect(material.fragmentShader).toContain('exp2(6.0 * vSharpness - 2.0)');
      expect(material.fragmentShader).toContain('exp(-K * pow(p, beta))');
      expect(material.fragmentShader).toContain('INV_ONE_MINUS_C');
      expect(material.fragmentShader).not.toContain('1.0 - p * p');
      expect(material.fragmentShader).not.toContain('LUXAR_SHARPNESS_TWO');

      // cap factor is computed in fragment from vT/vSegmentLength/etc.,
      // one ramp per endpoint (#796).
      expect(material.fragmentShader).toContain('capFactor');
      expect(material.fragmentShader).toContain('distFromStart');
      expect(material.fragmentShader).toContain('distFromEnd');
      expect(material.fragmentShader).toContain('vT');

      // Check for anti-aliasing and intensity scaling
      expect(material.fragmentShader).toContain('vPixelWidth');
      expect(material.fragmentShader).toContain('smoothstep');
      expect(material.fragmentShader).toContain('edgeAA');
      expect(material.fragmentShader).toContain('widthScale');
      expect(material.fragmentShader).toContain('minPixelWidth');

      // Check for discard outside line width
      expect(material.fragmentShader).toContain('discard');
      expect(material.fragmentShader).toContain('p >= 1.0');
    });
  });

  describe('methods', () => {
    it('should update camera parameters', () => {
      const material = new LineMaterial();
      const fov = (45 * Math.PI) / 180;
      const resolution = new THREE.Vector2(1920, 1080);

      material.updateCameraParams(fov, resolution);

      expect(material.uniforms.uResolution.value.x).toBe(1920);
      expect(material.uniforms.uResolution.value.y).toBe(1080);
    });

    it('should update opacity', () => {
      const material = new LineMaterial();

      material.updateOpacity(0.75);

      expect(material.uniforms.uOpacity.value).toBe(0.75);
    });

    it('should update gamma', () => {
      const material = new LineMaterial();

      material.updateGamma(2.2);

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(material.userData.gamma).toBe(2.2);
    });

    it('should clamp gamma to prevent division by zero', () => {
      const material = new LineMaterial();

      material.updateGamma(0);

      expect(material.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 0.001, 5);
      expect(material.userData.gamma).toBe(0.001);
    });

    it('should clone material with current values', () => {
      const original = new LineMaterial({
        opacity: 0.5,
        gamma: 2.2,
      });

      const cloned = original.clone();

      expect(cloned.uniforms.uOpacity.value).toBe(0.5);
      expect(cloned.uniforms.uInvGamma.value).toBeCloseTo(1.0 / 2.2, 5);
      expect(cloned.userData.gamma).toBe(2.2);

      // Ensure it's a new instance
      expect(cloned).not.toBe(original);
    });
  });

  describe('updateCameraParams nearCull write-through', () => {
    it('accepts nearCull = 0 (zero-diagonal scenes) instead of keeping a stale value', () => {
      // The old `nearCull > 0` gate silently KEPT the previous value —
      // with LRU-cached materials that could be the previous DATASET's
      // scene-scaled nearCull, re-creating the cross-geometry near-fade
      // divergence B9c fixed. The shader floors at 1e-20 (zero-guard
      // only), so writing 0 is safe and symmetric with the point/gsplat
      // wrappers.
      const material = new LineMaterial();
      material.updateCameraParams(1.0, new THREE.Vector2(100, 100), false, 5.0);
      expect(material.uniforms.uNearCull.value).toBe(5.0);
      material.updateCameraParams(1.0, new THREE.Vector2(100, 100), false, 0);
      expect(material.uniforms.uNearCull.value).toBe(0);
      // undefined = keep (the deliberate sentinel)
      material.updateCameraParams(1.0, new THREE.Vector2(100, 100), false);
      expect(material.uniforms.uNearCull.value).toBe(0);
    });
  });

  describe('shader correctness', () => {
    it('should use semicircle kernel model for joints (cap math now in fragment)', () => {
      const material = new LineMaterial();

      // cap factor at endpoints should be 0.5 for seamless joints.
      // Now computed in the fragment shader from interpolated vT, with
      // one ramp per endpoint (issue #796: a single nearest-endpoint
      // ramp was discontinuous on short segments).
      expect(material.fragmentShader).toContain('0.5 + 0.5');
      expect(material.fragmentShader).toContain('distFromStart');
      expect(material.fragmentShader).toContain('distFromEnd');
    });

    it('should handle clipped endpoints correctly (in fragment)', () => {
      const material = new LineMaterial();

      // Clipped/suppressed endpoints should lift THEIR OWN ramp to full
      // intensity (1.0), and the two per-endpoint caps combine with min()
      // (issue #796). Cap-clipping logic lives in the fragment shader.
      expect(material.fragmentShader).toContain(
        'mix(0.5 + 0.5 * startRamp, 1.0, vCapSuppressStart)'
      );
      expect(material.fragmentShader).toContain('mix(0.5 + 0.5 * endRamp, 1.0, vCapSuppressEnd)');
      expect(material.fragmentShader).toContain('min(startCap, endCap)');
    });

    it('should use world-space to pixel conversion', () => {
      const material = new LineMaterial();

      // Check for perspective-correct pixel width calculation. tan() is
      // no longer evaluated per-vertex — `uPerspectiveLineScale` is
      // precomputed CPU-side as resolution.y / tan(fov*0.5).
      expect(material.vertexShader).toContain('uPerspectiveLineScale');
      expect(material.vertexShader).toContain('uOrthoLineScale');
    });

    it('drives the pathological discard from the per-segment max width (issue #849)', () => {
      const material = new LineMaterial();

      // The discard must be ONE per-segment decision, not per-quad-vertex:
      // rawPixelWidth varies between the t=0 and t=1 corners of the shared
      // quad, so gating on it sentinels only half the quad and leaves a
      // visible wedge toward screen center during a close fly-by (#849).
      // The fix gates on segMaxPixelWidth (max over both clipped endpoints).
      expect(material.vertexShader).toContain('segMaxPixelWidth');
      expect(material.vertexShader).toContain('segMaxPixelWidth > maxPW * 2.0');
      expect(material.vertexShader).not.toContain('rawPixelWidth > maxPW * 2.0');
    });
  });

  describe('blending mode depth test configuration', () => {
    it('should have depthTest false for additive mode (ignores depth)', () => {
      const material = new LineMaterial({ blendingMode: 'additive' });

      // 'additive' ignores depth entirely (renders on top of everything)
      expect(material.userData.depthTest).toBe(false);
      expect(material.blending).toBe('AdditiveBlending');
    });

    it('should have depthTest true for luminous mode (respects depth occlusion)', () => {
      const material = new LineMaterial({ blendingMode: 'luminous' });

      // 'luminous' respects depth occlusion but uses same visual output as additive
      expect(material.userData.depthTest).toBe(true);
      expect(material.blending).toBe('AdditiveBlending'); // Same as additive
    });

    it('should have depthTest true for normal mode', () => {
      const material = new LineMaterial({ blendingMode: 'normal' });

      expect(material.userData.depthTest).toBe(true);
      expect(material.blending).toBe('NormalBlending');
    });

    it('should use simple alpha output in fragment shader', () => {
      const material = new LineMaterial();

      // No uLuminous uniform - shader always uses same output pattern
      expect(material.fragmentShader).not.toContain('uniform bool uLuminous');
      expect(material.fragmentShader).not.toContain('if (uLuminous)');

      // Check for alpha output for AdditiveBlending (SrcAlpha, One)
      expect(material.fragmentShader).toContain(
        'fragColor = vec4(finalColor, intensity * uOpacity)'
      );
    });

    it('should configure opaque mode correctly', () => {
      const material = new LineMaterial({ blendingMode: 'opaque' });

      expect(material.transparent).toBe(false);
      expect(material.depthWrite).toBe(true);
      expect(material.blending).toBe('NormalBlending');
      expect(material.userData.depthTest).toBe(true);
    });
  });

  describe('applyBlendingMode', () => {
    // The line-material mock at the top of this file strings out
    // some THREE constants (`AdditiveBlending`, `NormalBlending`,
    // `CustomBlending`, `AddEquation`, `OneFactor`) but leaves others
    // as their real numeric values (`MaxEquation`, `SrcAlphaFactor`,
    // `OneMinusSrcAlphaFactor`). Use real THREE constants for those.
    it('switches additive → max: blending becomes CustomBlending + MaxEquation', () => {
      const material = new LineMaterial({ blendingMode: 'additive' });
      expect(material.blending).toBe('AdditiveBlending');

      material.applyBlendingMode('max');

      expect(material.blending).toBe('CustomBlending');
      expect(material.blendEquation).toBe(THREE.MaxEquation);
      expect(material.blendSrc).toBe('OneFactor');
      expect(material.blendDst).toBe('OneFactor');
      expect(material.userData.blendingMode).toBe('max');
      expect(material.needsUpdate).toBe(true);
    });

    it('switches max → additive: blending resets, blendEquation back to AddEquation', () => {
      // Without applyBlendingMode resetting state, blendEquation would
      // strand at MaxEquation after the user switched modes via the
      // layers panel.
      const material = new LineMaterial({ blendingMode: 'max' });
      expect(material.blendEquation).toBe(THREE.MaxEquation);

      material.applyBlendingMode('additive');

      expect(material.blending).toBe('AdditiveBlending');
      expect(material.blendEquation).toBe('AddEquation');
      expect(material.userData.blendingMode).toBe('additive');
    });

    it('switches additive → luminous: blending unchanged, depthTest flips to true', () => {
      const material = new LineMaterial({ blendingMode: 'additive' });
      expect(material.depthTest).toBe(false);

      material.applyBlendingMode('luminous');

      expect(material.blending).toBe('AdditiveBlending');
      expect(material.depthTest).toBe(true);
      expect(material.userData.blendingMode).toBe('luminous');
    });
  });
});

describe('createLineQuadGeometry', () => {
  it('should create geometry with quad corners', () => {
    const geometry = createLineQuadGeometry();

    const quadCorner = geometry.getAttribute('aQuadCorner');
    expect(quadCorner).toBeDefined();
    expect(quadCorner.count).toBe(4); // 4 vertices per quad
    expect(quadCorner.itemSize).toBe(2);

    // Check corner values
    const array = quadCorner.array as Float32Array;
    expect(array[0]).toBe(-1); // First vertex x
    expect(array[1]).toBe(-1); // First vertex y
    expect(array[6]).toBe(1); // Last vertex x
    expect(array[7]).toBe(1); // Last vertex y
  });

  it('should have correct index buffer', () => {
    const geometry = createLineQuadGeometry();

    const index = geometry.index;
    expect(index).toBeDefined();
    expect(index!.count).toBe(6); // 2 triangles * 3 indices
  });
});

describe('createInstancedLinesMesh', () => {
  it('should create instanced mesh with correct attributes', () => {
    const config = {
      startPositions: new Float32Array([0, 0, 0, 1, 1, 1]),
      endPositions: new Float32Array([1, 0, 0, 2, 1, 1]),
      startColors: new Float32Array([1, 0, 0, 0, 1, 0]),
      endColors: new Float32Array([1, 0, 0, 0, 1, 0]),
      startWidths: new Float32Array([0.1, 0.1]),
      endWidths: new Float32Array([0.1, 0.1]),
      startSharpness: new Float32Array([1.0, 1.0]),
      endSharpness: new Float32Array([1.0, 1.0]),
      segmentLengths: new Float32Array([1.0, 1.414]),
      startJointCode: new Float32Array([0, 0]),
      endJointCode: new Float32Array([0, 0]),
      segmentCount: 2,
    };

    const material = new LineMaterial();
    const mesh = createInstancedLinesMesh(config, material);

    // Lines use THREE.Mesh with InstancedBufferGeometry (not InstancedMesh)
    // to avoid exceeding WebGL's 16 attribute location limit
    expect(mesh).toBeInstanceOf(THREE.Mesh);

    const geometry = mesh.geometry as THREE.InstancedBufferGeometry;
    expect(geometry.instanceCount).toBe(2);

    // Texture-backed storage: aSortedIndex (identity after a fresh build)
    // pair is the only per-instance data; the interleaved-era attributes
    // are gone.
    const sortedIndex = geometry.getAttribute('aSortedIndex');
    expect(sortedIndex).toBeDefined();
    expect(sortedIndex.array).toBeInstanceOf(Uint32Array);
    expect(Array.from((sortedIndex.array as Uint32Array).subarray(0, 2))).toEqual([0, 1]);
    expect(geometry.getAttribute('aStartPos')).toBeUndefined();
    expect(geometry.getAttribute('aStartColor')).toBeUndefined();
    expect(geometry.getAttribute('aSegmentLength')).toBeUndefined();

    // Per-segment data lives in the line texture at the documented
    // 6-texel offsets (segment 1 checked; 24-float stride).
    const texture = getLineTexture(geometry);
    expect(texture).not.toBeNull();
    const data = texture!.image.data as Float32Array;
    const o = 1 * LINE_FLOATS_PER_SEGMENT;
    // texel 0: startPos.xyz, startWidth
    expect(Array.from(data.subarray(o, o + 3))).toEqual([1, 1, 1]);
    expect(data[o + 3]).toBeCloseTo(0.1, 6);
    // texel 1: endPos.xyz, endWidth
    expect(Array.from(data.subarray(o + 4, o + 7))).toEqual([2, 1, 1]);
    expect(data[o + 7]).toBeCloseTo(0.1, 6);
    // texel 2: startColor.rgb, startSharpness
    expect(Array.from(data.subarray(o + 8, o + 12))).toEqual([0, 1, 0, 1]);
    // texel 3: endColor.rgb, endSharpness
    expect(Array.from(data.subarray(o + 12, o + 16))).toEqual([0, 1, 0, 1]);
    // texel 4: segmentLength, startJointCode, endJointCode
    expect(data[o + 16]).toBeCloseTo(1.414, 5);
    expect(data[o + 17]).toBe(0);
    expect(data[o + 18]).toBe(0);
    // texel 5: scalar identities (no scalars in config) + opaque alphas —
    // written UNCONDITIONALLY (fixed layout).
    expect(Array.from(data.subarray(o + 20, o + 24))).toEqual([0, 0, 1, 1]);
  });

  it('should compute bounding box and sphere (width-expanded)', () => {
    const config = {
      startPositions: new Float32Array([0, 0, 0]),
      endPositions: new Float32Array([10, 10, 10]),
      startColors: new Float32Array([1, 0, 0]),
      endColors: new Float32Array([0, 1, 0]),
      startWidths: new Float32Array([0.1]),
      endWidths: new Float32Array([0.1]),
      startSharpness: new Float32Array([1.0]),
      endSharpness: new Float32Array([1.0]),
      segmentLengths: new Float32Array([17.32]),
      startJointCode: new Float32Array([0]),
      endJointCode: new Float32Array([0]),
      segmentCount: 1,
    };

    const material = new LineMaterial();
    const mesh = createInstancedLinesMesh(config, material);

    const geometry = mesh.geometry;
    expect(geometry.boundingBox).toBeDefined();
    expect(geometry.boundingSphere).toBeDefined();

    // bounds are expanded by maxWidth so thick lines near the
    // frustum edge are not prematurely culled. Endpoints are at 0 and
    // 10; with width=0.1 the box extends to [-0.1, 10.1].
    expect(geometry.boundingBox!.min.x).toBeCloseTo(-0.1, 4);
    expect(geometry.boundingBox!.max.x).toBeCloseTo(10.1, 4);
  });

  it('width-only-zero lines have endpoint-only bounds (no expansion)', () => {
    const config = {
      startPositions: new Float32Array([0, 0, 0]),
      endPositions: new Float32Array([10, 0, 0]),
      startColors: new Float32Array([1, 0, 0]),
      endColors: new Float32Array([1, 0, 0]),
      startWidths: new Float32Array([0]),
      endWidths: new Float32Array([0]),
      startSharpness: new Float32Array([1.0]),
      endSharpness: new Float32Array([1.0]),
      segmentLengths: new Float32Array([10]),
      startJointCode: new Float32Array([0]),
      endJointCode: new Float32Array([0]),
      segmentCount: 1,
    };

    const material = new LineMaterial();
    const mesh = createInstancedLinesMesh(config, material);
    const geometry = mesh.geometry;
    expect(geometry.boundingBox!.min.x).toBeCloseTo(0, 5);
    expect(geometry.boundingBox!.max.x).toBeCloseTo(10, 5);
  });
});
