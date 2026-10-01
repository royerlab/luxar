/**
 * The refraction-shift edge apodization: outward shifts vanish on the screen
 * borders, the field is continuous and the identity away from them, and samples
 * stay on screen. The WebGL patch lands in three's real chunks where it must.
 *
 * The wrong answers here all render something plausible: a limiter that touches the
 * centre bends every glass a little; one that is discontinuous draws a seam where it
 * engages; one that is merely bounded still clamp-streaks at the border pixel.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as THREE from 'three';
import {
  REFRACTION_MIN_RAY_SCALE,
  REFRACTION_SHIFT_KNEE,
  TRANSMISSION_NORMAL_LINE,
  TRANSMISSION_PARS_INCLUDE,
  apodizeRefractionShiftGlsl,
  apodizeShiftAxis,
  apodizedShiftFraction,
  rayScaleForShiftFraction,
} from '../../../../../rendering/materials/mesh-physical/refraction-apodization';
import {
  pinTransmittedAlphaGlsl,
  PHYSICAL_PROGRAM_CACHE_KEY,
} from '../../../../../rendering/materials/mesh-physical/config';
import { PhysicalMeshMaterial } from '../../../../../rendering/materials/mesh-physical/material-glsl';
import {
  applyRefractionApodizationTSL,
  refractionRayScaleTSL,
} from '../../../../../rendering/materials/mesh-physical/refraction-apodization-tsl';
import { loadTslMaterials } from '../../../../../rendering/tsl/load';
import { requireTslMaterials } from '../../../../../rendering/tsl/slot';

beforeAll(async () => {
  await loadTslMaterials();
});

describe('apodizeShiftAxis — one axis of the shaped shift', () => {
  it('is the identity up to the knee, everywhere a shift stays inside it', () => {
    for (const room of [1, 0.5, 0.1, 0.01]) {
      for (const f of [0, 0.1, 0.3, REFRACTION_SHIFT_KNEE]) {
        expect(apodizeShiftAxis(f * room, room)).toBe(1);
      }
    }
  });

  it('never lets the shaped shift reach the border, however large the shift', () => {
    for (const room of [1, 0.5, 0.05, 1e-4]) {
      for (const shift of [0.6 * room, room, 2 * room, 10, 1e3]) {
        const shaped = shift * apodizeShiftAxis(shift, room);
        // Strictly below in exact arithmetic; tanh saturates to 1.0 in floating point
        // for a huge shift, which lands the sample ON the border — still on screen.
        expect(shaped).toBeLessThanOrEqual(room);
        expect(shaped).toBeGreaterThan(REFRACTION_SHIFT_KNEE * room);
      }
    }
  });

  it('is zero with no room — the field vanishes on the border', () => {
    for (const shift of [1e-6, 0.01, 1, 100]) expect(apodizeShiftAxis(shift, 0)).toBe(0);
  });

  it('is continuous with slope 1 through the knee, and the shaped shift is monotone', () => {
    const room = 0.4;
    const knee = REFRACTION_SHIFT_KNEE * room;
    const shaped = (x: number): number => x * apodizeShiftAxis(x, room);
    const h = 1e-6;
    expect(shaped(knee + h) - shaped(knee)).toBeCloseTo(h, 9);
    let prev = 0;
    for (let x = 0; x < 3 * room; x += room / 200) {
      const y = shaped(x);
      expect(y).toBeGreaterThanOrEqual(prev);
      prev = y;
    }
  });

  it('shrinks continuously as the room closes, for a fixed shift', () => {
    const shift = 0.05;
    let prev = Infinity;
    for (let step = 200; step >= 0; step--) {
      const y = shift * apodizeShiftAxis(shift, step / 1000);
      expect(y).toBeLessThanOrEqual(prev + 1e-12);
      expect(Math.abs(y - (prev === Infinity ? y : prev))).toBeLessThan(0.002);
      prev = y;
    }
    expect(prev).toBe(0);
  });
});

describe('apodizedShiftFraction — the whole vector keeps its direction', () => {
  it('is 1 in the centre for an ordinary shift, and 0 for outward shifts on the border', () => {
    expect(apodizedShiftFraction([0, 0], [0.05, -0.03])).toBe(1);
    expect(apodizedShiftFraction([0.3, -0.2], [0.1, 0.1])).toBe(1);
    for (const [b, s] of [
      [
        [1, 0],
        [0.04, 0],
      ],
      [
        [-1, 0.3],
        [-0.04, 0],
      ],
      [
        [0.2, 1],
        [0, 0.02],
      ],
      [
        [-0.5, -1],
        [0, -0.02],
      ],
    ]) {
      expect(apodizedShiftFraction(b, s)).toBe(0);
    }
  });

  it('preserves inward shifts near and on the border', () => {
    expect(apodizedShiftFraction([0.95, 0], [-0.2, 0])).toBe(1);
    expect(apodizedShiftFraction([-0.95, 0], [0.2, 0])).toBe(1);
    expect(apodizedShiftFraction([1, 0], [-0.2, 0])).toBe(1);
    expect(apodizedShiftFraction([0, -1], [0, 0.2])).toBe(1);
    expect(apodizedShiftFraction([0.95, 0], [0.2, 0])).toBeLessThan(1);
    const acrossScreen = apodizedShiftFraction([0.95, 0], [-3, 0]);
    expect(acrossScreen).toBeLessThan(1);
    expect(0.95 - 3 * acrossScreen).toBeGreaterThanOrEqual(-1);
  });

  it('takes the stricter axis, so the shaped shift is parallel to the unshaped one', () => {
    const b = [0.9, 0];
    const s = [0.08, 0.08];
    const kappa = apodizedShiftFraction(b, s);
    expect(kappa).toBeCloseTo(apodizeShiftAxis(0.08, 0.1), 12);
    expect(kappa).toBeLessThan(1);
  });
});

describe('rayScaleForShiftFraction — shortening the world ray by the exact amount', () => {
  it('moves the projected sample by exactly the wanted fraction of the screen shift', () => {
    const camera = new THREE.PerspectiveCamera(63, 1.6, 0.01, 100);
    camera.position.set(0.3, -0.2, 1.5);
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    const vp = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    const clip = (p: THREE.Vector3): THREE.Vector4 =>
      new THREE.Vector4(p.x, p.y, p.z, 1).applyMatrix4(vp);
    const p = new THREE.Vector3(0.5, 0.4, -0.3);
    const ray = new THREE.Vector3(0.3, -0.1, -0.6); // a strongly perspective ray (w changes a lot)
    const c0 = clip(p);
    const c1 = clip(p.clone().add(ray));
    const b = new THREE.Vector2(c0.x / c0.w, c0.y / c0.w);
    const t = new THREE.Vector2(c1.x / c1.w, c1.y / c1.w);
    for (const kappa of [0, 0.1, 0.5, 0.9, 1]) {
      const lambda = rayScaleForShiftFraction(kappa, c0.w, c1.w);
      const c = clip(p.clone().addScaledVector(ray, lambda));
      const got = new THREE.Vector2(c.x / c.w, c.y / c.w);
      const want = b.clone().lerp(t, kappa);
      expect(got.distanceTo(want)).toBeLessThan(1e-5);
    }
  });

  it('is the identity at kappa 1, floored at 0, and the floor for a ray ending behind the camera', () => {
    expect(rayScaleForShiftFraction(1, 2, 3)).toBe(1);
    expect(rayScaleForShiftFraction(0, 2, 3)).toBe(REFRACTION_MIN_RAY_SCALE);
    expect(rayScaleForShiftFraction(0.5, 2, -1)).toBe(REFRACTION_MIN_RAY_SCALE);
  });
});

describe('the shaped field over a whole frame, camera inside a refracting sphere', () => {
  /**
   * Trace three's own ray for every pixel of a coarse frame, shape it, and check the
   * properties the field must have where the streaks used to be.
   */
  function frame(shaped: boolean): { inside: boolean; shift: THREE.Vector2[][] } {
    const camera = new THREE.PerspectiveCamera(63, 1.6, 0.001, 100);
    const radius = 1;
    camera.position.set(0, 0, 0.6); // 0.6 R from the centre, looking through it
    camera.lookAt(0, 0, -1);
    camera.updateMatrixWorld();
    const vp = new THREE.Matrix4().multiplyMatrices(
      camera.projectionMatrix,
      camera.matrixWorldInverse
    );
    const inv = vp.clone().invert();
    const thickness = 0.2 * radius;
    const eta = 1 / 1.33;
    let inside = true;
    const rows: THREE.Vector2[][] = [];
    const W = 64;
    const H = 40;
    for (let j = 0; j < H; j++) {
      const row: THREE.Vector2[] = [];
      for (let i = 0; i < W; i++) {
        const ndc = new THREE.Vector2((2 * i) / (W - 1) - 1, (2 * j) / (H - 1) - 1);
        const far = new THREE.Vector4(ndc.x, ndc.y, 1, 1).applyMatrix4(inv);
        const dir = new THREE.Vector3(far.x / far.w, far.y / far.w, far.z / far.w)
          .sub(camera.position)
          .normalize();
        // Far wall of the sphere (the back face the camera sees from inside).
        const o = camera.position;
        const bq = o.dot(dir);
        const tHit = -bq + Math.sqrt(bq * bq - (o.lengthSq() - radius * radius));
        const pos = o.clone().addScaledVector(dir, tHit);
        const n = pos.clone().normalize().negate(); // three flips a back face's normal
        // GLSL refract(I, N, eta)
        const I = dir;
        const k = 1 - eta * eta * (1 - n.dot(I) ** 2);
        const r = I.clone()
          .multiplyScalar(eta)
          .addScaledVector(n, -(eta * n.dot(I) + Math.sqrt(k)));
        const ray = r.normalize().multiplyScalar(thickness);
        const c0 = new THREE.Vector4(pos.x, pos.y, pos.z, 1).applyMatrix4(vp);
        const c1 = new THREE.Vector4(pos.x + ray.x, pos.y + ray.y, pos.z + ray.z, 1).applyMatrix4(
          vp
        );
        const b = new THREE.Vector2(c0.x / c0.w, c0.y / c0.w);
        const t = new THREE.Vector2(c1.x / c1.w, c1.y / c1.w);
        let lambda = 1;
        if (shaped) {
          const kappa = apodizedShiftFraction([b.x, b.y], [t.x - b.x, t.y - b.y]);
          lambda = rayScaleForShiftFraction(kappa, c0.w, c1.w);
        }
        const c = new THREE.Vector4(pos.x, pos.y, pos.z, 1).applyMatrix4(vp).lerp(c1, lambda); // clip space is linear along the ray
        const s = new THREE.Vector2(c.x / c.w, c.y / c.w);
        if (Math.abs(s.x) > 1 + 1e-5 || Math.abs(s.y) > 1 + 1e-5) inside = false;
        row.push(s.sub(b));
      }
      rows.push(row);
    }
    return { inside, shift: rows };
  }

  it('the unshaped field samples off screen (the streaks), the shaped one never does', () => {
    expect(frame(false).inside).toBe(false);
    expect(frame(true).inside).toBe(true);
  });

  it('is zero on the border, untouched in the centre, and has no seam', () => {
    const raw = frame(false).shift;
    const { shift } = frame(true);
    const H = shift.length;
    const W = shift[0].length;
    for (let i = 0; i < W; i++) {
      expect(shift[0][i].length()).toBeLessThan(1e-5);
      expect(shift[H - 1][i].length()).toBeLessThan(1e-5);
    }
    for (let j = 0; j < H; j++) {
      expect(shift[j][0].length()).toBeLessThan(1e-5);
      expect(shift[j][W - 1].length()).toBeLessThan(1e-5);
    }
    // Centre pixels: identical to three's own shift.
    const cj = Math.floor(H / 2);
    const ci = Math.floor(W / 2);
    expect(shift[cj][ci].distanceTo(raw[cj][ci])).toBeLessThan(1e-9);
    // No seam: per axis the shaped shift changes no faster than the shift plus the room
    // (which closes by one grid step per pixel); keeping the direction lets the other
    // axis follow the stricter factor, hence the margin …
    const maxStep = (f: THREE.Vector2[][], horizontal: boolean): number => {
      let m = 0;
      for (let j = horizontal ? 0 : 1; j < H; j++)
        for (let i = horizontal ? 1 : 0; i < W; i++) {
          const prev = horizontal ? f[j][i - 1] : f[j - 1][i];
          m = Math.max(m, f[j][i].distanceTo(prev));
        }
      return m;
    };
    expect(maxStep(shift, true)).toBeLessThanOrEqual(2 * (maxStep(raw, true) + 2 / (W - 1)));
    expect(maxStep(shift, false)).toBeLessThanOrEqual(2 * (maxStep(raw, false) + 2 / (H - 1)));
    // … and the sample map never folds back on itself, so the edge band is squeezed,
    // not mirrored.
    for (let j = 0; j < H; j++)
      for (let i = 1; i < W; i++) {
        const x0 = (2 * (i - 1)) / (W - 1) - 1 + shift[j][i - 1].x;
        const x1 = (2 * i) / (W - 1) - 1 + shift[j][i].x;
        expect(x1).toBeGreaterThan(x0);
      }
  });
});

describe('the WebGL patch on three’s real chunks', () => {
  const fullPhysical = (): string =>
    pinTransmittedAlphaGlsl(
      THREE.ShaderLib.physical.fragmentShader,
      THREE.ShaderChunk.transmission_fragment
    );

  it('both anchors exist exactly once where the patch looks for them', () => {
    expect(THREE.ShaderChunk.transmission_fragment.split(TRANSMISSION_NORMAL_LINE)).toHaveLength(2);
    expect(THREE.ShaderLib.physical.fragmentShader.split(TRANSMISSION_PARS_INCLUDE)).toHaveLength(
      2
    );
    // The pars chunk defines the ray function the helper calls.
    expect(THREE.ShaderChunk.transmission_pars_fragment).toContain(
      'vec3 getVolumeTransmissionRay('
    );
  });

  it('defines the helpers after the parameters and rescales the ray before the refraction call', () => {
    const patched = apodizeRefractionShiftGlsl(fullPhysical());
    const helper = patched.indexOf('float luxarRefractionRayScale(');
    const apply = patched.indexOf('material.thickness *= luxarRayScale;');
    const call = patched.indexOf('vec4 transmitted = getIBLVolumeRefraction(');
    expect(helper).toBeGreaterThan(patched.indexOf(TRANSMISSION_PARS_INCLUDE));
    expect(helper).toBeLessThan(patched.indexOf('void main()'));
    expect(apply).toBeGreaterThan(patched.indexOf(TRANSMISSION_NORMAL_LINE));
    expect(apply).toBeLessThan(call);
    expect(patched).toContain('material.attenuationDistance *= luxarRayScale;');
    // The knee in the shader is the reference's knee.
    expect(patched).toContain(`${REFRACTION_SHIFT_KNEE.toFixed(4)} * room`);
    expect(patched).toContain('if ( c0.w <= 0.0 || c1.w <= 0.0 )');
    expect(patched).toContain('vec2 s = c1.xy / c1.w - b;');
    expect(patched).toContain('vec2 room = max( 1.0 - sign( s ) * b, 0.0 );');
    expect(patched).toContain('luxarApodizeShiftAxis( abs( s.x ), room.x )');
    expect(patched).toContain('luxarApodizeShiftAxis( abs( s.y ), room.y )');
    expect(patched).toContain('kappa * c0.w / ( kappa * c0.w + ( 1.0 - kappa ) * c1.w )');
  });

  it('leaves a shader without the expanded chunk alone', () => {
    const plain = 'void main() { gl_FragColor = vec4(1.0); }';
    expect(apodizeRefractionShiftGlsl(plain)).toBe(plain);
    const unpinned = THREE.ShaderLib.physical.fragmentShader; // include not expanded
    expect(apodizeRefractionShiftGlsl(unpinned)).toBe(unpinned);
  });

  it('the WebGL wrapper applies pin then apodization, under the fixed cache key', () => {
    const m = new PhysicalMeshMaterial({ transmission: 1.0 });
    const params = {
      fragmentShader: THREE.ShaderLib.physical.fragmentShader,
    } as unknown as THREE.WebGLProgramParametersWithUniforms;
    m.clone().onBeforeCompile(params);
    expect(params.fragmentShader).toContain('material.thickness *= luxarRayScale;');
    expect(params.fragmentShader).toBe(apodizeRefractionShiftGlsl(fullPhysical()));
    expect(m.customProgramCacheKey()).toBe(PHYSICAL_PROGRAM_CACHE_KEY);
  });
});

describe('the WebGPU twin', () => {
  it('overrides setupVariants on the physical TSL material', () => {
    const Ctor = requireTslMaterials().materials.meshPhysical as unknown as {
      prototype: { setupVariants: unknown };
    };
    expect(Object.prototype.hasOwnProperty.call(Ctor.prototype, 'setupVariants')).toBe(true);
  });

  it('builds the ray-scale node graph with and without dispersion', () => {
    // Compiling it needs a GPU (the cross-backend pixel check lives in E2E); this pins
    // that the graph is well-formed TSL, which is where a renamed three export breaks.
    expect((refractionRayScaleTSL(true) as { isNode?: boolean }).isNode).toBe(true);
    expect((refractionRayScaleTSL(false) as { isNode?: boolean }).isNode).toBe(true);
    expect(() => applyRefractionApodizationTSL(true)).not.toThrow();
  });
});
