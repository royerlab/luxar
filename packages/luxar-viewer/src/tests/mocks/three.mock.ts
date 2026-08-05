/**
 * Centralized THREE.js Mock
 *
 * Comprehensive mock of THREE.js classes and constants used across all tests.
 * This mock provides consistent, reusable implementations to avoid duplication.
 *
 * Usage:
 *   vi.mock('three', () => import('./mocks/three.mock'))
 */

import { vi } from 'vitest';

// ============================================================================
// Math Classes
// ============================================================================

/**
 * Vitest mock of THREE.Vector2. Constructs a 2D vector with working
 * set/copy/clone/add/sub/multiply(Scalar)/length/normalize/equals so tests can
 * exercise real 2D vector arithmetic without pulling in the full THREE build.
 */
export const Vector2 = vi.fn().mockImplementation((x = 0, y = 0) => ({
  x,
  y,
  set: vi.fn(function (this: any, newX: number, newY: number) {
    this.x = newX;
    this.y = newY;
    return this;
  }),
  copy: vi.fn(function (this: any, v: any) {
    this.x = v.x;
    this.y = v.y;
    return this;
  }),
  clone: vi.fn(function (this: any) {
    return new (Vector2 as any)(this.x, this.y);
  }),
  add: vi.fn(function (this: any, v: any) {
    this.x += v.x;
    this.y += v.y;
    return this;
  }),
  sub: vi.fn(function (this: any, v: any) {
    this.x -= v.x;
    this.y -= v.y;
    return this;
  }),
  multiply: vi.fn(function (this: any, v: any) {
    this.x *= v.x;
    this.y *= v.y;
    return this;
  }),
  multiplyScalar: vi.fn(function (this: any, s: number) {
    this.x *= s;
    this.y *= s;
    return this;
  }),
  length: vi.fn(function (this: any) {
    return Math.sqrt(this.x * this.x + this.y * this.y);
  }),
  normalize: vi.fn(function (this: any) {
    const len = this.length();
    if (len > 0) {
      this.x /= len;
      this.y /= len;
    }
    return this;
  }),
  equals: vi.fn(function (this: any, v: any) {
    return this.x === v.x && this.y === v.y;
  }),
}));

/**
 * Vitest mock of THREE.Vector3. A 3D vector with faithful implementations of
 * the common operations (set/copy/clone/add/sub/dot/cross/normalize/distanceTo/
 * fromArray/toArray). `applyMatrix4`/`applyQuaternion` are no-op passthroughs
 * and `getWorldDirection` returns a fixed -Z; tests that need real transforms
 * should not rely on those.
 */
export const Vector3 = vi.fn().mockImplementation((x = 0, y = 0, z = 0) => ({
  x,
  y,
  z,
  set: vi.fn(function (this: any, newX: number, newY: number, newZ: number) {
    this.x = newX;
    this.y = newY;
    this.z = newZ;
    return this;
  }),
  copy: vi.fn(function (this: any, v: any) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }),
  clone: vi.fn(function (this: any) {
    return new (Vector3 as any)(this.x, this.y, this.z);
  }),
  add: vi.fn(function (this: any, v: any) {
    this.x += v.x;
    this.y += v.y;
    this.z += v.z;
    return this;
  }),
  sub: vi.fn(function (this: any, v: any) {
    this.x -= v.x;
    this.y -= v.y;
    this.z -= v.z;
    return this;
  }),
  multiply: vi.fn(function (this: any, v: any) {
    this.x *= v.x;
    this.y *= v.y;
    this.z *= v.z;
    return this;
  }),
  multiplyScalar: vi.fn(function (this: any, s: number) {
    this.x *= s;
    this.y *= s;
    this.z *= s;
    return this;
  }),
  applyMatrix4: vi.fn(function (this: any) {
    return this;
  }),
  applyQuaternion: vi.fn(function (this: any) {
    return this;
  }),
  getWorldDirection: vi.fn(function (this: any) {
    return new (Vector3 as any)(0, 0, -1);
  }),
  distanceTo: vi.fn(function (this: any, v: any) {
    const dx = this.x - v.x;
    const dy = this.y - v.y;
    const dz = this.z - v.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }),
  length: vi.fn(function (this: any) {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
  }),
  normalize: vi.fn(function (this: any) {
    const len = this.length();
    if (len > 0) {
      this.x /= len;
      this.y /= len;
      this.z /= len;
    }
    return this;
  }),
  dot: vi.fn(function (this: any, v: any) {
    return this.x * v.x + this.y * v.y + this.z * v.z;
  }),
  cross: vi.fn(function (this: any, v: any) {
    const x = this.y * v.z - this.z * v.y;
    const y = this.z * v.x - this.x * v.z;
    const z = this.x * v.y - this.y * v.x;
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }),
  equals: vi.fn(function (this: any, v: any) {
    return this.x === v.x && this.y === v.y && this.z === v.z;
  }),
  fromArray: vi.fn(function (this: any, arr: number[], offset = 0) {
    this.x = arr[offset];
    this.y = arr[offset + 1];
    this.z = arr[offset + 2];
    return this;
  }),
  toArray: vi.fn(function (this: any, arr: number[] = [], offset = 0) {
    arr[offset] = this.x;
    arr[offset + 1] = this.y;
    arr[offset + 2] = this.z;
    return arr;
  }),
}));

/**
 * Vitest mock of THREE.Vector4. A minimal 4D vector supporting set/copy/clone;
 * used mainly as a plain data holder (e.g. render-target viewport/scissor).
 */
export const Vector4 = vi.fn().mockImplementation((x = 0, y = 0, z = 0, w = 1) => ({
  x,
  y,
  z,
  w,
  set: vi.fn(function (this: any, newX: number, newY: number, newZ: number, newW: number) {
    this.x = newX;
    this.y = newY;
    this.z = newZ;
    this.w = newW;
    return this;
  }),
  copy: vi.fn(function (this: any, v: any) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    this.w = v.w;
    return this;
  }),
  clone: vi.fn(function (this: any) {
    return new (Vector4 as any)(this.x, this.y, this.z, this.w);
  }),
}));

/**
 * Vitest mock of THREE.Quaternion. Implements set/copy/clone plus real
 * `setFromEuler` and Hamilton-product `multiply`, so orientation math in tests
 * behaves like the real class.
 */
export const Quaternion = vi.fn().mockImplementation((x = 0, y = 0, z = 0, w = 1) => ({
  x,
  y,
  z,
  w,
  set: vi.fn(function (this: any, newX: number, newY: number, newZ: number, newW: number) {
    this.x = newX;
    this.y = newY;
    this.z = newZ;
    this.w = newW;
    return this;
  }),
  copy: vi.fn(function (this: any, q: any) {
    this.x = q.x;
    this.y = q.y;
    this.z = q.z;
    this.w = q.w;
    return this;
  }),
  clone: vi.fn(function (this: any) {
    return new (Quaternion as any)(this.x, this.y, this.z, this.w);
  }),
  setFromEuler: vi.fn(function (this: any, euler: any) {
    // Simplified quaternion from Euler angles
    const cx = Math.cos(euler.x / 2);
    const cy = Math.cos(euler.y / 2);
    const cz = Math.cos(euler.z / 2);
    const sx = Math.sin(euler.x / 2);
    const sy = Math.sin(euler.y / 2);
    const sz = Math.sin(euler.z / 2);

    this.x = sx * cy * cz + cx * sy * sz;
    this.y = cx * sy * cz - sx * cy * sz;
    this.z = cx * cy * sz + sx * sy * cz;
    this.w = cx * cy * cz - sx * sy * sz;

    return this;
  }),
  multiply: vi.fn(function (this: any, q: any) {
    const qax = this.x,
      qay = this.y,
      qaz = this.z,
      qaw = this.w;
    const qbx = q.x,
      qby = q.y,
      qbz = q.z,
      qbw = q.w;

    this.x = qax * qbw + qaw * qbx + qay * qbz - qaz * qby;
    this.y = qay * qbw + qaw * qby + qaz * qbx - qax * qbz;
    this.z = qaz * qbw + qaw * qbz + qax * qby - qay * qbx;
    this.w = qaw * qbw - qax * qbx - qay * qby - qaz * qbz;

    return this;
  }),
  equals: vi.fn(function (this: any, q: any) {
    return this.x === q.x && this.y === q.y && this.z === q.z && this.w === q.w;
  }),
}));

/**
 * Vitest mock of THREE.Euler. Holds x/y/z angles plus rotation `order` and
 * supports set/copy/clone; consumed by the Quaternion mock's `setFromEuler`.
 */
export const Euler = vi.fn().mockImplementation((x = 0, y = 0, z = 0, order = 'XYZ') => ({
  x,
  y,
  z,
  order,
  set: vi.fn(function (this: any, newX: number, newY: number, newZ: number, newOrder?: string) {
    this.x = newX;
    this.y = newY;
    this.z = newZ;
    if (newOrder !== undefined) this.order = newOrder;
    return this;
  }),
  copy: vi.fn(function (this: any, euler: any) {
    this.x = euler.x;
    this.y = euler.y;
    this.z = euler.z;
    this.order = euler.order;
    return this;
  }),
  clone: vi.fn(function (this: any) {
    return new (Euler as any)(this.x, this.y, this.z, this.order);
  }),
}));

/**
 * Vitest mock of THREE.Matrix4. Backed by a real 16-element column-major
 * `elements` array with working set/identity/copy/clone/multiply(Matrices) and
 * make* factories. `compose`/`decompose` are simplified to only carry the
 * translation column (rotation/scale are ignored), which is sufficient for the
 * scene-graph matrix bookkeeping the tests exercise.
 */
export const Matrix4 = vi.fn().mockImplementation(() => {
  const elements = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  return {
    elements,
    set: vi.fn(function (
      this: any,
      n11: number,
      n12: number,
      n13: number,
      n14: number,
      n21: number,
      n22: number,
      n23: number,
      n24: number,
      n31: number,
      n32: number,
      n33: number,
      n34: number,
      n41: number,
      n42: number,
      n43: number,
      n44: number
    ) {
      const te = this.elements;
      te[0] = n11;
      te[1] = n21;
      te[2] = n31;
      te[3] = n41;
      te[4] = n12;
      te[5] = n22;
      te[6] = n32;
      te[7] = n42;
      te[8] = n13;
      te[9] = n23;
      te[10] = n33;
      te[11] = n43;
      te[12] = n14;
      te[13] = n24;
      te[14] = n34;
      te[15] = n44;
      return this;
    }),
    identity: vi.fn(function (this: any) {
      this.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
      return this;
    }),
    copy: vi.fn(function (this: any, m: any) {
      const te = this.elements;
      const me = m.elements;
      for (let i = 0; i < 16; i++) {
        te[i] = me[i];
      }
      return this;
    }),
    clone: vi.fn(function (this: any) {
      const m = new (Matrix4 as any)();
      return m.copy(this);
    }),
    multiply: vi.fn(function (this: any, m: any) {
      return this.multiplyMatrices(this, m);
    }),
    multiplyMatrices: vi.fn(function (this: any, a: any, b: any) {
      const ae = a.elements;
      const be = b.elements;
      const te = this.elements;

      const a11 = ae[0],
        a12 = ae[4],
        a13 = ae[8],
        a14 = ae[12];
      const a21 = ae[1],
        a22 = ae[5],
        a23 = ae[9],
        a24 = ae[13];
      const a31 = ae[2],
        a32 = ae[6],
        a33 = ae[10],
        a34 = ae[14];
      const a41 = ae[3],
        a42 = ae[7],
        a43 = ae[11],
        a44 = ae[15];

      const b11 = be[0],
        b12 = be[4],
        b13 = be[8],
        b14 = be[12];
      const b21 = be[1],
        b22 = be[5],
        b23 = be[9],
        b24 = be[13];
      const b31 = be[2],
        b32 = be[6],
        b33 = be[10],
        b34 = be[14];
      const b41 = be[3],
        b42 = be[7],
        b43 = be[11],
        b44 = be[15];

      te[0] = a11 * b11 + a12 * b21 + a13 * b31 + a14 * b41;
      te[4] = a11 * b12 + a12 * b22 + a13 * b32 + a14 * b42;
      te[8] = a11 * b13 + a12 * b23 + a13 * b33 + a14 * b43;
      te[12] = a11 * b14 + a12 * b24 + a13 * b34 + a14 * b44;

      te[1] = a21 * b11 + a22 * b21 + a23 * b31 + a24 * b41;
      te[5] = a21 * b12 + a22 * b22 + a23 * b32 + a24 * b42;
      te[9] = a21 * b13 + a22 * b23 + a23 * b33 + a24 * b43;
      te[13] = a21 * b14 + a22 * b24 + a23 * b34 + a24 * b44;

      te[2] = a31 * b11 + a32 * b21 + a33 * b31 + a34 * b41;
      te[6] = a31 * b12 + a32 * b22 + a33 * b32 + a34 * b42;
      te[10] = a31 * b13 + a32 * b23 + a33 * b33 + a34 * b43;
      te[14] = a31 * b14 + a32 * b24 + a33 * b34 + a34 * b44;

      te[3] = a41 * b11 + a42 * b21 + a43 * b31 + a44 * b41;
      te[7] = a41 * b12 + a42 * b22 + a43 * b32 + a44 * b42;
      te[11] = a41 * b13 + a42 * b23 + a43 * b33 + a44 * b43;
      te[15] = a41 * b14 + a42 * b24 + a43 * b34 + a44 * b44;

      return this;
    }),
    compose: vi.fn(function (this: any, position: any, _quaternion: any, _scale: any) {
      // Simplified compose - just sets position
      const te = this.elements;
      te[12] = position.x;
      te[13] = position.y;
      te[14] = position.z;
      return this;
    }),
    decompose: vi.fn(function (this: any, position: any, _quaternion: any, _scale: any) {
      // Simplified decompose - just extracts position
      const te = this.elements;
      position.set(te[12], te[13], te[14]);
      return this;
    }),
    makeTranslation: vi.fn(function (this: any, x: number, y: number, z: number) {
      this.set(1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z, 0, 0, 0, 1);
      return this;
    }),
    makeRotationX: vi.fn(function (this: any, theta: number) {
      const c = Math.cos(theta),
        s = Math.sin(theta);
      this.set(1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1);
      return this;
    }),
    makeRotationY: vi.fn(function (this: any, theta: number) {
      const c = Math.cos(theta),
        s = Math.sin(theta);
      this.set(c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1);
      return this;
    }),
    makeRotationZ: vi.fn(function (this: any, theta: number) {
      const c = Math.cos(theta),
        s = Math.sin(theta);
      this.set(c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
      return this;
    }),
    makeScale: vi.fn(function (this: any, x: number, y: number, z: number) {
      this.set(x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1);
      return this;
    }),
    fromArray: vi.fn(function (this: any, arr: number[], offset = 0) {
      for (let i = 0; i < 16; i++) {
        this.elements[i] = arr[i + offset];
      }
      return this;
    }),
    toArray: vi.fn(function (this: any, arr: number[] = [], offset = 0) {
      const te = this.elements;
      for (let i = 0; i < 16; i++) {
        arr[i + offset] = te[i];
      }
      return arr;
    }),
  };
});

/**
 * Vitest mock of THREE.Box3, an axis-aligned bounding box. Implements
 * set/expandByPoint/getCenter/getSize/clone/copy/makeEmpty/isEmpty against the
 * real Vector3 min/max so bounds computations work in tests.
 *
 * Caveat: `expandByScalar` is a non-functional stub — it calls
 * `Vector3.addScalar`, which this file's Vector3 mock does not implement, so
 * calling it throws. Tests must not use it.
 */
export const Box3 = vi.fn().mockImplementation((min?: any, max?: any) => ({
  min: min || new (Vector3 as any)(Infinity, Infinity, Infinity),
  max: max || new (Vector3 as any)(-Infinity, -Infinity, -Infinity),
  set: vi.fn(function (this: any, min: any, max: any) {
    this.min.copy(min);
    this.max.copy(max);
    return this;
  }),
  expandByPoint: vi.fn(function (this: any, point: any) {
    this.min.x = Math.min(this.min.x, point.x);
    this.min.y = Math.min(this.min.y, point.y);
    this.min.z = Math.min(this.min.z, point.z);
    this.max.x = Math.max(this.max.x, point.x);
    this.max.y = Math.max(this.max.y, point.y);
    this.max.z = Math.max(this.max.z, point.z);
    return this;
  }),
  expandByScalar: vi.fn(function (this: any, scalar: number) {
    this.min.addScalar(-scalar);
    this.max.addScalar(scalar);
    return this;
  }),
  getCenter: vi.fn(function (this: any, target: any) {
    target.x = (this.min.x + this.max.x) / 2;
    target.y = (this.min.y + this.max.y) / 2;
    target.z = (this.min.z + this.max.z) / 2;
    return target;
  }),
  getSize: vi.fn(function (this: any, target: any) {
    target.x = this.max.x - this.min.x;
    target.y = this.max.y - this.min.y;
    target.z = this.max.z - this.min.z;
    return target;
  }),
  clone: vi.fn(function (this: any) {
    return new (Box3 as any)(this.min.clone(), this.max.clone());
  }),
  copy: vi.fn(function (this: any, box: any) {
    this.min.copy(box.min);
    this.max.copy(box.max);
    return this;
  }),
  makeEmpty: vi.fn(function (this: any) {
    this.min.x = this.min.y = this.min.z = Infinity;
    this.max.x = this.max.y = this.max.z = -Infinity;
    return this;
  }),
  isEmpty: vi.fn(function (this: any) {
    return this.max.x < this.min.x || this.max.y < this.min.y || this.max.z < this.min.z;
  }),
}));

/**
 * Vitest mock of THREE.Color. Parses hex strings, packed hex numbers, or RGB
 * triples into normalized r/g/b in [0,1] and supports set/setRGB/setHex/copy/
 * clone/getHex, matching the real class closely enough for color-handling tests.
 */
export const Color = vi.fn().mockImplementation((r?: any, g?: number, b?: number) => {
  let _r = 0,
    _g = 0,
    _b = 0;

  if (typeof r === 'string') {
    // Handle hex color strings
    const hex = r.replace('#', '');
    _r = parseInt(hex.substring(0, 2), 16) / 255;
    _g = parseInt(hex.substring(2, 4), 16) / 255;
    _b = parseInt(hex.substring(4, 6), 16) / 255;
  } else if (typeof r === 'number') {
    if (g !== undefined && b !== undefined) {
      _r = r;
      _g = g;
      _b = b;
    } else {
      // Handle single number as hex
      _r = ((r >> 16) & 255) / 255;
      _g = ((r >> 8) & 255) / 255;
      _b = (r & 255) / 255;
    }
  }

  return {
    r: _r,
    g: _g,
    b: _b,
    set: vi.fn(function (this: any, value: any) {
      if (typeof value === 'string') {
        const hex = value.replace('#', '');
        this.r = parseInt(hex.substring(0, 2), 16) / 255;
        this.g = parseInt(hex.substring(2, 4), 16) / 255;
        this.b = parseInt(hex.substring(4, 6), 16) / 255;
      } else if (typeof value === 'number') {
        this.r = ((value >> 16) & 255) / 255;
        this.g = ((value >> 8) & 255) / 255;
        this.b = (value & 255) / 255;
      }
      return this;
    }),
    setRGB: vi.fn(function (this: any, r: number, g: number, b: number) {
      this.r = r;
      this.g = g;
      this.b = b;
      return this;
    }),
    setHex: vi.fn(function (this: any, hex: number) {
      this.r = ((hex >> 16) & 255) / 255;
      this.g = ((hex >> 8) & 255) / 255;
      this.b = (hex & 255) / 255;
      return this;
    }),
    copy: vi.fn(function (this: any, color: any) {
      this.r = color.r;
      this.g = color.g;
      this.b = color.b;
      return this;
    }),
    clone: vi.fn(function (this: any) {
      return new (Color as any)(this.r, this.g, this.b);
    }),
    getHex: vi.fn(function (this: any) {
      return (
        ((Math.floor(this.r * 255) << 16) ^
          (Math.floor(this.g * 255) << 8) ^
          Math.floor(this.b * 255)) >>>
        0
      );
    }),
  };
});

// ============================================================================
// Scene Graph Classes
// ============================================================================

/**
 * Vitest mock of THREE.Object3D, the scene-graph node base class. Provides a
 * real parent/children hierarchy with add/remove/clear, name/id lookups,
 * traverse(Visible), matrix/matrixWorld updates, and clone/copy. This is the
 * one full `class` in the mock (other node types extend it) so instanceof and
 * subclassing behave; unlike the real class it omits event dispatch and layers.
 */
export class Object3D {
  position = new (Vector3 as any)();
  rotation = new (Euler as any)();
  quaternion = new (Quaternion as any)();
  scale = new (Vector3 as any)(1, 1, 1);
  matrix = new (Matrix4 as any)();
  matrixWorld = new (Matrix4 as any)();
  matrixAutoUpdate = true;
  matrixWorldNeedsUpdate = false;
  visible = true;
  castShadow = false;
  receiveShadow = false;
  frustumCulled = true;
  renderOrder = 0;
  userData = {};
  parent: any = null;
  children: any[] = [];
  up = new (Vector3 as any)(0, 1, 0);
  name = '';
  type = 'Object3D';

  add = vi.fn(function (this: any, ...objects: any[]) {
    for (const object of objects) {
      if (object === this) {
        console.error("Object3D.add: object can't be added as a child of itself.", object);
        continue;
      }

      if (object.parent !== null) {
        object.parent.remove(object);
      }

      object.parent = this;
      this.children.push(object);
    }
    return this;
  });

  remove = vi.fn(function (this: any, ...objects: any[]) {
    for (const object of objects) {
      const index = this.children.indexOf(object);
      if (index !== -1) {
        object.parent = null;
        this.children.splice(index, 1);
      }
    }
    return this;
  });

  clear = vi.fn(function (this: any) {
    for (let i = 0; i < this.children.length; i++) {
      this.children[i].parent = null;
    }
    this.children.length = 0;
    return this;
  });

  getObjectByName = vi.fn(function (this: any, name: string): any {
    if (this.name === name) return this;
    for (const child of this.children) {
      const found = child.getObjectByName?.(name);
      if (found) return found;
    }
    return undefined;
  });

  getObjectById = vi.fn(function (this: any, id: number): any {
    if (this.id === id) return this;
    for (const child of this.children) {
      const found = child.getObjectById?.(id);
      if (found) return found;
    }
    return undefined;
  });

  traverse = vi.fn(function (this: any, callback: (object: any) => void) {
    callback(this);
    for (const child of this.children) {
      child.traverse?.(callback);
    }
  });

  traverseVisible = vi.fn(function (this: any, callback: (object: any) => void) {
    if (!this.visible) return;
    callback(this);
    for (const child of this.children) {
      child.traverseVisible?.(callback);
    }
  });

  updateMatrix = vi.fn(function (this: any) {
    this.matrix.compose(this.position, this.quaternion, this.scale);
    this.matrixWorldNeedsUpdate = true;
  });

  updateMatrixWorld = vi.fn(function (this: any, force = false) {
    if (this.matrixAutoUpdate) this.updateMatrix();

    if (this.matrixWorldNeedsUpdate || force) {
      if (this.parent === null) {
        this.matrixWorld.copy(this.matrix);
      } else {
        this.matrixWorld.multiplyMatrices(this.parent.matrixWorld, this.matrix);
      }

      this.matrixWorldNeedsUpdate = false;
      force = true;
    }

    // Update children
    for (const child of this.children) {
      child.updateMatrixWorld?.(force);
    }
  });

  lookAt = vi.fn(function (this: any, _x: any, _y?: number, _z?: number) {
    // Simplified lookAt implementation
    return this;
  });

  clone = vi.fn(function (this: any, recursive = true) {
    const cloned = new Object3D();
    cloned.name = this.name;
    cloned.visible = this.visible;
    cloned.userData = JSON.parse(JSON.stringify(this.userData));
    cloned.position.copy(this.position);
    cloned.quaternion.copy(this.quaternion);
    cloned.scale.copy(this.scale);

    if (recursive) {
      for (const child of this.children) {
        cloned.add(child.clone(true));
      }
    }

    return cloned;
  });

  copy = vi.fn(function (this: any, source: any, recursive = true) {
    this.name = source.name;
    this.visible = source.visible;
    this.userData = JSON.parse(JSON.stringify(source.userData));
    this.position.copy(source.position);
    this.quaternion.copy(source.quaternion);
    this.scale.copy(source.scale);

    if (recursive) {
      this.clear();
      for (const child of source.children) {
        this.add(child.clone(true));
      }
    }

    return this;
  });
}

/**
 * Vitest mock of THREE.Group: an Object3D whose `type` is `'Group'`, used to
 * bundle children under a single transform.
 */
export const Group = vi.fn().mockImplementation(() => {
  const obj = new Object3D();
  obj.type = 'Group';
  return obj;
});

/**
 * Vitest mock of THREE.Scene: the root Object3D (`type` `'Scene'`) with the
 * background/environment/fog/overrideMaterial fields stubbed to null.
 */
export const Scene = vi.fn().mockImplementation(() => {
  const obj = new Object3D();
  (obj as any).type = 'Scene';
  (obj as any).background = null;
  (obj as any).environment = null;
  (obj as any).fog = null;
  (obj as any).overrideMaterial = null;
  (obj as any).autoUpdate = true;
  return obj;
});

// ============================================================================
// Camera Classes
// ============================================================================

/**
 * Vitest mock of THREE.Camera: an Object3D carrying the matrixWorldInverse and
 * projectionMatrix(Inverse) Matrix4s; base for the perspective/ortho mocks.
 */
export const Camera = vi.fn().mockImplementation(() => {
  const obj = new Object3D();
  (obj as any).type = 'Camera';
  (obj as any).matrixWorldInverse = new (Matrix4 as any)();
  (obj as any).projectionMatrix = new (Matrix4 as any)();
  (obj as any).projectionMatrixInverse = new (Matrix4 as any)();
  return obj;
});

/**
 * Vitest mock of THREE.PerspectiveCamera. Stores fov/aspect/near/far/zoom and
 * stubs updateProjectionMatrix and the view-offset/film-metric helpers; the
 * projection matrix itself is not computed.
 */
export const PerspectiveCamera = vi
  .fn()
  .mockImplementation((fov = 50, aspect = 1, near = 0.1, far = 2000) => {
    const cam = Camera();
    cam.type = 'PerspectiveCamera';
    cam.fov = fov;
    cam.aspect = aspect;
    cam.near = near;
    cam.far = far;
    cam.zoom = 1;
    cam.focus = 10;
    cam.filmGauge = 35;
    cam.filmOffset = 0;
    cam.view = null;

    cam.updateProjectionMatrix = vi.fn();
    cam.setViewOffset = vi.fn();
    cam.clearViewOffset = vi.fn();
    cam.getEffectiveFOV = vi.fn(() => fov);
    cam.getFilmWidth = vi.fn(() => 35);
    cam.getFilmHeight = vi.fn(() => 24);

    return cam;
  });

/**
 * Vitest mock of THREE.OrthographicCamera. Stores the left/right/top/bottom/
 * near/far frustum extents and zoom and stubs updateProjectionMatrix and the
 * view-offset helpers; the projection matrix itself is not computed.
 */
export const OrthographicCamera = vi
  .fn()
  .mockImplementation((left = -1, right = 1, top = 1, bottom = -1, near = 0.1, far = 2000) => {
    const cam = Camera();
    cam.type = 'OrthographicCamera';
    cam.left = left;
    cam.right = right;
    cam.top = top;
    cam.bottom = bottom;
    cam.near = near;
    cam.far = far;
    cam.zoom = 1;
    cam.view = null;

    cam.updateProjectionMatrix = vi.fn();
    cam.setViewOffset = vi.fn();
    cam.clearViewOffset = vi.fn();

    return cam;
  });

// ============================================================================
// Geometry Classes
// ============================================================================

/**
 * Vitest mock of THREE.BufferAttribute. Wraps a typed array with a given
 * itemSize and implements real per-component get/set X/Y/Z/W accessors plus
 * clone/copy, so geometry-attribute reads and writes behave in tests.
 */
export const BufferAttribute = vi
  .fn()
  .mockImplementation((array: ArrayLike<number>, itemSize: number, normalized = false) => ({
    array,
    itemSize,
    count: array.length / itemSize,
    normalized,
    usage: 35044, // StaticDrawUsage
    updateRange: { offset: 0, count: -1 },
    version: 0,
    name: '',
    onUploadCallback: () => {},
    isBufferAttribute: true,

    set: vi.fn(function (this: any, value: any, offset = 0) {
      this.array.set(value, offset);
      return this;
    }),
    setX: vi.fn(function (this: any, index: number, x: number) {
      this.array[index * this.itemSize] = x;
      return this;
    }),
    setY: vi.fn(function (this: any, index: number, y: number) {
      this.array[index * this.itemSize + 1] = y;
      return this;
    }),
    setZ: vi.fn(function (this: any, index: number, z: number) {
      this.array[index * this.itemSize + 2] = z;
      return this;
    }),
    setW: vi.fn(function (this: any, index: number, w: number) {
      this.array[index * this.itemSize + 3] = w;
      return this;
    }),
    getX: vi.fn(function (this: any, index: number) {
      return this.array[index * this.itemSize];
    }),
    getY: vi.fn(function (this: any, index: number) {
      return this.array[index * this.itemSize + 1];
    }),
    getZ: vi.fn(function (this: any, index: number) {
      return this.array[index * this.itemSize + 2];
    }),
    getW: vi.fn(function (this: any, index: number) {
      return this.array[index * this.itemSize + 3];
    }),
    clone: vi.fn(function (this: any) {
      return new (BufferAttribute as any)(
        new (this.array.constructor as any)(this.array),
        this.itemSize,
        this.normalized
      );
    }),
    copy: vi.fn(function (this: any, source: any) {
      this.name = source.name;
      this.array = new (source.array.constructor as any)(source.array);
      this.itemSize = source.itemSize;
      this.count = source.count;
      this.normalized = source.normalized;
      this.usage = source.usage;
      return this;
    }),
    needsUpdate: false,
  }));

/**
 * Vitest mock of THREE.BufferGeometry. Maintains a real attributes map with
 * set/get/delete/hasAttribute, index, draw range, groups, and clone/copy; the
 * computeBounding* helpers create placeholder Box3/sphere objects rather than
 * computing true extents.
 */
export const BufferGeometry = vi.fn().mockImplementation(() => ({
  attributes: {} as Record<string, any>,
  index: null,
  drawRange: { start: 0, count: Infinity },
  groups: [] as any[],
  boundingBox: null,
  boundingSphere: null,
  userData: {},
  name: '',
  type: 'BufferGeometry',
  id: Math.random(),
  uuid: Math.random().toString(36),

  setAttribute: vi.fn(function (this: any, name: string, attribute: any) {
    this.attributes[name] = attribute;
    return this;
  }),
  getAttribute: vi.fn(function (this: any, name: string) {
    return this.attributes[name];
  }),
  deleteAttribute: vi.fn(function (this: any, name: string) {
    delete this.attributes[name];
    return this;
  }),
  hasAttribute: vi.fn(function (this: any, name: string) {
    return this.attributes[name] !== undefined;
  }),
  setIndex: vi.fn(function (this: any, index: any) {
    this.index = index;
    return this;
  }),
  setDrawRange: vi.fn(function (this: any, start: number, count: number) {
    this.drawRange.start = start;
    this.drawRange.count = count;
  }),
  addGroup: vi.fn(function (this: any, start: number, count: number, materialIndex = 0) {
    this.groups.push({ start, count, materialIndex });
  }),
  clearGroups: vi.fn(function (this: any) {
    this.groups.length = 0;
  }),
  computeBoundingBox: vi.fn(function (this: any) {
    if (this.boundingBox === null) {
      this.boundingBox = new (Box3 as any)();
    }
  }),
  computeBoundingSphere: vi.fn(function (this: any) {
    if (this.boundingSphere === null) {
      this.boundingSphere = { center: new (Vector3 as any)(), radius: 1 };
    }
  }),
  clone: vi.fn(function (this: any) {
    const cloned = new (BufferGeometry as any)();
    cloned.name = this.name;
    Object.keys(this.attributes).forEach((key) => {
      cloned.setAttribute(key, this.attributes[key].clone());
    });
    if (this.index) cloned.setIndex(this.index.clone());
    cloned.groups = JSON.parse(JSON.stringify(this.groups));
    cloned.drawRange = { ...this.drawRange };
    cloned.userData = JSON.parse(JSON.stringify(this.userData));
    return cloned;
  }),
  copy: vi.fn(function (this: any, source: any) {
    this.name = source.name;
    this.attributes = {};
    Object.keys(source.attributes).forEach((key) => {
      this.setAttribute(key, source.attributes[key].clone());
    });
    if (source.index) this.setIndex(source.index.clone());
    this.groups = JSON.parse(JSON.stringify(source.groups));
    this.drawRange = { ...source.drawRange };
    this.userData = JSON.parse(JSON.stringify(source.userData));
    return this;
  }),
  dispose: vi.fn(),
}));

/**
 * Vitest mock of THREE.BoxGeometry: a BufferGeometry tagged `'BoxGeometry'`
 * that records its width/height/depth parameters (no vertices are generated).
 */
export const BoxGeometry = vi.fn().mockImplementation((width = 1, height = 1, depth = 1) => {
  const geom = BufferGeometry();
  geom.type = 'BoxGeometry';
  geom.parameters = { width, height, depth };
  return geom;
});

// ============================================================================
// Material Classes
// ============================================================================

/**
 * Vitest mock of THREE.Material. Exposes the common material flags plus
 * clone/copy and a minimal EventDispatcher surface (addEventListener/
 * removeEventListener/dispatchEvent) so that `dispose()` fires a synchronous
 * `'dispose'` event — MaterialManager relies on this for automatic cleanup.
 */
export const Material = vi.fn().mockImplementation(() => ({
  type: 'Material',
  name: '',
  fog: true,
  blending: 1, // NormalBlending
  side: 0, // FrontSide
  vertexColors: false,
  opacity: 1,
  transparent: false,
  blendSrc: 204, // SrcAlphaFactor
  blendDst: 205, // OneMinusSrcAlphaFactor
  blendEquation: 100, // AddEquation
  depthFunc: 3, // LessEqualDepth
  depthTest: true,
  depthWrite: true,
  colorWrite: true,
  stencilWrite: false,
  stencilFunc: 519, // AlwaysStencilFunc
  stencilRef: 0,
  stencilMask: 0xff,
  stencilFail: 7680, // KeepStencilOp
  stencilZFail: 7680,
  stencilZPass: 7680,
  polygonOffset: false,
  polygonOffsetFactor: 0,
  polygonOffsetUnits: 0,
  dithering: false,
  alphaToCoverage: false,
  premultipliedAlpha: false,
  visible: true,
  toneMapped: true,
  userData: {},
  version: 0,
  needsUpdate: false,

  clone: vi.fn(function (this: any) {
    const cloned = new (Material as any)();
    return cloned.copy(this);
  }),
  copy: vi.fn(function (this: any, source: any) {
    this.name = source.name;
    this.fog = source.fog;
    this.blending = source.blending;
    this.side = source.side;
    this.vertexColors = source.vertexColors;
    this.opacity = source.opacity;
    this.transparent = source.transparent;
    this.userData = JSON.parse(JSON.stringify(source.userData));
    return this;
  }),
  // Minimal EventDispatcher surface — MaterialManager subscribes to
  // the synchronous `dispose` event so it can clean up automatically.
  _listeners: {} as Record<string, ((...args: unknown[]) => void)[]>,
  addEventListener: vi.fn(function (
    this: any,
    type: string,
    listener: (...args: unknown[]) => void
  ) {
    this._listeners ??= {};
    (this._listeners[type] ??= []).push(listener);
  }),
  removeEventListener: vi.fn(function (
    this: any,
    type: string,
    listener: (...args: unknown[]) => void
  ) {
    if (!this._listeners?.[type]) return;
    this._listeners[type] = this._listeners[type].filter((l: unknown) => l !== listener);
  }),
  dispatchEvent: vi.fn(function (this: any, event: { type: string }) {
    this._listeners?.[event.type]?.forEach((l: (e: unknown) => void) => l(event));
  }),
  dispose: vi.fn(function (this: any) {
    this.dispatchEvent?.({ type: 'dispose', target: this });
  }),
}));

/**
 * Vitest mock of THREE.ShaderMaterial. Extends the Material mock with
 * defines/uniforms/vertex+fragmentShader from its parameters and overrides
 * `clone` to deep-copy shader source, uniform values (cloning clonable
 * uniform objects), and the blending/side/depth flags — matching the real
 * clone semantics the material tests depend on.
 */
export const ShaderMaterial = vi.fn().mockImplementation((parameters: any = {}) => {
  const mat = Material();
  mat.type = 'ShaderMaterial';
  mat.defines = parameters.defines || {};
  mat.uniforms = parameters.uniforms || {};
  mat.vertexShader = parameters.vertexShader || '';
  mat.fragmentShader = parameters.fragmentShader || '';
  mat.linewidth = 1;
  mat.wireframe = false;
  mat.wireframeLinewidth = 1;
  mat.lights = false;
  mat.clipping = false;
  mat.extensions = {
    derivatives: false,
    fragDepth: false,
    drawBuffers: false,
    shaderTextureLOD: false,
  };

  // Copy parameters
  Object.assign(mat, parameters);

  // Override clone to deep-copy shader-specific properties
  mat.clone = vi.fn(function (this: any) {
    const cloned = new (ShaderMaterial as any)({
      vertexShader: this.vertexShader,
      fragmentShader: this.fragmentShader,
      defines: { ...this.defines },
      uniforms: Object.fromEntries(
        Object.entries(this.uniforms).map(([k, u]: [string, any]) => [
          k,
          {
            value:
              u &&
              typeof u.value === 'object' &&
              u.value !== null &&
              typeof u.value.clone === 'function'
                ? u.value.clone()
                : u?.value,
          },
        ])
      ),
    });
    // Copy material base properties
    cloned.transparent = this.transparent;
    cloned.depthWrite = this.depthWrite;
    cloned.depthTest = this.depthTest;
    cloned.toneMapped = this.toneMapped;
    cloned.blending = this.blending;
    cloned.blendEquation = this.blendEquation;
    cloned.blendSrc = this.blendSrc;
    cloned.blendDst = this.blendDst;
    cloned.blendEquationAlpha = this.blendEquationAlpha;
    cloned.blendSrcAlpha = this.blendSrcAlpha;
    cloned.blendDstAlpha = this.blendDstAlpha;
    cloned.side = this.side;
    cloned.userData = JSON.parse(JSON.stringify(this.userData || {}));
    return cloned;
  });

  return mat;
});

/**
 * Vitest mock of THREE.PointsMaterial: the Material mock with point-specific
 * fields (color/map/alphaMap/size/sizeAttenuation), overridable via parameters.
 */
export const PointsMaterial = vi.fn().mockImplementation((parameters: any = {}) => {
  const mat = Material();
  mat.type = 'PointsMaterial';
  mat.color = new (Color as any)(0xffffff);
  mat.map = null;
  mat.alphaMap = null;
  mat.size = 1;
  mat.sizeAttenuation = true;

  Object.assign(mat, parameters);

  return mat;
});

// ============================================================================
// Mesh and Points Classes
// ============================================================================

/**
 * Vitest mock of THREE.Mesh: an Object3D (`type` `'Mesh'`) holding the supplied
 * geometry and material references.
 */
export const Mesh = vi.fn().mockImplementation((geometry?: any, material?: any) => {
  const obj = new Object3D();
  (obj as any).type = 'Mesh';
  (obj as any).geometry = geometry || null;
  (obj as any).material = material || null;
  return obj;
});

/**
 * Vitest mock of THREE.Points: an Object3D (`type` `'Points'`) holding the
 * supplied geometry and material references.
 */
export const Points = vi.fn().mockImplementation((geometry?: any, material?: any) => {
  const obj = new Object3D();
  (obj as any).type = 'Points';
  (obj as any).geometry = geometry || null;
  (obj as any).material = material || null;
  return obj;
});

// ============================================================================
// Renderer Classes
// ============================================================================

/**
 * Vitest mock of THREE.WebGLRenderer. Returns a renderer-shaped object whose
 * every draw/state method (render/setSize/clear/etc.) is a no-op spy, with a
 * real (or created) canvas as `domElement` and a stub `getContext` reporting
 * the requested context attributes. Lets rendering code run headlessly without
 * a real WebGL context.
 */
export const WebGLRenderer = vi.fn().mockImplementation((parameters: any = {}) => {
  const canvas = parameters.canvas || document.createElement('canvas');

  return {
    domElement: canvas,
    context: null,
    autoClear: true,
    autoClearColor: true,
    autoClearDepth: true,
    autoClearStencil: true,
    sortObjects: true,
    clippingPlanes: [],
    localClippingEnabled: false,
    outputColorSpace: SRGBColorSpace,
    toneMapping: NoToneMapping,
    toneMappingExposure: 1,
    shadowMap: {
      enabled: false,
      autoUpdate: true,
      needsUpdate: false,
      type: PCFShadowMap,
    },
    xr: {
      enabled: false,
      isPresenting: false,
    },

    setSize: vi.fn(),
    setPixelRatio: vi.fn(),
    getPixelRatio: vi.fn(() => 1),
    getSize: vi.fn((target: any) => {
      if (target) {
        target.set(canvas.width, canvas.height);
        return target;
      }
      return new (Vector2 as any)(canvas.width, canvas.height);
    }),
    getDrawingBufferSize: vi.fn((target: any) => {
      if (target) {
        target.set(canvas.width, canvas.height);
        return target;
      }
      return new (Vector2 as any)(canvas.width, canvas.height);
    }),
    setViewport: vi.fn(),
    getViewport: vi.fn((target: any) => {
      return target.set(0, 0, canvas.width, canvas.height);
    }),
    setScissor: vi.fn(),
    setScissorTest: vi.fn(),
    setClearColor: vi.fn(),
    getClearColor: vi.fn(() => new (Color as any)(0x000000)),
    setClearAlpha: vi.fn(),
    getClearAlpha: vi.fn(() => 1),
    clear: vi.fn(),
    clearColor: vi.fn(),
    clearDepth: vi.fn(),
    clearStencil: vi.fn(),
    render: vi.fn(),
    renderBufferDirect: vi.fn(),
    compile: vi.fn(),
    dispose: vi.fn(),
    getContext: vi.fn(() => ({
      getContextAttributes: () => ({
        alpha: parameters.alpha ?? true,
        antialias: parameters.antialias ?? false,
        depth: parameters.depth ?? true,
        premultipliedAlpha: parameters.premultipliedAlpha ?? true,
        preserveDrawingBuffer: parameters.preserveDrawingBuffer ?? false,
        stencil: parameters.stencil ?? true,
      }),
      canvas,
    })),
    forceContextLoss: vi.fn(),
    forceContextRestore: vi.fn(),
    getMaxAnisotropy: vi.fn(() => 16),
    getPrecision: vi.fn(() => 'highp'),
    readRenderTargetPixels: vi.fn(),
    copyFramebufferToTexture: vi.fn(),
    copyTextureToTexture: vi.fn(),
    initTexture: vi.fn(),
    resetState: vi.fn(),
  };
});

// ============================================================================
// Texture Classes
// ============================================================================

/**
 * Vitest mock of THREE.Texture. A plain data holder carrying the standard
 * wrap/filter/format/encoding fields and stubbed updateMatrix/clone/copy/
 * dispose; no GPU upload happens.
 */
export const Texture = vi.fn().mockImplementation((image?: any) => ({
  id: Math.random(),
  uuid: Math.random().toString(36),
  name: '',
  image: image || null,
  mipmaps: [],
  mapping: 300, // UVMapping
  wrapS: 1001, // ClampToEdgeWrapping
  wrapT: 1001,
  magFilter: 1006, // LinearFilter
  minFilter: 1008, // LinearMipmapLinearFilter
  anisotropy: 1,
  format: 1023, // RGBAFormat
  type: 1009, // UnsignedByteType
  offset: new (Vector2 as any)(0, 0),
  repeat: new (Vector2 as any)(1, 1),
  center: new (Vector2 as any)(0, 0),
  rotation: 0,
  matrixAutoUpdate: true,
  matrix: new (Matrix4 as any)(),
  generateMipmaps: true,
  premultiplyAlpha: false,
  flipY: true,
  unpackAlignment: 4,
  encoding: LinearEncoding,
  version: 0,
  needsUpdate: false,
  userData: {},

  updateMatrix: vi.fn(),
  clone: vi.fn(),
  copy: vi.fn(),
  dispose: vi.fn(),
}));

/**
 * Vitest mock of THREE.WebGLRenderTarget. Holds width/height/depth plus a mock
 * Texture and viewport/scissor vectors. On a size change `setSize` records the
 * new dimensions, but it also writes through `texture.image` (which is `null`
 * on this mock's Texture), so resizing currently throws — tests should
 * construct render targets at their final size rather than resizing them.
 */
export const WebGLRenderTarget = vi
  .fn()
  .mockImplementation((width = 1, height = 1, options: any = {}) => ({
    width,
    height,
    depth: 1,
    texture: new (Texture as any)(),
    depthBuffer: options.depthBuffer ?? true,
    stencilBuffer: options.stencilBuffer ?? false,
    depthTexture: options.depthTexture ?? null,
    samples: options.samples ?? 0,
    scissorTest: false,
    viewport: new (Vector4 as any)(0, 0, width, height),
    scissor: new (Vector4 as any)(0, 0, width, height),

    setSize: vi.fn(function (this: any, w: number, h: number, depth = 1) {
      if (this.width !== w || this.height !== h || this.depth !== depth) {
        this.width = w;
        this.height = h;
        this.depth = depth;
        this.texture.image.width = w;
        this.texture.image.height = h;
        this.dispose();
      }
    }),
    clone: vi.fn(),
    copy: vi.fn(),
    dispose: vi.fn(),
  }));

// ============================================================================
// Utility Classes
// ============================================================================

/**
 * Vitest mock of THREE.Frustum. Carries six planes but every intersection test
 * (intersectsObject/Box/Sphere/containsPoint) returns true, so culling never
 * hides geometry in tests.
 */
export const Frustum = vi.fn().mockImplementation(() => ({
  planes: [
    { normal: new (Vector3 as any)(), constant: 0 },
    { normal: new (Vector3 as any)(), constant: 0 },
    { normal: new (Vector3 as any)(), constant: 0 },
    { normal: new (Vector3 as any)(), constant: 0 },
    { normal: new (Vector3 as any)(), constant: 0 },
    { normal: new (Vector3 as any)(), constant: 0 },
  ],
  setFromProjectionMatrix: vi.fn(),
  intersectsObject: vi.fn(() => true),
  intersectsBox: vi.fn(() => true),
  intersectsSphere: vi.fn(() => true),
  containsPoint: vi.fn(() => true),
  clone: vi.fn(),
  copy: vi.fn(),
}));

/**
 * Vitest mock of THREE.Raycaster. Exposes the ray/near/far/params shape, but
 * setFromCamera is a no-op and intersectObject(s) always return an empty array
 * (no real intersection math).
 */
export const Raycaster = vi.fn().mockImplementation(() => ({
  ray: {
    origin: new (Vector3 as any)(),
    direction: new (Vector3 as any)(),
  },
  near: 0,
  far: Infinity,
  camera: null,
  layers: { mask: 1 },
  params: {
    Mesh: {},
    Line: { threshold: 1 },
    LOD: {},
    Points: { threshold: 1 },
    Sprite: {},
  },

  set: vi.fn(),
  setFromCamera: vi.fn(),
  intersectObject: vi.fn(() => []),
  intersectObjects: vi.fn(() => []),
}));

/**
 * Vitest mock of THREE's Timer (addons). Tracks delta/elapsed from
 * performance.now() with a working reset/update/getDelta/getElapsed and
 * timescale, so frame-timing code advances realistically in tests.
 */
export const Timer = vi.fn().mockImplementation(() => ({
  _previousTime: 0,
  _delta: 0,
  _elapsed: 0,
  _timescale: 1,

  connect: vi.fn(),
  dispose: vi.fn(),
  reset: vi.fn(function (this: any) {
    this._delta = 0;
    this._elapsed = 0;
  }),
  update: vi.fn(function (this: any) {
    const now = performance.now();
    this._delta = this._previousTime === 0 ? 0 : (now - this._previousTime) / 1000;
    this._previousTime = now;
    this._elapsed += this._delta;
    return this;
  }),
  getDelta: vi.fn(function (this: any) {
    return this._delta;
  }),
  getElapsed: vi.fn(function (this: any) {
    return this._elapsed;
  }),
  getTimescale: vi.fn(function (this: any) {
    return this._timescale;
  }),
  setTimescale: vi.fn(function (this: any, value: number) {
    this._timescale = value;
    return this;
  }),
}));

// ============================================================================
// Constants
// ============================================================================

/** THREE.js blending-mode constants (mirror the real numeric enum values). */
export const NoBlending = 0;
export const NormalBlending = 1;
export const AdditiveBlending = 2;
export const SubtractiveBlending = 3;
export const MultiplyBlending = 4;
export const CustomBlending = 5;

/** THREE.js depth-comparison-function constants. */
export const NeverDepth = 0;
export const AlwaysDepth = 1;
export const LessDepth = 2;
export const LessEqualDepth = 3;
export const EqualDepth = 4;
export const GreaterEqualDepth = 5;
export const GreaterDepth = 6;
export const NotEqualDepth = 7;

/** THREE.js material face-side constants. */
export const FrontSide = 0;
export const BackSide = 1;
export const DoubleSide = 2;

/** THREE.js legacy vertex/face color-mode constants. */
export const NoColors = 0;
export const FaceColors = 1;
export const VertexColors = 2;

/** THREE.js shading-model constants. */
export const FlatShading = 1;
export const SmoothShading = 2;

/** THREE.js texture mapping/wrapping/filtering constants. */
export const UVMapping = 300;
export const ClampToEdgeWrapping = 1001;
export const RepeatWrapping = 1000;
export const MirroredRepeatWrapping = 1002;
export const NearestFilter = 1003;
export const LinearFilter = 1006;
export const LinearMipmapLinearFilter = 1008;

/** THREE.js texture pixel-format constants. */
export const AlphaFormat = 1019;
export const RGBFormat = 1022;
export const RGBAFormat = 1023;

/** THREE.js texture pixel-type constants. */
export const UnsignedByteType = 1009;
export const FloatType = 1015;
export const HalfFloatType = 1016;

/** THREE.js legacy texture-encoding constants. */
export const LinearEncoding = 3000;
export const sRGBEncoding = 3001;

/** THREE.js color-space string constants (the newer color-management API). */
export const NoColorSpace = '';
export const SRGBColorSpace = 'srgb';
export const LinearSRGBColorSpace = 'srgb-linear';

/** THREE.js tone-mapping-operator constants. */
export const NoToneMapping = 0;
export const LinearToneMapping = 1;
export const ReinhardToneMapping = 2;
export const CineonToneMapping = 3;
export const ACESFilmicToneMapping = 4;

/** THREE.js shadow-map-algorithm constants. */
export const BasicShadowMap = 0;
export const PCFShadowMap = 1;
export const VSMShadowMap = 3;

/** THREE.js buffer-attribute usage-hint constants (GL enum values). */
export const StaticDrawUsage = 35044;
export const DynamicDrawUsage = 35048;
export const StreamDrawUsage = 35040;

/** THREE.js GLSL-version constant selecting GLSL ES 3.00 shaders. */
export const GLSL3 = '300 es';

// ============================================================================
// Export All
// ============================================================================

export default {
  // Math
  Vector2,
  Vector3,
  Vector4,
  Quaternion,
  Euler,
  Matrix4,
  Box3,
  Color,

  // Scene graph
  Object3D,
  Group,
  Scene,

  // Cameras
  Camera,
  PerspectiveCamera,
  OrthographicCamera,

  // Geometry
  BufferAttribute,
  BufferGeometry,
  BoxGeometry,

  // Materials
  Material,
  ShaderMaterial,
  PointsMaterial,

  // Mesh
  Mesh,
  Points,

  // Renderer
  WebGLRenderer,

  // Textures
  Texture,
  WebGLRenderTarget,

  // Utilities
  Frustum,
  Raycaster,
  Timer,

  // Constants
  NoBlending,
  NormalBlending,
  AdditiveBlending,
  SubtractiveBlending,
  MultiplyBlending,
  CustomBlending,
  NeverDepth,
  AlwaysDepth,
  LessDepth,
  LessEqualDepth,
  EqualDepth,
  GreaterEqualDepth,
  GreaterDepth,
  NotEqualDepth,
  FrontSide,
  BackSide,
  DoubleSide,
  NoColors,
  FaceColors,
  VertexColors,
  FlatShading,
  SmoothShading,
  UVMapping,
  ClampToEdgeWrapping,
  RepeatWrapping,
  MirroredRepeatWrapping,
  NearestFilter,
  LinearFilter,
  LinearMipmapLinearFilter,
  AlphaFormat,
  RGBFormat,
  RGBAFormat,
  UnsignedByteType,
  FloatType,
  HalfFloatType,
  LinearEncoding,
  sRGBEncoding,
  NoColorSpace,
  SRGBColorSpace,
  LinearSRGBColorSpace,
  NoToneMapping,
  LinearToneMapping,
  ReinhardToneMapping,
  CineonToneMapping,
  ACESFilmicToneMapping,
  BasicShadowMap,
  PCFShadowMap,
  VSMShadowMap,
  StaticDrawUsage,
  DynamicDrawUsage,
  StreamDrawUsage,
  GLSL3,
};
