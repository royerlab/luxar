/**
 * First-order ambisonic (FOA) playback: a sound FIELD rather than a point
 * source, rotated against the camera so the field stays fixed to the world as
 * the visitor turns (`SOUND_SPEC.md` §6, Phase 4).
 *
 * Built from plain Web Audio nodes, no library:
 *
 * ```
 * 4-ch source ─► splitter ─► W ──────────────────────────► rot[W] ─┐
 *                        ├─► Y ─┬─ g[Y][Y] ─► rot[Y] ─┐           │
 *                        ├─► Z ─┼─ g[Y][Z] ─► rot[Y]  ├─ 3×3 gains │ decode gains ─► merger(2) ─► out
 *                        └─► X ─┴─ g[Y][X] ─► rot[Y]  ┘ (9 total)  │
 *                                              …                    ┘
 * ```
 *
 * The rotation of a first-order field IS the 3×3 rotation of its dipole
 * components (X, Y, Z), so nine gain nodes do the whole job; the omni W passes
 * straight through. The stereo decode is a pair of first-order virtual
 * cardioids at ±60° (`L = W + X·cos60 + Y·sin60`, mirrored for R), which keeps
 * front sources centred, hard left/right at the sides, and the back softer.
 *
 * Channel order and normalisation follow **AmbiX** (ACN order `W, Y, Z, X`,
 * SN3D), the interchange convention every current encoder writes. Axes: AmbiX
 * X = front, Y = left, Z = up. The bed is authored in the WORLD frame with
 * front = the viewer's −Z (what an identity camera looks at), left = −X,
 * up = +Y; {@link FoaDecoder.setCameraQuaternion} folds that mapping in, so a
 * source that was in front of an identity camera moves to the listener's
 * right after the camera yaws 90° to the left.
 *
 * Why not Omnitone (the spec's named library): it decodes to BINAURAL only —
 * wrong on the kiosk's room speakers — fetches its HRIR set from a CDN at
 * runtime (an offline kiosk cannot), and has not been maintained for years.
 * The virtual-microphone decode here is speaker-agnostic and dependency-free;
 * a headphone HRTF decode can be added behind the same `input`/`output` pair.
 *
 * @module audio/foa-decoder
 */

import * as THREE from 'three';

/** Number of channels of a first-order ambisonic stream. */
export const FOA_CHANNELS = 4;

/** ACN channel indices (AmbiX). */
const W = 0;
const Y = 1;
const Z = 2;
const X = 3;

/** Virtual microphone half-angle: cardioids at ±60° from the front. */
const MIC_AZIMUTH = Math.PI / 3;
/** Cardioid pattern weights: `w·W + w·(X cosθ + Y sinθ)`, halved for headroom. */
const MIC_GAIN = 0.5;

/** Smoothing time constant for gain changes as the camera turns (seconds). */
const ROTATION_SMOOTHING_S = 0.02;

/**
 * AmbiX axes → world axes (columns are the images of ambix X, Y, Z): front
 * (+X ambix) is world −Z, left (+Y ambix) is world −X, up (+Z ambix) is world +Y.
 */
const AMBIX_TO_WORLD = new THREE.Matrix3().set(0, -1, 0, 0, 0, 1, -1, 0, 0);
const WORLD_TO_AMBIX = AMBIX_TO_WORLD.clone().transpose();

const _cameraRotation = new THREE.Matrix3();
const _tmp = new THREE.Matrix3();
const _rotation = new THREE.Matrix3();

/**
 * The pure rotation the decoder applies to the field's (X, Y, Z) dipoles for a
 * camera with world quaternion `q`: `Mᵀ · R_cᵀ · M`, i.e. world → camera frame,
 * expressed in AmbiX axes. Exported for the tests, which pin the geometry.
 */
export function foaRotationForCamera(
  q: THREE.Quaternion,
  out = new THREE.Matrix3()
): THREE.Matrix3 {
  _cameraRotation.setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q)).transpose();
  _tmp.multiplyMatrices(_cameraRotation, AMBIX_TO_WORLD);
  return out.multiplyMatrices(WORLD_TO_AMBIX, _tmp);
}

/** The FOA rotate-and-decode graph. Wire `source → input` and `output → wherever`. */
export class FoaDecoder {
  /** Feed the 4-channel source here. */
  readonly input: ChannelSplitterNode;
  /** Stereo out. */
  readonly output: ChannelMergerNode;
  /** `rotationGains[i][j]`: rotated dipole i (Y, Z, X order as ACN 1..3) ← source dipole j. */
  private readonly rotationGains: GainNode[][] = [];
  private readonly nodes: AudioNode[] = [];
  private readonly lastRotation = new THREE.Matrix3();
  private disposed = false;

  constructor(readonly context: BaseAudioContext) {
    const ctx = context;
    this.input = ctx.createChannelSplitter(FOA_CHANNELS);
    this.output = ctx.createChannelMerger(2);
    // Rotated-channel sum buses.
    const rot: GainNode[] = [];
    for (let i = 0; i < FOA_CHANNELS; i++) {
      const g = ctx.createGain();
      g.gain.value = 1;
      rot.push(g);
      this.nodes.push(g);
    }
    // W passes through.
    this.input.connect(rot[W], W);
    // The 3×3 rotation over the dipoles (ACN 1..3 = Y, Z, X).
    for (const i of [Y, Z, X]) {
      const row: GainNode[] = [];
      for (const j of [Y, Z, X]) {
        const g = ctx.createGain();
        g.gain.value = i === j ? 1 : 0;
        this.input.connect(g, j);
        g.connect(rot[i]);
        row.push(g);
        this.nodes.push(g);
      }
      this.rotationGains.push(row);
    }
    // Virtual cardioid pair → stereo.
    const cos = Math.cos(MIC_AZIMUTH);
    const sin = Math.sin(MIC_AZIMUTH);
    const decode: Array<[number, number, number]> = [
      // [channel, left weight, right weight]
      [W, MIC_GAIN, MIC_GAIN],
      [X, MIC_GAIN * cos, MIC_GAIN * cos],
      [Y, MIC_GAIN * sin, -MIC_GAIN * sin],
    ];
    for (const [channel, left, right] of decode) {
      const gl = ctx.createGain();
      gl.gain.value = left;
      rot[channel].connect(gl);
      gl.connect(this.output, 0, 0);
      const gr = ctx.createGain();
      gr.gain.value = right;
      rot[channel].connect(gr);
      gr.connect(this.output, 0, 1);
      this.nodes.push(gl, gr);
    }
    this.lastRotation.identity();
  }

  /**
   * Point the field at the camera: the world-fixed bed is heard from the
   * camera's orientation. Cheap and idempotent — unchanged rotations return
   * without touching a param.
   */
  setCameraQuaternion(q: THREE.Quaternion): void {
    if (this.disposed) return;
    foaRotationForCamera(q, _rotation);
    const a = _rotation.elements;
    const b = this.lastRotation.elements;
    let changed = false;
    for (let k = 0; k < 9; k++) {
      if (Math.abs(a[k] - b[k]) > 1e-4) {
        changed = true;
        break;
      }
    }
    if (!changed) return;
    this.lastRotation.copy(_rotation);
    this.applyRotation(_rotation);
  }

  /** The rotation matrix currently applied (tests). */
  get rotation(): THREE.Matrix3 {
    return this.lastRotation.clone();
  }

  private applyRotation(m: THREE.Matrix3): void {
    // Matrix3.elements is column-major: element (row r, col c) = elements[c * 3 + r].
    // Our dipole order in the gain grid is (Y, Z, X) = ambix axes (y, z, x) →
    // matrix rows/cols in ambix (x, y, z) order are indices (1, 2, 0).
    const axisOf = [1, 2, 0]; // grid index → ambix axis index
    const now = this.context.currentTime;
    for (let gi = 0; gi < 3; gi++) {
      for (let gj = 0; gj < 3; gj++) {
        const value = m.elements[axisOf[gj] * 3 + axisOf[gi]];
        const param = this.rotationGains[gi][gj].gain;
        param.setTargetAtTime(value, now, ROTATION_SMOOTHING_S);
      }
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    try {
      this.input.disconnect();
      for (const n of this.nodes) n.disconnect();
      this.output.disconnect();
    } catch {
      /* a closed context throws on disconnect; nothing left to release */
    }
  }
}

/**
 * A three `Audio` whose source runs through a {@link FoaDecoder} before the
 * node gain — the same play/stop/loop/fade surface the other voices have, so
 * `SoundNode` treats a bed and a field alike. Non-spatial by construction: a
 * field has no position, it has an orientation.
 */
export class FoaAudio extends THREE.Audio<GainNode> {
  private foaConnected = false;

  constructor(
    listener: THREE.AudioListener,
    readonly decoder: FoaDecoder
  ) {
    super(listener);
  }

  override connect(): this {
    if (this.source === null) return this;
    this.source.connect(this.decoder.input);
    this.decoder.output.connect(this.getOutput());
    this.foaConnected = true;
    return this;
  }

  override disconnect(): this {
    if (!this.foaConnected) return this;
    this.source?.disconnect(this.decoder.input);
    this.decoder.output.disconnect(this.getOutput());
    this.foaConnected = false;
    return this;
  }
}
