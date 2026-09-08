/**
 * `FoaDecoder` — the graph shape (4-ch splitter, 3×3 rotation gains, virtual
 * cardioid stereo decode) and the rotation geometry against the camera.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import * as THREE from 'three';
import { FoaAudio, FoaDecoder, foaRotationForCamera } from '../../../audio/foa-decoder';
import {
  FakeAudioBuffer,
  FakeAudioContext,
  FakeChannelNode,
  FakeGainNode,
  installFakeAudioContext,
} from '../../mocks/fake-audio-context.mock';

let ctx: FakeAudioContext;

beforeEach(() => {
  ctx = installFakeAudioContext('running');
});

function yaw(deg: number): THREE.Quaternion {
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), (deg * Math.PI) / 180);
}
function pitch(deg: number): THREE.Quaternion {
  return new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), (deg * Math.PI) / 180);
}
function rotated(q: THREE.Quaternion, ambix: [number, number, number]): number[] {
  return new THREE.Vector3(...ambix)
    .applyMatrix3(foaRotationForCamera(q))
    .toArray()
    .map((c) => Math.round(c * 1000) / 1000);
}

describe('foaRotationForCamera (AmbiX x=front, y=left, z=up)', () => {
  it('an identity camera leaves the field alone', () => {
    expect(rotated(new THREE.Quaternion(), [1, 0, 0])).toEqual([1, 0, 0]);
    expect(rotated(new THREE.Quaternion(), [0, 1, 0])).toEqual([0, 1, 0]);
  });

  it("turning the camera 90° left puts a front source on the listener's right", () => {
    expect(rotated(yaw(90), [1, 0, 0])).toEqual([0, -1, 0]);
    // ... and a source on the world-left is now in front.
    expect(rotated(yaw(90), [0, 1, 0])).toEqual([1, 0, 0]);
  });

  it('pitching the camera up 90° puts a front source below the listener', () => {
    expect(rotated(pitch(90), [1, 0, 0])).toEqual([0, 0, -1]);
  });
});

describe('FoaDecoder graph', () => {
  it('splits four channels, rotates the three dipoles through nine gains, decodes to a stereo merger', () => {
    const decoder = new FoaDecoder(ctx as unknown as BaseAudioContext);
    const input = decoder.input as unknown as FakeChannelNode;
    const output = decoder.output as unknown as FakeChannelNode;
    expect(input.channels).toBe(4);
    expect(output.channels).toBe(2);
    // W passes to its bus once; each dipole channel feeds three rotation gains.
    const byOutput = new Map<number, number>();
    for (const link of input.channelLinks)
      byOutput.set(link.output, (byOutput.get(link.output) ?? 0) + 1);
    expect(byOutput.get(0)).toBe(1);
    expect(byOutput.get(1)).toBe(3);
    expect(byOutput.get(2)).toBe(3);
    expect(byOutput.get(3)).toBe(3);
    // Identity rotation at rest: the diagonal gains are 1, the rest 0.
    // The nine rotation gains are the ones feeding a sum bus (one output that is
    // not the merger); the sum buses themselves feed two decode gains each.
    const rotationGains = ctx.gains.filter(
      (g) => g.outputs.size === 1 && !g.outputs.has(output as never)
    );
    expect(rotationGains).toHaveLength(9);
    expect(rotationGains.filter((g) => g.gain.value === 1)).toHaveLength(3);
    expect(rotationGains.filter((g) => g.gain.value === 0)).toHaveLength(6);
    // Six decode gains land on the merger: three per ear.
    const decodeGains = ctx.gains.filter((g) => g.outputs.has(output as never));
    expect(decodeGains).toHaveLength(6);
    const leftInputs = decodeGains.filter((g) => g.connections[0].input === 0).length;
    const rightInputs = decodeGains.filter((g) => g.connections[0].input === 1).length;
    expect(leftInputs).toBe(3);
    expect(rightInputs).toBe(3);
  });

  it('setCameraQuaternion schedules the rotation onto the gains and skips unchanged rotations', () => {
    const decoder = new FoaDecoder(ctx as unknown as BaseAudioContext);
    const before = ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0);
    decoder.setCameraQuaternion(new THREE.Quaternion());
    expect(ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0)).toBe(before); // identity = at rest
    decoder.setCameraQuaternion(yaw(90));
    const after = ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0);
    expect(after).toBe(before + 9);
    // A front source now on the right: the (Y' ← X) gain is -1, (X' ← X) is 0.
    const targets = ctx.gains
      .flatMap((g) => g.gain.calls)
      .filter((c) => c.method === 'setTargetAtTime')
      .map((c) => Math.round((c.value ?? 0) * 1000) / 1000);
    expect(targets.filter((v) => v === -1)).toHaveLength(1);
    expect(targets.filter((v) => v === 1)).toHaveLength(2); // Z'←Z and X'←Y
    expect(decoder.rotation.elements.map((e) => Math.round(e * 1000) / 1000)).toEqual(
      foaRotationForCamera(yaw(90)).elements.map((e) => Math.round(e * 1000) / 1000)
    );
    decoder.setCameraQuaternion(yaw(90));
    expect(ctx.gains.reduce((n, g) => n + g.gain.calls.length, 0)).toBe(after);
  });

  it('FoaAudio routes source → decoder → gain, and disconnects symmetrically', () => {
    const listener = new THREE.AudioListener();
    const decoder = new FoaDecoder(ctx as unknown as BaseAudioContext);
    const audio = new FoaAudio(listener, decoder);
    audio.setBuffer(new FakeAudioBuffer(1, 48000, 4) as unknown as AudioBuffer);
    audio.play();
    const source = ctx.sources[0];
    expect(source.outputs.has(decoder.input as never)).toBe(true);
    expect(
      (decoder.output as unknown as FakeChannelNode).outputs.has(
        audio.gain as unknown as FakeGainNode
      )
    ).toBe(true);
    audio.disconnect();
    expect(source.outputs.has(decoder.input as never)).toBe(false);
    expect((decoder.output as unknown as FakeChannelNode).outputs.size).toBe(0);
    decoder.dispose();
  });
});
