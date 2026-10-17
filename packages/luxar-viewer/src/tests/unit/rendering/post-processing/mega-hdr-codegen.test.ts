import * as THREE from 'three';
import { context } from 'three/tsl';
import { WGSLNodeBuilder, type NodeMaterial } from 'three/webgpu';
import { describe, expect, it } from 'vitest';

import { MegaShaderTSLMaterial } from '../../../../rendering/post-processing/mega/material-tsl';

function fragmentFor(material: MegaShaderTSLMaterial): string {
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
      compatibilityMode: false,
      capabilities: { getUniformBufferLimit: () => 65536 },
      utils: { getTextureSampleData: () => ({ samples: 1, primarySamples: 1, isMSAA: false }) },
    },
  };
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
  const builder = new WGSLNodeBuilder(
    mesh,
    renderer as ConstructorParameters<typeof WGSLNodeBuilder>[1]
  ) as unknown as {
    camera: THREE.Camera;
    scene: THREE.Scene;
    material: NodeMaterial;
    fragmentShader: string;
    build(): void;
  };
  builder.camera = new THREE.PerspectiveCamera();
  builder.scene = new THREE.Scene();
  builder.material = material;
  builder.build();
  return builder.fragmentShader;
}

describe('mega-shader HDR WGSL', () => {
  it.each([
    ['Reinhard', THREE.ReinhardToneMapping],
    ['Cineon', THREE.CineonToneMapping],
    ['ACES', THREE.ACESFilmicToneMapping],
    ['AgX', THREE.AgXToneMapping],
    ['Neutral', THREE.NeutralToneMapping],
  ])('emits the peak uniform and exponential shoulder for %s', (_name, toneMapping) => {
    const material = new MegaShaderTSLMaterial({ toneMapping, hdrOutputPeak: 3 });
    const wgsl = fragmentFor(material);
    const peak = wgsl.match(/object\.nodeUniform(\d+) > 1\.0/);
    expect(peak).not.toBeNull();
    expect(wgsl).toContain(`nodeUniform${peak?.[1]} : f32`);
    expect(wgsl).toContain('exp(');
    expect(wgsl).toMatch(/> 1\.0/);
    material.dispose();
  });

  it('keeps the shoulder out of SDR and linear-LDR capture builds', () => {
    const hdr = new MegaShaderTSLMaterial({ hdrOutputPeak: 3 });
    const sdr = new MegaShaderTSLMaterial();
    const hdrWgsl = fragmentFor(hdr);
    const sdrWgsl = fragmentFor(sdr);
    hdr.toggleLinearLdrCapture(true);
    const captureWgsl = fragmentFor(hdr);
    expect(hdrWgsl).toContain('exp(');
    expect(sdrWgsl).not.toContain('exp(');
    expect(captureWgsl).not.toContain('exp(');
    hdr.dispose();
    sdr.dispose();
  });

  it.each([THREE.LinearToneMapping, THREE.NoToneMapping])(
    'clamps Linear/None at the peak without the shoulder',
    (toneMapping) => {
      const material = new MegaShaderTSLMaterial({ toneMapping, hdrOutputPeak: 3 });
      const wgsl = fragmentFor(material);
      expect(wgsl).not.toContain('exp(');
      expect(wgsl).toContain('clamp(');
      material.dispose();
    }
  );
});
