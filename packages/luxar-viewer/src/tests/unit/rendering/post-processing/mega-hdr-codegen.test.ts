import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { MegaShaderTSLMaterial } from '../../../../rendering/post-processing/mega/material-tsl';
import { HDR_SLOPE_STEP } from '../../../../rendering/post-processing/mega/shader.tsl';
import { wgslFragmentFor } from '../../helpers/wgsl-build';

function fragmentFor(material: MegaShaderTSLMaterial): string {
  return wgslFragmentFor(material, new THREE.PerspectiveCamera(), new THREE.PlaneGeometry(2, 2));
}

describe('mega-shader HDR WGSL', () => {
  it.each([
    ['Reinhard', THREE.ReinhardToneMapping, 'reinhardToneMapping', 0.5, 'reinhardToneMapping'],
    ['Cineon', THREE.CineonToneMapping, 'cineonToneMapping', 0.5, '6.2'],
    ['ACES', THREE.ACESFilmicToneMapping, 'acesFilmicToneMapping', 0.5, '0.59719'],
    ['AgX', THREE.AgXToneMapping, 'agxToneMapping', 0.5, '12.47393'],
    ['Neutral', THREE.NeutralToneMapping, 'neutralToneMapping', 0.8, '0.76'],
    [
      'Reinhard with bloom',
      THREE.ReinhardToneMapping,
      'reinhardToneMapping',
      0.5,
      'reinhardToneMapping',
    ],
  ])(
    'emits the peak uniform and exponential shoulder for %s',
    (_name, toneMapping, fn, knee, operatorMarker) => {
      const material = new MegaShaderTSLMaterial({
        toneMapping,
        hdrOutputPeak: 3,
        ...(_name === 'Reinhard with bloom' ? { bloom: { enabled: true } } : {}),
      });
      const wgsl = fragmentFor(material);
      const kneePattern = String(knee).replace('.', '\\.');
      const branch = wgsl.match(
        new RegExp(
          `if \\( \\( \\( (nodeVar\\d+) > ${kneePattern} \\) && \\( object\\.(nodeUniform\\d+) > 1\\.0 \\) \\) \\) \\{`
        )
      );
      expect(branch).not.toBeNull();
      const [, level, peak] = branch!;
      expect(wgsl).toContain(`${peak} : f32`);
      expect(wgsl).toContain(operatorMarker);
      const body = wgsl.slice(branch!.index, wgsl.indexOf('\t} else {', branch!.index));
      const room = body.match(
        new RegExp(
          `(nodeVar\\d+) = max\\( \\( object\\.${peak} - ${fn}\\( vec3<f32>\\( ${kneePattern} \\), 1\\.0 \\)\\.x \\), 0\\.0001 \\);`
        )
      );
      expect(room).not.toBeNull();
      const slope = `( ( ( - ( ( ${fn}( vec3<f32>( ${knee + HDR_SLOPE_STEP} ), 1.0 ).x - ${fn}( vec3<f32>( ${knee - HDR_SLOPE_STEP} ), 1.0 ).x ) / ${2 * HDR_SLOPE_STEP} ) ) * ( ${level} - ${knee} ) ) / ${room![1]} )`;
      const shoulder = `( object.${peak} - ( ${room![1]} * exp( ${slope} ) ) )`;
      const extended = body.match(/(nodeVar\d+) = min\( \( (nodeVar\d+) \* vec3<f32>\(/);
      expect(extended).not.toBeNull();
      expect(body).toContain(
        `${extended![1]} = min( ( ${extended![2]} * vec3<f32>( ( ${shoulder} / max( ${fn}( vec3<f32>( ${level} ), 1.0 ).x, 0.000001 ) ) ) ), vec3<f32>( object.${peak} ) );`
      );
      expect(wgsl.slice(wgsl.indexOf('\t} else {', branch!.index))).toContain(
        `${extended![1]} = ${extended![2]};`
      );
      material.dispose();
    }
  );

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
      expect(wgsl).toMatch(/clamp\([^\n]+, vec3<f32>\( object\.nodeUniform\d+ \) \)/);
      material.dispose();
    }
  );
});
