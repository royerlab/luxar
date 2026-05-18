/**
 * Unit tests for the effect-orchestrator partitioner.
 */

import { describe, it, expect } from 'vitest';
import {
  buildOrderedEffects,
  isConvolutionEffectName,
  isUVTransformEffectName,
  partitionEffectsIntoPasses,
} from '../../../../rendering/post-processing/effect-orchestrator';

const e = (name: string) => ({ effect: name, name }) as { effect: string; name: string };

describe('isUVTransformEffectName', () => {
  it('classifies ChromaticLensDistortion as UV-transform', () => {
    expect(isUVTransformEffectName('ChromaticLensDistortion')).toBe(true);
  });

  it('returns false for non-UV-transform names', () => {
    for (const name of ['Bloom', 'DOF', 'AO', 'ToneMapping', 'Vignette', 'SMAA', 'FXAA']) {
      expect(isUVTransformEffectName(name)).toBe(false);
    }
  });
});

describe('isConvolutionEffectName', () => {
  it('classifies Bloom as convolution', () => {
    expect(isConvolutionEffectName('Bloom')).toBe(true);
  });

  it('returns false for non-convolution names', () => {
    for (const name of ['ChromaticLensDistortion', 'DOF', 'ToneMapping', 'Vignette', 'SMAA']) {
      expect(isConvolutionEffectName(name)).toBe(false);
    }
  });
});

describe('partitionEffectsIntoPasses', () => {
  it('returns empty partition for no effects', () => {
    expect(partitionEffectsIntoPasses([])).toEqual({
      passA: [],
      passB: [],
      passANames: [],
      passBNames: [],
      splitAt: null,
    });
  });

  it('all effects fit in pass A when none are convolution + UV-transform together', () => {
    const partition = partitionEffectsIntoPasses([
      e('Bloom'),
      e('DOF'),
      e('AO'),
      e('ToneMapping'),
      e('Vignette'),
      e('SMAA'),
    ]);
    expect(partition.splitAt).toBeNull();
    expect(partition.passANames).toEqual(['Bloom', 'DOF', 'AO', 'ToneMapping', 'Vignette', 'SMAA']);
    expect(partition.passBNames).toEqual([]);
  });

  it('Bloom + ChromaticLensDistortion: split at the second offender', () => {
    const partition = partitionEffectsIntoPasses([e('Bloom'), e('ChromaticLensDistortion')]);
    expect(partition.passANames).toEqual(['Bloom']);
    expect(partition.passBNames).toEqual(['ChromaticLensDistortion']);
    expect(partition.splitAt).toBe('ChromaticLensDistortion');
  });

  it('ChromaticLensDistortion + Bloom: split at the second offender', () => {
    const partition = partitionEffectsIntoPasses([e('ChromaticLensDistortion'), e('Bloom')]);
    expect(partition.passANames).toEqual(['ChromaticLensDistortion']);
    expect(partition.passBNames).toEqual(['Bloom']);
    expect(partition.splitAt).toBe('Bloom');
  });

  it('once split, all remaining effects go to Pass B', () => {
    const partition = partitionEffectsIntoPasses([
      e('Bloom'),
      e('ChromaticLensDistortion'),
      e('ToneMapping'),
      e('Vignette'),
      e('SMAA'),
    ]);
    expect(partition.passANames).toEqual(['Bloom']);
    expect(partition.passBNames).toEqual([
      'ChromaticLensDistortion',
      'ToneMapping',
      'Vignette',
      'SMAA',
    ]);
    expect(partition.splitAt).toBe('ChromaticLensDistortion');
  });

  it('single Bloom only: no split', () => {
    const partition = partitionEffectsIntoPasses([e('Bloom')]);
    expect(partition.passANames).toEqual(['Bloom']);
    expect(partition.passBNames).toEqual([]);
    expect(partition.splitAt).toBeNull();
  });

  it('single ChromaticLensDistortion only: no split', () => {
    const partition = partitionEffectsIntoPasses([e('ChromaticLensDistortion')]);
    expect(partition.passANames).toEqual(['ChromaticLensDistortion']);
    expect(partition.passBNames).toEqual([]);
    expect(partition.splitAt).toBeNull();
  });

  it('preserves the input effect references in their assigned passes', () => {
    const bloom = { kind: 'b' };
    const chromatic = { kind: 'c' };
    const tone = { kind: 't' };
    const partition = partitionEffectsIntoPasses([
      { effect: bloom, name: 'Bloom' },
      { effect: chromatic, name: 'ChromaticLensDistortion' },
      { effect: tone, name: 'ToneMapping' },
    ]);
    expect(partition.passA).toEqual([bloom]);
    expect(partition.passB).toEqual([chromatic, tone]);
  });

  it('input list is iterated in order — splitAt reports the FIRST offender', () => {
    // Two convolution + UV-transform pairs in a single list — the partitioner
    // splits at the first occurrence and dumps everything past it into Pass B.
    const partition = partitionEffectsIntoPasses([
      e('Bloom'),
      e('ChromaticLensDistortion'),
      e('Bloom'),
      e('ChromaticLensDistortion'),
    ]);
    expect(partition.passANames).toEqual(['Bloom']);
    expect(partition.passBNames).toEqual([
      'ChromaticLensDistortion',
      'Bloom',
      'ChromaticLensDistortion',
    ]);
    expect(partition.splitAt).toBe('ChromaticLensDistortion');
  });
});

describe('buildOrderedEffects', () => {
  it('empty slots produce an empty list', () => {
    const ordered = buildOrderedEffects<string>({ smaaEnabled: false, fxaaEnabled: false });
    expect(ordered).toEqual([]);
  });

  it('only-defined slots contribute, in canonical visual order', () => {
    const ordered = buildOrderedEffects<string>({
      bloom: 'b',
      toneMapping: 'tm',
      smaaEnabled: false,
      fxaaEnabled: false,
    });
    expect(ordered.map((e) => e.name)).toEqual(['Bloom', 'ToneMapping']);
  });

  it('full pipeline yields the canonical order', () => {
    const ordered = buildOrderedEffects<string>({
      bloom: 'b',
      dof: 'd',
      ao: 'ao',
      chromaticLensDistortion: 'cld',
      detectorNoise: 'dn',
      toneMapping: 'tm',
      vignette: 'v',
      smaa: 's',
      fxaa: 'f',
      smaaEnabled: true,
      fxaaEnabled: true,
    });
    expect(ordered.map((e) => e.name)).toEqual([
      'Bloom',
      'DOF',
      'AO',
      'ChromaticLensDistortion',
      'DetectorNoise',
      'ToneMapping',
      'Vignette',
      'SMAA', // SMAA wins over FXAA when both flags are true
    ]);
  });

  it('AA selection: SMAA wins over FXAA when both enabled', () => {
    const ordered = buildOrderedEffects<string>({
      smaa: 's',
      fxaa: 'f',
      smaaEnabled: true,
      fxaaEnabled: true,
    });
    expect(ordered.map((e) => e.name)).toEqual(['SMAA']);
  });

  it('AA selection: FXAA used when smaaEnabled=false', () => {
    const ordered = buildOrderedEffects<string>({
      smaa: 's',
      fxaa: 'f',
      smaaEnabled: false,
      fxaaEnabled: true,
    });
    expect(ordered.map((e) => e.name)).toEqual(['FXAA']);
  });

  it('AA selection: skipped when both flags are false', () => {
    const ordered = buildOrderedEffects<string>({
      smaa: 's',
      fxaa: 'f',
      smaaEnabled: false,
      fxaaEnabled: false,
    });
    expect(ordered).toEqual([]);
  });

  it('AA selection: smaaEnabled=true but smaa undefined falls through to FXAA', () => {
    const ordered = buildOrderedEffects<string>({
      fxaa: 'f',
      smaaEnabled: true,
      fxaaEnabled: true,
    });
    expect(ordered.map((e) => e.name)).toEqual(['FXAA']);
  });

  it('output passes through partitionEffectsIntoPasses correctly', () => {
    const ordered = buildOrderedEffects<string>({
      bloom: 'bloom-marker',
      chromaticLensDistortion: 'cld-marker',
      smaaEnabled: false,
      fxaaEnabled: false,
    });
    const partition = partitionEffectsIntoPasses(ordered);
    // Bloom is convolution, ChromaticLensDistortion is UV-transform → split.
    expect(partition.passANames).toEqual(['Bloom']);
    expect(partition.passBNames).toEqual(['ChromaticLensDistortion']);
    expect(partition.splitAt).toBe('ChromaticLensDistortion');
  });
});
