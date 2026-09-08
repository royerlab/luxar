/**
 * Sound popover — the rail's right-click surface for the sound layer: master
 * gain and the three bus gains, plus the panning model. The left-click on the
 * same button is the mute toggle. Only reachable when the loaded scene has sound
 * nodes (the rail hides the button otherwise).
 *
 * @module ui/rail-panels/audio-popover
 */

import { makePopoverGui } from './popover-gui';
import type { AudioBusName, AudioPatch, AudioState, PanningModel } from '../../types/audio';

/** The engine surface the popover drives (a subset of `AudioEngine`). */
export interface AudioPopoverContext {
  getState(): AudioState;
  setAudio(patch: AudioPatch): void;
}

const BUS_LABELS: Record<AudioBusName, string> = {
  ambient: 'Ambient bed',
  voice: 'Voice (narration)',
  effects: 'Effects',
};

/** Build the popover into `host`; returns the teardown the rail runs on close. */
export function buildAudioPopover(host: HTMLElement, ctx: AudioPopoverContext): () => void {
  const gui = makePopoverGui(host, 'Sound');
  const state = ctx.getState();
  const model = {
    muted: state.muted,
    masterGain: state.masterGain,
    panningModel: state.panningModel as PanningModel,
    ambient: state.buses.ambient,
    voice: state.buses.voice,
    effects: state.buses.effects,
  };

  gui
    .add(model, 'muted')
    .name('Mute')
    .onChange((v: boolean) => ctx.setAudio({ muted: v }));
  gui
    .add(model, 'masterGain', 0, 1.5, 0.01)
    .name('Master gain')
    .onChange((v: number) => ctx.setAudio({ masterGain: v }));
  gui
    .add(model, 'panningModel', ['equalpower', 'HRTF'])
    .name('Panning')
    .onChange((v: PanningModel) => ctx.setAudio({ panningModel: v }));

  const buses = gui.addFolder('Buses');
  for (const bus of ['ambient', 'voice', 'effects'] as AudioBusName[]) {
    buses
      .add(model, bus, 0, 1.5, 0.01)
      .name(BUS_LABELS[bus])
      .onChange((v: number) => ctx.setAudio({ buses: { [bus]: v } }));
  }

  return () => gui.destroy();
}
