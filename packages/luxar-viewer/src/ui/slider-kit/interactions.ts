import { applyModifierTier } from '../../utils/cross-layer/modifier-tiers';
import { normalizeWheelDeltaWithAxisFallback } from '../../utils/wheel-delta';
import { SLIDER_INTERACTION_HINT } from './hints';

/** Callbacks and values required by the shared slider gesture binding. */
export interface SliderInteractionOptions {
  input: HTMLInputElement;
  baseStep: number;
  resetValue: () => void;
  getValue: () => number;
  setValue: (value: number) => void;
}

/** Attach wheel, arrow-key, and double-click reset interactions to a slider. */
export function attachSliderInteractions(options: SliderInteractionOptions): () => void {
  const { input } = options;
  const applyStep = (direction: 1 | -1, event: { shiftKey: boolean; ctrlKey: boolean }): void => {
    const step = applyModifierTier(options.baseStep, event);
    options.setValue(options.getValue() + direction * step);
  };
  const onWheel = (event: WheelEvent): void => {
    if (input.disabled) return;
    event.preventDefault();
    const delta = normalizeWheelDeltaWithAxisFallback(event);
    if (delta !== 0) applyStep(delta < 0 ? 1 : -1, event);
  };
  const onKeyDown = (event: KeyboardEvent): void => {
    if (input.disabled) return;
    const direction =
      event.key === 'ArrowRight' || event.key === 'ArrowUp'
        ? 1
        : event.key === 'ArrowLeft' || event.key === 'ArrowDown'
          ? -1
          : 0;
    if (direction === 0) return;
    event.preventDefault();
    event.stopPropagation();
    applyStep(direction, event);
  };
  const onDoubleClick = (): void => {
    if (input.disabled) return;
    options.resetValue();
  };
  input.title = SLIDER_INTERACTION_HINT;
  input.addEventListener('wheel', onWheel, { passive: false });
  input.addEventListener('keydown', onKeyDown);
  input.addEventListener('dblclick', onDoubleClick);
  return () => {
    input.removeEventListener('wheel', onWheel);
    input.removeEventListener('keydown', onKeyDown);
    input.removeEventListener('dblclick', onDoubleClick);
  };
}
