/** Modifier state accepted from navigation options or DOM keyboard events. */
export interface ModifierState {
  shift?: boolean;
  ctrl?: boolean;
  shiftKey?: boolean;
  ctrlKey?: boolean;
}

/** Optional overrides for the standard fine and coarse tier factors. */
export interface ModifierTierConfig {
  fineDivisor?: number;
  coarseMultiplier?: number;
}

function isActive(value: boolean | undefined, eventValue: boolean | undefined): boolean {
  return value === true || eventValue === true;
}

/** Apply the shared base/fine/coarse/extra-fine modifier ladder to a step. */
export function applyModifierTier(
  step: number,
  modifiers: ModifierState,
  config: ModifierTierConfig = {}
): number {
  const fineDivisor = config.fineDivisor ?? 10;
  const coarseMultiplier = config.coarseMultiplier ?? 10;
  const shift = isActive(modifiers.shift, modifiers.shiftKey);
  const ctrl = isActive(modifiers.ctrl, modifiers.ctrlKey);
  const tier = (shift ? 1 : 0) + (ctrl ? 2 : 0);
  switch (tier) {
    case 1:
      return step / fineDivisor;
    case 2:
      return step * coarseMultiplier;
    case 3:
      return step / fineDivisor / fineDivisor;
    default:
      return step;
  }
}
