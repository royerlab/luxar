import type { AppConfig } from '../../types';

/**
 * Validate control configuration (ConfigRange consistency)
 */
export function validateControls(config: AppConfig, errors: string[], _warnings: string[]): void {
  const { controls } = config;

  // Validate all ConfigRange objects: min < max and min <= default <= max
  const ranges: Array<{ name: string; range: { min: number; max: number; default: number } }> = [
    { name: 'fly.movement.speed', range: controls.fly.movement.speed },
    { name: 'fly.movement.acceleration', range: controls.fly.movement.acceleration },
    { name: 'fly.movement.damping', range: controls.fly.movement.damping },
    { name: 'fly.rotation.speed', range: controls.fly.rotation.speed },
    { name: 'fly.rotation.damping', range: controls.fly.rotation.damping },
    { name: 'fly.look.mouseSpeed', range: controls.fly.look.mouseSpeed },
    { name: 'orbit.autoRotate.speed', range: controls.orbit.autoRotate.speed },
    {
      name: 'orbit.autoDolly.amplitudePercent',
      range: controls.orbit.autoDolly.amplitudePercent,
    },
    { name: 'orbit.autoDolly.period', range: controls.orbit.autoDolly.period },
    { name: 'orbit.zoom.speed', range: controls.orbit.zoom.speed },
    { name: 'orbit.damping.factor', range: controls.orbit.damping.factor },
  ];

  for (const { name, range } of ranges) {
    // NaN check on every range field — without this, any of
    // {min, max, default} could be NaN and silently pass.
    if (
      !Number.isFinite(range.min) ||
      !Number.isFinite(range.max) ||
      !Number.isFinite(range.default)
    ) {
      errors.push(
        `Invalid controls.${name}: non-finite values (min=${range.min}, max=${range.max}, default=${range.default})`
      );
      continue;
    }
    if (range.min >= range.max) {
      errors.push(`Invalid controls.${name}: min (${range.min}) >= max (${range.max})`);
    }
    if (range.default < range.min || range.default > range.max) {
      errors.push(
        `Invalid controls.${name}: default (${range.default}) outside [${range.min}, ${range.max}]`
      );
    }
  }
}
