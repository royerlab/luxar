/**
 * Pure cycle helper for the V-key camera control toggle.
 *
 * Extracted from input-handler.ts so the cycle order is unit-testable
 * without instantiating SceneManager / OrbitControls. The full toggle
 * (which mutates SceneManager + InputContextManager + RenderingControls)
 * stays in input-handler.ts and calls this for the next-state lookup.
 *
 * @module input/handlers/control-mode-cycle
 */

/** Camera control modes the V key cycles through. */
export type ControlType = 'orbit' | 'fly' | 'ortho';

/**
 * Return the next control type in the cycle: orbit → fly → ortho → orbit.
 * Defensive: any unknown current mode (shouldn't happen in production)
 * resets to `orbit`.
 *
 * @param current - The currently-active control type.
 */
export function nextControlType(current: ControlType | string): ControlType {
  switch (current) {
    case 'orbit':
      return 'fly';
    case 'fly':
      return 'ortho';
    case 'ortho':
      return 'orbit';
    default:
      return 'orbit';
  }
}
