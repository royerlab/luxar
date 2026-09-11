/**
 * What a remote controller is allowed to call, and what it is not.
 *
 * The spec's rule is *"the wire surface is the embedder API"*
 * (`docs/guides/specs/REMOTE_CONTROL_SPEC.md` §3.2), which is true but not
 * literally true — a handful of `LuxarApp` members cannot cross a JSON socket
 * at all. An unwritten exception list is exactly the thing that rots, so the
 * rule is stated here precisely:
 *
 * > The wire surface is every public `LuxarApp` method, minus
 * > {@link CONTROL_EXCLUDED_METHODS} (each with a recorded reason), plus
 * > {@link CONTROL_WIRE_ONLY_METHODS}.
 *
 * A lock test (`src/tests/unit/core/app/control/method-policy.test.ts`) scans
 * `core/app.ts` and fails if any public member is in neither list, so the next
 * embedder method cannot land outside the policy unnoticed.
 *
 * ## Threat model
 *
 * The designed deployment is a kiosk on a LAN the operator owns, and the hub
 * is open unless `luxar serve --control-token` is used. Anything that can
 * reach the socket can therefore drive the display. That is a deliberate
 * choice, not an oversight — but it means two things are worth knowing:
 *
 * - `dispose` is **excluded outright**. It tears the viewer down in one frame
 *   and `init` is not on the wire, so there is no way back: a single frame
 *   would blank an exhibit until someone power-cycles it. Every other verb is
 *   recoverable by sending another one, which is what keeps it on the list.
 * - `switchDataset` **is** allowed (a controller swapping scenes is a real use
 *   case), but the dispatcher runs its argument through the viewer's own
 *   `normalizeDataSourceUrl` first. `LuxarApp.switchDataset` itself validates
 *   nothing, so without that step any LAN peer could point the display at an
 *   arbitrary URL. Rejecting `file:`/`javascript:`/`data:` is not a narrowing
 *   of the capability; it is the validation the direct caller already relies
 *   on the URL parser to have done.
 */

/** Subscribe to an embedder event by name. Replaces `on`, which takes a function. */
export const CONTROL_SUBSCRIBE = 'subscribe';
/** Stop forwarding an embedder event by name. */
export const CONTROL_UNSUBSCRIBE = 'unsubscribe';

/**
 * Methods that exist only on the wire, because their in-process form takes a
 * callback. Kept separate from the allow-list so the lock test can tell
 * "deliberately not a `LuxarApp` method" from "forgot to classify".
 */
export const CONTROL_WIRE_ONLY_METHODS: readonly string[] = [
  CONTROL_SUBSCRIBE,
  CONTROL_UNSUBSCRIBE,
];

/**
 * Every `LuxarApp` method a controller may call.
 *
 * A `readonly string[]` rather than a `Record` on purpose: the Python side
 * locks itself against this list with `read_ts_string_literals`, which parses
 * array literals and unions, not object literals.
 */
export const CONTROL_ALLOWED_METHODS: readonly string[] = [
  'awaitDimensionUpdate',
  'captureSnapshot',
  'flyTo',
  'getAudioState',
  'getCameraPose',
  'getDatasetFault',
  'getDimensions',
  'getLayers',
  'getRenderingSettings',
  'getViewerState',
  'playSound',
  'recenterCamera',
  'resize',
  'restoreSnapshot',
  'screenshot',
  'setAudio',
  'setCameraPose',
  'setDimensionValue',
  'setInputEnabled',
  'setLayer',
  'setRenderingSettings',
  'shortcutForAction',
  'stopSound',
  'switchDataset',
];

/** Public `LuxarApp` members deliberately kept off the wire. */
export const CONTROL_EXCLUDED_METHODS: readonly string[] = [
  'components',
  'dispose',
  'init',
  'initialized',
  'on',
  'popContext',
  'pushContext',
  'registerBinding',
  'registerContext',
  'unregisterBinding',
  'unregisterContext',
];

/**
 * Why each excluded member is excluded.
 *
 * Documentation with teeth: the lock test asserts every name in
 * {@link CONTROL_EXCLUDED_METHODS} has an entry here, so an exclusion cannot
 * be added without saying why.
 */
export const CONTROL_EXCLUSION_REASONS: Readonly<Record<string, string>> = {
  components: 'returns live engine objects (renderer, scene, controls) — not serialisable',
  dispose:
    'a one-frame remote kill switch with no recovery, since init() is not on the wire; ' +
    'every other verb can be undone by sending another',
  init: 'takes a canvas and a container element, and re-initialising a live app is not remote work',
  initialized: 'a property accessor, not a callable method; getViewerState() reports readiness',
  on: 'takes a listener function; subscribe/unsubscribe carry the same capability by name',
  popContext: 'mutates an input-context stack whose depth a controller cannot observe',
  pushContext: 'mutates an input-context stack whose depth a controller cannot observe',
  registerBinding: 'KeyBinding.handler is a required function, so the argument cannot be encoded',
  registerContext: 'ContextConfig carries handler functions, so the argument cannot be encoded',
  unregisterBinding: 'pairs with registerBinding, which cannot be encoded',
  unregisterContext: 'pairs with registerContext, which cannot be encoded',
};

const ALLOWED = new Set<string>([...CONTROL_ALLOWED_METHODS, ...CONTROL_WIRE_ONLY_METHODS]);
const EXCLUDED = new Set<string>(CONTROL_EXCLUDED_METHODS);

/** Whether a controller may invoke `method`. */
export function isControlMethodAllowed(method: string): boolean {
  return ALLOWED.has(method);
}

/**
 * Why `method` was refused, phrased for a controller's error message.
 *
 * Distinguishes "deliberately not exposed" from "no such method", because the
 * two send an integrator looking in completely different places.
 */
export function controlRefusalReason(method: string): string {
  const reason = EXCLUDED.has(method) ? CONTROL_EXCLUSION_REASONS[method] : undefined;
  if (reason !== undefined) return `method '${method}' is not exposed to controllers: ${reason}`;
  if (EXCLUDED.has(method)) return `method '${method}' is not exposed to controllers`;
  return `unknown method '${method}'`;
}
