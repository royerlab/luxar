/**
 * The viewer's `Object3D.layers` bits.
 *
 * Luxar draws everything on three's default layer 0: the main camera, the pick pass,
 * the environment `CubeCamera` and the blend warm-up's compile objects all assume it,
 * and nothing persistent lives anywhere else. The one other bit is used TRANSIENTLY —
 * set and restored inside a single render call, so no other camera can ever observe
 * it — by the WebGL refraction split (`post-processing/post-processing-manager/
 * refraction-split.ts`), which needs to draw `refract_data` glass in a second pass
 * without the rest of the scene. Keep it that way: a mesh parked on a non-default
 * layer between frames would vanish from every other pass at once.
 *
 * @module rendering/render-layers
 */

/** Three's default layer; every camera in the viewer sees exactly this one. */
export const RENDER_LAYER_DEFAULT = 0;

/**
 * The layer the refraction split moves refracting glass onto for the duration of one
 * render call (pass A hides it, pass B draws only it), restoring the previous mask in a
 * `finally`. Never persistent.
 */
export const RENDER_LAYER_REFRACTING_GLASS = 1;
