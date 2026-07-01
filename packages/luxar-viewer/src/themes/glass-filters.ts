/**
 * SVG Filter Definitions for Liquid Glass (Geometry-Aware Refraction)
 *
 * Creates authentic glass refraction based on element geometry, not random noise.
 * Technique: Convert SourceAlpha → Height Map → Normal Map → Displacement
 *
 * Key concept: Element acts like convex lens - thick in center, thin at edges.
 * Refraction is strongest at edges where curvature is highest.
 *
 * Based on advanced glassmorphism techniques with Sobel edge detection.
 */

import { getViewerContainer } from '../utils/viewer-container';

export interface GlassFilterParams {
  // Geometry parameters
  blurRadius: number; // Edge curve width - larger = thicker glass feel (10-30)
  refractionScale: number; // Lens strength - how much light bends (10-50)

  // Visual enhancements
  chromaticStrength: number; // RGB separation at edges (0-5)
  specularIntensity?: number; // Rim light brightness (0-1, optional)
}

/**
 * Default glass filter parameters
 * EASILY ADJUSTABLE - Change these to customize the glass effect!
 */
export const defaultGlassParams: GlassFilterParams = {
  blurRadius: 35, // Soft, thick edge curve (increased for more visible effect)
  refractionScale: 80, // Strong refraction (doubled for testing)
  chromaticStrength: 5, // Strong color splitting (rainbow edges)
  specularIntensity: 0.5, // Brighter rim light
};

/**
 * Inject geometry-aware glass filters into DOM
 *
 * Creates filters that refract based on element shape, not random noise.
 * Thicker glass in center → no refraction. Edges curve → strong refraction.
 */
export function injectGlassFilters(params: GlassFilterParams = defaultGlassParams): void {
  // Check if already injected
  if (document.getElementById('luxar-glass-filters')) {
    return;
  }

  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.id = 'luxar-glass-filters';
  svg.setAttribute('width', '0');
  svg.setAttribute('height', '0');
  svg.style.position = 'absolute';
  svg.style.overflow = 'hidden';

  // Sobel kernels for gradient detection (edge detection)
  // These detect the slope/curvature of the glass surface
  const sobelX = '-1 0 1 -2 0 2 -1 0 1'; // Horizontal gradient
  const sobelY = '-1 -2 -1 0 0 0 1 2 1'; // Vertical gradient

  svg.innerHTML = `
    <defs>
      <!-- Main Liquid Glass Filter: Geometry-Aware Refraction -->
      <filter id="luxar-liquid-refraction" x="-50%" y="-50%" width="200%" height="200%" color-interpolation-filters="sRGB">

        <!-- STEP 1: Create Height Map from element shape -->
        <!-- Blur SourceAlpha = "hill" shape (thick center, thin edges) -->
        <feGaussianBlur
          in="SourceAlpha"
          stdDeviation="${params.blurRadius}"
          result="heightMap"/>

        <!-- STEP 2: Calculate Gradients (Sobel Edge Detection) -->
        <!-- X-gradient: How fast height changes horizontally -->
        <feConvolveMatrix
          in="heightMap"
          order="3"
          kernelMatrix="${sobelX}"
          preserveAlpha="false"
          bias="0.5"
          result="gradX"/>

        <!-- Y-gradient: How fast height changes vertically -->
        <feConvolveMatrix
          in="heightMap"
          order="3"
          kernelMatrix="${sobelY}"
          preserveAlpha="false"
          bias="0.5"
          result="gradY"/>

        <!-- STEP 3: Build Normal Map (slope directions) -->
        <!-- Map X-gradient to Red channel -->
        <feColorMatrix
          in="gradX"
          type="matrix"
          values="0 0 0 1 0
                  0 0 0 0 0
                  0 0 0 0 0
                  0 0 0 1 0"
          result="normalR"/>

        <!-- Map Y-gradient to Green channel -->
        <feColorMatrix
          in="gradY"
          type="matrix"
          values="0 0 0 0 0
                  0 0 0 1 0
                  0 0 0 0 0
                  0 0 0 1 0"
          result="normalG"/>

        <!-- Combine into single normal map -->
        <feBlend
          in="normalR"
          in2="normalG"
          mode="screen"
          result="normalMap"/>

        <!-- STEP 4: Apply Displacement (Refraction) -->
        <!-- Center (flat) = no displacement. Edges (curved) = strong displacement -->
        <feDisplacementMap
          in="SourceGraphic"
          in2="normalMap"
          scale="${params.refractionScale}"
          xChannelSelector="R"
          yChannelSelector="G"
          result="refracted"/>

        <!-- STEP 5: Chromatic Aberration (RGB splitting) -->
        <!-- Red channel (shifted right) -->
        <feOffset
          in="refracted"
          dx="${params.chromaticStrength}"
          dy="0"
          result="redShift"/>
        <feColorMatrix
          in="redShift"
          type="matrix"
          values="1 0 0 0 0
                  0 0 0 0 0
                  0 0 0 0 0
                  0 0 0 1 0"
          result="redChannel"/>

        <!-- Green channel (no shift) -->
        <feColorMatrix
          in="refracted"
          type="matrix"
          values="0 0 0 0 0
                  0 1 0 0 0
                  0 0 0 0 0
                  0 0 0 1 0"
          result="greenChannel"/>

        <!-- Blue channel (shifted left) -->
        <feOffset
          in="refracted"
          dx="${-params.chromaticStrength}"
          dy="0"
          result="blueShift"/>
        <feColorMatrix
          in="blueShift"
          type="matrix"
          values="0 0 0 0 0
                  0 0 0 0 0
                  0 0 1 0 0
                  0 0 0 1 0"
          result="blueChannel"/>

        <!-- Combine RGB channels -->
        <feComposite
          in="redChannel"
          in2="greenChannel"
          operator="arithmetic"
          k2="1" k3="1"
          result="rg"/>
        <feComposite
          in="rg"
          in2="blueChannel"
          operator="arithmetic"
          k2="1" k3="1"
          result="rgb"/>

        <!-- STEP 6: Mask to element shape -->
        <feComposite
          in="rgb"
          in2="SourceAlpha"
          operator="in"
          result="final"/>

        ${
          params.specularIntensity && params.specularIntensity > 0
            ? `
        <!-- OPTIONAL: Specular Rim Light (glossy highlight) -->
        <feGaussianBlur in="heightMap" stdDeviation="5" result="specMap"/>
        <feColorMatrix
          in="specMap"
          type="matrix"
          values="0 0 0 0 ${params.specularIntensity}
                  0 0 0 0 ${params.specularIntensity}
                  0 0 0 0 ${params.specularIntensity}
                  0 0 0 1 0"
          result="specular"/>
        <feBlend
          in="final"
          in2="specular"
          mode="screen"
          result="withSpecular"/>
        <feComposite
          in="withSpecular"
          in2="SourceAlpha"
          operator="in"/>
        `
            : ''
        }

      </filter>
    </defs>
  `;

  getViewerContainer().appendChild(svg);
}

/**
 * Remove glass filters from DOM
 */
export function removeGlassFilters(): void {
  const svg = document.getElementById('luxar-glass-filters');
  if (svg) {
    svg.remove();
  }
}

/**
 * Single marker class for glass-enabled surfaces.
 *
 * Any panel that opts into the glass themes adds `luxar-glass-surface` to its
 * root element (see the `ui/*` panel constructors). The theme CSS
 * (`frosted-glass.css` / `liquid-glass.css`) and this refraction injector all
 * key off this one class — there is no per-panel list to keep in sync across
 * the three files. To make a new panel glass-aware, add the class at its
 * creation site; nothing here changes.
 */
const GLASS_SURFACE_SELECTOR = '.luxar-glass-surface';

/**
 * Inject real DOM elements for the glass refraction layer
 *
 * SVG filters on ::before pseudo-elements don't work reliably across browsers.
 * This function creates real DOM elements that can properly receive the SVG filter.
 *
 * Must be called after the glass panels are created in the DOM.
 */
export function injectGlassRefractionLayers(): void {
  const panels = document.querySelectorAll(GLASS_SURFACE_SELECTOR);
  panels.forEach((panel) => {
    // Skip if already has a refraction layer
    if (panel.querySelector('.luxar-glass-refraction')) {
      return;
    }

    // Create the refraction layer element
    const refractionLayer = document.createElement('div');
    refractionLayer.className = 'luxar-glass-refraction';
    refractionLayer.setAttribute('aria-hidden', 'true');

    // Insert as first child so it's behind all content
    panel.insertBefore(refractionLayer, panel.firstChild);
  });
}

/**
 * Remove all glass refraction layer elements from the DOM
 */
export function removeGlassRefractionLayers(): void {
  const layers = document.querySelectorAll('.luxar-glass-refraction');
  layers.forEach((layer) => layer.remove());
}

/**
 * Set up a MutationObserver to automatically inject glass refraction layers
 * when new glass-enabled panels are added to the DOM.
 *
 * Returns a cleanup function to disconnect the observer.
 */
export function setupGlassRefractionObserver(): () => void {
  const observer = new MutationObserver((mutations) => {
    let shouldInject = false;

    for (const mutation of mutations) {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // Check if any added node is (or contains) a glass surface.
        mutation.addedNodes.forEach((node) => {
          if (node instanceof HTMLElement) {
            if (
              node.matches(GLASS_SURFACE_SELECTOR) ||
              node.querySelector(GLASS_SURFACE_SELECTOR)
            ) {
              shouldInject = true;
            }
          }
        });
      }

      if (shouldInject) break;
    }

    if (shouldInject) {
      // Small delay to ensure DOM is fully updated
      requestAnimationFrame(() => {
        injectGlassRefractionLayers();
      });
    }
  });

  // Scope the observer to the viewer container (the element panels actually
  // mount into — the embedder's `container` or `document.body` by default).
  // Using subtree: true on document.body is expensive; scoping to the
  // container reduces DOM-mutation noise and stays correct under embedding.
  observer.observe(getViewerContainer(), {
    childList: true,
    subtree: true,
  });

  // Return cleanup function
  return () => observer.disconnect();
}
