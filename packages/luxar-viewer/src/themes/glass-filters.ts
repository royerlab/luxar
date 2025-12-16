/**
 * SVG Filter Definitions for Glass Effects
 *
 * Provides reusable SVG filters for creating authentic glass refraction,
 * chromatic aberration, and distance-based distortion effects.
 *
 * Usage: Call injectGlassFilters() when theme is applied
 */

export interface GlassFilterParams {
  // Noise/texture parameters
  noiseFrequency: number; // Base frequency for turbulence (0.005-0.02)
  noiseOctaves: number; // Detail level (1-4)
  noiseSeed: number; // Random seed

  // Distortion parameters
  displacementScale: number; // Refraction strength (20-150)
  blurAmount: number; // Smoothness of distortion (1-5)

  // Chromatic aberration
  chromaticStrength: number; // Color separation (0-3)

  // Edge enhancement
  edgeDistortionMultiplier: number; // Stronger distortion at edges (1-3)
}

/**
 * Default glass filter parameters
 * Adjust these for different glass effects!
 */
export const defaultGlassParams: GlassFilterParams = {
  // Subtle noise for glass texture
  noiseFrequency: 0.012, // Medium frequency for fine texture
  noiseOctaves: 3, // Good detail without being too busy
  noiseSeed: 42, // Random seed for texture pattern

  // Moderate refraction
  displacementScale: 60, // Visible but not extreme distortion
  blurAmount: 2.5, // Smooth, glass-like distortion

  // Subtle chromatic aberration
  chromaticStrength: 1.5, // Slight color splitting at edges

  // Edge-aware distortion
  edgeDistortionMultiplier: 2.0, // 2x stronger at panel edges
};

/**
 * Inject SVG filter definitions into the DOM
 *
 * Call this when applying Liquid Glass theme to add the necessary
 * SVG filters for refraction effects.
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

  svg.innerHTML = `
    <defs>
      <!-- Main Glass Distortion Filter -->
      <filter id="luxar-glass-distortion" x="-50%" y="-50%" width="200%" height="200%">
        <!-- Turbulence for glass texture/imperfections -->
        <feTurbulence
          type="fractalNoise"
          baseFrequency="${params.noiseFrequency} ${params.noiseFrequency}"
          numOctaves="${params.noiseOctaves}"
          seed="${params.noiseSeed}"
          result="noise"/>

        <!-- Blur the noise for smoother glass -->
        <feGaussianBlur
          in="noise"
          stdDeviation="${params.blurAmount}"
          result="smoothNoise"/>

        <!-- Displace pixels = refraction effect -->
        <feDisplacementMap
          in="SourceGraphic"
          in2="smoothNoise"
          scale="${params.displacementScale}"
          xChannelSelector="R"
          yChannelSelector="G"
          result="distorted"/>
      </filter>

      <!-- Chromatic Aberration Filter (color splitting like thick glass) -->
      <filter id="luxar-glass-chromatic" x="-20%" y="-20%" width="140%" height="140%">
        <!-- Red channel offset -->
        <feOffset in="SourceGraphic" dx="${params.chromaticStrength}" dy="0" result="offsetR"/>
        <feColorMatrix in="offsetR" type="matrix"
          values="1 0 0 0 0
                  0 0 0 0 0
                  0 0 0 0 0
                  0 0 0 1 0" result="redChannel"/>

        <!-- Blue channel offset (opposite direction) -->
        <feOffset in="SourceGraphic" dx="${-params.chromaticStrength}" dy="0" result="offsetB"/>
        <feColorMatrix in="offsetB" type="matrix"
          values="0 0 0 0 0
                  0 0 0 0 0
                  0 0 1 0 0
                  0 0 0 1 0" result="blueChannel"/>

        <!-- Green channel (no offset) -->
        <feColorMatrix in="SourceGraphic" type="matrix"
          values="0 0 0 0 0
                  0 1 0 0 0
                  0 0 0 0 0
                  0 0 0 1 0" result="greenChannel"/>

        <!-- Combine all channels -->
        <feComposite in="redChannel" in2="greenChannel" operator="arithmetic" k1="0" k2="1" k3="1" k4="0" result="rg"/>
        <feComposite in="rg" in2="blueChannel" operator="arithmetic" k1="0" k2="1" k3="1" k4="0"/>
      </filter>

      <!-- Combined: Distortion + Chromatic Aberration -->
      <filter id="luxar-liquid-glass-full" x="-50%" y="-50%" width="200%" height="200%">
        <!-- First apply displacement -->
        <feTurbulence
          type="fractalNoise"
          baseFrequency="${params.noiseFrequency} ${params.noiseFrequency}"
          numOctaves="${params.noiseOctaves}"
          seed="${params.noiseSeed}"
          result="noise"/>
        <feGaussianBlur in="noise" stdDeviation="${params.blurAmount}" result="smoothNoise"/>
        <feDisplacementMap
          in="SourceGraphic"
          in2="smoothNoise"
          scale="${params.displacementScale}"
          xChannelSelector="R"
          yChannelSelector="G"
          result="displaced"/>

        <!-- Then add subtle chromatic aberration -->
        <feOffset in="displaced" dx="${params.chromaticStrength * 0.5}" dy="0" result="r"/>
        <feOffset in="displaced" dx="${-params.chromaticStrength * 0.5}" dy="0" result="b"/>

        <!-- Blend channels -->
        <feBlend in="r" in2="displaced" mode="screen" result="rb"/>
        <feBlend in="rb" in2="b" mode="screen"/>
      </filter>

      <!-- Edge-Aware Gradient Mask (for stronger distortion at edges) -->
      <filter id="luxar-glass-edge-mask">
        <!-- Create radial gradient from center -->
        <feGaussianBlur in="SourceAlpha" stdDeviation="40" result="blur"/>
        <feColorMatrix in="blur" type="matrix"
          values="1 0 0 0 0
                  0 1 0 0 0
                  0 0 1 0 0
                  0 0 0 ${params.edgeDistortionMultiplier} 0" result="edgeMask"/>
      </filter>
    </defs>
  `;

  document.body.appendChild(svg);
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
 * Update glass filter parameters dynamically
 */
export function updateGlassFilters(params: Partial<GlassFilterParams>): void {
  removeGlassFilters();
  injectGlassFilters({ ...defaultGlassParams, ...params });
}
