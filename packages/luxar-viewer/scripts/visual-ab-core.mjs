/** Shared screenshot capture and image scoring for opt-in visual A/B scripts. */

const DEFAULT_SCORE_SIZE = 256;
const BLOWN_CHANNEL = 250;

function assertRgb(rgb, size, label) {
  const expected = size * size * 3;
  if (rgb.length !== expected) {
    throw new Error(
      `${label} has ${rgb.length} channels; expected ${expected} for ${size}x${size}`
    );
  }
}

function toGrey(rgb) {
  const grey = new Array(rgb.length / 3);
  for (let pixel = 0; pixel < grey.length; pixel++) {
    const offset = pixel * 3;
    grey[pixel] = 0.299 * rgb[offset] + 0.587 * rgb[offset + 1] + 0.114 * rgb[offset + 2];
  }
  return grey;
}

/** Global SSIM over 8x8 box windows on two equal-length greyscale arrays. */
export function ssim(a, b, size) {
  const k1 = 0.01;
  const k2 = 0.03;
  const dynamicRange = 255;
  const c1 = (k1 * dynamicRange) ** 2;
  const c2 = (k2 * dynamicRange) ** 2;
  const windowSize = 8;
  let sum = 0;
  let windowCount = 0;
  for (let y = 0; y + windowSize <= size; y += windowSize) {
    for (let x = 0; x + windowSize <= size; x += windowSize) {
      let meanA = 0;
      let meanB = 0;
      for (let j = 0; j < windowSize; j++) {
        for (let i = 0; i < windowSize; i++) {
          const index = (y + j) * size + x + i;
          meanA += a[index];
          meanB += b[index];
        }
      }
      const sampleCount = windowSize * windowSize;
      meanA /= sampleCount;
      meanB /= sampleCount;
      let varianceA = 0;
      let varianceB = 0;
      let covariance = 0;
      for (let j = 0; j < windowSize; j++) {
        for (let i = 0; i < windowSize; i++) {
          const index = (y + j) * size + x + i;
          const deltaA = a[index] - meanA;
          const deltaB = b[index] - meanB;
          varianceA += deltaA * deltaA;
          varianceB += deltaB * deltaB;
          covariance += deltaA * deltaB;
        }
      }
      varianceA /= sampleCount - 1;
      varianceB /= sampleCount - 1;
      covariance /= sampleCount - 1;
      sum +=
        ((2 * meanA * meanB + c1) * (2 * covariance + c2)) /
        ((meanA * meanA + meanB * meanB + c1) * (varianceA + varianceB + c2));
      windowCount++;
    }
  }
  if (windowCount === 0) throw new Error('SSIM requires an image at least 8x8 pixels');
  return sum / windowCount;
}

/** Normalised cross-correlation (Pearson) of two greyscale arrays. */
export function ncc(a, b) {
  const sampleCount = a.length;
  let meanA = 0;
  let meanB = 0;
  for (let index = 0; index < sampleCount; index++) {
    meanA += a[index];
    meanB += b[index];
  }
  meanA /= sampleCount;
  meanB /= sampleCount;
  let numerator = 0;
  let varianceA = 0;
  let varianceB = 0;
  for (let index = 0; index < sampleCount; index++) {
    const deltaA = a[index] - meanA;
    const deltaB = b[index] - meanB;
    numerator += deltaA * deltaB;
    varianceA += deltaA * deltaA;
    varianceB += deltaB * deltaB;
  }
  if (varianceA === 0 || varianceB === 0) {
    return a.every((value, index) => value === b[index]) ? 1 : 0;
  }
  return numerator / Math.sqrt(varianceA * varianceB);
}

function srgbChannelToLinear(channel) {
  const value = channel / 255;
  return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
}

function rgbToLab(red, green, blue) {
  const linearRed = srgbChannelToLinear(red);
  const linearGreen = srgbChannelToLinear(green);
  const linearBlue = srgbChannelToLinear(blue);
  const x = (0.4124564 * linearRed + 0.3575761 * linearGreen + 0.1804375 * linearBlue) / 0.95047;
  const y = 0.2126729 * linearRed + 0.7151522 * linearGreen + 0.072175 * linearBlue;
  const z = (0.0193339 * linearRed + 0.119192 * linearGreen + 0.9503041 * linearBlue) / 1.08883;
  const pivot = (value) => (value > 216 / 24389 ? Math.cbrt(value) : (841 / 108) * value + 4 / 29);
  const fx = pivot(x);
  const fy = pivot(y);
  const fz = pivot(z);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

function meanDeltaE(reference, candidate) {
  let sum = 0;
  const pixelCount = reference.length / 3;
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const offset = pixel * 3;
    const labReference = rgbToLab(reference[offset], reference[offset + 1], reference[offset + 2]);
    const labCandidate = rgbToLab(candidate[offset], candidate[offset + 1], candidate[offset + 2]);
    sum += Math.hypot(
      labReference[0] - labCandidate[0],
      labReference[1] - labCandidate[1],
      labReference[2] - labCandidate[2]
    );
  }
  return sum / pixelCount;
}

function blownPixelFraction(rgb) {
  let blown = 0;
  const pixelCount = rgb.length / 3;
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const offset = pixel * 3;
    if (Math.max(rgb[offset], rgb[offset + 1], rgb[offset + 2]) >= BLOWN_CHANNEL) blown++;
  }
  return blown / pixelCount;
}

export function scoreImagePair(referenceRgb, candidateRgb, size = DEFAULT_SCORE_SIZE) {
  assertRgb(referenceRgb, size, 'reference image');
  assertRgb(candidateRgb, size, 'candidate image');
  const referenceGrey = toGrey(referenceRgb);
  const candidateGrey = toGrey(candidateRgb);
  const referenceBlown = blownPixelFraction(referenceRgb);
  const candidateBlown = blownPixelFraction(candidateRgb);
  return {
    ssim: ssim(referenceGrey, candidateGrey, size),
    ncc: ncc(referenceGrey, candidateGrey),
    meanDeltaE: meanDeltaE(referenceRgb, candidateRgb),
    blownPixelFraction: {
      reference: referenceBlown,
      candidate: candidateBlown,
      delta: candidateBlown - referenceBlown,
    },
  };
}

export function evaluateThresholds(score, thresholds) {
  const checks = {
    ssim: score.ssim >= thresholds.minSsim,
    meanDeltaE: score.meanDeltaE <= thresholds.maxMeanDeltaE,
    blownPixelFractionDelta:
      score.blownPixelFraction.delta <= thresholds.maxBlownPixelFractionDelta,
  };
  return { pass: Object.values(checks).every(Boolean), checks };
}

/** Capture a canvas locator and decode a fixed-size RGB comparison image in-page. */
export async function captureCanvasImage(page, canvas, size = DEFAULT_SCORE_SIZE) {
  const png = await canvas.screenshot();
  const stats = await page.evaluate(
    async ({ base64, comparisonSize }) => {
      const image = new Image();
      image.src = `data:image/png;base64,${base64}`;
      await image.decode();
      const full = document.createElement('canvas');
      full.width = image.width;
      full.height = image.height;
      const fullContext = full.getContext('2d');
      if (!fullContext) throw new Error('2D canvas context unavailable');
      fullContext.drawImage(image, 0, 0);
      const fullData = fullContext.getImageData(0, 0, image.width, image.height).data;
      let lit = 0;
      let nearWhite = 0;
      let lumaSum = 0;
      const pixelCount = image.width * image.height;
      for (let offset = 0; offset < fullData.length; offset += 4) {
        const maximum = Math.max(fullData[offset], fullData[offset + 1], fullData[offset + 2]);
        if (maximum > 16) lit++;
        if (maximum > 235) nearWhite++;
        lumaSum += (fullData[offset] + fullData[offset + 1] + fullData[offset + 2]) / 3;
      }
      const small = document.createElement('canvas');
      small.width = comparisonSize;
      small.height = comparisonSize;
      const smallContext = small.getContext('2d');
      if (!smallContext) throw new Error('2D canvas context unavailable');
      smallContext.drawImage(image, 0, 0, comparisonSize, comparisonSize);
      const smallData = smallContext.getImageData(0, 0, comparisonSize, comparisonSize).data;
      const rgb = new Array(comparisonSize * comparisonSize * 3);
      for (let pixel = 0; pixel < comparisonSize * comparisonSize; pixel++) {
        rgb[pixel * 3] = smallData[pixel * 4];
        rgb[pixel * 3 + 1] = smallData[pixel * 4 + 1];
        rgb[pixel * 3 + 2] = smallData[pixel * 4 + 2];
      }
      return {
        width: image.width,
        height: image.height,
        litFraction: lit / pixelCount,
        nearWhiteFraction: nearWhite / pixelCount,
        meanLuma: lumaSum / pixelCount,
        rgb,
      };
    },
    { base64: png.toString('base64'), comparisonSize: size }
  );
  return { png, ...stats, grey: toGrey(stats.rgb) };
}
