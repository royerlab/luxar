import { ArrayDecoder, type ArrayMetadata } from '../../../array-decoder/decoder';
import type { EncodingType } from './encoding-types';

/**
 * Detect encoding type from array metadata. Priority order matches the
 * Python encoder spec — broadcasted → array_ref → LUT → quantized → direct.
 */
export function detectEncoding(attrs: ArrayMetadata | undefined): EncodingType {
  if (!attrs?.encoding) return 'direct';

  const enc = attrs.encoding;
  ArrayDecoder.validateEncodingMetadata(enc);

  if (enc.name === 'broadcasted') return 'broadcasted';
  if (enc.name === 'array_ref') return 'array_ref';
  if (ArrayDecoder.isLUTEncodingName(enc.name)) return 'lut';
  if (ArrayDecoder.isQuantizedEncoding(attrs)) return 'quantized';
  if (ArrayDecoder.isDirectEncodingName(enc.name)) return 'direct';
  // Per-channel quantized arrays (log/signed-log perchannel) load as raw integer
  // levels (direct); the consumer applies per-channel dequant (the gsplats loader
  // does so in loadCholeskyRanges when recombining the split Cholesky).
  if (ArrayDecoder.isPerChannelQuantEncodingName(enc.name)) return 'direct';

  throw new Error(`Unknown encoding name: ${enc.name}`);
}
