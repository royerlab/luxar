import { describe, it, expect } from 'vitest';
import { escapeHtml } from '../../../utils/escape-html';

describe('escapeHtml', () => {
  it('should escape HTML tags', () => {
    expect(escapeHtml('<script>alert("xss")</script>')).toBe(
      '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
    );
  });
  it('should escape ampersands', () => {
    expect(escapeHtml('foo & bar')).toBe('foo &amp; bar');
  });
  it('should escape double quotes', () => {
    expect(escapeHtml('a "quoted" value')).toBe('a &quot;quoted&quot; value');
  });
  it('should pass through safe strings', () => {
    expect(escapeHtml('hello world 123')).toBe('hello world 123');
  });
  it('should handle empty string', () => {
    expect(escapeHtml('')).toBe('');
  });
});
