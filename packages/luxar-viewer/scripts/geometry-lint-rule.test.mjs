import { describe, expect, it } from 'vitest';
import { ESLint } from 'eslint';

const eslint = new ESLint({
  overrideConfig: { rules: { 'max-lines-per-function': 'off' } },
});
const filePath = 'src/types/geometry-capabilities.ts';

async function restrictedCount(source) {
  const [result] = await eslint.lintText(source, { filePath });
  return result.messages.filter((message) => message.ruleId === 'no-restricted-syntax').length;
}

describe('geometry subset lint rule', () => {
  it.each([
    "const types = new Set(['points', 'lines']);",
    "const types = new Set<string>(['points', 'lines'] as const);",
    "const types = new Set<string>(['points', 'lines'] satisfies string[]);",
    "const allowed = ['points', 'lines'].includes(type);",
    "const allowed = (['points', 'lines'] as const).includes(type);",
    "const allowed = (['points', 'lines'] satisfies string[]).includes(type);",
    "const allowed = type === 'points' || type === 'lines';",
    "const allowed = 'points' === type || 'lines' === type;",
  ])('catches a hand-written subset: %s', async (source) => {
    expect(await restrictedCount(source)).toBeGreaterThan(0);
  });

  it('leaves a single-type check alone', async () => {
    expect(await restrictedCount("const allowed = type === 'points';")).toBe(0);
  });
});
