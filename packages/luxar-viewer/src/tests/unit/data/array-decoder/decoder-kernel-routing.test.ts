import { readFileSync } from 'node:fs';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

const DECODER_SOURCE_URL = new URL('../../../../data/array-decoder/decoder.ts', import.meta.url);
const INLINE_FLOAT_ARITHMETIC_ALLOWLIST = new Set(['ArrayDecoder.makePerChannelDequant']);
const QUANTIZATION_DIVISORS = new Set([255, 65_535]);

interface Violation {
  functionName: string;
  line: number;
  expression: string;
}

function containingFunctionName(node: ts.Node): string {
  for (let current: ts.Node | undefined = node; current; current = current.parent) {
    if (ts.isMethodDeclaration(current)) {
      const className = ts.isClassDeclaration(current.parent)
        ? (current.parent.name?.getText() ?? '<anonymous class>')
        : '<object>';
      return `${className}.${current.name.getText()}`;
    }
    if (ts.isFunctionDeclaration(current)) {
      return current.name?.getText() ?? '<anonymous function>';
    }
  }
  return '<module>';
}

function isMathExponentialCall(node: ts.Node): boolean {
  if (!ts.isCallExpression(node)) return false;

  const { expression } = node;
  return (
    ts.isPropertyAccessExpression(expression) &&
    expression.expression.getText() === 'Math' &&
    (expression.name.text === 'exp' || expression.name.text === 'expm1')
  );
}

function isQuantizationDivision(node: ts.Node): boolean {
  if (!ts.isBinaryExpression(node) || node.operatorToken.kind !== ts.SyntaxKind.SlashToken) {
    return false;
  }

  let divisor = node.right;
  while (ts.isParenthesizedExpression(divisor)) divisor = divisor.expression;

  return (
    (ts.isNumericLiteral(divisor) && QUANTIZATION_DIVISORS.has(Number(divisor.text))) ||
    (ts.isIdentifier(divisor) && /^max_?int$/i.test(divisor.text))
  );
}

function findInlineFloatArithmetic(source: string): Violation[] {
  const sourceFile = ts.createSourceFile(
    'decoder.ts',
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS
  );
  const violations: Violation[] = [];

  function visit(node: ts.Node): void {
    if (isMathExponentialCall(node) || isQuantizationDivision(node)) {
      const functionName = containingFunctionName(node);
      if (!INLINE_FLOAT_ARITHMETIC_ALLOWLIST.has(functionName)) {
        violations.push({
          functionName,
          line: sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1,
          expression: node.getText(sourceFile),
        });
      }
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return violations;
}

describe('ArrayDecoder kernel routing', () => {
  it('keeps inline float dequantization inside the explicit allowlist', () => {
    const source = readFileSync(DECODER_SOURCE_URL, 'utf8');

    expect(findInlineFloatArithmetic(source)).toEqual([]);
  });

  it('detects representative inline dequantization arithmetic', () => {
    const source = `
      class ArrayDecoder {
        makePerChannelDequant() {
          return Math.exp(1) + Math.expm1(2) + 3 / 255;
        }
        decodeInline() {
          return Math.exp(1) + Math.expm1(2) + 3 / 255 + 4 / 65535 + 5 / max_int;
        }
      }
    `;

    expect(findInlineFloatArithmetic(source)).toEqual([
      { functionName: 'ArrayDecoder.decodeInline', line: 7, expression: 'Math.exp(1)' },
      { functionName: 'ArrayDecoder.decodeInline', line: 7, expression: 'Math.expm1(2)' },
      { functionName: 'ArrayDecoder.decodeInline', line: 7, expression: '3 / 255' },
      { functionName: 'ArrayDecoder.decodeInline', line: 7, expression: '4 / 65535' },
      { functionName: 'ArrayDecoder.decodeInline', line: 7, expression: '5 / max_int' },
    ]);
  });
});
