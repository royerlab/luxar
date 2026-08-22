/**
 * Structural tripwire for ArrayDecoder's float-kernel routing contract.
 *
 * Every scalar dequantization path must reach a function imported from
 * `wasm/typescript/decode.ts`, directly or through another guarded method. The
 * arithmetic scan covers every TypeScript file beside ArrayDecoder so splitting
 * decoder.ts cannot silently move an inline implementation out of sight.
 *
 * This deliberately does not resolve arbitrary aliases or destructured Math
 * calls such as `const { exp } = Math; exp(x)`. It targets accidental inline
 * implementations while the positive routing assertion remains the primary
 * guard against differently-written arithmetic.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { describe, expect, it } from 'vitest';

const THIS_FILE = fileURLToPath(import.meta.url);
const ARRAY_DECODER_DIR = resolve(dirname(THIS_FILE), '../../../../data/array-decoder');
const DECODER_KERNEL_MODULE = '../../wasm/typescript/decode';
const NON_KERNEL_DECODER_METHOD_ALLOWLIST = new Set([
  'ArrayDecoder.decodeBroadcasted',
  'ArrayDecoder.decodeLUT',
  'ArrayDecoder.decodeLUTIndices',
]);
const INLINE_FLOAT_ARITHMETIC_ALLOWLIST = new Set(['ArrayDecoder.makePerChannelDequant']);
const QUANTIZATION_DIVISORS = new Set([255, 65_535]);

interface DecoderSource {
  fileName: string;
  source: string;
}

interface Violation {
  fileName: string;
  functionName: string;
  line: number;
  expression: string;
}

function decoderSources(): DecoderSource[] {
  return readdirSync(ARRAY_DECODER_DIR)
    .filter((entry) => entry.endsWith('.ts'))
    .sort()
    .map((fileName) => ({
      fileName,
      source: readFileSync(resolve(ARRAY_DECODER_DIR, fileName), 'utf8'),
    }));
}

function containingFunctionName(node: ts.Node): string {
  for (let current: ts.Node | undefined = node; current; current = current.parent) {
    if (ts.isMethodDeclaration(current) || ts.isPropertyDeclaration(current)) {
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

function isDivision(node: ts.Node): node is ts.BinaryExpression {
  return ts.isBinaryExpression(node) && node.operatorToken.kind === ts.SyntaxKind.SlashToken;
}

function isKnownQuantizationDivision(node: ts.BinaryExpression): boolean {
  let divisor = node.right;
  while (ts.isParenthesizedExpression(divisor)) divisor = divisor.expression;

  return (
    (ts.isNumericLiteral(divisor) && QUANTIZATION_DIVISORS.has(Number(divisor.text))) ||
    (ts.isIdentifier(divisor) && /^max_?int$/i.test(divisor.text))
  );
}

function requiresKernelRoute(functionName: string): boolean {
  const prefix = 'ArrayDecoder.';
  if (!functionName.startsWith(prefix)) return false;
  const methodName = functionName.slice(prefix.length);
  return (
    (methodName.startsWith('decode') || methodName.startsWith('dequantize')) &&
    !NON_KERNEL_DECODER_METHOD_ALLOWLIST.has(functionName)
  );
}

function findInlineFloatArithmetic({ fileName, source }: DecoderSource): Violation[] {
  const sourceFile = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true);
  const violations: Violation[] = [];

  function visit(node: ts.Node): void {
    const functionName = containingFunctionName(node);
    const guardedDivision =
      isDivision(node) && (requiresKernelRoute(functionName) || isKnownQuantizationDivision(node));

    if (
      (isMathExponentialCall(node) || guardedDivision) &&
      !INLINE_FLOAT_ARITHMETIC_ALLOWLIST.has(functionName)
    ) {
      violations.push({
        fileName,
        functionName,
        line: sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1,
        expression: node.getText(sourceFile),
      });
    }
    ts.forEachChild(node, visit);
  }

  visit(sourceFile);
  return violations;
}

function importedKernelNames(sourceFile: ts.SourceFile): Set<string> {
  const names = new Set<string>();
  for (const statement of sourceFile.statements) {
    if (
      !ts.isImportDeclaration(statement) ||
      !ts.isStringLiteral(statement.moduleSpecifier) ||
      statement.moduleSpecifier.text !== DECODER_KERNEL_MODULE
    ) {
      continue;
    }
    for (const element of statement.importClause?.namedBindings &&
    ts.isNamedImports(statement.importClause.namedBindings)
      ? statement.importClause.namedBindings.elements
      : []) {
      names.add(element.name.text);
    }
  }
  return names;
}

function arrayDecoderMethods(sourceFile: ts.SourceFile): Map<string, ts.MethodDeclaration> {
  const methods = new Map<string, ts.MethodDeclaration>();
  function visit(node: ts.Node): void {
    if (
      ts.isMethodDeclaration(node) &&
      ts.isClassDeclaration(node.parent) &&
      node.parent.name?.text === 'ArrayDecoder'
    ) {
      methods.set(`ArrayDecoder.${node.name.getText(sourceFile)}`, node);
    }
    ts.forEachChild(node, visit);
  }
  visit(sourceFile);
  return methods;
}

function missingKernelRoutes(source: string): string[] {
  const sourceFile = ts.createSourceFile(
    'decoder.ts',
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TS
  );
  const importedKernels = importedKernelNames(sourceFile);
  const methods = arrayDecoderMethods(sourceFile);

  function directlyCallsKernel(method: ts.MethodDeclaration): boolean {
    const callableNames = new Set(importedKernels);
    let routed = false;

    function visit(node: ts.Node): void {
      if (routed) return;
      if (
        ts.isBinaryExpression(node) &&
        node.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
        ts.isIdentifier(node.left) &&
        ts.isIdentifier(node.right) &&
        importedKernels.has(node.right.text)
      ) {
        callableNames.add(node.left.text);
      }
      if (
        ts.isCallExpression(node) &&
        ts.isIdentifier(node.expression) &&
        callableNames.has(node.expression.text)
      ) {
        routed = true;
        return;
      }
      ts.forEachChild(node, visit);
    }

    ts.forEachChild(method, visit);
    return routed;
  }

  function reachesKernel(functionName: string, seen = new Set<string>()): boolean {
    if (seen.has(functionName)) return false;
    seen.add(functionName);

    const method = methods.get(functionName);
    if (!method) return false;
    if (directlyCallsKernel(method)) return true;

    let routed = false;
    function visit(node: ts.Node): void {
      if (routed) return;
      if (
        ts.isCallExpression(node) &&
        ts.isPropertyAccessExpression(node.expression) &&
        node.expression.expression.kind === ts.SyntaxKind.ThisKeyword &&
        reachesKernel(`ArrayDecoder.${node.expression.name.text}`, new Set(seen))
      ) {
        routed = true;
        return;
      }
      ts.forEachChild(node, visit);
    }
    ts.forEachChild(method, visit);
    return routed;
  }

  return [...methods.keys()].filter(
    (functionName) => requiresKernelRoute(functionName) && !reachesKernel(functionName)
  );
}

describe('ArrayDecoder kernel routing', () => {
  it('routes every scalar float path through the shared TypeScript kernels', () => {
    const decoder = decoderSources().find(({ fileName }) => fileName === 'decoder.ts');
    expect(
      decoder,
      'array-decoder/decoder.ts must remain part of the scanned source set'
    ).toBeDefined();

    const missing = missingKernelRoutes(decoder!.source);
    expect(
      missing,
      `${missing.join(', ')} no longer reaches a kernel imported from ` +
        '`src/wasm/typescript/decode.ts`. Restore that route to preserve the ' +
        'ArrayDecoder README Priority-Dispatch Order.'
    ).toEqual([]);
  });

  it('keeps inline float dequantization inside the explicit allowlist', () => {
    const violations = decoderSources().flatMap(findInlineFloatArithmetic);

    expect(
      violations,
      'Move inline float decoding to `src/wasm/typescript/decode.ts` and call it ' +
        'through the ArrayDecoder README Priority-Dispatch Order.'
    ).toEqual([]);
  });

  it('detects representative inline dequantization arithmetic', () => {
    const source = `
      class ArrayDecoder {
        makePerChannelDequant = () => Math.exp(1) + Math.expm1(2) + 3 / 255;
        decodeInline() {
          const scale = 1 / maxCode;
          return Math.exp(1) + Math.expm1(2) + 3 / 255 + 4 / 65535 + 5 / max_int + scale;
        }
      }
    `;

    expect(findInlineFloatArithmetic({ fileName: 'synthetic.ts', source })).toEqual([
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 5,
        expression: '1 / maxCode',
      },
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 6,
        expression: 'Math.exp(1)',
      },
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 6,
        expression: 'Math.expm1(2)',
      },
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 6,
        expression: '3 / 255',
      },
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 6,
        expression: '4 / 65535',
      },
      {
        fileName: 'synthetic.ts',
        functionName: 'ArrayDecoder.decodeInline',
        line: 6,
        expression: '5 / max_int',
      },
    ]);
  });

  it('detects a violation spliced into the real decoder source', () => {
    const decoder = decoderSources().find(({ fileName }) => fileName === 'decoder.ts')!;
    const mutated = decoder.source.replace(
      'private decodeGeologScalar(',
      'private decodeGeologScalarViolation = () => 1 / 255;\n\n  private decodeGeologScalar('
    );

    expect(findInlineFloatArithmetic({ fileName: decoder.fileName, source: mutated })).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          fileName: 'decoder.ts',
          functionName: 'ArrayDecoder.decodeGeologScalarViolation',
          expression: '1 / 255',
        }),
      ])
    );
  });

  it('rejects a guarded method that replaces kernel routing with inline math', () => {
    const decoder = decoderSources().find(({ fileName }) => fileName === 'decoder.ts')!;
    const mutated = decoder.source.replace(
      /[ ]{2}private decodeGeologScalar\([\s\S]*?\n[ ]{2}}\n\n[ ]{2}\/\*\*/,
      `  private decodeGeologScalar(): Float32Array {
    return new Float32Array([1 / 4_294_967_295]);
  }

  /**`
    );

    expect(missingKernelRoutes(mutated)).toContain('ArrayDecoder.decodeGeologScalar');
  });
});
