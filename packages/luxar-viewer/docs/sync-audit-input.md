# Input Package Synchronization Audit Report

**Audit Date**: 2025-12-08
**Package**: `luxar-viewer.input`
**Version**: 1.0.0 (per SPECIFICATIONS.md)
**Auditor**: Claude Sonnet 4.5

---

## Executive Summary

This audit examines the synchronization between technical specifications (SPECIFICATIONS.md), implementation documentation (README.md), and actual code for the input package. The input package provides context-aware keyboard and mouse input handling with intelligent conflict resolution.

**Overall Status**: ⚠️ **MODERATE MISALIGNMENT**

**Key Findings**:
- **Critical Discrepancy**: InputContext enum values differ between SPEC and implementation
- **Missing Implementation**: Context-specific key filtering maps not fully implemented as specified
- **Documentation Gap**: README is more comprehensive than SPECIFICATIONS in some areas
- **Utility Functions**: Well-factored utilities are present but not documented in SPEC
- **Good Architecture**: Core concepts are sound, but details have diverged

---

## Detailed Findings

### 1. InputContext Enum Discrepancy

**Severity**: 🔴 **CRITICAL**

#### SPECIFICATIONS.md (lines 32-39):
```typescript
enum InputContext {
  NAVIGATION = 0,      // Default 3D navigation (orbit/arcball)
  FLY_CONTROLS = 1,    // Fly mode active (WASD enabled)
  TYPING = 2,          // Text input focused (all shortcuts disabled)
  DIMENSION_NAV = 3,   // nD dimension navigation (1-9, [, ])
  UI_OVERLAY = 4,      // UI panel open (Tab, Enter, Esc)
  MODAL = 5,           // Modal dialog (highest priority)
}
```

#### Actual Implementation (input-context-manager.ts, lines 20-26):
```typescript
export enum InputContext {
  NAVIGATION = 'navigation',           // String literal, not number
  FLY_CONTROLS = 'fly_controls',       // Underscore, not camelCase
  TYPING = 'typing',                   // String literal, not number
  UI_INTERACTION = 'ui_interaction',   // Different name than spec
  DIMENSION_NAV = 'dimension_nav',     // Underscore, not camelCase
}
```

**Differences**:
1. **Type**: SPEC uses numeric enum (0-5), implementation uses string enums
2. **Values**: SPEC has `UI_OVERLAY` and `MODAL`, implementation has only `UI_INTERACTION` (missing MODAL)
3. **Naming**: Implementation uses snake_case strings, SPEC implies numeric ordering
4. **Priority System**: SPEC explicitly defines priority by enum value order, implementation uses separate priority field

**Impact**: The priority-based context system described in SPEC (where higher enum value = higher priority) is NOT how the implementation works. Implementation uses a separate `priority` field in `ContextConfig`.

---

### 2. Context-Specific Key Sets

**Severity**: 🟡 **MODERATE**

#### SPECIFICATIONS.md (lines 146-178):
Defines explicit `CONTEXT_KEYS` mapping for each context:
```typescript
const CONTEXT_KEYS = {
  [InputContext.NAVIGATION]: ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'h', 'p', 'r', 'v', 'f', 'c', 'm', 'o', 'n'],
  [InputContext.FLY_CONTROLS]: ['w', 'a', 's', 'd', 'q', 'e', 'i', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'],
  [InputContext.DIMENSION_NAV]: ['1', '2', '3', '4', '5', '6', '7', '8', '9', '[', ']'],
  [InputContext.UI_OVERLAY]: ['Tab', 'Enter', 'Escape'],
  [InputContext.MODAL]: ['Escape', 'Enter'],
};
```

#### Actual Implementation (input-context-manager.ts, lines 73-117):
Uses `allowedKeys` and `blockedKeys` arrays in `ContextConfig`:
```typescript
// Navigation context
blockedKeys: [...config.input.keyboard.flyModeKeys], // Block WASD in orbit mode

// Fly controls context
allowedKeys: [...config.input.keyboard.flyModeKeys, 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'],

// Typing context
allowedKeys: [], // No shortcuts while typing

// Dimension navigation context
allowedKeys: [...config.input.keyboard.dimensionKeys],
```

**Differences**:
1. **Structure**: SPEC has a flat map, implementation uses `ContextConfig` objects
2. **Configuration Source**: Implementation pulls keys from central config, SPEC hard-codes them
3. **Navigation Keys**: SPEC lists all navigation keys explicitly, implementation uses `blockedKeys` approach
4. **Missing Context**: SPEC defines `UI_OVERLAY` and `MODAL` key sets, implementation only has `UI_INTERACTION`

**Assessment**: Implementation is more flexible (config-driven), but deviates from SPEC's explicit design.

---

### 3. Context Stack Implementation

**Severity**: 🟢 **ALIGNED**

Both SPEC and implementation correctly implement a context stack:

#### SPECIFICATIONS.md (lines 56-74):
```typescript
class InputContextManager {
  private contextStack: InputContext[] = [InputContext.NAVIGATION];
  getCurrentContext(): InputContext;
  pushContext(context: InputContext): void;
  popContext(): InputContext | undefined;
}
```

#### Actual Implementation (input-context-manager.ts, lines 60-138):
```typescript
private currentContext: InputContext = InputContext.NAVIGATION;
private contextStack: InputContext[] = [];

public pushContext(context: InputContext): void { ... }
public popContext(): void { ... }
public setContext(context: InputContext): void { ... }
```

**Minor Difference**: Implementation has separate `currentContext` field and empty initial stack, vs SPEC's stack-only approach with `[NAVIGATION]` initial state. Both work correctly.

---

### 4. Typing Detection

**Severity**: 🟢 **ALIGNED**

Both SPEC and implementation correctly detect typing in input fields.

#### SPECIFICATIONS.md (lines 205-227):
```typescript
function isTypingInInput(): boolean {
  const active = document.activeElement;
  if (!active) return false;

  if (active.tagName === 'INPUT') {
    const type = (active as HTMLInputElement).type;
    const textTypes = ['text', 'search', 'url', 'email', 'password', 'tel', 'number'];
    return textTypes.includes(type);
  }

  if (active.tagName === 'TEXTAREA') return true;
  if (active.getAttribute('contenteditable') === 'true') return true;

  return false;
}
```

#### Actual Implementation (input-handler.ts, lines 705-717):
```typescript
private isTypingInInput(): boolean {
  const activeElement = document.activeElement;
  if (!activeElement) return false;

  const tagName = activeElement.tagName.toLowerCase();
  return (
    tagName === 'input' ||
    tagName === 'textarea' ||
    tagName === 'select' ||  // Additional check
    activeElement.getAttribute('contenteditable') === 'true'
  );
}
```

**Differences**:
1. Implementation adds `select` element check (good addition)
2. Implementation doesn't check input types (accepts all input types)
3. SPEC's approach is more precise (filters specific text input types)

**Assessment**: Implementation is more permissive but functionally correct.

---

### 5. Key Routing Algorithm

**Severity**: 🟡 **MODERATE**

#### SPECIFICATIONS.md (lines 96-115):
Defines a 5-step algorithm:
```typescript
function handleKeyDown(event: KeyboardEvent): void {
  // 1. Check if typing in text field
  if (isTypingInInput()) return;

  // 2. Get current context
  const context = contextManager.getCurrentContext();

  // 3. Check if current context should handle this key
  if (!shouldHandleKey(event.key, context)) return;

  // 4. Prevent default browser behavior
  event.preventDefault();

  // 5. Route to appropriate handler
  routeKeyEvent(event, context);
}
```

#### Actual Implementation:

**Split Across Two Classes**:

1. **input-handler.ts** (lines 350-525): Handles specific key events directly
2. **input-context-manager.ts** (lines 204-248): Provides `handleKeyEvent()` method

**Issue**: The routing is NOT centralized as SPEC describes. Instead:
- `input-handler.ts` directly handles most keys in large switch statements
- `input-context-manager.ts` provides a binding system that is NOT used for most keys
- Fly control keys are checked early and routed separately

**Code Example** (input-handler.ts, lines 350-374):
```typescript
private onKeyDown(event: KeyboardEvent): void {
  // Check if fly controls are active and should handle this key
  const flyControls = this.sceneManager.controls.getFlyControls();
  if (flyControls && flyControls.enabled) {
    const flyKeys = [...config.input.keyboard.flyModeKeys, 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];
    const keyLower = event.key.toLowerCase();

    const isFlyKey = flyKeys.some((k) => k.toLowerCase() === keyLower) || event.key.startsWith('Arrow');

    if (isFlyKey) {
      if (!this.isTypingInInput()) {
        flyControls.handleKeyDown(event);
        return;
      }
    }
  }

  // Handle remaining keys that aren't context-specific
  switch (event.key) {
    case 'Shift': ...
    case 'h': ...
    case 'c': ...
    // ... 150+ lines of switch cases
  }
}
```

**Assessment**: Implementation does NOT follow SPEC's clean routing algorithm. It's a pragmatic but less elegant solution.

---

### 6. Modifier Key Handling

**Severity**: 🟢 **ALIGNED**

#### SPECIFICATIONS.md (lines 259-273):
```typescript
function getEffectiveKey(event: KeyboardEvent): string {
  let key = event.key;
  const modifiers = [];
  if (event.ctrlKey || event.metaKey) modifiers.push('Control');
  if (event.altKey) modifiers.push('Alt');
  if (event.shiftKey) modifiers.push('Shift');

  if (modifiers.length > 0) {
    key = modifiers.join('+') + '+' + key;
  }
  return key;
}
```

#### Actual Implementation (input-context-manager.ts, lines 341-350):
```typescript
private getBindingKeyFromEvent(event: KeyboardEvent): string {
  const parts = [event.key.toLowerCase()];

  if (event.ctrlKey) parts.push('ctrl');
  if (event.shiftKey) parts.push('shift');
  if (event.altKey) parts.push('alt');
  if (event.metaKey) parts.push('meta');

  return parts.sort().join('+');
}
```

**Differences**:
1. Implementation distinguishes `ctrlKey` and `metaKey` separately (more precise)
2. Implementation sorts modifier parts (canonical ordering)
3. Implementation always lowercases key

**Assessment**: Implementation is better than SPEC.

---

### 7. Undocumented Components

**Severity**: 🟡 **MODERATE**

#### Missing from SPECIFICATIONS.md:

1. **input-handler-utils.ts** (346 lines) - Pure utility functions:
   - `getNonDisplayedDimensions()`
   - `calculateStepSize()`
   - `calculateNextPosition()`
   - `mapKeyToDimension()`
   - `formatDimensionValue()`
   - `generateNavigationHelp()`
   - `isNavigationKey()`
   - `calculateFovChange()`
   - `shouldBlockShortcut()`

2. **NavigationConfig interface** - Configuration for keyboard navigation
3. **Dimension navigation algorithms** - Step size calculation, wrap-around logic
4. **FOV adjustment** - Field of view change calculations
5. **Key binding system** - `KeyBinding` and `ContextConfig` interfaces

**These are significant omissions from the specification.**

---

### 8. Global Keys vs Context Keys

**Severity**: 🟡 **MODERATE**

#### SPECIFICATIONS.md (lines 189-191):
```typescript
// Global keys always handled
const globalKeys = ['h', 'Space', 'Escape', 'm', 'Control+l'];
return contextKeys.includes(key) || globalKeys.includes(key);
```

#### Actual Implementation:
**No explicit global keys list**. Instead, global keys are handled in the main switch statement in `input-handler.ts`, outside of the context system.

**Examples** (input-handler.ts):
- Line 382-386: 'h' always handled (help)
- Line 431-440: Ctrl+L always handled (debug console)
- Line 443-458: 'm' handled (data monitor)
- Line 487-493: Space handled conditionally

**Assessment**: Implementation is more complex but functional. No centralized "global keys" concept as SPEC describes.

---

### 9. Conflict Resolution

**Severity**: 🟢 **ALIGNED (Conceptually)**

Both SPEC and implementation prevent WASD keys from working in non-fly modes, though they achieve it differently.

#### SPECIFICATIONS.md (lines 289-316):
Shows priority-based resolution with explicit context checks.

#### Actual Implementation:
- Lines 352-374: Early check for fly controls
- Lines 369: `if (!this.isTypingInInput())` prevents typing conflicts
- Lines 73-80 (context-manager): `blockedKeys: [...config.input.keyboard.flyModeKeys]` in navigation mode

**Assessment**: Core conflict resolution logic is sound, though implementation details differ.

---

### 10. README Accuracy

**Severity**: 🟢 **GOOD**

The README.md accurately describes the implemented system (not the specified system). Key sections:

- **Lines 92-102**: Correctly lists implemented `InputContext` enum values (string-based)
- **Lines 139-149**: Correctly describes priority system using separate priority field
- **Lines 151-166**: Correctly shows context stack usage
- **Lines 196-228**: Accurately describes keyboard event routing (though implementation differs from SPEC)
- **Lines 303-349**: Documents typing detection accurately

**The README documents what was built, not what was specified.**

---

## Missing from SPECIFICATIONS.md

### Critical Omissions:

1. **Utility Functions Module** (`input-handler-utils.ts`)
   - Dimension navigation algorithms
   - Step size calculation logic
   - Position clamping and wrapping
   - Value formatting for display

2. **Navigation Configuration**
   - `NavigationConfig` interface
   - Step size multipliers
   - Fine/coarse control divisors
   - Wrap-around behavior

3. **Integration with Config System**
   - How input system reads from central config
   - Relationship between `config.input.keyboard` and context manager

4. **Key Binding System Details**
   - `KeyBinding` interface
   - `ContextConfig` interface
   - How bindings are registered and managed
   - Passthrough behavior for unhandled keys

5. **Dimension Navigation Details**
   - How number keys (1-9) map to dimensions
   - How [ ] keys navigate with step sizes
   - Discrete vs continuous dimension handling
   - Boundary clamping logic

6. **FOV Control**
   - Shift+wheel FOV adjustment
   - FOV calculation algorithms
   - Min/max clamping

7. **Mouse Event Handling**
   - Mouse button routing
   - Touch event handling
   - Canvas interaction detection

---

## Recommendations

### 1. Update SPECIFICATIONS.md (Priority: HIGH)

**Action Items**:

- [ ] Change `InputContext` enum to use string literals matching implementation
- [ ] Remove `UI_OVERLAY` and `MODAL` contexts (not implemented) OR implement them
- [ ] Document the priority field in `ContextConfig` as the actual priority mechanism
- [ ] Add section on `input-handler-utils.ts` utility functions
- [ ] Document `NavigationConfig` interface and navigation algorithms
- [ ] Add section on dimension navigation (step calculation, position updates)
- [ ] Document key binding registration system
- [ ] Clarify that key routing is split between `InputHandler` and `InputContextManager`

### 2. Align Implementation to SPEC (Priority: MEDIUM)

**Option A: Implement Missing SPEC Features**:
- [ ] Add `MODAL` context for highest-priority modal dialogs
- [ ] Centralize key routing as SPEC describes
- [ ] Implement explicit `CONTEXT_KEYS` mapping
- [ ] Use numeric enum values with priority based on enum order

**Option B: Accept Current Implementation** (Recommended):
- [ ] Update SPEC to match current implementation
- [ ] Document rationale for string-based enums
- [ ] Document split routing architecture
- [ ] Acknowledge config-driven key lists are better than hard-coded

### 3. Documentation Improvements (Priority: MEDIUM)

- [ ] Add "Implementation Notes" section to SPEC explaining deviations
- [ ] Cross-reference between SPEC and README
- [ ] Add examples showing actual API usage
- [ ] Document the relationship between `InputHandler` and `InputContextManager`
- [ ] Add diagrams showing event flow through the system

### 4. Code Refactoring (Priority: LOW)

**If time permits**:
- [ ] Centralize key routing as SPEC originally envisioned
- [ ] Extract global keys list to config
- [ ] Consolidate switch statements into cleaner routing table
- [ ] Make context system actually control routing (not just track state)

---

## Version History Audit

### SPECIFICATIONS.md Changelog:

```markdown
- **v1.0.0** (2025-01-30): Initial specification
  - Context-based input routing with priority system
  - Context stack for nested states
  - Automatic typing detection
  - Mode-specific key filtering (WASD only in fly mode)
  - Modifier key support (Ctrl, Alt, Shift)
  - Conflict resolution algorithms
```

**Issue**: Version date is 2025-01-30, but audit date is 2025-12-08. If this is a typo (should be 2024-01-30), specification is ~11 months old. If genuine future date, there's a timeline error.

**Recommendation**: Verify and correct the version date.

---

## Synchronization Matrix

| Component | SPEC | README | Code | Status |
|-----------|------|--------|------|--------|
| InputContext enum values | Numeric (0-5) | String literals | String literals | ❌ MISALIGNED |
| InputContext names | 6 contexts | 5 contexts | 5 contexts | ❌ MISALIGNED |
| Priority system | Enum order | Priority field | Priority field | ⚠️ DIFFERENT |
| Context stack | Present | Present | Present | ✅ ALIGNED |
| Typing detection | Present | Present | Present | ✅ ALIGNED |
| Key routing | Centralized | Distributed | Distributed | ⚠️ DIFFERENT |
| Modifier handling | Basic | Advanced | Advanced | ✅ ALIGNED |
| Global keys | Explicit list | Implicit | Implicit | ⚠️ DIFFERENT |
| Context key sets | Hard-coded map | Config-driven | Config-driven | ⚠️ DIFFERENT |
| Utility functions | Not documented | Not documented | Implemented | ❌ MISSING |
| Dimension navigation | Basic | Detailed | Detailed | ⚠️ PARTIAL |
| FOV control | Not documented | Brief mention | Implemented | ❌ MISSING |
| Key binding system | Not detailed | Present | Present | ⚠️ PARTIAL |

**Legend**:
- ✅ ALIGNED: SPEC, README, and code all match
- ⚠️ DIFFERENT: Core concept present but implementation differs
- ❌ MISALIGNED: Significant difference or missing entirely
- ❌ MISSING: Present in code but not in SPEC

---

## Code Quality Assessment

### Strengths:

1. **Clean Separation**: Utility functions extracted to separate module
2. **Type Safety**: Good use of TypeScript types and interfaces
3. **Configurability**: Input behavior driven by central config
4. **Context Awareness**: Core context system works well
5. **Typing Detection**: Properly prevents shortcuts during text input

### Weaknesses:

1. **Routing Complexity**: Split between two classes, not centralized
2. **Large Switch Statement**: 150+ line switch in `onKeyDown()`
3. **Manual Key Lists**: Fly keys checked manually in multiple places
4. **Binding System Unused**: Context manager has binding registration but most keys bypass it
5. **Documentation Drift**: SPEC and implementation have diverged significantly

---

## Testing Recommendations

### Unit Tests Needed:

1. **Context Manager**:
   - Context stack push/pop behavior
   - Context switching
   - Priority resolution
   - Key filtering by context

2. **Utility Functions**:
   - Step size calculation
   - Position clamping
   - Dimension mapping
   - FOV calculation

3. **Typing Detection**:
   - Input element detection
   - Contenteditable handling
   - Select element handling

4. **Conflict Resolution**:
   - WASD blocked in orbit mode
   - WASD enabled in fly mode
   - All shortcuts blocked while typing

### Integration Tests Needed:

1. **Context Transitions**:
   - Orbit → Fly → Typing → Orbit
   - Key behavior changes with context

2. **Dimension Navigation**:
   - Number key selection
   - [ ] navigation with various step sizes
   - Discrete vs continuous dimensions

3. **Global Keys**:
   - Help (H) works in all contexts
   - Debug console (Ctrl+L) works in all contexts
   - Fullscreen (Space) context awareness

---

## Risk Assessment

### High Risk:

1. **Context System Mismatch**: SPEC describes numeric priority, code uses string enums and config-based priority
   - **Impact**: Future developers may implement wrong system
   - **Mitigation**: Update SPEC immediately

2. **Missing MODAL Context**: SPEC defines it, code doesn't implement it
   - **Impact**: Modals may not block input correctly
   - **Mitigation**: Implement or remove from SPEC

### Medium Risk:

1. **Undocumented Utilities**: `input-handler-utils.ts` has complex algorithms not in SPEC
   - **Impact**: Hard to maintain or re-implement
   - **Mitigation**: Document algorithms in SPEC

2. **Distributed Routing**: Key routing split across multiple places
   - **Impact**: Hard to trace execution path
   - **Mitigation**: Centralize or document architecture clearly

### Low Risk:

1. **README Accuracy**: README documents implementation, not specification
   - **Impact**: Confusion about intended design
   - **Mitigation**: Add note explaining SPEC vs implementation

---

## Conclusion

The input package has **moderate misalignment** between SPECIFICATIONS.md and actual implementation. The core concepts are sound and the code works well, but significant details have diverged:

1. **Critical Issue**: InputContext enum type and values don't match
2. **Architecture Drift**: Key routing is distributed, not centralized as specified
3. **Missing Documentation**: Utility functions and algorithms undocumented in SPEC
4. **Missing Implementation**: MODAL context specified but not implemented

**Recommendation**: **Update SPECIFICATIONS.md to match implementation** rather than refactoring code to match SPEC. The current implementation is more flexible and pragmatic than the original specification.

**Priority Actions**:
1. Fix InputContext enum documentation (HIGH)
2. Document utility functions and algorithms (HIGH)
3. Add implementation notes explaining deviations (MEDIUM)
4. Decide on MODAL context: implement or remove (MEDIUM)

---

**Audit Complete**

Files Examined:
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/input/SPECIFICATIONS.md`
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/input/README.md`
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/input/input-handler.ts`
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/input/input-context-manager.ts`
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/input/input-handler-utils.ts`
- `/Users/loic.royer/workspace/python/luxar/packages/luxar-viewer/src/config/index.ts` (input section)
