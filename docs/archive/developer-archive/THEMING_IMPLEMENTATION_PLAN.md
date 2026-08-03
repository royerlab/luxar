> **⚠️ Archived — implemented plan, not maintained.** The work described here **shipped**; this plan is kept for design history and was not updated afterwards, so details may not match current code. Do not treat it as current guidance. See [the archive README](../README.md) for status labels and retention policy.

> **Status**: ✅ **ALL PHASES COMPLETE** - Implementation 100% Finished + Final Polish
> **Estimated Effort**: 3-4 weeks (15-18 working days)
> **Priority**: Medium (Quality of Life improvement)
> **Complexity**: Medium-High
> **Last Updated**: 2025-12-15

## Implementation Progress

### ✅ Phase 1: Foundation (Days 1-5) - **100% COMPLETE**
- ✅ Theme type definitions created
- ✅ ThemeManager implemented with CSS variable injection
- ✅ Three themes created (dark, light, high-contrast)
- ✅ CSS infrastructure established (reset, utilities, base styles)
- ✅ UIComponent base class created

### ✅ Phase 2: Component Migration (Days 6-10) - **100% COMPLETE**
- ✅ helpers.ts migrated (error dialog, help overlay, loading indicator)
- ✅ dimension-sliders.ts migrated (complex slider controls)
- ✅ debug-console.ts migrated (removed applyStyles() method, proper BEM naming)
- ✅ dataset-browser.ts migrated (file browser UI)

### ✅ Phase 3: Complex Components (Days 11-15) - **100% COMPLETE**
- ✅ data-loading-monitor.css created (comprehensive styling)
- ✅ Key template functions refactored (renderMetricCard, renderProgressBar, renderRecommendation)
- ✅ rendering-controls.css created (lil-gui theme integration with CSS variables)
- ✅ Third-party library theming complete (lil-gui fully themed)

### ✅ Phase 4: Polish & Testing (Days 16-20) - **100% COMPLETE**
- ✅ Theme selector added to rendering controls
- ✅ URL parameter support (?theme=light)
- ✅ Theme persistence via localStorage
- ✅ ThemeManager.dispose() and resetInstance() methods with full JSDoc
- ✅ UIComponent.addEventListener() with comprehensive JSDoc documenting arrow function binding fix
- ✅ Documentation complete and verified

### 📊 Current Metrics
- **CSS Bundle**: 39.82 KB (gzipped: 5.78 KB)
- **JS Bundle Reduction**: -13 KB
- **Inline Styles Removed**: 400+
- **CSS Classes Created**: 130+
- **Components Migrated**: 8 fully complete (helpers, dimension-sliders, debug-console, dataset-browser, data-loading-monitor, rendering-controls, loading-indicator, help-overlay)
- **Themes Available**: 3 (runtime switchable)
- **Tests**: 1030 passing ✅
- **TypeScript**: 0 errors ✅
- **Documentation**: Complete JSDoc coverage ✅

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Architectural Vision](#architectural-vision)
4. [Detailed Implementation Plan](#detailed-implementation-plan)
5. [Migration Strategy](#migration-strategy)
6. [Testing Strategy](#testing-strategy)
7. [Rollout Plan](#rollout-plan)
8. [Appendix](#appendix)

---

## Executive Summary

### Problem Statement

The Luxar viewer currently uses **CSS-in-JS with partial config consolidation**, resulting in:
- ❌ **162 hardcoded color values** scattered across 8 UI files
- ❌ **No theme switching capability** - dark theme only
- ❌ **Tight coupling** between styling and component logic
- ❌ **Difficult maintenance** - style changes require editing multiple files
- ❌ **Poor accessibility** - no high-contrast mode support
- ❌ **Inconsistent styling** - same colors defined differently in different places

### Solution Overview

Implement a **three-layer architecture** with modular themes:

```
Layer 1: Theme System (colors, spacing, typography)
         ↓
Layer 2: Component Styles (CSS files, design tokens)
         ↓
Layer 3: Component Logic (TypeScript, no styling)
```

### Expected Benefits

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| Themes available | 1 (dark) | 3+ | Infinite extensibility |
| Hardcoded colors | 162 | 0 | 100% elimination |
| Style reusability | ~10% | ~80% | Via utility classes |
| CSS bundle size | 0KB (in JS) | ~20KB (cached) | Better caching |
| Theme switch time | N/A | <100ms | Instant |
| Accessibility | AA | AAA | High-contrast support |
| Maintenance effort | High | Low | Single file edits |

### Success Criteria

- ✅ Support 3+ themes (dark, light, high-contrast) with instant switching
- ✅ 0 hardcoded colors in component code
- ✅ Complete separation: logic → templates → styles
- ✅ CSS hot reload works (HMR)
- ✅ All existing functionality preserved
- ✅ Performance maintained or improved
- ✅ All tests passing

---

## Current State Analysis

### Existing Architecture

**Current file**: `src/config/index.ts`

```typescript
config.ui.styles = {
  colors: {
    success: '#4CAF50',
    warning: '#FFC107',
    error: '#f44336',
    // ... 20 more colors
  },
  spacing: {
    panelPadding: 15,
    sectionGap: 15,
    // ... 15 more values
  },
  effects: {
    borderRadius: 8,
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
    // ... 10 more values
  },
};
```

**Current usage pattern**:
```typescript
// Some files use config (good!)
element.style.color = config.ui.styles.colors.primaryText;

// But many still hardcode (bad!)
element.style.background = 'rgba(30, 30, 30, 0.95)';
element.style.color = '#e0e0e0';
element.style.padding = '20px';
```

### Problems Breakdown

#### Problem 1: Inconsistent Config Usage
- Only 27 references to `config.ui.styles` across all UI files
- 162 hardcoded rgba/hex values still remain
- No enforcement mechanism

#### Problem 2: No Runtime Theme Switching
- Config values are static at build time
- No mechanism to swap themes without reloading
- Cannot support user preferences

#### Problem 3: CSS-in-JS Limitations
- Large JavaScript bundle (styles in code)
- No CSS caching
- No browser DevTools CSS editing
- Poor developer experience (no syntax highlighting)

#### Problem 4: Tight Coupling
- Styling mixed with business logic
- Cannot reuse styles across components
- Hard to maintain consistency

### Current Metrics

```bash
Analyzed: src/ui/**/*.ts
─────────────────────────────
Inline styles:        352 occurrences
cssText assignments:  182 occurrences
Hardcoded colors:     162 occurrences
Config references:    27 occurrences
CSS files:            0 files
Theme support:        0% (dark theme only, hardcoded)
```

---

## Architectural Vision

### Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ LAYER 3: COMPONENT LOGIC (TypeScript)                       │
│                                                              │
│  • Business logic and state management                       │
│  • Event handling and user interactions                      │
│  • Data transformation and validation                        │
│  • API calls and data fetching                               │
│  • NO STYLING CODE                                           │
│                                                              │
│  Example:                                                    │
│  class ErrorDialog extends UIComponent {                     │
│    handleDismiss() { this.close(); }                         │
│    validateMessage() { ... }                                 │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
                            ↓ uses
┌─────────────────────────────────────────────────────────────┐
│ LAYER 2: COMPONENT TEMPLATES (Structure + Classes)          │
│                                                              │
│  • DOM structure (HTML)                                      │
│  • Layout primitives (flex, grid)                            │
│  • CSS class names (BEM methodology)                         │
│  • Design tokens (semantic meaning)                          │
│  • NO INLINE STYLES, NO COLORS, NO MAGIC NUMBERS             │
│                                                              │
│  Example:                                                    │
│  <div class="luxar-error-dialog luxar-surface-elevated">    │
│    <div class="luxar-error-dialog__header">                 │
│      <h2 class="luxar-text-error luxar-text-xl">...</h2>    │
│    </div>                                                    │
│  </div>                                                      │
└─────────────────────────────────────────────────────────────┘
                            ↓ styled by
┌─────────────────────────────────────────────────────────────┐
│ LAYER 1: THEME SYSTEM (Appearance)                          │
│                                                              │
│  • CSS custom properties (runtime theme switching)           │
│  • Component stylesheets (.css files)                        │
│  • Utility classes (spacing, typography, colors)             │
│  • Theme definitions (dark, light, high-contrast)            │
│  • Design tokens (semantic color names)                      │
│                                                              │
│  Example:                                                    │
│  .luxar-error-dialog {                                       │
│    background: var(--luxar-surface-elevated);                │
│    color: var(--luxar-text-primary);                         │
│    padding: var(--luxar-spacing-12);                         │
│    border-radius: var(--luxar-radius-lg);                    │
│  }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### File Organization

```
src/
├── themes/
│   ├── index.ts                  # Export all themes
│   ├── types.ts                  # Theme interface
│   ├── theme-manager.ts          # Runtime theme switching
│   ├── themes/
│   │   ├── dark.theme.ts         # Dark theme definition
│   │   ├── light.theme.ts        # Light theme definition
│   │   ├── high-contrast.theme.ts
│   │   ├── scientific-blue.theme.ts  # Example custom theme
│   │   └── README.md             # How to create custom themes
│   └── utils.ts                  # Theme utilities
│
├── styles/
│   ├── index.css                 # Main stylesheet entry point
│   ├── reset.css                 # CSS reset/normalize
│   ├── variables.css             # CSS custom properties (generated)
│   ├── base/
│   │   ├── typography.css        # Font loading, base text styles
│   │   ├── layout.css            # Grid system, containers
│   │   └── utilities.css         # Utility classes
│   ├── components/
│   │   ├── button.css            # Shared button styles
│   │   ├── panel.css             # Shared panel styles
│   │   ├── error-dialog.css
│   │   ├── help-overlay.css
│   │   ├── dimension-sliders.css
│   │   ├── debug-console.css
│   │   ├── dataset-browser.css
│   │   ├── data-loading-monitor.css
│   │   ├── performance-monitor.css
│   │   └── rendering-controls.css
│   └── themes/
│       ├── dark.css              # Dark theme CSS variable values
│       ├── light.css             # Light theme overrides
│       └── high-contrast.css     # High contrast overrides
│
└── ui/
    └── components/
        ├── base/
        │   └── ui-component.ts   # Base class with lifecycle management
        ├── error-dialog.ts       # Logic only (no styles!)
        ├── help-overlay.ts       # Logic only
        └── ... (other components)
```

---

## Detailed Implementation Plan

### **PHASE 1: Foundation (Week 1 - Days 1-5)**

#### Day 1: Theme System Core

**Task 1.1: Create Theme Type Definitions** (2 hours)
- File: `packages/luxar-viewer/src/themes/types.ts`
- Create `Theme` interface with all properties
- Create `ThemeColors`, `ThemeTypography`, `ThemeSpacing`, `ThemeEffects` sub-interfaces
- Add JSDoc documentation for each field
- Include examples in comments

**Task 1.2: Create Dark Theme** (2 hours)
- File: `packages/luxar-viewer/src/themes/themes/dark.theme.ts`
- Migrate all values from `config.ui.styles` to theme object
- Add missing colors from hardcoded values analysis
- Organize semantically (not by component)
- Full type checking

**Task 1.3: Create ThemeManager Class** (3 hours)
- File: `packages/luxar-viewer/src/themes/theme-manager.ts`
- Singleton pattern with `getInstance()`
- Methods: `registerTheme()`, `setTheme()`, `getCurrentTheme()`
- CSS variable injection via `applyThemeVariables()`
- LocalStorage persistence
- Observer pattern for theme change events
- **Critical**: Proper cleanup of old theme variables

**Task 1.4: Initial Integration** (1 hour)
- Import ThemeManager in `src/core/main.ts`
- Call `ThemeManager.getInstance()` on app init
- Verify CSS variables are set in DevTools

**Verification**:
```bash
# Check CSS variables are injected
# Open DevTools → Elements → :root
# Should see: --luxar-bg-primary, --luxar-text-primary, etc.
```

**Deliverables**:
- [ ] `src/themes/types.ts` (~150 lines)
- [ ] `src/themes/themes/dark.theme.ts` (~200 lines)
- [ ] `src/themes/theme-manager.ts` (~250 lines)
- [ ] Integration in `main.ts` (~5 lines)
- [ ] All TypeScript compiles
- [ ] CSS variables visible in DevTools

---

#### Day 2: CSS Infrastructure

**Task 2.1: Create CSS Directory Structure** (1 hour)
```bash
mkdir -p packages/luxar-viewer/src/styles/{base,components,themes}
touch packages/luxar-viewer/src/styles/{index,reset,variables}.css
touch packages/luxar-viewer/src/styles/base/{typography,layout,utilities}.css
```

**Task 2.2: Create CSS Reset** (30 min)
- File: `src/styles/reset.css`
- Modern CSS reset (based on Josh Comeau's reset)
- Ensure canvas/WebGL remains untouched

```css
/* src/styles/reset.css */
*, *::before, *::after {
  box-sizing: border-box;
}

* {
  margin: 0;
}

body {
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

img, picture, video, canvas, svg {
  display: block;
  max-width: 100%;
}

/* IMPORTANT: Don't reset the main canvas! */
canvas#app {
  /* Preserve existing styles from index.html */
}

button, input, textarea, select {
  font: inherit;
}
```

**Task 2.3: Create Base Utilities** (2 hours)
- File: `src/styles/base/utilities.css`
- Utility classes for common patterns
- Follow Tailwind-like convention but Luxar-prefixed

```css
/* Flexbox utilities */
.luxar-flex { display: flex; }
.luxar-flex-col { flex-direction: column; }
.luxar-items-center { align-items: center; }
.luxar-justify-between { justify-content: space-between; }
.luxar-gap-2 { gap: var(--luxar-spacing-2); }
.luxar-gap-4 { gap: var(--luxar-spacing-4); }
.luxar-gap-8 { gap: var(--luxar-spacing-8); }

/* Spacing utilities (padding) */
.luxar-p-2 { padding: var(--luxar-spacing-2); }
.luxar-p-4 { padding: var(--luxar-spacing-4); }
.luxar-p-8 { padding: var(--luxar-spacing-8); }
.luxar-p-12 { padding: var(--luxar-spacing-12); }

/* Spacing utilities (margin) */
.luxar-m-2 { margin: var(--luxar-spacing-2); }
.luxar-m-4 { margin: var(--luxar-spacing-4); }
.luxar-mb-4 { margin-bottom: var(--luxar-spacing-4); }
.luxar-mb-8 { margin-bottom: var(--luxar-spacing-8); }

/* Text utilities */
.luxar-text-primary { color: var(--luxar-text-primary); }
.luxar-text-secondary { color: var(--luxar-text-secondary); }
.luxar-text-muted { color: var(--luxar-text-muted); }
.luxar-text-sm { font-size: var(--luxar-text-sm); }
.luxar-text-md { font-size: var(--luxar-text-md); }
.luxar-text-lg { font-size: var(--luxar-text-lg); }
.luxar-text-bold { font-weight: var(--luxar-font-bold); }

/* Surface utilities */
.luxar-surface-primary { background: var(--luxar-bg-primary); }
.luxar-surface-secondary { background: var(--luxar-bg-secondary); }
.luxar-surface-elevated {
  background: var(--luxar-bg-secondary);
  backdrop-filter: var(--luxar-blur-md);
  box-shadow: var(--luxar-shadow-lg);
}

/* Border utilities */
.luxar-rounded-none { border-radius: var(--luxar-radius-none); }
.luxar-rounded-sm { border-radius: var(--luxar-radius-sm); }
.luxar-rounded-md { border-radius: var(--luxar-radius-md); }
.luxar-rounded-lg { border-radius: var(--luxar-radius-lg); }
.luxar-rounded-full { border-radius: var(--luxar-radius-full); }

/* Effect utilities */
.luxar-shadow-sm { box-shadow: var(--luxar-shadow-sm); }
.luxar-shadow-md { box-shadow: var(--luxar-shadow-md); }
.luxar-shadow-lg { box-shadow: var(--luxar-shadow-lg); }
.luxar-blur-sm { backdrop-filter: var(--luxar-blur-sm); }
.luxar-blur-md { backdrop-filter: var(--luxar-blur-md); }

/* Transition utilities */
.luxar-transition { transition: var(--luxar-transition-normal); }
.luxar-transition-fast { transition: var(--luxar-transition-fast); }
```

**Task 2.4: Vite Integration** (1 hour)
- Update `vite.config.ts` to import CSS
- Ensure proper CSS module support
- Configure CSS minification for production

```typescript
// vite.config.ts
import { defineConfig } from 'vite';

export default defineConfig({
  css: {
    devSourcemap: true, // CSS source maps for debugging
  },
  build: {
    cssCodeSplit: false, // Single CSS bundle for now
  },
});
```

**Task 2.5: Import CSS in Main** (30 min)
```typescript
// src/core/main.ts
import '../styles/index.css';  // ← Add this line

// Rest of imports...
import { ThemeManager } from '../themes/theme-manager';

// Initialize theme system
ThemeManager.getInstance();
```

**Verification**:
```bash
pnpm dev
# Open http://localhost:5173
# Check Network tab → should see CSS file loaded
# Check Elements → :root → should see CSS variables
```

**Deliverables**:
- [ ] CSS directory structure created
- [ ] `reset.css` implemented
- [ ] `utilities.css` with ~50 utility classes
- [ ] Vite configured for CSS
- [ ] CSS imported in main.ts
- [ ] Dev server shows CSS loading

---

#### Day 3-4: Light Theme + High Contrast Theme

**Task 3.1: Create Light Theme** (3 hours)
- File: `src/themes/themes/light.theme.ts`
- Invert dark theme colors
- Ensure sufficient contrast ratios (WCAG AA minimum)
- Test readability on bright displays

```typescript
// Key differences from dark theme:
export const lightTheme: Theme = {
  id: 'light',
  name: 'Light Theme',

  colors: {
    background: {
      primary: '#ffffff',
      secondary: 'rgba(250, 250, 250, 0.95)',
      tertiary: 'rgba(240, 240, 240, 0.8)',
      overlay: 'rgba(0, 0, 0, 0.3)',  // Inverted
    },
    text: {
      primary: '#1a1a1a',           // Inverted
      secondary: '#666666',
      muted: 'rgba(0, 0, 0, 0.6)',  // Inverted
    },
    border: {
      default: 'rgba(0, 0, 0, 0.15)',  // Darker borders for light bg
      subtle: 'rgba(0, 0, 0, 0.08)',
      strong: 'rgba(0, 0, 0, 0.25)',
    },
    // Semantic colors stay mostly the same (maybe adjust brightness)
  },
};
```

**Task 3.2: Create High Contrast Theme** (3 hours)
- File: `src/themes/themes/high-contrast.theme.ts`
- WCAG AAA contrast ratios (7:1 minimum)
- Pure colors for maximum differentiation
- No subtle effects (no transparency, minimal shadows)

```typescript
export const highContrastTheme: Theme = {
  id: 'high-contrast',
  name: 'High Contrast',

  colors: {
    background: {
      primary: '#000000',        // Pure black
      secondary: '#000000',
      tertiary: '#1a1a1a',
      overlay: 'rgba(0, 0, 0, 0.95)',
    },
    text: {
      primary: '#ffffff',        // Pure white (21:1 contrast!)
      secondary: '#ffffff',
      muted: '#cccccc',          // Still high contrast
    },
    semantic: {
      success: '#00ff00',        // Pure green (max contrast)
      warning: '#ffff00',        // Pure yellow
      error: '#ff0000',          // Pure red
      info: '#00ffff',           // Pure cyan
    },
  },
  effects: {
    borderRadius: {
      // No rounded corners for clarity
      none: '0px',
      sm: '0px',
      md: '0px',
      lg: '0px',
      full: '0px',
    },
    shadow: {
      // Stronger, more visible shadows
      sm: '0 0 0 2px #ffffff',
      md: '0 0 0 3px #ffffff',
      lg: '0 0 0 4px #ffffff',
    },
  },
};
```

**Task 3.3: Contrast Validation** (2 hours)
- Create utility to calculate contrast ratios
- Validate all text/background combinations
- Ensure WCAG AAA compliance

```typescript
// src/themes/utils.ts
export function getContrastRatio(fg: string, bg: string): number {
  // Implement WCAG contrast ratio calculation
  // https://www.w3.org/TR/WCAG21/#dfn-contrast-ratio
}

export function validateTheme(theme: Theme): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check text/background contrast
  const textBgContrast = getContrastRatio(
    theme.colors.text.primary,
    theme.colors.background.primary
  );

  if (textBgContrast < 4.5) {
    errors.push(`Text contrast too low: ${textBgContrast} (min 4.5 for AA)`);
  }

  // Check semantic colors
  // ... more validation

  return { valid: errors.length === 0, errors };
}
```

**Deliverables**:
- [ ] `light.theme.ts` with full light theme
- [ ] `high-contrast.theme.ts` with AAA-compliant theme
- [ ] Contrast validation utility
- [ ] All themes registered in ThemeManager
- [ ] Manual testing of each theme

---

#### Day 5: UIComponent Base Class

**Task 5.1: Create UIComponent Base Class** (4 hours)
- File: `src/ui/components/base/ui-component.ts`
- Managed event listeners (prevents memory leaks!)
- Theme subscription
- Lifecycle methods
- Disposal pattern

```typescript
// src/ui/components/base/ui-component.ts
export abstract class UIComponent<TConfig = any> {
  protected element: HTMLElement;
  protected config: TConfig;

  // Managed event listeners - automatic cleanup!
  private eventListeners: Map<
    EventTarget,
    Map<string, EventListenerObject>
  > = new Map();

  // Theme subscription
  private themeUnsubscribe: (() => void) | null = null;

  constructor(config: TConfig) {
    this.config = config;
    this.element = this.render();
    this.attachEventListeners();
    this.subscribeToTheme();
  }

  // ==================== Abstract Methods ====================
  // Subclasses MUST implement these

  /**
   * Render the component DOM structure.
   * Should return a root element with appropriate class names.
   * NO INLINE STYLES - use CSS classes only!
   */
  protected abstract render(): HTMLElement;

  /**
   * Get the base CSS class name for this component.
   * Used for BEM-style naming.
   */
  protected abstract getClassName(): string;

  // ==================== Optional Lifecycle Hooks ====================

  /**
   * Attach event listeners to the component.
   * Use this.addEventListener() for automatic cleanup!
   */
  protected attachEventListeners(): void {}

  /**
   * Called when theme changes.
   * Override for theme-specific logic (e.g., canvas colors).
   */
  protected onThemeChange(theme: Theme): void {}

  /**
   * Called before disposal.
   * Override for custom cleanup logic.
   */
  protected onDispose(): void {}

  // ==================== Managed Event Listeners ====================

  /**
   * Add an event listener with automatic cleanup tracking.
   * Prevents memory leaks - all listeners removed on dispose()!
   */
  protected addEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K,
    listener: (this: HTMLElement, ev: HTMLElementEventMap[K]) => any,
    options?: AddEventListenerOptions
  ): void {
    // Store for cleanup
    if (!this.eventListeners.has(target)) {
      this.eventListeners.set(target, new Map());
    }

    const boundListener = listener.bind(this.element);
    const listenerObj: EventListenerObject = {
      handleEvent: boundListener as any,
    };

    this.eventListeners.get(target)!.set(type, listenerObj);
    target.addEventListener(type, listenerObj, options);
  }

  /**
   * Remove a specific event listener.
   */
  protected removeEventListener<K extends keyof HTMLElementEventMap>(
    target: EventTarget,
    type: K
  ): void {
    const listeners = this.eventListeners.get(target);
    if (!listeners) return;

    const listener = listeners.get(type);
    if (listener) {
      target.removeEventListener(type, listener);
      listeners.delete(type);
    }
  }

  // ==================== Theme Management ====================

  private subscribeToTheme(): void {
    this.themeUnsubscribe = ThemeManager.getInstance().onChange((theme) => {
      this.onThemeChange(theme);
    });
  }

  // ==================== Public API ====================

  /**
   * Show the component (add to DOM).
   */
  public show(): void {
    if (!this.element.parentNode) {
      document.body.appendChild(this.element);
    }
    this.element.classList.add(`${this.getClassName()}--visible`);
  }

  /**
   * Hide the component (keep in DOM).
   */
  public hide(): void {
    this.element.classList.remove(`${this.getClassName()}--visible`);
  }

  /**
   * Toggle visibility.
   */
  public toggle(): void {
    const isVisible = this.element.classList.contains(`${this.getClassName()}--visible`);
    if (isVisible) {
      this.hide();
    } else {
      this.show();
    }
  }

  /**
   * Check if component is visible.
   */
  public isVisible(): boolean {
    return this.element.classList.contains(`${this.getClassName()}--visible`);
  }

  /**
   * Dispose the component - cleanup all resources.
   * NO MORE MEMORY LEAKS!
   */
  public dispose(): void {
    // 1. Remove all tracked event listeners
    for (const [target, listeners] of this.eventListeners) {
      for (const [type, listener] of listeners) {
        target.removeEventListener(type, listener);
      }
    }
    this.eventListeners.clear();

    // 2. Unsubscribe from theme changes
    if (this.themeUnsubscribe) {
      this.themeUnsubscribe();
      this.themeUnsubscribe = null;
    }

    // 3. Custom cleanup
    this.onDispose();

    // 4. Remove from DOM
    this.element.remove();
  }
}
```

**Task 5.2: Create Tests for UIComponent** (2 hours)
- File: `src/tests/unit/ui/ui-component.test.ts`
- Test lifecycle methods
- Test event listener cleanup
- Test theme subscription
- Test disposal

**Deliverables**:
- [ ] `ui-component.ts` base class (~200 lines)
- [ ] Tests for UIComponent (~150 lines)
- [ ] All tests passing

---

### **PHASE 2: Component Migration (Week 2 - Days 6-10)**

#### Day 6: Migrate helpers.ts (Error Dialog + Help Overlay)

**Task 6.1: Extract Error Dialog CSS** (2 hours)
- File: `src/styles/components/error-dialog.css`
- Convert all inline styles to CSS
- Use BEM naming: `.luxar-error-dialog`, `.luxar-error-dialog__header`, etc.
- Replace colors with CSS variables

```css
/* src/styles/components/error-dialog.css */

.luxar-error-dialog {
  /* Positioning */
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);

  /* Sizing */
  max-width: 520px;
  width: 90%;

  /* Theming */
  background: var(--luxar-bg-secondary);
  color: var(--luxar-text-primary);
  border: 1px solid var(--luxar-error);
  border-radius: var(--luxar-radius-lg);
  padding: var(--luxar-spacing-12);

  /* Effects */
  backdrop-filter: var(--luxar-blur-md);
  box-shadow: var(--luxar-shadow-xl);
  z-index: var(--luxar-z-modal);

  /* Animation */
  opacity: 0;
  animation: luxar-fade-in 0.2s ease forwards;
}

@keyframes luxar-fade-in {
  to { opacity: 1; }
}

.luxar-error-dialog__header {
  display: flex;
  align-items: center;
  gap: var(--luxar-spacing-6);
  margin-bottom: var(--luxar-spacing-8);
  padding-bottom: var(--luxar-spacing-6);
  border-bottom: 1px solid var(--luxar-border-subtle);
}

.luxar-error-dialog__icon {
  font-size: var(--luxar-text-2xl);
}

.luxar-error-dialog__title {
  margin: 0;
  font-size: var(--luxar-text-xl);
  font-weight: var(--luxar-font-semibold);
  color: var(--luxar-error);
}

.luxar-error-dialog__message {
  margin-bottom: var(--luxar-spacing-8);
  line-height: var(--luxar-line-relaxed);
  color: var(--luxar-text-primary);
}

.luxar-error-dialog__guidance {
  margin-top: var(--luxar-spacing-8);
  padding: var(--luxar-spacing-6);
  background: rgba(255, 255, 255, 0.05);
  border-radius: var(--luxar-radius-sm);
  border-left: 3px solid var(--luxar-success);
}

.luxar-error-dialog__guidance-title {
  font-weight: var(--luxar-font-semibold);
  margin-bottom: var(--luxar-spacing-4);
  color: var(--luxar-success);
}

.luxar-error-dialog__dismiss {
  margin-top: var(--luxar-spacing-8);
  font-size: var(--luxar-text-sm);
  color: var(--luxar-text-muted);
  text-align: center;
  font-style: italic;
}

/* Theme-specific overrides */
[data-theme="light"] .luxar-error-dialog {
  border-color: #ffcccc;
  background: rgba(255, 255, 255, 0.98);
}

[data-theme="light"] .luxar-error-dialog__guidance {
  background: rgba(0, 0, 0, 0.03);
}

[data-theme="high-contrast"] .luxar-error-dialog {
  border: 3px solid var(--luxar-error);
  border-radius: 0;
  background: #000000;
}
```

**Task 6.2: Refactor showError() Function** (2 hours)
```typescript
// src/ui/helpers.ts - BEFORE (50 lines of styling)
export function showError(message: string) {
  const errorDiv = document.createElement('div');
  errorDiv.style.backgroundColor = config.ui.styles.colors.panelBg;
  errorDiv.style.color = config.ui.styles.colors.primaryText;
  // ... 30 more lines of inline styles
}

// AFTER (5 lines of logic, styling in CSS)
export function showError(message: string) {
  // Remove existing
  const existing = document.getElementById('error-message');
  if (existing) existing.remove();

  // Create with CSS classes
  const errorDiv = document.createElement('div');
  errorDiv.id = 'error-message';
  errorDiv.className = 'luxar-error-dialog';
  errorDiv.setAttribute('role', 'alertdialog');
  errorDiv.setAttribute('aria-modal', 'true');
  errorDiv.setAttribute('aria-labelledby', 'error-title');
  errorDiv.setAttribute('aria-describedby', 'error-message-text');

  errorDiv.innerHTML = `
    <div class="luxar-error-dialog__header">
      <div class="luxar-error-dialog__icon">⚠️</div>
      <div id="error-title" class="luxar-error-dialog__title">Unable to Load Dataset</div>
    </div>
    <div id="error-message-text" class="luxar-error-dialog__message">${message}</div>
    <div class="luxar-error-dialog__guidance">
      <div class="luxar-error-dialog__guidance-title">💡 How to Load a Dataset:</div>
      <div class="luxar-error-dialog__guidance-content">
        <!-- Guidance content -->
      </div>
    </div>
    <div class="luxar-error-dialog__dismiss">Click anywhere or press Escape to dismiss</div>
  `;

  // Event listeners
  errorDiv.addEventListener('click', () => errorDiv.remove());
  errorDiv.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      errorDiv.remove();
    }
  });

  document.body.appendChild(errorDiv);
}
```

**Task 6.3: Similar for Help Overlay** (2 hours)
- Create `src/styles/components/help-overlay.css`
- Refactor `showHelpOverlay()` to use CSS classes
- Remove all inline styles

**Verification**:
```bash
# Test error dialog
pnpm dev
# Trigger an error (e.g., load invalid dataset)
# Verify styling looks identical
# Switch themes → verify colors update

# Test help overlay
# Press 'H' key
# Verify styling looks identical
# Switch themes → verify colors update
```

**Deliverables**:
- [ ] `error-dialog.css` (~80 lines)
- [ ] `help-overlay.css` (~100 lines)
- [ ] Refactored `showError()` and `showHelpOverlay()`
- [ ] 0 inline styles in these functions
- [ ] Visual parity with current design
- [ ] Theme switching works

---

#### Day 7-8: Migrate dimension-sliders.ts

**Task 7.1: Extract CSS** (4 hours)
- File: `src/styles/components/dimension-sliders.css`
- Complex component with custom slider styling
- Progress bars, thumb indicators, dropdowns
- Requires careful attention to layout

```css
/* src/styles/components/dimension-sliders.css */

.luxar-dimension-sliders {
  /* Container */
  position: fixed;
  left: 50%;
  bottom: var(--luxar-spacing-10);
  transform: translateX(-50%);

  /* Sizing */
  width: 80%;
  max-width: 800px;
  min-width: 400px;
  max-height: 240px;

  /* Theming */
  background: var(--luxar-bg-secondary);
  border-radius: var(--luxar-radius-md);
  padding: var(--luxar-spacing-8);
  backdrop-filter: var(--luxar-blur-md);
  box-shadow: var(--luxar-shadow-lg);
  z-index: var(--luxar-z-base);

  /* Typography */
  font-family: var(--luxar-font-base);
  font-size: var(--luxar-text-md);
  color: var(--luxar-text-primary);

  /* Behavior */
  user-select: none;
  overflow-y: auto;
}

.luxar-dimension-sliders__header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--luxar-spacing-5);
  padding-bottom: var(--luxar-spacing-3);
  border-bottom: 1px solid var(--luxar-border-strong);
}

.luxar-dimension-sliders__title {
  font-weight: var(--luxar-font-bold);
  font-size: var(--luxar-text-lg);
}

.luxar-dimension-sliders__status {
  font-family: var(--luxar-font-mono);
  font-size: var(--luxar-text-md);
  color: var(--luxar-text-primary);
  opacity: 0.8;
}

/* Individual slider */
.luxar-dimension-slider {
  margin-bottom: var(--luxar-spacing-6);
}

.luxar-dimension-slider__label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--luxar-spacing-2);
}

.luxar-dimension-slider__name {
  font-weight: var(--luxar-font-medium);
}

.luxar-dimension-slider__value {
  font-family: var(--luxar-font-mono);
  font-size: var(--luxar-text-md);
  color: var(--luxar-success);
  cursor: help;
}

/* Custom slider track */
.luxar-dimension-slider__track {
  position: relative;
  height: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 10px;
  overflow: hidden;
}

.luxar-dimension-slider__progress {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  background: rgba(76, 175, 80, 0.3);
  transition: width 3ms ease-out;
  pointer-events: none;
}

.luxar-dimension-slider__input {
  position: absolute;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  opacity: 0;
  cursor: pointer;
  z-index: 10;
}

.luxar-dimension-slider__thumb {
  position: absolute;
  width: 16px;
  height: 16px;
  background: var(--luxar-success);
  border-radius: 50%;
  top: 50%;
  transform: translateY(-50%);
  box-shadow: var(--luxar-shadow-sm);
  pointer-events: none;
  transition: left 3ms ease-out;
}

/* Categorical dropdowns */
.luxar-dimension-dropdown {
  display: flex;
  flex-direction: column;
  gap: var(--luxar-spacing-2);
}

.luxar-dimension-dropdown__label {
  font-size: var(--luxar-text-md);
  font-weight: var(--luxar-font-medium);
  color: var(--luxar-text-primary);
}

.luxar-dimension-dropdown__select {
  width: 100%;
  padding: 6px 8px;
  background: rgba(255, 255, 255, 0.1);
  color: var(--luxar-text-primary);
  border: 1px solid var(--luxar-border-default);
  border-radius: var(--luxar-radius-sm);
  font-size: var(--luxar-text-sm);
  cursor: pointer;
  outline: none;
  font-family: inherit;
  transition: var(--luxar-transition-fast);
}

.luxar-dimension-dropdown__select:hover {
  background: rgba(255, 255, 255, 0.15);
  border-color: var(--luxar-success);
}

.luxar-dimension-dropdown__select:focus {
  border-color: var(--luxar-success);
  background: rgba(255, 255, 255, 0.15);
}

/* Theme overrides */
[data-theme="light"] .luxar-dimension-sliders {
  background: rgba(255, 255, 255, 0.95);
}

[data-theme="light"] .luxar-dimension-slider__track {
  background: rgba(0, 0, 0, 0.1);
}

[data-theme="light"] .luxar-dimension-dropdown__select {
  background: rgba(0, 0, 0, 0.05);
  color: var(--luxar-text-primary);
  border-color: rgba(0, 0, 0, 0.2);
}
```

**Task 7.2: Refactor DimensionSliders Class** (4 hours)
- Convert to extend UIComponent
- Replace inline styles with class names
- Use managed event listeners
- Test thoroughly

**Verification Checklist**:
- [ ] Visual parity with old version
- [ ] All sliders work (continuous + discrete)
- [ ] All dropdowns work (categorical)
- [ ] Hover states work
- [ ] Keyboard navigation works
- [ ] Theme switching updates colors
- [ ] No memory leaks (verify with profiling)
- [ ] All tests pass

**Deliverables**:
- [ ] `dimension-sliders.css` (~200 lines)
- [ ] Refactored DimensionSliders class
- [ ] 0 inline styles
- [ ] All functionality preserved

---

#### Day 9-10: Migrate debug-console.ts + dataset-browser.ts

**Task 9.1: Debug Console CSS** (2 hours)
- File: `src/styles/components/debug-console.css`
- Already has some CSS via injected `<style>` tag
- Extract to external file
- Add theme support

**Task 9.2: Dataset Browser CSS** (2 hours)
- File: `src/styles/components/dataset-browser.css`
- Modal styling
- File list styling
- Breadcrumb navigation

**Task 9.3: Refactor Both Components** (4 hours)
- Update to use CSS classes
- Migrate to UIComponent pattern (optional for now)
- Test thoroughly

**Deliverables**:
- [ ] `debug-console.css` (~150 lines)
- [ ] `dataset-browser.css` (~120 lines)
- [ ] Both components refactored
- [ ] Theme switching works

---

### **PHASE 3: Complex Components (Week 3 - Days 11-15)**

#### Day 11-13: Migrate data-loading-monitor.ts

**Challenge**: Most complex component (1,582 lines, dynamic templates)

**Task 11.1: Extract Monitor CSS** (4 hours)
- File: `src/styles/components/data-loading-monitor.css`
- Complex: tabs, mini view, expanded view, 4 different tabs
- Many nested elements

**Task 11.2: Refactor Templates** (6 hours)
- Update `data-monitor-templates.ts` to use CSS classes
- Replace all inline styles
- Maintain template functions (keep HTML generation pattern)

```typescript
// BEFORE
export function renderMetricCard(title: string, value: string): string {
  return `
    <div style="background: ${MonitorColors.sectionBg}; padding: ${monitorConfig.padding.compact}px; border-radius: 4px;">
      <div style="font-size: 10px; color: ${MonitorColors.muted};">${title}</div>
      <div style="font-size: 24px; font-weight: bold;">${value}</div>
    </div>
  `;
}

// AFTER
export function renderMetricCard(title: string, value: string): string {
  return `
    <div class="luxar-metric-card luxar-surface-tertiary luxar-rounded-sm luxar-p-4">
      <div class="luxar-metric-card__title luxar-text-sm luxar-text-muted">${title}</div>
      <div class="luxar-metric-card__value luxar-text-3xl luxar-text-bold">${value}</div>
    </div>
  `;
}
```

**Task 11.3: Update DataLoadingMonitor Class** (4 hours)
- Migrate to UIComponent (optional)
- Use CSS classes throughout
- Test mini/expanded modes
- Test all 4 tabs

**Deliverables**:
- [ ] `data-loading-monitor.css` (~300 lines)
- [ ] Updated template functions (0 inline styles)
- [ ] Refactored DataLoadingMonitor class
- [ ] All modes work (mini, expanded)
- [ ] All tabs work (overview, cache, performance, insights)

---

#### Day 14-15: Third-Party Library Theming

**Task 14.1: lil-gui Theming** (2 hours)
- File: `src/styles/components/rendering-controls.css`
- Override lil-gui CSS variables
- Match Luxar theme system

```css
/* src/styles/components/rendering-controls.css */

/* Override lil-gui variables */
.lil-gui {
  --background-color: var(--luxar-bg-secondary) !important;
  --text-color: var(--luxar-text-primary) !important;
  --title-background-color: var(--luxar-bg-tertiary) !important;
  --title-text-color: var(--luxar-text-primary) !important;
  --widget-color: rgba(255, 255, 255, 0.1) !important;
  --hover-color: rgba(255, 255, 255, 0.15) !important;
  --focus-color: var(--luxar-success) !important;
  --number-color: var(--luxar-success) !important;
  --string-color: var(--luxar-info) !important;
}

/* Light theme overrides */
[data-theme="light"] .lil-gui {
  --background-color: rgba(255, 255, 255, 0.95) !important;
  --widget-color: rgba(0, 0, 0, 0.08) !important;
  --hover-color: rgba(0, 0, 0, 0.12) !important;
}

/* High contrast overrides */
[data-theme="high-contrast"] .lil-gui {
  --background-color: #000000 !important;
  --focus-color: #00ff00 !important;
  --border-width: 2px !important;
}
```

**Task 14.2: stats.js Theming** (1 hour)
- File: `src/styles/components/performance-monitor.css`
- Override stats.js canvas colors
- May require JavaScript updates to canvas rendering

**Task 14.3: Theme Subscription in Third-Party Wrappers** (2 hours)
```typescript
// src/ui/rendering-controls.ts
export class RenderingControls {
  constructor() {
    // ... existing code

    // Subscribe to theme changes
    ThemeManager.getInstance().onChange((theme) => {
      this.updateLibraryTheming(theme);
    });
  }

  private updateLibraryTheming(theme: Theme): void {
    // Update lil-gui CSS variables
    const gui = this.gui.domElement;
    gui.style.setProperty('--background-color', theme.colors.background.secondary);
    gui.style.setProperty('--text-color', theme.colors.text.primary);
    // ... update other variables
  }
}
```

**Deliverables**:
- [ ] lil-gui themed correctly
- [ ] stats.js themed correctly
- [ ] Theme switching updates third-party libs
- [ ] Visual consistency maintained

---

### **PHASE 4: Polish & Testing (Week 4 - Days 16-20)**

#### Day 16: Theme Selector UI

**Task 16.1: Add Theme Dropdown to Rendering Controls** (2 hours)
```typescript
// In rendering-controls setup
private setupThemeControls(gui: GUI): void {
  const themeFolder = gui.addFolder('🎨 Theme');

  const themes = ThemeManager.getInstance().getAllThemes();
  const themeOptions = themes.reduce((acc, theme) => {
    acc[theme.name] = theme.id;
    return acc;
  }, {} as Record<string, string>);

  const settings = {
    currentTheme: ThemeManager.getInstance().getCurrentTheme().id,
  };

  themeFolder
    .add(settings, 'currentTheme', themeOptions)
    .name('Active Theme')
    .onChange((themeId: string) => {
      ThemeManager.getInstance().setTheme(themeId);
      this.saveSettings(); // Persist
    });

  themeFolder.close(); // Collapsed by default
}
```

**Task 16.2: Add URL Parameter Support** (1 hour)
```typescript
// Support ?theme=light in URL
const params = new URLSearchParams(window.location.search);
const themeParam = params.get('theme');
if (themeParam) {
  ThemeManager.getInstance().setTheme(themeParam);
}
```

**Task 16.3: Theme Persistence** (1 hour)
- Save theme preference to localStorage
- Restore on app load
- Per-user preference

**Deliverables**:
- [ ] Theme selector in rendering controls
- [ ] URL parameter support
- [ ] Persistence working
- [ ] Manual testing of theme switching

---

#### Day 17-18: Comprehensive Testing

**Task 17.1: Visual Regression Tests** (4 hours)
- Use Playwright for screenshots
- Capture each component in each theme
- Compare against baselines

```typescript
// src/tests/e2e/theme-visual-regression.spec.ts
import { test, expect } from '@playwright/test';

const themes = ['dark', 'light', 'high-contrast'];
const components = ['error-dialog', 'help-overlay', 'dimension-sliders', 'data-monitor'];

for (const theme of themes) {
  for (const component of components) {
    test(`${component} - ${theme} theme`, async ({ page }) => {
      await page.goto(`/?theme=${theme}&debug`);

      // Trigger component (e.g., press H for help)
      if (component === 'help-overlay') {
        await page.keyboard.press('h');
      }

      // Wait for component to appear
      const selector = `.luxar-${component}`;
      await page.waitForSelector(selector);

      // Take screenshot
      const element = await page.$(selector);
      const screenshot = await element!.screenshot();

      // Compare with baseline
      expect(screenshot).toMatchSnapshot(`${component}-${theme}.png`);
    });
  }
}
```

**Task 17.2: Theme Switching Tests** (2 hours)
- Test rapid theme switching
- Verify no memory leaks
- Verify all CSS variables update

```typescript
// src/tests/unit/themes/theme-manager.test.ts
describe('ThemeManager', () => {
  it('should switch themes without memory leaks', () => {
    const manager = ThemeManager.getInstance();

    // Switch themes 100 times
    for (let i = 0; i < 100; i++) {
      manager.setTheme('dark');
      manager.setTheme('light');
      manager.setTheme('high-contrast');
    }

    // Check no memory growth
    // (In real test, use memory profiling)
    expect(manager.getAllThemes().length).toBe(3);
  });

  it('should update all CSS variables when theme changes', () => {
    const manager = ThemeManager.getInstance();

    manager.setTheme('dark');
    const darkBg = getComputedStyle(document.documentElement)
      .getPropertyValue('--luxar-bg-primary');
    expect(darkBg).toBe('#111111');

    manager.setTheme('light');
    const lightBg = getComputedStyle(document.documentElement)
      .getPropertyValue('--luxar-bg-primary');
    expect(lightBg).toBe('#ffffff');
  });
});
```

**Task 17.3: Accessibility Testing** (2 hours)
- Test with screen reader (NVDA/JAWS)
- Verify high-contrast theme meets WCAG AAA
- Check color contrast ratios

**Deliverables**:
- [ ] Visual regression test suite
- [ ] Theme switching tests
- [ ] Accessibility audit passed
- [ ] All screenshots baseline captured

---

#### Day 19: Documentation

**Task 19.1: Create Theme Creation Guide** (3 hours)
- File: `docs/guides/developer/THEME_CREATION_GUIDE.md`
- How to create custom themes
- Design token reference
- Color palette guidelines
- Accessibility requirements

**Task 19.2: Update Component Documentation** (2 hours)
- Update `src/ui/README.md`
- Document CSS class naming conventions
- Update examples to show CSS classes

**Task 19.3: API Documentation** (1 hour)
- Document ThemeManager API
- Document UIComponent base class
- Add JSDoc to all public methods

**Deliverables**:
- [ ] `THEME_CREATION_GUIDE.md`
- [ ] Updated `ui/README.md`
- [ ] Complete API documentation

---

#### Day 20: Final Polish & Review

**Task 20.1: Code Review Prep** (2 hours)
- Self-review all changes
- Run full test suite
- Check TypeScript compilation
- Run linter
- Update CHANGELOG.md

**Task 20.2: Performance Validation** (2 hours)
- Measure theme switch performance
- Ensure no regression in render performance
- Check bundle size impact

```bash
# Before
JS bundle: 450KB
CSS bundle: 0KB (in JS)
Total: 450KB

# After
JS bundle: 400KB (removed CSS-in-JS)
CSS bundle: 25KB (new, cacheable)
Total: 425KB (-25KB, better caching)
```

**Task 20.3: Migration Verification** (2 hours)
- Visual comparison: old vs new
- Functionality checklist (all features work)
- Cross-browser testing

**Deliverables**:
- [ ] All tests passing
- [ ] Performance validated
- [ ] Bundle size acceptable
- [ ] Ready for PR

---

## Migration Strategy

### Backward Compatibility During Transition

**Strategy**: Support both systems simultaneously

```typescript
// src/ui/helpers.ts - Hybrid approach
export function showError(message: string) {
  if (USE_NEW_THEME_SYSTEM) {
    // New: CSS classes
    return showErrorWithCSS(message);
  } else {
    // Old: Inline styles (fallback)
    return showErrorInline(message);
  }
}

// Toggle via feature flag
const USE_NEW_THEME_SYSTEM =
  import.meta.env.VITE_USE_THEME_SYSTEM === 'true' ||
  localStorage.getItem('luxar-theme-system') === 'enabled';
```

### Component-by-Component Migration

**Order** (lowest risk to highest):

1. ✅ **helpers.ts** (error dialog, help overlay) - Independent, no dependencies
2. ✅ **dataset-browser.ts** - Simple modal, low complexity
3. ✅ **performance-monitor.ts** - Minimal styling, mostly third-party
4. ✅ **debug-console.ts** - Already has CSS injection pattern
5. ✅ **dimension-sliders.ts** - Medium complexity, custom controls
6. ✅ **data-loading-monitor.ts** - Highest complexity, many templates

**Migration Checklist per Component**:
- [ ] Extract CSS to dedicated file
- [ ] Replace inline styles with classes
- [ ] Add to `styles/index.css` import
- [ ] Test in all 3 themes
- [ ] Visual regression test
- [ ] Verify no memory leaks
- [ ] Update tests if needed
- [ ] Mark as complete

---

## Testing Strategy

### Test Categories

#### 1. Unit Tests (Per Component)

```typescript
// Example: error-dialog.test.ts
describe('ErrorDialog Theming', () => {
  it('should apply correct CSS classes', () => {
    const dialog = new ErrorDialog({ message: 'Test error' });
    dialog.show();

    const element = document.querySelector('.luxar-error-dialog');
    expect(element).toBeTruthy();
    expect(element?.classList.contains('luxar-surface-elevated')).toBe(true);
  });

  it('should update when theme changes', () => {
    const dialog = new ErrorDialog({ message: 'Test' });
    dialog.show();

    ThemeManager.getInstance().setTheme('light');

    const bgColor = getComputedStyle(dialog.element).backgroundColor;
    expect(bgColor).toBe('rgba(255, 255, 255, 0.98)'); // Light theme background
  });
});
```

#### 2. Visual Regression Tests (E2E)

```typescript
// All components × All themes = Matrix testing
// src/tests/e2e/theme-regression.spec.ts

const testMatrix = [
  { component: 'error-dialog', trigger: async (page) => { /* trigger error */ } },
  { component: 'help-overlay', trigger: async (page) => page.keyboard.press('h') },
  { component: 'dimension-sliders', trigger: async (page) => page.keyboard.press('n') },
  { component: 'data-monitor-mini', trigger: async (page) => page.keyboard.press('m') },
  { component: 'data-monitor-expanded', trigger: async (page) => {
    await page.keyboard.press('m');
    await page.keyboard.press('m');
  }},
];

for (const { component, trigger } of testMatrix) {
  for (const theme of ['dark', 'light', 'high-contrast']) {
    test(`${component} in ${theme} theme`, async ({ page }) => {
      await page.goto(`/?theme=${theme}&debug`);
      await trigger(page);
      await expect(page.locator(`.luxar-${component}`)).toHaveScreenshot();
    });
  }
}
```

#### 3. Performance Tests

```typescript
// src/tests/performance/theme-switching.test.ts
test('theme switching performance', async () => {
  const manager = ThemeManager.getInstance();
  const iterations = 100;

  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    manager.setTheme('dark');
    manager.setTheme('light');
  }
  const end = performance.now();

  const avgTime = (end - start) / (iterations * 2);

  // Theme switch should be < 5ms
  expect(avgTime).toBeLessThan(5);
});
```

#### 4. Accessibility Tests

```typescript
// src/tests/a11y/theme-contrast.test.ts
import { getContrastRatio } from '../../themes/utils';

describe('Theme Accessibility', () => {
  it('dark theme should meet WCAG AA', () => {
    const theme = ThemeManager.getInstance().getTheme('dark')!;

    const contrast = getContrastRatio(
      theme.colors.text.primary,
      theme.colors.background.primary
    );

    expect(contrast).toBeGreaterThanOrEqual(4.5); // AA requirement
  });

  it('high-contrast theme should meet WCAG AAA', () => {
    const theme = ThemeManager.getInstance().getTheme('high-contrast')!;

    const contrast = getContrastRatio(
      theme.colors.text.primary,
      theme.colors.background.primary
    );

    expect(contrast).toBeGreaterThanOrEqual(7.0); // AAA requirement
  });
});
```

---

## Rollout Plan

### Pre-Release (Internal Testing)

**Week 1-3**: Development
- Implement all phases
- Internal dogfooding
- Fix bugs discovered

**Week 4**: Stabilization
- Address feedback
- Performance tuning
- Documentation polish

### Release Strategy

**Option A: Feature Flag (Recommended)**
```typescript
// Enable for testing
localStorage.setItem('luxar-theme-system', 'enabled');

// Or via URL
// http://localhost:5173/?theme-system=enabled&theme=light
```

**Benefits**:
- Safe rollout
- Can disable if issues found
- Gradual user adoption
- A/B testing possible

**Option B: Direct Release**
- Ship to all users immediately
- Higher risk
- Faster feedback

### Monitoring Post-Release

**Metrics to track**:
- Theme switching usage (which themes are popular?)
- Performance impact (bundle size, load time)
- Error rates (any theme-related bugs?)
- User feedback (survey or analytics)

---

## Appendix

### A. Code Examples

#### Example 1: Creating a Custom Theme

```typescript
// my-theme.ts
import { Theme } from '@luxar/viewer/themes';

export const myCustomTheme: Theme = {
  id: 'custom-ocean',
  name: 'Ocean Blue',
  description: 'Cool blue theme inspired by deep ocean',

  colors: {
    background: {
      primary: '#0a1929',       // Deep ocean
      secondary: 'rgba(15, 30, 45, 0.95)',
      tertiary: 'rgba(20, 40, 60, 0.8)',
      overlay: 'rgba(0, 0, 0, 0.7)',
    },
    text: {
      primary: '#e3f2fd',       // Light blue-white
      secondary: '#90caf9',     // Sky blue
      muted: 'rgba(227, 242, 253, 0.6)',
    },
    semantic: {
      success: '#26a69a',       // Teal
      warning: '#ffa726',       // Coral
      error: '#ef5350',         // Salmon red
      info: '#42a5f5',          // Ocean blue
      highlight: '#00e5ff',     // Bright cyan
    },
    interactive: {
      default: '#1976d2',
      hover: '#2196f3',
      active: '#1565c0',
      focus: '#64b5f6',
      disabled: '#607d8b',
    },
    visualization: {
      hot: '#ff6b6b',
      warm: '#feca57',
      cold: '#48dbfb',
      neutral: '#a29bfe',
    },
  },

  // Inherit spacing, typography, effects from dark theme
  spacing: darkTheme.spacing,
  typography: {
    ...darkTheme.typography,
    fontFamily: {
      ...darkTheme.typography.fontFamily,
      base: '"Inter", -apple-system, sans-serif', // Custom font
    },
  },
  effects: darkTheme.effects,
  zIndex: darkTheme.zIndex,
};

// Register your theme
ThemeManager.getInstance().registerTheme(myCustomTheme);
```

#### Example 2: Component Migration Pattern

**BEFORE** (Current):
```typescript
// 50 lines of inline styling
export class DimensionSliders {
  private createSlidersContainer(): HTMLElement {
    const container = document.createElement('div');
    container.style.position = 'fixed';
    container.style.bottom = '20px';
    container.style.left = '50%';
    container.style.transform = 'translateX(-50%)';
    container.style.backgroundColor = 'rgba(30, 30, 30, 0.9)';
    container.style.borderRadius = '8px';
    container.style.padding = '15px';
    container.style.backdropFilter = 'blur(10px)';
    container.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.3)';
    // ... 20 more lines of styling
    return container;
  }
}
```

**AFTER** (Proposed):
```typescript
// 5 lines - styling in CSS
export class DimensionSliders extends UIComponent {
  protected getClassName(): string {
    return 'luxar-dimension-sliders';
  }

  protected render(): HTMLElement {
    const container = document.createElement('div');
    container.className = this.getClassName();
    // All styling via CSS - no inline styles!
    return container;
  }
}

/* CSS file handles all styling */
/* src/styles/components/dimension-sliders.css */
.luxar-dimension-sliders {
  position: fixed;
  bottom: var(--luxar-spacing-10);
  left: 50%;
  transform: translateX(-50%);
  background: var(--luxar-bg-secondary);
  border-radius: var(--luxar-radius-md);
  padding: var(--luxar-spacing-8);
  backdrop-filter: var(--luxar-blur-md);
  box-shadow: var(--luxar-shadow-lg);
  /* ... all styles from CSS variables */
}
```

#### Example 3: Design Token Usage

```typescript
// BEFORE: Direct color references
element.style.color = '#4CAF50';
element.style.background = 'rgba(30, 30, 30, 0.95)';

// AFTER: Semantic tokens
element.className = 'luxar-text-success luxar-surface-secondary';

// Or if you need dynamic styling:
element.style.color = 'var(--luxar-success)';
element.style.background = 'var(--luxar-bg-secondary)';
```

---

### B. CSS Variable Reference

#### Complete Variable List

```css
/* ============================================
   LUXAR THEME SYSTEM - CSS CUSTOM PROPERTIES
   ============================================ */

:root {
  /* ========== Colors ========== */

  /* Background */
  --luxar-bg-primary: #111111;
  --luxar-bg-secondary: rgba(30, 30, 30, 0.95);
  --luxar-bg-tertiary: rgba(0, 0, 0, 0.3);
  --luxar-bg-overlay: rgba(0, 0, 0, 0.5);

  /* Text */
  --luxar-text-primary: #e0e0e0;
  --luxar-text-secondary: #888888;
  --luxar-text-muted: rgba(255, 255, 255, 0.6);
  --luxar-text-disabled: rgba(255, 255, 255, 0.4);
  --luxar-text-inverse: #111111;

  /* Semantic */
  --luxar-success: #4CAF50;
  --luxar-warning: #FFC107;
  --luxar-error: #f44336;
  --luxar-info: #2196F3;
  --luxar-highlight: #00a0ff;

  /* Interactive */
  --luxar-interactive-default: rgba(255, 255, 255, 0.1);
  --luxar-interactive-hover: rgba(255, 255, 255, 0.15);
  --luxar-interactive-active: rgba(255, 255, 255, 0.2);
  --luxar-interactive-focus: rgba(76, 175, 80, 0.3);
  --luxar-interactive-disabled: rgba(255, 255, 255, 0.05);

  /* Border */
  --luxar-border-default: rgba(255, 255, 255, 0.1);
  --luxar-border-subtle: rgba(255, 255, 255, 0.05);
  --luxar-border-strong: rgba(255, 255, 255, 0.2);
  --luxar-border-focus: rgba(76, 175, 80, 0.5);

  /* Visualization */
  --luxar-viz-hot: #ff6b6b;
  --luxar-viz-warm: #FFC107;
  --luxar-viz-cold: #4CAF50;
  --luxar-viz-neutral: #9E9E9E;

  /* ========== Spacing (8px grid) ========== */
  --luxar-spacing-0: 0px;
  --luxar-spacing-1: 2px;
  --luxar-spacing-2: 4px;
  --luxar-spacing-3: 6px;
  --luxar-spacing-4: 8px;
  --luxar-spacing-5: 10px;
  --luxar-spacing-6: 12px;
  --luxar-spacing-8: 16px;
  --luxar-spacing-10: 20px;
  --luxar-spacing-12: 24px;
  --luxar-spacing-16: 32px;
  --luxar-spacing-20: 40px;

  /* ========== Typography ========== */

  /* Font families */
  --luxar-font-base: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif;
  --luxar-font-mono: "Monaco", "Menlo", "Ubuntu Mono", monospace;
  --luxar-font-display: "Inter", -apple-system, sans-serif;

  /* Font sizes */
  --luxar-text-xs: 9px;
  --luxar-text-sm: 10px;
  --luxar-text-base: 11px;
  --luxar-text-md: 12px;
  --luxar-text-lg: 14px;
  --luxar-text-xl: 16px;
  --luxar-text-2xl: 18px;
  --luxar-text-3xl: 24px;

  /* Font weights */
  --luxar-font-normal: 400;
  --luxar-font-medium: 500;
  --luxar-font-semibold: 600;
  --luxar-font-bold: 700;

  /* Line heights */
  --luxar-line-tight: 1.2;
  --luxar-line-normal: 1.4;
  --luxar-line-relaxed: 1.6;

  /* ========== Effects ========== */

  /* Border radius */
  --luxar-radius-none: 0px;
  --luxar-radius-sm: 4px;
  --luxar-radius-md: 8px;
  --luxar-radius-lg: 12px;
  --luxar-radius-full: 9999px;

  /* Shadows */
  --luxar-shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
  --luxar-shadow-md: 0 4px 12px rgba(0, 0, 0, 0.3);
  --luxar-shadow-lg: 0 8px 24px rgba(0, 0, 0, 0.4);
  --luxar-shadow-xl: 0 12px 48px rgba(0, 0, 0, 0.5);

  /* Blur */
  --luxar-blur-none: none;
  --luxar-blur-sm: blur(4px);
  --luxar-blur-md: blur(10px);
  --luxar-blur-lg: blur(20px);

  /* Opacity */
  --luxar-opacity-disabled: 0.4;
  --luxar-opacity-secondary: 0.6;
  --luxar-opacity-hover: 0.8;
  --luxar-opacity-full: 1.0;

  /* Transitions */
  --luxar-transition-fast: all 0.1s ease;
  --luxar-transition-normal: all 0.2s ease;
  --luxar-transition-slow: all 0.3s ease;

  /* ========== Z-Index ========== */
  --luxar-z-base: 100;
  --luxar-z-dropdown: 1000;
  --luxar-z-modal: 2000;
  --luxar-z-popover: 3000;
  --luxar-z-tooltip: 4000;
}
```

---

### C. BEM Naming Convention

**Pattern**: `.luxar-{component}__{element}--{modifier}`

**Examples**:
```css
/* Component */
.luxar-error-dialog { }

/* Element */
.luxar-error-dialog__header { }
.luxar-error-dialog__title { }
.luxar-error-dialog__message { }
.luxar-error-dialog__icon { }

/* Modifier */
.luxar-error-dialog--visible { }
.luxar-error-dialog--closing { }

/* Element + Modifier */
.luxar-error-dialog__message--warning { }
```

**Benefits**:
- Clear ownership (which component)
- Prevents naming conflicts
- Easy to understand hierarchy
- Grep-friendly

---

### D. Effort Breakdown

| Phase | Tasks | Days | Hours | Complexity |
|-------|-------|------|-------|------------|
| **Phase 1: Foundation** | Theme system core, CSS infra, base themes | 5 | 40 | Medium |
| **Phase 2: Simple Components** | helpers, dataset-browser, debug-console | 5 | 40 | Low-Medium |
| **Phase 3: Complex Components** | dimension-sliders, data-monitor | 5 | 40 | High |
| **Phase 4: Polish & Testing** | Visual regression, docs, review | 5 | 40 | Medium |
| **Total** | All phases | **20** | **160** | Medium-High |

**Note**: Estimates assume:
- 1 developer working full-time
- Moderate familiarity with codebase
- Standard development environment
- ~8 working hours per day

**Risk Buffer**: Add 20-25% for unexpected issues (→ 24-25 days total)

---

### E. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Hardcoded colors** | 0 | `grep -r "rgba\|#[0-9a-f]" src/ui \| wc -l` |
| **Inline styles** | <10 (exceptions allowed) | `grep -r "\.style\." src/ui \| wc -l` |
| **CSS coverage** | 100% of components | Visual inspection |
| **Theme switch time** | <100ms | Performance.now() |
| **Bundle size** | <450KB total | Build output |
| **Contrast ratios** | All ≥4.5 (AA) | Automated testing |
| **Memory leaks** | 0 | Chrome DevTools profiling |
| **Test coverage** | 100% of themes × components | Test matrix |

---

### F. Quick Reference Checklists

#### Component Migration Checklist

When migrating a component to the new theme system:

- [ ] Create CSS file in `src/styles/components/`
- [ ] Extract all inline styles to CSS
- [ ] Replace hardcoded colors with CSS variables
- [ ] Use BEM naming convention
- [ ] Add theme-specific overrides (light, high-contrast)
- [ ] Update component to use CSS classes
- [ ] Remove all inline style assignments
- [ ] Test in all 3 themes
- [ ] Take visual regression screenshots
- [ ] Update component tests
- [ ] Update documentation
- [ ] Mark component as migrated

#### Theme Creation Checklist

When creating a new theme:

- [ ] Create `{name}.theme.ts` file
- [ ] Implement full `Theme` interface
- [ ] Validate contrast ratios (WCAG AA minimum)
- [ ] Test with all components
- [ ] Create theme-specific CSS overrides if needed
- [ ] Add to theme registry
- [ ] Document unique characteristics
- [ ] Take screenshots for gallery

#### Pre-Release Checklist

Before releasing theme system:

- [ ] All components migrated
- [ ] All 3 themes working
- [ ] 0 hardcoded colors remain
- [ ] All tests passing (1030+)
- [ ] Visual regression tests pass
- [ ] Accessibility audit passed
- [ ] Performance acceptable
- [ ] Documentation complete
- [ ] Code reviewed
- [ ] Feature flag ready
- [ ] Rollback plan in place

---

## Conclusion

This implementation plan provides a **complete roadmap** for transforming the Luxar viewer into a **modular, themeable application** with proper separation of concerns.

**Key Achievements**:
- ✅ Modular theme system with runtime switching
- ✅ Clean separation: logic → structure → style
- ✅ 0 hardcoded colors in component code
- ✅ Support for light/dark/high-contrast themes
- ✅ Better maintainability and developer experience
- ✅ Improved accessibility (WCAG AAA capable)
- ✅ Better performance (CSS caching, smaller JS bundle)

**Timeline**: 3-4 weeks
**Complexity**: Medium-High
**Risk**: Low (with gradual migration strategy)
**ROI**: High (better UX, easier maintenance, accessibility compliance)

---

**Next Steps**:
1. Review this plan with team
2. Get approval and prioritization
3. Create feature branch: `feature/modular-theming`
4. Begin Phase 1, Day 1 implementation
5. Weekly check-ins to track progress

**Questions? Issues?** See:
- Technical lead for architecture review
- Design team for theme color palettes
- Accessibility expert for WCAG validation
