# Luxar UI Design System

This document defines the consistent design principles and specifications for all UI panels in Luxar. All new UI components must follow these guidelines to maintain visual consistency across the application.

## Core Design Principles

### 1. Visual Hierarchy
- **Minimalist approach**: Clean, uncluttered interfaces with clear information hierarchy
- **Floating panels**: UI elements float over the 3D scene with semi-transparent backgrounds
- **No hard borders**: Use subtle separators or spacing instead of visible borders
- **Dark theme**: Consistent dark UI that doesn't distract from the 3D visualization

### 2. Layout & Positioning

#### Panel Positioning
- **Fixed positioning**: All panels use `position: fixed`
- **Standard margins**: 20px from screen edges
- **Bottom panels**: `bottom: 20px` (dimension sliders, performance stats)
- **Top-right panels**: `top: 20px; right: 20px` (help overlay, controls)
- **Centered modals**: `top: 50%; left: 50%; transform: translate(-50%, -50%)`

#### Panel Dimensions
- **Width constraints**: 
  - Small panels: 300-400px
  - Medium panels: 400-600px  
  - Large panels: 600-800px
  - Always use `max-width: 90vw` for responsiveness
- **Height constraints**:
  - Use `max-height: 80vh` for tall panels
  - Allow scrolling with `overflow-y: auto` when needed

### 3. Typography

#### Font Stack
```css
font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, "Segoe UI", Roboto, sans-serif;
```

#### Font Sizes
- **Panel titles**: 14px, bold
- **Section headers**: 12px, bold or 600 weight
- **Body text**: 11px or 12px
- **Small text/labels**: 9px or 10px
- **Monospace data**: Use `font-family: monospace` for values

#### Text Colors
- **Primary text**: `#e0e0e0` or `rgba(224, 224, 224, 1)`
- **Secondary text**: `#888` or `rgba(255, 255, 255, 0.6)`
- **Muted text**: `#666` or `rgba(255, 255, 255, 0.4)`
- **Headers**: Slightly brighter, can use semantic colors

### 4. Colors & Theming

#### Background Colors
- **Panel backgrounds**: `rgba(30, 30, 30, 0.9)` or `rgba(30, 30, 30, 0.95)`
- **Section backgrounds**: `rgba(0, 0, 0, 0.3)` for nested content
- **Hover states**: Slightly lighter, e.g., `rgba(40, 40, 40, 0.9)`

#### Semantic Colors
- **Success/Good**: `#4CAF50` (green)
- **Warning/Caution**: `#FFC107` or `#FF9800` (amber/orange)
- **Error/Bad**: `#f44336` or `#ff6b6b` (red)
- **Info/Primary**: `#2196F3` (blue)
- **Secondary**: `#9C27B0` (purple)

#### Status Indicators
- Use color to indicate status (green = good, red = bad, amber = warning)
- Combine with icons for better accessibility
- Maintain sufficient contrast ratios

### 5. Spacing & Layout

#### Padding
- **Panel padding**: 15px (compact) or 20px (spacious)
- **Section padding**: 10px to 15px
- **Inline spacing**: 5px to 10px gaps

#### Margins
- **Section separation**: 10px to 15px between sections
- **Title margins**: `margin-bottom: 10px` for headers
- **Element spacing**: 8px gap in grid layouts

#### Borders & Separators
- **No visible panel borders**: Rely on backdrop and shadow
- **Section separators**: `border-bottom: 1px solid rgba(255, 255, 255, 0.1)` or `0.2`
- **Padding with borders**: 6px to 8px `padding-bottom` when using border separators

### 6. Visual Effects

#### Backdrop & Shadows
- **Backdrop filter**: `backdrop-filter: blur(10px)` for glass effect
- **Box shadow**: `box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3)` for subtle depth
- **Larger panels**: Can use stronger shadow like `0 8px 32px rgba(0, 0, 0, 0.4)`

#### Border Radius
- **Standard radius**: 8px for panels
- **Small elements**: 4px for buttons and inputs
- **Large modals**: 12px for prominent dialogs

#### Transparency
- **Panel opacity**: 0.9 to 0.95 for backgrounds
- **Full opacity text**: Keep text at 100% opacity for readability

### 7. Interactive Elements

#### Buttons
- **Minimal style**: Often just text with hover effects
- **Close buttons**: Simple "×" character, 24px font size
- **Hover states**: Change color or opacity on hover
- **No borders**: Avoid button borders, use background color changes

#### Inputs & Controls
- **Dark backgrounds**: `rgba(0, 0, 0, 0.3)` for input fields
- **Subtle borders**: If needed, use very subtle borders
- **Focus states**: Use color highlights, not borders

### 8. Z-Index Hierarchy
- **Base content**: 0-99
- **UI panels**: 100-999
- **Modals/Overlays**: 1000-1999
- **Critical overlays**: 2000+ (performance stats, debug)

### 9. Responsive Design
- **Max width constraints**: Always include `max-width` with vw units
- **Flexible layouts**: Use CSS Grid or Flexbox
- **Overflow handling**: Use `overflow-y: auto` for scrollable content
- **Font scaling**: Consider using relative units for better scaling

### 10. Accessibility
- **ARIA attributes**: Include appropriate roles and labels
- **Keyboard navigation**: Support Escape to close, Enter to confirm
- **Tooltips**: Use `title` attributes for additional context
- **Contrast**: Maintain WCAG AA contrast ratios

## Example Panel Structure

```html
<div id="panel-name" style="
  position: fixed;
  top: 20px;
  right: 20px;
  width: 400px;
  max-width: 90vw;
  background: rgba(30, 30, 30, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
  color: #e0e0e0;
  z-index: 100;
">
  <!-- Header -->
  <div style="
    padding: 15px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    justify-content: space-between;
    align-items: center;
  ">
    <h3 style="margin: 0; font-size: 14px; font-weight: bold;">Panel Title</h3>
    <button style="
      background: none;
      border: none;
      color: #999;
      font-size: 24px;
      cursor: pointer;
    ">×</button>
  </div>
  
  <!-- Content -->
  <div style="padding: 15px;">
    <!-- Section -->
    <div style="margin-bottom: 15px;">
      <h4 style="
        margin: 0 0 10px 0;
        font-size: 12px;
        color: #4CAF50;
        font-weight: 600;
      ">Section Title</h4>
      <div style="font-size: 11px;">
        Content here...
      </div>
    </div>
  </div>
</div>
```

## Implementation Checklist

When creating a new UI panel, ensure:

- [ ] Uses standard color palette (dark theme)
- [ ] Follows typography guidelines (font stack, sizes)
- [ ] Maintains consistent spacing (15px/20px padding)
- [ ] No hard borders on panels (use shadows and backdrop)
- [ ] Section separators are subtle (0.1 or 0.2 opacity)
- [ ] Includes backdrop blur effect
- [ ] Has appropriate z-index for its layer
- [ ] Responsive with max-width constraints
- [ ] Supports keyboard navigation where applicable
- [ ] Uses semantic colors consistently
- [ ] Follows the standard panel structure
- [ ] Tooltips on complex metrics/controls