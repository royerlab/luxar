// @vitest-environment jsdom
/**
 * `LabeledToggle` — the boolean sibling of `LabeledSlider`: same control-group DOM
 * shape (the panel's label lookups depend on it), onChange only on user changes, and
 * the inert contract (disabled + dimmed + reason as hover text).
 */

import { describe, it, expect, vi } from 'vitest';
import { LabeledToggle } from '../../../../ui/layers/labeled-toggle';

function make(initial = false) {
  const container = document.createElement('div');
  const onChange = vi.fn();
  const toggle = new LabeledToggle({
    container,
    label: 'Refract data',
    initialChecked: initial,
    onChange,
  });
  const input = container.querySelector('input') as HTMLInputElement;
  const group = container.querySelector('.luxar-layers-panel__control-group') as HTMLElement;
  return { container, toggle, input, group, onChange };
}

describe('LabeledToggle', () => {
  it('renders the slider-shaped control group with the label in a span and a checkbox', () => {
    const { group, input } = make(true);
    expect(group.querySelector('.luxar-layers-panel__control-label span')?.textContent).toBe(
      'Refract data'
    );
    expect(input.type).toBe('checkbox');
    expect(input.checked).toBe(true);
    expect(input.getAttribute('aria-label')).toBe('Refract data');
  });

  it('fires onChange on a user change only, not on setChecked', () => {
    const { toggle, input, onChange } = make(false);
    toggle.setChecked(true);
    expect(toggle.isChecked()).toBe(true);
    expect(onChange).not.toHaveBeenCalled();
    input.checked = false;
    input.dispatchEvent(new Event('change'));
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('inert = disabled + dimmed + the reason as hover text; null restores', () => {
    const { toggle, input, group } = make();
    toggle.setInert('Refracts nothing until Transmission is above 0.');
    expect(input.disabled).toBe(true);
    expect(group.classList.contains('luxar-layers-panel__control-group--inert')).toBe(true);
    expect(toggle.getInertReason()).toBe('Refracts nothing until Transmission is above 0.');
    toggle.setInert(null);
    expect(input.disabled).toBe(false);
    expect(group.classList.contains('luxar-layers-panel__control-group--inert')).toBe(false);
    expect(toggle.getInertReason()).toBeNull();
  });

  it('setVisible hides the whole group; dispose removes it and detaches the listener', () => {
    const { container, toggle, group, input, onChange } = make();
    toggle.setVisible(false);
    expect(group.style.display).toBe('none');
    toggle.setVisible(true);
    expect(group.style.display).toBe('');
    toggle.dispose();
    expect(container.children).toHaveLength(0);
    input.dispatchEvent(new Event('change'));
    expect(onChange).not.toHaveBeenCalled();
  });
});
