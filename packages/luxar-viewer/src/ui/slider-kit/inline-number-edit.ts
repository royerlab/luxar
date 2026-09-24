/** Value accessors and callbacks for an editable numeric readout. */
export interface InlineNumberEditOptions {
  ariaLabel: string;
  getValue: () => number;
  formatValue: (value: number) => string;
  onCommit: (value: number) => void;
}

/** Turn a readout into an accessible click/keyboard inline number editor. */
export function attachInlineNumberEdit(
  element: HTMLElement,
  options: InlineNumberEditOptions
): () => void {
  element.setAttribute('role', 'button');
  element.tabIndex = 0;
  element.setAttribute('aria-label', options.ariaLabel);
  element.title = 'Click to edit';

  const beginEdit = (): void => {
    if (element.nextElementSibling?.classList.contains('luxar-slider-kit__inline-input')) return;
    const input = document.createElement('input');
    input.type = 'text';
    input.inputMode = 'decimal';
    input.className = 'luxar-slider-kit__inline-input luxar-range-slider__bound-input';
    input.value = options.formatValue(options.getValue());
    input.setAttribute('aria-label', options.ariaLabel);
    input.style.width = `${Math.max(element.offsetWidth, 28)}px`;
    element.style.display = 'none';
    element.parentElement?.insertBefore(input, element.nextSibling);
    input.focus();
    input.select();

    let finished = false;
    const restore = (): void => {
      input.remove();
      element.style.display = '';
    };
    const commit = (): void => {
      if (finished) return;
      finished = true;
      const parsed = parseFloat(input.value);
      if (Number.isFinite(parsed)) options.onCommit(parsed);
      restore();
    };
    input.addEventListener('keydown', (event) => {
      event.stopPropagation();
      if (event.key === 'Enter') commit();
      else if (event.key === 'Escape') {
        finished = true;
        restore();
      }
    });
    input.addEventListener('blur', commit);
  };

  const onKeyDown = (event: KeyboardEvent): void => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    event.stopPropagation();
    beginEdit();
  };
  element.addEventListener('click', beginEdit);
  element.addEventListener('keydown', onKeyDown);
  return () => {
    element.removeEventListener('click', beginEdit);
    element.removeEventListener('keydown', onKeyDown);
  };
}
