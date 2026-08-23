// @vitest-environment jsdom
/**
 * Unit tests for `ui/help-overlay/type-to-filter.ts` (issue #1922).
 *
 * These are real DOM behaviours: a container element, a real `<input>`, and
 * real dispatched `KeyboardEvent`s. Nothing is mocked — the assertions are on
 * observable state (`document.activeElement`, the input's `value`, and the
 * side effect of the input's OWN `input` listener) so a mutant that moves
 * focus without landing the character, or lands the character without firing
 * the filter, still fails.
 */
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { installTypeToFilter } from '../../../ui/help-overlay/type-to-filter';
import { isTypingInInput } from '../../../input/input-handler/commands/focus-utils';

interface Harness {
  container: HTMLElement;
  input: HTMLInputElement;
  /** A focusable non-typing descendant (stands in for a listing row). */
  row: HTMLElement;
  /** Everything the container's own `input` handler has rendered so far. */
  filtered: string[];
  /** Keys that reached a document-level listener (the "global bindings"). */
  escaped: string[];
  release: () => void;
}

const ITEMS = ['alpha', 'beta', 'help', 'gamma'];

function mount(options?: { passthroughKeys?: readonly string[]; filterAvailable?: boolean }) {
  const container = document.createElement('div');
  const input = document.createElement('input');
  input.type = 'text';
  const row = document.createElement('div');
  row.tabIndex = 0;
  container.appendChild(input);
  container.appendChild(row);
  document.body.appendChild(container);

  const harness: Harness = {
    container,
    input,
    row,
    filtered: [...ITEMS],
    escaped: [],
    release: () => {},
  };

  // The filter's real handler — the thing `installTypeToFilter` must trigger.
  input.addEventListener('input', () => {
    const q = input.value.toLowerCase();
    harness.filtered = ITEMS.filter((name) => name.includes(q));
  });

  // Stand-in for InputHandler's document-level keydown: records anything the
  // forwarder let propagate.
  const documentListener = (e: KeyboardEvent) => harness.escaped.push(e.key);
  document.addEventListener('keydown', documentListener);

  const releaseForwarder = installTypeToFilter(
    container,
    () => (options?.filterAvailable === false ? null : input),
    { passthroughKeys: options?.passthroughKeys }
  );
  harness.release = () => {
    releaseForwarder();
    document.removeEventListener('keydown', documentListener);
  };
  return harness;
}

/** Dispatch a keydown from whatever currently holds focus. */
function press(init: KeyboardEventInit & { key: string }): KeyboardEvent {
  const event = new KeyboardEvent('keydown', { bubbles: true, cancelable: true, ...init });
  (document.activeElement ?? document.body).dispatchEvent(event);
  return event;
}

describe('installTypeToFilter', () => {
  let harness: Harness;

  beforeEach(() => {
    document.body.innerHTML = '';
  });

  afterEach(() => {
    harness?.release();
    document.body.innerHTML = '';
  });

  it('focuses the container — not a typing surface — on install', () => {
    harness = mount();

    expect(document.activeElement).toBe(harness.container);
    expect(harness.container.getAttribute('tabindex')).toBe('-1');
    // The exact predicate InputHandler's typing guard uses: it must be false,
    // otherwise the panel's toggle key gets swallowed (issue #1922).
    expect(isTypingInInput(document.activeElement)).toBe(false);
  });

  it('leaves an existing tabindex alone', () => {
    const container = document.createElement('div');
    container.setAttribute('tabindex', '0');
    document.body.appendChild(container);
    const release = installTypeToFilter(container, () => null);
    expect(container.getAttribute('tabindex')).toBe('0');
    release();
  });

  it('a printable key focuses the filter, lands the character, and filters', () => {
    harness = mount();

    const event = press({ key: 'b' });

    expect(document.activeElement).toBe(harness.input);
    expect(harness.input.value).toBe('b');
    // The filter's own `input` listener ran — this is the assertion that
    // fails if the forwarder sets `value` without dispatching.
    expect(harness.filtered).toEqual(['beta']);
    // Neither double-inserted by the browser nor leaked to global bindings.
    expect(event.defaultPrevented).toBe(true);
    expect(harness.escaped).toEqual([]);
  });

  it('appends subsequent forwarded characters rather than replacing', () => {
    harness = mount();

    press({ key: 'g' });
    // Focus is now in the input, so the forwarder must NOT handle this one —
    // the browser inserts it. Simulate that, then forward from outside again.
    harness.input.value = 'ga';
    harness.input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(harness.filtered).toEqual(['gamma']);

    harness.container.focus();
    press({ key: 'z' });
    expect(harness.input.value).toBe('gaz');
    expect(harness.filtered).toEqual([]);
  });

  it('does not double-handle keys typed while the filter already has focus', () => {
    harness = mount();
    harness.input.focus();

    const event = press({ key: 'b' });

    // Untouched: the browser's own text insertion must do the work.
    expect(event.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
  });

  it.each(['Escape', 'Tab', 'Enter', 'ArrowDown', 'ArrowUp', 'F1', 'Backspace', 'Home', 'End'])(
    'lets the non-printable key %s through untouched',
    (key) => {
      harness = mount();

      const event = press({ key });

      expect(event.defaultPrevented).toBe(false);
      expect(document.activeElement).toBe(harness.container);
      expect(harness.input.value).toBe('');
      expect(harness.escaped).toEqual([key]);
    }
  );

  it('lets Space through — it is a global shortcut, not a filter character', () => {
    harness = mount();

    const event = press({ key: ' ' });

    expect(event.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
    expect(harness.escaped).toEqual([' ']);
  });

  it.each([{ ctrlKey: true }, { metaKey: true }, { altKey: true }])(
    'lets modified keys through untouched (%o)',
    (modifier) => {
      harness = mount();

      const event = press({ key: 's', ...modifier });

      expect(event.defaultPrevented).toBe(false);
      expect(harness.input.value).toBe('');
      expect(harness.escaped).toEqual(['s']);
    }
  );

  it('lets a shifted printable key through when it is the passthrough key', () => {
    harness = mount({ passthroughKeys: ['h'] });

    // The panel's own toggle key must reach the global binding so the panel
    // can close — both cases, since the global bindings lowercase the key.
    const lower = press({ key: 'h' });
    const upper = press({ key: 'H', shiftKey: true });

    expect(lower.defaultPrevented).toBe(false);
    expect(upper.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
    expect(harness.escaped).toEqual(['h', 'H']);
    expect(document.activeElement).toBe(harness.container);
  });

  it('still forwards every other printable key when a passthrough key is set', () => {
    harness = mount({ passthroughKeys: ['h'] });

    press({ key: 'e' });

    expect(harness.input.value).toBe('e');
    expect(harness.filtered).toEqual(['beta', 'help']);
  });

  it('ignores IME composition keystrokes', () => {
    harness = mount();

    const composing = press({ key: 'a', isComposing: true });
    expect(composing.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');

    // Legacy spelling of the same thing.
    const legacy = press({ key: 'a', keyCode: 229 });
    expect(legacy.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
  });

  it('passes the key through when the filter is unavailable', () => {
    harness = mount({ filterAvailable: false });

    const event = press({ key: 'b' });

    expect(event.defaultPrevented).toBe(false);
    expect(harness.escaped).toEqual(['b']);
    expect(harness.input.value).toBe('');
  });

  it('ignores a key an inner handler already consumed', () => {
    harness = mount();
    // A descendant (e.g. a listing row) claims the key first; the event then
    // bubbles to the container already `defaultPrevented`.
    harness.row.addEventListener('keydown', (e) => e.preventDefault());
    harness.row.focus();

    press({ key: 'b' });

    expect(harness.input.value).toBe('');
    expect(document.activeElement).toBe(harness.row);
  });

  it('forwards a key pressed while a non-typing descendant holds focus', () => {
    harness = mount();
    harness.row.focus();

    press({ key: 'b' });

    expect(document.activeElement).toBe(harness.input);
    expect(harness.input.value).toBe('b');
    expect(harness.filtered).toEqual(['beta']);
  });

  it('cleanup removes the listener and is idempotent', () => {
    harness = mount();
    const releaseForwarder = harness.release;

    releaseForwarder();
    expect(() => releaseForwarder()).not.toThrow();

    harness.container.focus();
    const event = press({ key: 'b' });

    expect(event.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
    expect(harness.filtered).toEqual([...ITEMS]);
  });
});
