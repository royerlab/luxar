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
import { isTypingInInput } from '../../../utils/dom/focus';

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

function mount(options?: {
  passthroughKeys?: readonly string[];
  filterAvailable?: boolean;
  withFirstItem?: boolean;
}) {
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
    {
      passthroughKeys: options?.passthroughKeys,
      resolveFirstItem: options?.withFirstItem ? () => row : undefined,
    }
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
  // Reset between cases: a test that never calls `mount()` would otherwise
  // still see the previous test's harness and double-release it in afterEach.
  let harness: Harness | undefined;

  beforeEach(() => {
    harness = undefined;
    document.body.innerHTML = '';
  });

  afterEach(() => {
    harness?.release();
    harness = undefined;
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

  it.each(['Enter', 'ArrowDown', 'ArrowUp', 'F1', 'Backspace', 'Home', 'End', 'PageDown'])(
    'contains the non-printable key %s inside the modal',
    (key) => {
      harness = mount();

      const event = press({ key });

      // Not typed into the filter, not `preventDefault`ed (native in-panel
      // behaviour such as scrolling still applies) — but it must NOT reach
      // the global bindings behind an `aria-modal` dialog, or `Home`/`End`
      // would jump the selected dimension and `Shift`+arrows would change the
      // animation speed while a panel is open.
      expect(event.defaultPrevented).toBe(false);
      expect(document.activeElement).toBe(harness.container);
      expect(harness.input.value).toBe('');
      expect(harness.escaped).toEqual([]);
    }
  );

  it.each(['Escape', 'Tab'])('still lets %s reach the global handler', (key) => {
    harness = mount();

    // Escape dismisses the panel via the global handler and Tab drives the
    // focus trap; containment must not swallow either.
    const event = press({ key });

    expect(event.defaultPrevented).toBe(false);
    expect(harness.escaped).toEqual([key]);
  });

  it('contains Space rather than typing it', () => {
    harness = mount();

    const event = press({ key: ' ' });

    // Space is not a filter character (a leading space matches nothing and it
    // scrolls a `tabindex="-1"` container). It also must not reach the global
    // bindings from inside a modal.
    expect(event.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
    expect(harness.escaped).toEqual([]);
  });

  it('contains keys that bubble up from an inner control', () => {
    harness = mount();
    harness.row.focus();

    // The listener is on the container, so an inner control's key bubbles
    // through it — and must be contained there too. Gating containment on the
    // event TARGET would mean a single `Tab` (or one `ArrowDown` into the
    // listing) handed `Home`/`End`/`Shift`+arrows/fly arrows straight back to
    // the scene behind an `aria-modal` panel.
    const home = press({ key: 'Home' });
    const enter = press({ key: 'Enter' });

    expect(harness.escaped).toEqual([]);
    // Never `preventDefault`ed: the row's own handlers already ran on the way
    // up, and native scrolling / browser shortcuts must survive.
    expect(home.defaultPrevented).toBe(false);
    expect(enter.defaultPrevented).toBe(false);
  });

  it.each(['Escape', 'Tab'])('still lets %s out from an inner control', (key) => {
    harness = mount();
    harness.row.focus();

    // The two exemptions are unconditional on the event target: Escape
    // dismisses the panel from anywhere inside it, Tab drives the focus trap.
    press({ key });

    expect(harness.escaped).toEqual([key]);
  });

  it('does not pass a passthrough key through from an inner control', () => {
    harness = mount({ passthroughKeys: ['o'] });
    harness.row.focus();

    // The passthrough exemption is gated on `event.target === container`, so
    // once focus has left the shell the key is an ordinary character again
    // (here: forwarded into the filter). Otherwise typing a path starting
    // with `o` into a panel field would close the panel.
    const event = press({ key: 'o' });

    expect(harness.escaped).toEqual([]);
    expect(harness.input.value).toBe('o');
    expect(event.defaultPrevented).toBe(true);
  });

  it('contains a passthrough key typed into the filter itself', () => {
    harness = mount({ passthroughKeys: ['o'] });
    harness.input.focus();

    // Focus is on a typing surface: the browser types the character, and the
    // key must not ALSO reach the global binding and close the panel.
    const event = press({ key: 'o' });

    expect(harness.escaped).toEqual([]);
    expect(event.defaultPrevented).toBe(false);
  });

  it.each([{ ctrlKey: true }, { metaKey: true }, { altKey: true }])(
    'does not type modified keys into the filter (%o)',
    (modifier) => {
      harness = mount();

      const event = press({ key: 's', ...modifier });

      expect(event.defaultPrevented).toBe(false);
      expect(harness.input.value).toBe('');
      // Contained like every other non-forwarded key: an application
      // shortcut aimed at the scene must not fire from inside the modal.
      expect(harness.escaped).toEqual([]);
    }
  );

  it('types an AltGr-composed character instead of dropping it', () => {
    harness = mount();

    // Windows reports AltGr as ctrlKey + altKey with an ordinary printable
    // `key` (AltGr+E = €, AltGr+2 = @). Rejecting it as "modified" made every
    // dataset whose name starts with @ / € / ~ / \ / | unreachable by typing.
    const event = press({ key: '@', ctrlKey: true, altKey: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(harness.input);
    expect(harness.input.value).toBe('@');
    expect(harness.escaped).toEqual([]);
  });

  it('lets the unshifted passthrough key through, in either letter case', () => {
    harness = mount({ passthroughKeys: ['h'] });

    // The panel's own toggle key must reach the global binding so the panel
    // can close. CapsLock reports `H` with `shiftKey === false`, and the
    // global lookup lowercases, so that case must pass through too.
    const lower = press({ key: 'h' });
    const capsLocked = press({ key: 'H' });

    expect(lower.defaultPrevented).toBe(false);
    expect(capsLocked.defaultPrevented).toBe(false);
    expect(harness.input.value).toBe('');
    expect(harness.escaped).toEqual(['h', 'H']);
    expect(document.activeElement).toBe(harness.container);
  });

  it('types Shift+the passthrough key into the filter (it is not a binding)', () => {
    harness = mount({ passthroughKeys: ['h'] });

    // `getBindingKeyFromEvent` spells a shifted `H` as "h+shift", and only
    // "h" is registered — so passing Shift+H through would make it a dead
    // key. It is an ordinary printable character instead.
    const event = press({ key: 'H', shiftKey: true });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(harness.input);
    expect(harness.input.value).toBe('H');
    expect(harness.escaped).toEqual([]);
  });

  it('steers ArrowDown into the listing when a first item is supplied', () => {
    harness = mount({ withFirstItem: true });

    const event = press({ key: 'ArrowDown' });

    expect(event.defaultPrevented).toBe(true);
    expect(document.activeElement).toBe(harness.row);
    expect(harness.escaped).toEqual([]);
  });

  it('contains ArrowDown when there is no listing to enter', () => {
    harness = mount();

    const event = press({ key: 'ArrowDown' });

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(harness.container);
    expect(harness.escaped).toEqual([]);
  });

  it('leaves Shift+ArrowDown out of the listing affordance', () => {
    harness = mount({ withFirstItem: true });

    // Shift+ArrowDown is the animation-speed binding; it is contained by the
    // modal rather than repurposed as list navigation.
    const event = press({ key: 'ArrowDown', shiftKey: true });

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(harness.container);
    expect(harness.escaped).toEqual([]);
  });

  it('still forwards every other printable key when a passthrough key is set', () => {
    harness = mount({ passthroughKeys: ['h'] });

    press({ key: 'e' });

    expect(harness.input.value).toBe('e');
    expect(harness.filtered).toEqual(['beta', 'help']);
  });

  it.each([{ isComposing: true }, { keyCode: 229 }])(
    'hands an IME composition to the filter without preventing it (%o)',
    (composing) => {
      harness = mount();

      const event = press({ key: 'a', ...composing });

      // The composition must RETARGET to the input: composing against the
      // non-editable container drops the first character outright (every
      // CJK/IME and European dead-key layout). So focus moves but the event
      // is left alone — the composer, not this forwarder, lands the text.
      expect(document.activeElement).toBe(harness.input);
      expect(event.defaultPrevented).toBe(false);
      expect(harness.input.value).toBe('');
      // Not prevented, but still CONTAINED: a composition keystroke is aimed
      // at the panel, so it must not also reach the global bindings behind
      // the modal.
      expect(harness.escaped).toEqual([]);
    }
  );

  it('does not re-focus on a composition keystroke aimed at the filter', () => {
    harness = mount();
    harness.input.focus();
    harness.input.value = 'x';

    const event = press({ key: 'a', isComposing: true });

    expect(event.defaultPrevented).toBe(false);
    expect(document.activeElement).toBe(harness.input);
    expect(harness.input.value).toBe('x');
    expect(harness.escaped).toEqual([]);
  });

  it('contains the key when the filter is unavailable', () => {
    harness = mount({ filterAvailable: false });

    const event = press({ key: 'b' });

    // Nothing to type into — but the panel is still modal, so the key must
    // not toggle a scene control behind it either.
    expect(event.defaultPrevented).toBe(false);
    expect(harness.escaped).toEqual([]);
    expect(harness.input.value).toBe('');
  });

  it('contains — rather than releases — a key an inner handler already consumed', () => {
    harness = mount();
    // A descendant (e.g. a listing row's arrow navigation, which calls
    // preventDefault WITHOUT stopPropagation) claims the key first; the event
    // then bubbles to the container already `defaultPrevented`.
    harness.row.addEventListener('keydown', (e) => e.preventDefault());
    harness.row.focus();

    press({ key: 'b' });
    press({ key: 'Home' });

    // Not forwarded into the filter (the inner handler owns it)…
    expect(harness.input.value).toBe('');
    expect(document.activeElement).toBe(harness.row);
    // …and not handed to the global bindings either.
    expect(harness.escaped).toEqual([]);
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
