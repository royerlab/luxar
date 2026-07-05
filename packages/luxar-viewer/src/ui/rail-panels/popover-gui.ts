/**
 * Shared helper for rail popovers that host the custom GUI.
 *
 * A rail popover (see {@link ControlRailPopover}) is itself the glass surface;
 * the GUI mounted inside it should read as flat content. This helper creates a
 * GUI parented to the popover host, strips its own glass marker (so the
 * liquid-glass ::before/::after layers don't double up), and shows it. The
 * popover CSS neutralizes the GUI's fixed positioning / frame.
 *
 * @module ui/rail-panels/popover-gui
 */

import GUI from '../gui';

/**
 * Create a GUI mounted inside a rail popover `host`, styled as flat content.
 *
 * @param host - The popover body element to mount into.
 * @param title - Panel title shown in the GUI header.
 * @param width - Nominal width (CSS forces 100% inside the popover; kept for
 *   the GUI's internal layout math).
 */
export function makePopoverGui(host: HTMLElement, title: string, width = 264): GUI {
  const gui = new GUI({ title, width, container: host, closeFolders: false });
  // The popover container is the glass surface — drop the nested marker so the
  // glass themes don't render a second refraction layer inside it.
  gui.domElement.classList.remove('luxar-glass-surface');
  gui.show();
  return gui;
}
