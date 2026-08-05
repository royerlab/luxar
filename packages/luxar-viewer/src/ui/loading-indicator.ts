/**
 * Loading indicator overlay — centered spinner shown during scene load or
 * dataset switching. Auto-positions over the viewport with a translucent
 * background; removed via hideLoadingIndicator() or element.remove().
 */

import { getViewerContainer } from '../utils/viewer-container';

/**
 * Create and mount the centered "Loading scene..." indicator into the viewer
 * container. The indicator is a small box fixed to the viewport centre carrying
 * a spinner and a text label (it does not cover the viewport); it is given a
 * fixed id so {@link hideLoadingIndicator} can find and remove it later.
 *
 * @returns The mounted indicator element, so the caller can update its text or
 *   remove it directly.
 */
export function showLoadingIndicator(): HTMLElement {
  const loadingDiv = document.createElement('div');
  loadingDiv.id = 'luxar-loading-indicator';
  loadingDiv.className = 'luxar-loading-indicator';

  const spinner = document.createElement('div');
  spinner.className = 'luxar-loading-indicator__spinner';

  const text = document.createElement('div');
  text.className = 'luxar-loading-indicator__text';
  text.textContent = 'Loading scene...';
  text.id = 'luxar-loading-text';

  loadingDiv.appendChild(spinner);
  loadingDiv.appendChild(text);
  getViewerContainer().appendChild(loadingDiv);

  return loadingDiv;
}

/**
 * Remove the loading overlay created by {@link showLoadingIndicator}, if one is
 * currently mounted. A no-op when no indicator is present, so it is safe to call
 * unconditionally on load completion or error.
 */
export function hideLoadingIndicator() {
  const loadingDiv = document.getElementById('luxar-loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }
}
