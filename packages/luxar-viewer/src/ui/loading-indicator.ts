/**
 * Loading indicator overlay — centered spinner shown during scene load or
 * dataset switching. Auto-positions over the viewport with a translucent
 * background; removed via hideLoadingIndicator() or element.remove().
 */

import { getViewerContainer } from '../utils/viewer-container';

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

export function hideLoadingIndicator() {
  const loadingDiv = document.getElementById('luxar-loading-indicator');
  if (loadingDiv) {
    loadingDiv.remove();
  }
}
