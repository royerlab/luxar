/**
 * Settings rail popover — viewer-wide preferences reached from the gear icon.
 *
 * Hosts the theme picker plus the persisted global preferences
 * (`config/user-settings.ts` → localStorage `luxar.settings`) as sibling
 * folders: Input, Performance, Caching, Advanced.
 *
 * Application model per control (see user-settings.ts for the full story):
 *   - LIVE values (FOV wheel sensitivity, idle timeout, web-worker routing)
 *     take effect immediately via `applyLiveConfigOverrides`.
 *   - STARTUP values (worker pool size, prefetch concurrency, cache tier
 *     gates + pool budget, renderer backend) are threaded through bootstrap
 *     on the next load; changing one shows the reload hint row, whose state
 *     diffs against the boot snapshot (`reloadRequired`) so it survives
 *     popover close/reopen and clears when a value is set back.
 *
 * URL parameters always win over stored preferences at boot (one-way
 * disable flags compose with `||`, value params with `??` — bootstrap.ts).
 *
 * @module ui/rail-panels/settings-popover
 */

import { setupThemeControls } from '../rendering-controls/setup/theme-setup';
import { FOLDER_ICONS } from '../rendering-controls/folder-icons';
import { makePopoverGui } from './popover-gui';
import {
  defaultUserSettings,
  loadUserSettings,
  saveUserSettings,
  applyLiveConfigOverrides,
  reloadRequired,
  USER_SETTINGS_RANGES,
} from '../../config/user-settings';
import { showToast } from '../toast';
import type { SceneLoader } from '../../data/scene-loader';

export interface SettingsPopoverContext {
  /** Re-render after a change so the effect paints immediately. */
  triggerAnimation: () => void;
  /**
   * Lazy scene-loader accessor — the loader exists only after the first
   * scene load. Used by the Caching folder (budget readout + Clear Caches).
   */
  getSceneLoader: () => SceneLoader | null;
}

const MB = 1024 * 1024;

/** Human line for the resolved budgets, e.g. "heap · L0 412 · L1 206 · S 618 MB". */
function describeBudgets(loader: SceneLoader | null): string {
  const budgets = loader?.getCacheBudgets();
  if (!budgets) return 'Budgets resolve on scene load';
  const mb = (bytes: number): string => `${Math.round(bytes / MB)}`;
  return (
    `Resolved (${budgets.source}): ` +
    `L0 ${mb(budgets.l0Bytes)} · L1 ${mb(budgets.l1Bytes)} · S ${mb(budgets.sliceBytes)} MB`
  );
}

/**
 * Build the Settings popover into `host`. Returns a teardown that disposes the
 * GUI when the popover closes.
 */
export function buildSettingsPopover(host: HTMLElement, ctx: SettingsPopoverContext): () => void {
  let teardown = buildContent(host, ctx);
  return () => teardown();

  /** Rebuild in place (Reset All) — swap the teardown the closure returns. */
  function rebuild(): void {
    teardown();
    host.innerHTML = '';
    teardown = buildContent(host, ctx);
  }

  function buildContent(el: HTMLElement, context: SettingsPopoverContext): () => void {
    const gui = makePopoverGui(el, 'Settings');
    const settings = loadUserSettings();

    // ── Reload hint row (below the GUI): shown whenever a reload-required
    // value differs from what was applied at boot.
    const hint = document.createElement('div');
    hint.className = 'luxar-control-rail__popover-hint';
    const hintText = document.createElement('span');
    hintText.textContent = 'Some changes apply after reload ';
    const reloadBtn = document.createElement('button');
    reloadBtn.type = 'button';
    reloadBtn.className = 'luxar-gui__button';
    reloadBtn.textContent = 'Reload';
    reloadBtn.style.width = 'auto';
    reloadBtn.style.marginLeft = '8px';
    reloadBtn.addEventListener('click', () => location.reload());
    hint.appendChild(hintText);
    hint.appendChild(reloadBtn);
    el.appendChild(hint);

    const updateHint = (): void => {
      hint.style.display = reloadRequired(settings) ? '' : 'none';
    };
    updateHint();

    /** Persist + apply live values + refresh the reload hint. */
    const commit = (): void => {
      saveUserSettings(settings);
      applyLiveConfigOverrides(settings);
      updateHint();
      context.triggerAnimation();
    };

    // ── Theme (unchanged — persists via ThemeManager's own storage).
    setupThemeControls({ gui, triggerAnimation: context.triggerAnimation });

    // ── Input
    const inputFolder = gui.addFolder('Input', FOLDER_ICONS.input);
    const fovRange = USER_SETTINGS_RANGES.fovSensitivity;
    inputFolder
      .add(settings.input, 'fovSensitivity', fovRange.min, fovRange.max, 0.01)
      .name('FOV Sensitivity')
      .onChange(commit);

    // ── Performance
    const perfFolder = gui.addFolder('Performance', FOLDER_ICONS.performance);
    // Idle timeout is edited in seconds (friendlier unit than ms).
    const perfProxy = { idleTimeoutS: settings.performance.idleTimeoutMs / 1000 };
    perfFolder
      .add(
        perfProxy,
        'idleTimeoutS',
        USER_SETTINGS_RANGES.idleTimeoutMs.min / 1000,
        USER_SETTINGS_RANGES.idleTimeoutMs.max / 1000,
        0.5
      )
      .name('Idle Pause (s)')
      .onChange((v: number) => {
        settings.performance.idleTimeoutMs = Math.round(v * 1000);
        commit();
      });
    perfFolder.add(settings.performance, 'useWebWorkers').name('Web Workers').onChange(commit);
    perfFolder
      .add(
        settings.performance,
        'workerCount',
        USER_SETTINGS_RANGES.workerCount.min,
        USER_SETTINGS_RANGES.workerCount.max,
        1
      )
      .name('Workers (0 = auto)')
      .onChange(commit);
    perfFolder
      .add(
        settings.performance,
        'networkMaxConcurrent',
        USER_SETTINGS_RANGES.networkMaxConcurrent.min,
        USER_SETTINGS_RANGES.networkMaxConcurrent.max,
        1
      )
      .name('Prefetch Limit')
      .onChange(commit);

    // ── Caching
    const cacheFolder = gui.addFolder('Caching', FOLDER_ICONS.caching);
    cacheFolder.add(settings.caching, 'enabled').name('Cache').onChange(commit);
    cacheFolder.add(settings.caching, 'sliceCache').name('Slice Cache').onChange(commit);
    cacheFolder.add(settings.caching, 'prefetch').name('Prefetch').onChange(commit);
    const budgetMBControl = () =>
      cacheFolder
        .add(
          settings.caching,
          'budgetMB',
          USER_SETTINGS_RANGES.budgetMB.min,
          USER_SETTINGS_RANGES.budgetMB.max,
          128
        )
        .name('Budget (MB)')
        .onChange(commit);
    cacheFolder
      .add(settings.caching, 'budgetMode', {
        'Auto (heap-aware)': 'auto',
        Custom: 'custom',
      })
      .name('Budget')
      .onChange(() => {
        commit();
        if (settings.caching.budgetMode === 'custom') {
          mbControl.show();
        } else {
          mbControl.hide();
        }
      });
    const mbControl = budgetMBControl();
    if (settings.caching.budgetMode !== 'custom') mbControl.hide();

    // Read-only resolved-budget line (source + per-tier MB). `explicit` here
    // also self-documents that a `?cacheBudgetMB=` URL param won over the UI.
    const budgetNote = document.createElement('div');
    budgetNote.className = 'luxar-control-rail__popover-note';
    budgetNote.textContent = describeBudgets(context.getSceneLoader());
    cacheFolder.domElement?.appendChild(budgetNote);

    cacheFolder
      .add(
        {
          clearCaches: async () => {
            const loader = context.getSceneLoader();
            if (!loader) {
              showToast('No scene loaded — nothing to clear');
              return;
            }
            try {
              await loader.clearAllCaches();
              showToast('Caches cleared (L0 · L1 · L2 · S-cache)');
            } catch {
              showToast('Cache clear failed — see console');
            }
            context.triggerAnimation();
          },
        },
        'clearCaches'
      )
      .name('Clear Caches');

    // ── Advanced
    const advFolder = gui.addFolder('Advanced', FOLDER_ICONS.advanced);
    advFolder
      .add(settings.advanced, 'renderer', {
        Auto: 'auto',
        WebGL: 'webgl',
        WebGPU: 'webgpu',
      })
      .name('Renderer')
      .onChange(commit);
    advFolder
      .add(
        {
          resetAll: () => {
            const defaults = defaultUserSettings();
            saveUserSettings(defaults);
            applyLiveConfigOverrides(defaults);
            showToast('Settings reset to defaults');
            rebuild();
          },
        },
        'resetAll'
      )
      .name('Reset All Settings');

    return () => gui.destroy();
  }
}
