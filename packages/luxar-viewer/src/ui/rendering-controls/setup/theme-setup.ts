/**
 * Theme picker setup for the rendering-controls panel.
 *
 * Creates the "🎨 Theme" folder with a single dropdown bound to
 * `ThemeManager`. The theme manager itself handles persistence;
 * we just need to call `setTheme()` and trigger a re-render so
 * the change is visible immediately.
 */

import type GUI from '../../gui';
import { ThemeManager } from '../../../themes/theme-manager';
import { log, Modules } from '../../../utils/log';
import { FOLDER_ICONS } from '../folder-icons';

export interface ThemeSetupContext {
  gui: GUI;
  triggerAnimation: () => void;
}

export function setupThemeControls(context: ThemeSetupContext): void {
  const { gui, triggerAnimation } = context;
  const themeFolder = gui.addFolder('Theme', FOLDER_ICONS.theme);

  themeFolder.domElement?.setAttribute(
    'title',
    'Theme: Choose the visual appearance of the viewer UI\n\n' +
      'Themes change the background, panel colors, and overall look.\n' +
      'Your choice is automatically saved and restored next session.'
  );

  const themeManager = ThemeManager.getInstance();
  const themes = themeManager.getAllThemes();

  const themeOptions = themes.reduce(
    (acc, theme) => {
      acc[theme.name] = theme.id;
      return acc;
    },
    {} as Record<string, string>
  );

  const themeSettings = {
    theme: themeManager.getCurrentTheme().id,
  };

  const themeControl = themeFolder
    .add(themeSettings, 'theme', themeOptions)
    .name('Active Theme')
    .onChange((themeId: string) => {
      themeManager.setTheme(themeId);
      log.info(Modules.RENDERER, `Theme changed to: ${themeId}`);
      triggerAnimation();
    });

  themeControl.domElement.setAttribute(
    'title',
    'Switch between visual themes\n' +
      '• Dark: Default scientific visualization theme\n' +
      '• Light: Bright theme for well-lit environments\n' +
      '• Frosted Glass: Subtle translucent glassmorphism\n' +
      '• Liquid Glass: True glass effect with inner glow and tint'
  );

  themeFolder.close();
}
