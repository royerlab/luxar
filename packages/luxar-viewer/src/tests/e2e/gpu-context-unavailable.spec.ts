/** Startup should explain when the browser cannot create either GPU context. */
import { test, expect } from '@playwright/test';

for (const renderer of ['webgl', 'webgpu'] as const) {
  test(`shows a GPU access remedy for ${renderer} when contexts are blocked`, async ({ page }) => {
    const teardownErrors: string[] = [];
    page.on('console', (entry) => {
      if (/Error disposing sceneManager|component\(s\) threw during teardown/.test(entry.text())) {
        teardownErrors.push(entry.text());
      }
    });
    await page.addInitScript(() => {
      Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true });
      const getContext = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function (this: HTMLCanvasElement, ...args) {
        if (args[0] === 'webgl2') return null;
        return Reflect.apply(getContext, this, args);
      } as typeof getContext;
    });

    await page.goto(`/?renderer=${renderer}`);
    const message = page.locator('#luxar-error-message-text');
    await expect(message).toContainText('browser is not providing GPU access');
    await expect(message).toContainText('Reload the page or restart the browser');
    await expect(message).toContainText('chrome://gpu');
    expect(teardownErrors).toEqual([]);
  });
}
