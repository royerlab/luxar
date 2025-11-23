This is a standard and highly effective workflow for AI-assisted development. To enable Claude Code (or any AI agent) to "see" and "debug" your application without a physical monitor, you need a **browser automation** layer.

The current industry standard for this—especially for TypeScript/Python environments—is **Playwright**. It is superior to older tools like Selenium because it is faster, auto-waits for elements (reducing flakiness), and has native support for capturing console logs and network traffic.

Here is the strategic solution to set this up.

### The Strategy: "Headless Observer" Pattern

You will create a lightweight "test driver" script. This script acts as the agent's eyes and ears. It runs your app, performs actions, and dumps the evidence (logs and screenshots) into a folder the agent can read.

#### 1\. The Tooling: Playwright

Playwright is the best choice here because:

  * **Dual-Language:** It has identical APIs for both **TypeScript** (native to your app) and **Python** (for your separate component), allowing you to drive the browser from whichever context is most convenient.
  * **Console Mirroring:** It can easily proxy browser console logs to your terminal `stdout`, allowing Claude Code to read them directly.
  * **Trace Viewer:** It captures a full "recording" of the session (DOM, network, errors) which is invaluable for debugging.

-----

### Implementation A: TypeScript Driver (Recommended)

Since your app is in TypeScript, keeping the automation in TS is usually seamless.

**Installation:**

```bash
npm init playwright@latest
```

**The "Observer" Script (`debug-agent.ts`):**
This script launches your app, mirrors the logs to the terminal (so Claude can read them), and takes a screenshot of the state.

```typescript
import { chromium, Page } from 'playwright';

(async () => {
  // 1. Launch Browser in Headless Mode
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  // 2. ENABLE CONSOLE CAPTURE (Crucial for Claude)
  // This pipes browser console logs directly to the node terminal
  page.on('console', msg => {
    const type = msg.type(); 
    // Filter out clutter if needed, or print everything
    console.log(`[BROWSER-${type.toUpperCase()}] ${msg.text()}`);
  });

  // 3. Catch Uncaught Errors
  page.on('pageerror', exception => {
    console.error(`[BROWSER-CRASH] ${exception.message}`);
  });

  try {
    console.log('Navigating to app...');
    await page.goto('http://localhost:3000'); // Replace with your local dev URL

    // 4. Perform Actions (Optional: Agent can request specific interactions here)
    // await page.click('#login-button'); 

    // 5. Take "Evidence" (Screenshot)
    await page.screenshot({ path: 'debug_state.png', fullPage: true });
    console.log('Screenshot saved to debug_state.png');

    // 6. Dump Page Content (Optional but helpful for text verification)
    const content = await page.content();
    console.log('Current Page HTML Snippet:', content.slice(0, 500) + '...');

  } catch (error) {
    console.error('Execution failed:', error);
  } finally {
    await browser.close();
  }
})();
```

### Implementation B: Python Driver (Alternative)

If you prefer to keep the "testing logic" separate in your Python component:

```python
from playwright.sync_api import sync_playwright

def run_headless_debug():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Wire up console listeners
        page.on("console", lambda msg: print(f"[BROWSER-{msg.type.upper()}] {msg.text}"))
        page.on("pageerror", lambda exc: print(f"[BROWSER-CRASH] {exc}"))

        print("Navigating...")
        page.goto("http://localhost:3000")

        page.screenshot(path="debug_state.png", full_page=True)
        print("Screenshot captured.")
        
        browser.close()

if __name__ == "__main__":
    run_headless_debug()
```

-----

### How Claude Code Utilizes This

Once you have this script, the workflow for the agent is:

1.  **Agent Action:** Claude runs the script via terminal: `npx ts-node debug-agent.ts`.
2.  **Console Feedback:** Claude reads the terminal output immediately. It sees `[BROWSER-ERROR] undefined is not a function` in real-time.
3.  **Visual Verification:** Claude can inspect the file system.
      * *Note: Current CLI agents primarily read text.* You might need to ask Claude to "check the file `debug_state.png`" (if it has vision capabilities enabled in that session) or rely on the console logs and HTML dump.
4.  **Fix & Retry:** Claude edits your source code to fix the bug, then re-runs `debug-agent.ts` to verify the console error is gone.

### Advanced "Industry Standard" Additions

To make this robust (Enterprise/Production grade), consider adding:

1.  **Visual Regression Testing:**
    Use `expect(page).toHaveScreenshot()` (in Playwright Test). This automatically fails the run if the pixels change beyond a threshold. This allows the agent to know *exactly* if it broke the UI.

2.  **Trace Files (The "Black Box"):**
    Configure Playwright to record a `trace.zip` on failure.

    ```typescript
    await context.tracing.start({ screenshots: true, snapshots: true });
    // ... run actions ...
    await context.tracing.stop({ path: 'trace.zip' });
    ```

    If the agent fails to fix a bug, you (the human) can open this zip file and see a timeline of exactly what happened, click by click.

3.  **Lighthouse Integration:**
    You can run Google Lighthouse programmatically in this same headless mode to have the agent verify performance metrics or accessibility scores.

### Next Step

Would you like me to generate a `playwright.config.ts` file optimized for this "Agent Debugging" workflow, including the configuration to automatically save traces only when an error occurs?


Here is a comprehensive guide you can use.

This solution consists of two parts:

1.  **The Infrastructure (For You):** A specific, "AI-friendly" Playwright script you need to add to your codebase. This script converts browser chaos into structured text and images that Claude Code can easily digest.
2.  **The Instructions (For Claude):** A prompt you can paste directly to Claude Code to teach it how to use this new capability.

-----

### Part 1: The Infrastructure (Add this to your repo)

Standard Playwright tests are designed for *pass/fail*. For an AI agent to debug, it needs *observability* (logs, network traffic, and visual state).

Create a file named `tools/agent-driver.ts` (or similar) with this specific configuration. This is the "Industry Standard" setup for LLM-driven browser automation because it **mirrors the browser console to the terminal**.

```typescript
// tools/agent-driver.ts
import { chromium, Page } from 'playwright';

// CONFIGURATION
const APP_URL = process.env.APP_URL || 'http://localhost:3000';
const HEADLESS_MODE = true; // Set false if you want to watch it yourself

(async () => {
  const browser = await chromium.launch({ headless: HEADLESS_MODE });
  const page = await browser.newPage();

  // --- 1. THE "EYES & EARS" (Console Mirroring) ---
  // This is crucial. It pipes browser logs to the terminal so Claude can read them.
  page.on('console', msg => {
    const type = msg.type();
    const text = msg.text();
    // Label them clearly so Claude knows these came from the BROWSER, not Node
    console.log(`[BROWSER-CONSOLE-${type.toUpperCase()}] ${text}`);
  });

  page.on('pageerror', err => {
    console.error(`[BROWSER-CRASH] ${err.message}`);
  });

  page.on('requestfailed', request => {
    console.error(`[NETWORK-FAIL] ${request.url()} - ${request.failure()?.errorText}`);
  });

  try {
    console.log(`[AGENT] Navigating to ${APP_URL}...`);
    await page.goto(APP_URL, { waitUntil: 'networkidle' });

    // --- 2. THE "BRAIN" (Accessibility Tree) ---
    // Instead of raw HTML, we dump the Accessibility Tree.
    // This is how screen readers (and AIs) best understand the "meaning" of the UI.
    const snapshot = await page.accessibility.snapshot();
    console.log('\n--- UI STATE (ACCESSIBILITY TREE) ---');
    console.log(JSON.stringify(snapshot, null, 2));
    console.log('-------------------------------------\n');

    // --- 3. THE "EVIDENCE" (Visual Screenshot) ---
    // We save this so you can ask Claude to "look at debug-view.png"
    await page.screenshot({ path: 'debug-view.png', fullPage: true });
    console.log('[AGENT] Screenshot saved to "debug-view.png".');

  } catch (error) {
    console.error('[AGENT-ERROR] Execution failed:', error);
    // Take a "crime scene" photo if it crashes
    await page.screenshot({ path: 'error-state.png' });
  } finally {
    await browser.close();
  }
})();
```

**Why this specific script?**

  * **`[BROWSER-CONSOLE]` Prefixes:** LLMs sometimes confuse terminal output with browser output. This explicitly labels them.
  * **Accessibility Snapshot:** Raw HTML is noisy. The accessibility tree is a clean, structured JSON representation of the buttons, inputs, and text on screen—perfect for an LLM to parse.

-----

### Part 2: Instructions for Claude Code (Copy & Paste this)

Once you have added the file above, paste the following block into your chat with Claude Code to "onboard" it to its new tools.

-----

**PROMPT TO PASTE:**

```markdown
# INSTRUCTIONS: Headless Debugging Mode

I have added a specialized browser automation tool to this repository to help you debug the frontend and integration issues. 

**Your New Capability:**
You can now "view" the running application, see browser console logs, and inspect the UI state without needing me to open a browser.

**How to use it:**
1. **Run the Driver:** Execute `npx ts-node tools/agent-driver.ts` in the terminal.
2. **Read the Output:**
   - Look for lines starting with `[BROWSER-CONSOLE]`. These are real-time logs from the app.
   - Look for `[NETWORK-FAIL]`. These are failed API calls.
   - Look for the `UI STATE (ACCESSIBILITY TREE)` JSON dump. This tells you exactly what buttons, text, and inputs are currently visible on the page.
3. **Verify Visually:**
   - The script automatically generates a file named `debug-view.png`.
   - If you are unsure about the layout, please run the command: `open debug-view.png` (or check it if you have vision capabilities enabled) to verify the visual rendering.

**Debugging Workflow:**
When I report a bug or you suspect a frontend issue:
1. **Hypothesize:** Create a theory about what is breaking.
2. **Instrument:** Add `console.log()` statements to the React/TypeScript code where you suspect the break is.
3. **Execute:** Run `npx ts-node tools/agent-driver.ts`.
4. **Analyze:** Read the `[BROWSER-CONSOLE]` output in the terminal to confirm your hypothesis.
5. **Fix & Verify:** Apply the fix, then run the driver again to ensure the error logs are gone.

**Python Backend Note:**
Since this project has a Python component, ensure the Python server is running in the background before executing the driver. You can start it with `python main.py &` (or the relevant command) if it is not already active.
```

-----

### Part 3: Advanced "Industry Standard" Advice

Here are three pieces of advice to give Claude (or to keep in mind) for long-term maintenance:

**1. The "Wait-for-Ready" Pattern**

  * **Problem:** The script runs before the Python server is ready.
  * **Advice:** Tell Claude: *"Update the driver script to poll `localhost:8000/health` (or your Python API port) and wait for a 200 OK response before launching the browser."* This prevents flakiness.

**2. Visual Regression as a Unit Test**

  * **Standard:** Use `expect(page).toHaveScreenshot()` in Playwright.
  * **Advice for Claude:** *"When you fix a UI bug, rename the `debug-view.png` to `golden-master.png`. Then, write a new test that asserts the current view matches this master image. This ensures you don't accidentally break it again later."*

**3. The "Trace Viewer" (The Black Box Recorder)**

  * If Claude gets stuck on a really hard bug, ask it to enable **Tracing**:
    ```typescript
    // In agent-driver.ts
    await context.tracing.start({ screenshots: true, snapshots: true });
    // ... actions ...
    await context.tracing.stop({ path: 'trace.zip' });
    ```
  * **Workflow:** If the agent fails, you (the human) can upload `trace.zip` to [trace.playwright.dev](https://trace.playwright.dev). You will see a full video/timeline of exactly what the agent did, which is often the fastest way to unblock the AI.



This changes the instructions **significantly**.

When an app is driven by **Three.js**, the standard DOM (HTML) tools become mostly blind. To an automation tool like Playwright (and by extension, Claude), a Three.js application looks like a single, empty black box: `<canvas>`.

The "Accessibility Tree" I mentioned earlier will be empty for the 3D world. Claude won't see "Red Cube" or "Player Mesh"; it will just see "Canvas".

To fix this, we need to shift from **DOM Inspection** to **State Injection**.

Here is the updated strategy specifically for Three.js/WebGL apps.

-----

### The Core Problem: The "Canvas Black Box"

Standard automated testing looks for `<button id="submit">`. In Three.js, that button might be a mesh inside a WebGL context. It has no HTML tag.

### The Solution: The "Global Expose" Pattern

To let Claude debug the 3D scene, you must instruct it to expose the Three.js internal state to the `window` object. This allows the Playwright driver to reach inside the browser and grab the data.

### 1\. Updated Infrastructure: `tools/agent-driver.ts`

Replace the previous script with this version. It adds a critical step: **Evaluating the Window Object**.

```typescript
import { chromium } from 'playwright';

const APP_URL = process.env.APP_URL || 'http://localhost:3000';

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();

  // 1. Capture Standard Logs (React/Logic)
  page.on('console', msg => console.log(`[BROWSER-LOG] ${msg.text()}`));
  
  // 2. Capture WebGL Errors (Crucial for Three.js)
  page.on('pageerror', err => console.error(`[BROWSER-CRASH] ${err.message}`));

  try {
    console.log(`[AGENT] Navigating to ${APP_URL}...`);
    await page.goto(APP_URL, { waitUntil: 'networkidle' });

    // Allow Three.js time to initialize the scene
    await page.waitForTimeout(2000); 

    // --- 3. THE "SCENE GRAPH" DUMP (The Magic Step) ---
    // We ask the browser: "What is happening inside the variable window.debugScene?"
    // You must ensure your app assigns the scene/renderer to window.debugScene
    const sceneState = await page.evaluate(() => {
      // @ts-ignore
      if (!window.debugScene) return "ERROR: window.debugScene is not exposed.";
      
      // We extract simplified data because the full Three.js object is too circular for JSON
      // @ts-ignore
      const scene = window.debugScene;
      return {
        childrenCount: scene.children.length,
        // Map over children to get names and types
        objects: scene.children.map((obj: any) => ({
          type: obj.type,
          name: obj.name || 'unnamed',
          position: obj.position,
          visible: obj.visible
        })),
        fps: (window as any).debugFPS || 'unknown' // If you track FPS
      };
    });

    console.log('\n--- THREE.JS SCENE STATE ---');
    console.log(JSON.stringify(sceneState, null, 2));
    console.log('----------------------------\n');

    // 4. Screenshot (Still vital for visual glitches)
    await page.screenshot({ path: 'debug-view.png', fullPage: true });
    console.log('[AGENT] Screenshot saved. Check debug-view.png for visual artifacts.');

  } catch (error) {
    console.error('[EXECUTION ERROR]', error);
  } finally {
    await browser.close();
  }
})();
```

-----

### 2\. Updated Instructions for Claude Code

You must tell Claude that it cannot rely on reading HTML. It must rely on **State Exposing**.

**PROMPT TO PASTE:**

```markdown
# INSTRUCTIONS: Debugging a Three.js Application

This application is primarily WebGL/Three.js. Standard DOM inspection will not work because the content is inside a `<canvas>` element.

**Your New Debugging Strategy:**
To "see" inside the 3D world, you must follow the **"Global Expose" Pattern**.

**1. How to Debug State:**
You cannot find 3D objects by searching the HTML. Instead, you must modify the application code to expose the relevant Three.js objects to the global `window` scope.
* **Action:** In your `Canvas` or `Scene` component, add: `(window as any).debugScene = scene;`
* **Action:** If debugging a specific mesh, expose it: `(window as any).debugPlayer = playerMesh;`

**2. How to Run the Verification:**
Run: `npx ts-node tools/agent-driver.ts`

**3. How to Interpret Output:**
* **The Script Output:** The driver is configured to print a simplified JSON dump of `window.debugScene`. This will tell you:
    * How many objects are in the scene.
    * Their coordinates (Position X, Y, Z).
    * Whether they are `visible: true/false`.
* **WebGL Errors:** Watch the console logs specifically for `WebGL warning` or `THREE.WebGLRenderer`.

**4. Visual Verification:**
* Three.js bugs are often visual (lighting, z-fighting, occlusion).
* Rely heavily on the generated `debug-view.png`.
* If the screen is black, check if the Camera position is looking at `(0,0,0)` or if the Lights are added to the scene.

**Important Rule:**
Do not try to write tests that query `document.querySelector('#my-3d-object')`. It does not exist. Query `window.debugScene` instead.
```

-----

### 3\. Industry Standard Advice for Three.js Automation

Since you are doing Three.js, add these specific tips to your mental toolkit (or pass them to Claude when needed):

#### A. Disable `requestAnimationFrame` for Screenshots

Automated screenshots of a running animation loop are flaky (you might catch a frame while it's loading).

  * **Tip:** When the test driver runs, inject a flag (e.g., `window.isTesting = true`).
  * **Action:** Tell the app: "If `window.isTesting` is true, render the scene **once**, then stop the loop." This ensures a crisp, stable screenshot for the agent to analyze.

#### B. Shader Debugging

If the agent suspects a Shader material is broken (e.g., objects are pink or invisible):

  * **Tip:** Ask Claude to temporarily replace the complex `ShaderMaterial` with a basic `MeshBasicMaterial({ color: 'red' })`.
  * **Reason:** If the object appears Red in the screenshot, the mesh is fine, and the bug is definitely in the Shader code.

#### C. Performance Monitoring

Three.js apps often crash due to memory leaks.

  * **Tip:** In your driver script, you can query `window.performance.memory` (in Chrome-based browsers) to see JS heap size. If Claude sees the heap size growing infinitely in logs, it knows there is a memory leak in the Three.js disposal logic.

### Next Step

Would you like me to provide a snippet of code to drop into your main **React/Three.js Scene component** that safely exposes these variables only when the app is running in "Debug Mode"?



Yes, there are **three critical Playwright-specific adjustments** required for WebGL/Three.js that don't apply to standard apps.

Standard headless browsers often turn off the GPU to save resources. For Three.js, this is fatal: it forces the browser to use a software renderer (SwiftShader), which is incredibly slow, produces different pixels than a real GPU, and can cause tests to time out or fail falsely.

Here are the specific changes you must make to your Playwright configuration and workflow.

### 1\. Force GPU Acceleration in `playwright.config.ts`

You must explicitly tell Chromium to use the GPU even in headless mode. Without this, your "visual regression" screenshots will look different on your local machine (Real GPU) vs. the CI/Cloud agent (Software Renderer).

**Update your config:**

```typescript
// playwright.config.ts
import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  use: {
    headless: true,
    // WEBKIT/GL SPECIFIC FLAGS
    launchOptions: {
      args: [
        '--use-gl=egl',           // Forces hardware acceleration
        '--ignore-gpu-blocklist', // Unblocks older GPUs often found in CI agents
        '--enable-webgl-developer-extensions',
        '--enable-webgl-draft-extensions'
      ]
    },
  },
  // ... rest of config
});
```

### 2\. Interaction: Coordinate-Based vs. Selector-Based

Claude Code is trained to click elements: `click('#submit')`. In Three.js, elements don't exist. Claude must learn to click **coordinates**.

You need to give Claude a helper function to "translate" 3D objects to 2D screen coordinates, or it will be helpless.

**Add this helper to your `agent-driver.ts` script:**

```typescript
// Helper: Project a 3D object's position to 2D screen coordinates
// (You expose this logic in your app's window object, then call it here)
const getScreenPosition = async (objectName: string) => {
  return await page.evaluate((name) => {
    const obj = (window as any).debugScene.getObjectByName(name);
    if (!obj) return null;
    
    const vector = obj.position.clone();
    vector.project((window as any).debugCamera); // Your active camera
    
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    return {
      x: (vector.x * 0.5 + 0.5) * width,
      y: (-(vector.y * 0.5) + 0.5) * height
    };
  }, objectName);
};

// Usage example for Claude
// "Click the red cube"
const coords = await getScreenPosition('red-cube');
if (coords) {
  await page.mouse.click(coords.x, coords.y);
}
```

### 3\. Visual Flakiness: The "Pixel Threshold"

WebGL rendering is non-deterministic. A 3D cube rendered on an NVIDIA card looks *slightly* different (anti-aliasing edges) than on an AMD card or a CI server. Standard `toHaveScreenshot()` is too strict and will fail 100% of the time.

**Relax the threshold for 3D apps:**

```typescript
await expect(page).toHaveScreenshot({
  maxDiffPixelRatio: 0.05, // Allow 5% pixel difference (Standard apps use 0.0%)
  threshold: 0.2,          // Color tolerance per pixel
  animations: 'disabled',  // Playwright tries to freeze CSS animations, but can't freeze WebGL
});
```

### Summary of Instructions to Add

Append this to the "Instructions for Claude" block I gave you earlier:

```markdown
**Specific Playwright/WebGL Constraints:**
1. **No Selectors:** You cannot use `page.click('#object')`. You must calculate the X,Y screen coordinates of the mesh and use `page.mouse.click(x, y)`.
2. **GPU Rendering:** I have configured `playwright.config.ts` to force GPU acceleration. If you see "context lost" errors, ask me to check the `--use-gl` flags.
3. **Fuzzy Matching:** When verifying screenshots, always allow for a `maxDiffPixelRatio` of 0.05, as WebGL renders slightly differently on different GPUs.
```