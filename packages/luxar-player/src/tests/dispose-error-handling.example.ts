/**
 * Example demonstrating robust error handling in DataLoadingMonitor disposal
 *
 * This example shows how the monitor gracefully handles various failure scenarios
 * during cleanup, ensuring that partial failures don't prevent other cleanup
 * operations from completing.
 */

import { DataLoadingMonitor } from '../ui/data-loading-monitor';
import type { LoaderMonitor } from '../ui/data-monitor-types';

// Example 1: Normal disposal - all operations succeed
function normalDisposal() {
  console.log('\n=== Example 1: Normal Disposal ===');

  const container = document.createElement('div');
  document.body.appendChild(container);

  const monitor = new DataLoadingMonitor(container);
  monitor.show();

  // Dispose normally
  monitor.dispose();
  console.log('✅ Monitor disposed successfully');
}

// Example 2: Disposal with loader failure
function disposalWithLoaderFailure() {
  console.log('\n=== Example 2: Disposal with Loader Failure ===');

  const container = document.createElement('div');
  document.body.appendChild(container);

  const monitor = new DataLoadingMonitor(container);

  // Create a loader that will fail during cleanup
  const problematicLoader: LoaderMonitor = {
    addEventListener: () => {},
    removeEventListener: () => {
      throw new Error('Network connection lost - cannot unsubscribe');
    },
    getMetrics: () => ({}) as any,
    getActiveQueries: () => [],
  };

  monitor.connectLoader('/problematic', problematicLoader);
  monitor.show();

  // Dispose will log warnings but complete successfully
  monitor.dispose();
  console.log('✅ Monitor disposed despite loader failure');
}

// Example 3: Disposal with DOM manipulation failure
function disposalWithDOMFailure() {
  console.log('\n=== Example 3: Disposal with DOM Failure ===');

  const container = document.createElement('div');
  document.body.appendChild(container);

  const monitor = new DataLoadingMonitor(container);
  monitor.show();

  // Simulate a situation where the panel is already removed
  const panel = container.querySelector('.luxar-data-monitor') as HTMLElement;
  if (panel) {
    // Remove panel prematurely
    panel.remove();

    // Override remove to simulate failure
    panel.remove = () => {
      throw new Error('Panel already removed from DOM');
    };
  }

  // Dispose will handle the error gracefully
  monitor.dispose();
  console.log('✅ Monitor disposed despite DOM error');
}

// Example 4: Critical failure that throws
function disposalWithCriticalFailure() {
  console.log('\n=== Example 4: Disposal with Critical Failure ===');

  const container = document.createElement('div');
  document.body.appendChild(container);

  const monitor = new DataLoadingMonitor(container);

  // Simulate a critical internal state corruption
  // (In practice, this would be very rare)
  (monitor as any).loaders = {
    clear: () => {
      throw new Error('Critical: Internal state corrupted');
    },
    values: () => [],
    entries: () => [],
  };

  try {
    monitor.dispose();
  } catch (error) {
    console.log('⚠️ Critical error during disposal:', (error as Error).message);
    console.log('✅ Critical errors are properly reported');
  }
}

// Example 5: Multiple concurrent failures
function disposalWithMultipleFailures() {
  console.log('\n=== Example 5: Disposal with Multiple Failures ===');

  const container = document.createElement('div');
  document.body.appendChild(container);

  const monitor = new DataLoadingMonitor(container);

  // Add multiple problematic loaders
  for (let i = 0; i < 5; i++) {
    const loader: LoaderMonitor = {
      addEventListener: () => {},
      removeEventListener: () => {
        if (i % 2 === 0) {
          throw new Error(`Loader ${i} cleanup failed`);
        }
      },
      getMetrics: () => ({}) as any,
      getActiveQueries: () => [],
    };
    monitor.connectLoader(`/loader${i}`, loader);
  }

  monitor.show();

  // Dispose will continue despite multiple failures
  monitor.dispose();
  console.log('✅ Monitor disposed successfully despite multiple failures');
  console.log('   - Failed loaders: 0, 2, 4');
  console.log('   - Successful loaders: 1, 3');
  console.log('   - All resources cleaned up');
}

// Run examples
export function runErrorHandlingExamples() {
  console.log('🧪 DataLoadingMonitor Disposal Error Handling Examples');
  console.log('='.repeat(50));

  normalDisposal();
  disposalWithLoaderFailure();
  disposalWithDOMFailure();
  disposalWithCriticalFailure();
  disposalWithMultipleFailures();

  console.log('\n' + '='.repeat(50));
  console.log('✅ All examples completed successfully');
  console.log('\nKey takeaways:');
  console.log("1. Non-critical errors are logged but don't stop cleanup");
  console.log('2. All cleanup operations are attempted even if some fail');
  console.log('3. Critical errors (rare) are properly thrown');
  console.log('4. Resources are freed even in error scenarios');
  console.log('5. Error messages provide debugging information');
}
