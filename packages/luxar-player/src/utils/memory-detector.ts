/**
 * Minimal automatic memory detection for adaptive cache sizing
 */

export interface MemoryInfo {
  recommendedCacheMB: number;
  confidence: 'high' | 'medium' | 'low';
  source: 'api' | 'device' | 'default';
}

/**
 * Detect available memory and recommend cache size
 */
export function detectMemory(): MemoryInfo {
  // Try Chrome's memory API first (most accurate - gives actual JS heap size)
  if ((performance as any).memory) {
    const mem = (performance as any).memory;
    const heapLimitMB = mem.jsHeapSizeLimit / (1024 * 1024);
    const usedMB = mem.usedJSHeapSize / (1024 * 1024);
    const availableMB = heapLimitMB - usedMB;
    
    // Use 80% of available heap memory for cache
    // Since lazy loading is the primary memory consumer, we can be aggressive
    const recommended = Math.round(availableMB * 0.8);
    
    // Apply reasonable bounds: min 128MB, max 4GB
    const bounded = Math.min(4096, Math.max(128, recommended));
    
    console.log(`[Memory API] Heap limit: ${heapLimitMB.toFixed(0)}MB, Used: ${usedMB.toFixed(0)}MB, Available: ${availableMB.toFixed(0)}MB → Cache: ${bounded}MB`);

    return {
      recommendedCacheMB: bounded,
      confidence: 'high',
      source: 'api',
    };
  }

  // Try device memory API (gives total device RAM in GB)
  if ((navigator as any).deviceMemory) {
    const deviceGB = (navigator as any).deviceMemory;
    
    // Estimate available JS heap based on device RAM
    // Browsers typically allow ~25-50% of system RAM for JS heap
    // We'll use a conservative estimate
    let estimatedHeapMB: number;
    let recommendedCacheMB: number;
    
    if (deviceGB <= 2) {
      // Very low memory: assume 256MB heap, use 200MB cache (80%)
      estimatedHeapMB = 256;
      recommendedCacheMB = 200;
    } else if (deviceGB <= 4) {
      // Mobile/low memory: assume 512MB heap, use 400MB cache (80%)
      estimatedHeapMB = 512;
      recommendedCacheMB = 400;
    } else if (deviceGB <= 8) {
      // Mid-range: assume 2GB heap, use 1.6GB cache (80%)
      estimatedHeapMB = 2048;
      recommendedCacheMB = 1600;
    } else if (deviceGB <= 16) {
      // Good desktop: assume 4GB heap, use 3.2GB cache (80%)
      estimatedHeapMB = 4096;
      recommendedCacheMB = 3200;
    } else {
      // High-end: assume 8GB heap, use 6.4GB cache (80%)
      estimatedHeapMB = 8192;
      recommendedCacheMB = 6400;
    }
    
    console.log(`[Device Memory] RAM: ${deviceGB}GB, Estimated heap: ${estimatedHeapMB}MB → Cache: ${recommendedCacheMB}MB`);

    return {
      recommendedCacheMB: recommendedCacheMB,
      confidence: 'medium',
      source: 'device',
    };
  }

  // Fallback: use conservative defaults based on platform detection
  const isMobile =
    /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.innerWidth <= 768;
  
  const recommended = isMobile ? 256 : 1024;
  console.log(`[Fallback] Platform: ${isMobile ? 'mobile' : 'desktop'} → Cache: ${recommended}MB`);

  return {
    recommendedCacheMB: recommended,
    confidence: 'low',
    source: 'default',
  };
}

/**
 * Monitor memory pressure and adjust cache recommendation
 */
export class MemoryMonitor {
  private callback?: (newSizeMB: number) => void;
  private currentSizeMB: number;
  private intervalId?: number;

  constructor(initialSizeMB: number, callback?: (newSizeMB: number) => void) {
    this.currentSizeMB = initialSizeMB;
    this.callback = callback;
  }

  start(): void {
    // Only monitor if we have the performance.memory API
    if (!(performance as any).memory) return;

    this.intervalId = window.setInterval(() => {
      const mem = (performance as any).memory;
      const usagePercent = mem.usedJSHeapSize / mem.jsHeapSizeLimit;

      // Adjust cache size based on memory pressure
      let newSize = this.currentSizeMB;

      if (usagePercent > 0.85) {
        // Critical: reduce by half
        newSize = Math.max(128, this.currentSizeMB * 0.5);
      } else if (usagePercent > 0.7) {
        // High: reduce by 25%
        newSize = Math.max(256, this.currentSizeMB * 0.75);
      }

      // Only notify if significant change
      if (Math.abs(newSize - this.currentSizeMB) > 64) {
        this.currentSizeMB = Math.round(newSize);
        if (this.callback) {
          this.callback(this.currentSizeMB);
        }
        console.log(
          `💾 [Luxar] Adjusted cache to ${this.currentSizeMB}MB (memory pressure: ${(usagePercent * 100).toFixed(0)}%)`
        );
      }
    }, 10000); // Check every 10 seconds
  }

  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = undefined;
    }
  }
}
