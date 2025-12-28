/**
 * Generic Event Queue for async event processing.
 *
 * Provides a non-blocking queue that producers can push to
 * and consumers can drain in batches. Designed for decoupling
 * event producers from consumers.
 *
 * @example
 * ```typescript
 * const queue = new EventQueue<MyEvent>();
 *
 * // Producer (non-blocking)
 * queue.push({ type: 'load', data: {...} });
 *
 * // Consumer (batch processing)
 * const events = queue.drain();
 * for (const event of events) {
 *   processEvent(event);
 * }
 * ```
 */
export class EventQueue<T> {
  private queue: T[] = [];
  private maxSize: number;

  /**
   * Create a new EventQueue.
   * @param maxSize - Maximum queue size (oldest events dropped when exceeded)
   */
  constructor(maxSize: number = 10000) {
    this.maxSize = maxSize;
  }

  /**
   * Push an event to the queue.
   * Non-blocking - returns immediately.
   * If queue is at max size, oldest event is dropped.
   */
  push(event: T): void {
    if (this.queue.length >= this.maxSize) {
      this.queue.shift(); // Drop oldest
    }
    this.queue.push(event);
  }

  /**
   * Push multiple events to the queue.
   */
  pushAll(events: T[]): void {
    for (const event of events) {
      this.push(event);
    }
  }

  /**
   * Drain all events from the queue.
   * Returns the events and clears the queue atomically.
   * This is the primary consumption method for batch processing.
   */
  drain(): T[] {
    const events = this.queue;
    this.queue = [];
    return events;
  }

  /**
   * Peek at the next event without removing it.
   */
  peek(): T | undefined {
    return this.queue[0];
  }

  /**
   * Get the current queue length.
   */
  get length(): number {
    return this.queue.length;
  }

  /**
   * Check if the queue is empty.
   */
  get isEmpty(): boolean {
    return this.queue.length === 0;
  }

  /**
   * Clear all events from the queue.
   */
  clear(): void {
    this.queue = [];
  }

  /**
   * Get the maximum queue size.
   */
  get capacity(): number {
    return this.maxSize;
  }
}
