/**
 * Unit tests for EventQueue
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { EventQueue } from '../../../ui/data-loading-monitor/event-queue';

interface TestEvent {
  type: string;
  data: number;
}

describe('EventQueue', () => {
  let queue: EventQueue<TestEvent>;

  beforeEach(() => {
    queue = new EventQueue<TestEvent>();
  });

  describe('basic operations', () => {
    it('should start empty', () => {
      expect(queue.isEmpty).toBe(true);
      expect(queue.length).toBe(0);
    });

    it('should push events', () => {
      queue.push({ type: 'test', data: 1 });
      expect(queue.length).toBe(1);
      expect(queue.isEmpty).toBe(false);
    });

    it('should push multiple events', () => {
      queue.push({ type: 'a', data: 1 });
      queue.push({ type: 'b', data: 2 });
      queue.push({ type: 'c', data: 3 });
      expect(queue.length).toBe(3);
    });

    it('should pushAll events', () => {
      const events = [
        { type: 'a', data: 1 },
        { type: 'b', data: 2 },
        { type: 'c', data: 3 },
      ];
      queue.pushAll(events);
      expect(queue.length).toBe(3);
    });
  });

  describe('drain', () => {
    it('should drain all events and clear the queue', () => {
      queue.push({ type: 'a', data: 1 });
      queue.push({ type: 'b', data: 2 });

      const events = queue.drain();

      expect(events).toHaveLength(2);
      expect(events[0]).toEqual({ type: 'a', data: 1 });
      expect(events[1]).toEqual({ type: 'b', data: 2 });
      expect(queue.isEmpty).toBe(true);
      expect(queue.length).toBe(0);
    });

    it('should return empty array when queue is empty', () => {
      const events = queue.drain();
      expect(events).toEqual([]);
    });

    it('should allow multiple drains', () => {
      queue.push({ type: 'a', data: 1 });
      const first = queue.drain();

      queue.push({ type: 'b', data: 2 });
      const second = queue.drain();

      expect(first).toHaveLength(1);
      expect(second).toHaveLength(1);
      expect(first[0].type).toBe('a');
      expect(second[0].type).toBe('b');
    });
  });

  describe('peek', () => {
    it('should peek at first event without removing', () => {
      queue.push({ type: 'a', data: 1 });
      queue.push({ type: 'b', data: 2 });

      const peeked = queue.peek();

      expect(peeked).toEqual({ type: 'a', data: 1 });
      expect(queue.length).toBe(2); // Not removed
    });

    it('should return undefined for empty queue', () => {
      expect(queue.peek()).toBeUndefined();
    });
  });

  describe('clear', () => {
    it('should clear all events', () => {
      queue.push({ type: 'a', data: 1 });
      queue.push({ type: 'b', data: 2 });

      queue.clear();

      expect(queue.isEmpty).toBe(true);
      expect(queue.length).toBe(0);
    });

    it('should be safe to clear empty queue', () => {
      queue.clear();
      expect(queue.isEmpty).toBe(true);
    });
  });

  describe('capacity and overflow', () => {
    it('should have default capacity', () => {
      expect(queue.capacity).toBe(10000);
    });

    it('should respect custom capacity', () => {
      const smallQueue = new EventQueue<TestEvent>(5);
      expect(smallQueue.capacity).toBe(5);
    });

    it('should drop oldest events when at capacity', () => {
      const smallQueue = new EventQueue<TestEvent>(3);

      smallQueue.push({ type: 'a', data: 1 });
      smallQueue.push({ type: 'b', data: 2 });
      smallQueue.push({ type: 'c', data: 3 });
      smallQueue.push({ type: 'd', data: 4 }); // Should drop 'a'

      expect(smallQueue.length).toBe(3);

      const events = smallQueue.drain();
      expect(events[0].type).toBe('b'); // 'a' was dropped
      expect(events[1].type).toBe('c');
      expect(events[2].type).toBe('d');
    });

    it('should handle rapid overflow correctly', () => {
      const smallQueue = new EventQueue<TestEvent>(2);

      // Push 10 events into a queue with capacity 2
      for (let i = 0; i < 10; i++) {
        smallQueue.push({ type: `event-${i}`, data: i });
      }

      expect(smallQueue.length).toBe(2);

      const events = smallQueue.drain();
      expect(events[0].type).toBe('event-8');
      expect(events[1].type).toBe('event-9');
    });
  });

  describe('concurrent usage simulation', () => {
    it('should handle interleaved push and drain', () => {
      // Simulate producer pushing while consumer drains
      queue.push({ type: 'a', data: 1 });
      queue.push({ type: 'b', data: 2 });

      const batch1 = queue.drain();

      queue.push({ type: 'c', data: 3 });

      const batch2 = queue.drain();

      expect(batch1).toHaveLength(2);
      expect(batch2).toHaveLength(1);
      expect(batch2[0].type).toBe('c');
    });
  });
});
