/**
 * Spatial Query Visualizer Component
 *
 * Visualizes spatial index grid and query patterns in real-time.
 * Shows which cells are occupied, cached, and being queried.
 */

import type { MonitorEvent, GridCellState } from '../data-monitor-types';

export class SpatialQueryVisualizer {
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  // Grid state
  private gridShape: number[] = [];
  private gridCells: Map<string, GridCellState> = new Map();
  private lastQueryBounds: { min: number[]; max: number[] } | null = null;

  // Visualization settings
  private viewDimensions: [number, number] = [0, 1]; // Which 2D slice to show
  private cellSize = 10; // Pixels per cell
  private colors = {
    empty: 'rgba(50, 50, 50, 0.3)',
    occupied: 'rgba(100, 100, 100, 0.5)',
    cached: 'rgba(76, 175, 80, 0.7)',
    loading: 'rgba(255, 193, 7, 0.9)',
    queried: 'rgba(33, 150, 243, 0.5)',
    queryBounds: 'rgba(255, 255, 255, 0.8)',
  };

  /**
   * Handle monitor events
   */
  handleEvent(event: MonitorEvent): void {
    switch (event.type) {
      case 'query':
        this.handleQueryEvent(event);
        break;
      case 'load':
        this.handleLoadEvent(event);
        break;
      case 'cache-hit':
        this.handleCacheHitEvent(event);
        break;
    }

    this.render();
  }

  /**
   * Handle query event
   */
  private handleQueryEvent(event: MonitorEvent): void {
    if (event.data.gridBounds) {
      this.lastQueryBounds = event.data.gridBounds;
    }

    // Mark cells as queried
    if (event.data.cells) {
      // This would need actual cell coordinates from the event
      // For now, just track that a query happened
    }
  }

  /**
   * Handle load event
   */
  private handleLoadEvent(event: MonitorEvent): void {
    // Mark cells as loaded/cached
    if (event.data.ranges) {
      // Update cell states based on loaded ranges
    }
  }

  /**
   * Handle cache hit event
   */
  private handleCacheHitEvent(_event: MonitorEvent): void {
    // Update cell cache status
  }

  /**
   * Initialize canvas
   */
  initializeCanvas(canvasId: string): void {
    this.canvas = document.getElementById(canvasId) as HTMLCanvasElement;
    if (this.canvas) {
      this.ctx = this.canvas.getContext('2d');
      this.render();
    }
  }

  /**
   * Render the spatial grid
   */
  private render(): void {
    if (!this.canvas || !this.ctx) return;

    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, width, height);

    if (this.gridShape.length < 2) {
      // No grid data yet
      ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.font = '12px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No spatial index data', width / 2, height / 2);
      return;
    }

    // Calculate grid dimensions for selected view
    const gridWidth = this.gridShape[this.viewDimensions[0]] || 10;
    const gridHeight = this.gridShape[this.viewDimensions[1]] || 10;

    // Calculate cell size to fit canvas
    this.cellSize = Math.min(
      (width - 20) / gridWidth,
      (height - 20) / gridHeight,
      20 // Max cell size
    );

    // Calculate offset to center grid
    const offsetX = (width - gridWidth * this.cellSize) / 2;
    const offsetY = (height - gridHeight * this.cellSize) / 2;

    // Draw grid cells
    for (let y = 0; y < gridHeight; y++) {
      for (let x = 0; x < gridWidth; x++) {
        const cellKey = `${x},${y}`;
        const cell = this.gridCells.get(cellKey);

        const px = offsetX + x * this.cellSize;
        const py = offsetY + y * this.cellSize;

        // Determine cell color
        let color = this.colors.empty;
        if (cell) {
          if (cell.isLoading) {
            color = this.colors.loading;
          } else if (cell.isCached) {
            color = this.colors.cached;
          } else if (cell.isOccupied) {
            color = this.colors.occupied;
          }
        }

        // Draw cell
        ctx.fillStyle = color;
        ctx.fillRect(px, py, this.cellSize - 1, this.cellSize - 1);

        // Draw point count if significant
        if (cell && cell.points > 0 && this.cellSize > 15) {
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.font = '9px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(
            this.formatCount(cell.points),
            px + this.cellSize / 2,
            py + this.cellSize / 2 + 3
          );
        }
      }
    }

    // Draw query bounds if available
    if (this.lastQueryBounds) {
      ctx.strokeStyle = this.colors.queryBounds;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);

      const minX = this.lastQueryBounds.min[this.viewDimensions[0]] || 0;
      const minY = this.lastQueryBounds.min[this.viewDimensions[1]] || 0;
      const maxX = this.lastQueryBounds.max[this.viewDimensions[0]] || gridWidth;
      const maxY = this.lastQueryBounds.max[this.viewDimensions[1]] || gridHeight;

      const x = offsetX + minX * this.cellSize;
      const y = offsetY + minY * this.cellSize;
      const w = (maxX - minX) * this.cellSize;
      const h = (maxY - minY) * this.cellSize;

      ctx.strokeRect(x, y, w, h);
      ctx.setLineDash([]);
    }

    // Draw labels
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Dimensions: ${this.viewDimensions[0]} × ${this.viewDimensions[1]}`, 5, 15);
    ctx.fillText(`Grid: ${gridWidth} × ${gridHeight}`, 5, 30);
  }

  /**
   * Render mini grid for compact view
   */
  renderMiniGrid(): string {
    // Simple HTML-based mini visualization
    // const gridWidth = this.gridShape[0] || 10;
    // const gridHeight = this.gridShape[1] || 10;

    let cached = 0;
    let occupied = 0;

    for (const cell of this.gridCells.values()) {
      if (cell.isCached) cached++;
      if (cell.isOccupied) occupied++;
    }

    const cachePercent = occupied > 0 ? (cached / occupied) * 100 : 0;

    return `
      <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        font-size: 10px;
        color: rgba(255, 255, 255, 0.7);
      ">
        <span style="color: #4CAF50;">${cached}</span>
        <span style="opacity: 0.5;">/</span>
        <span>${occupied}</span>
        <span style="opacity: 0.5; margin-left: 8px;">cells</span>
        <span style="color: #2196F3; margin-left: 8px;">${cachePercent.toFixed(0)}%</span>
      </div>
    `;
  }

  /**
   * Update grid metadata
   */
  updateGridMetadata(shape: number[], _origin: number[], _cellSize: number[]): void {
    this.gridShape = shape;
    // Could also store origin and cellSize for more accurate visualization
    this.render();
  }

  /**
   * Update cell state
   */
  updateCell(x: number, y: number, state: Partial<GridCellState>): void {
    const key = `${x},${y}`;
    const existing = this.gridCells.get(key) || {
      x,
      y,
      isOccupied: false,
      isCached: false,
      isLoading: false,
      isQueried: false,
      points: 0,
    };

    this.gridCells.set(key, { ...existing, ...state });
    this.render();
  }

  /**
   * Set view dimensions
   */
  setViewDimensions(dim1: number, dim2: number): void {
    this.viewDimensions = [dim1, dim2];
    this.render();
  }

  /**
   * Format point count
   */
  private formatCount(n: number): string {
    if (n >= 1000000) return (n / 1000000).toFixed(0) + 'M';
    if (n >= 1000) return (n / 1000).toFixed(0) + 'K';
    return n.toString();
  }

  /**
   * Clear visualization
   */
  clear(): void {
    this.gridCells.clear();
    this.lastQueryBounds = null;
    this.render();
  }

  /**
   * Dispose
   */
  dispose(): void {
    this.canvas = null;
    this.ctx = null;
    this.gridCells.clear();
  }
}
