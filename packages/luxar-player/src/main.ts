// Luxar - A web viewer for arbitrarily large 3D scenes
// Copyright (c) 2024 The Luxar Authors

import { LuxarApp } from "./app";
import { CONFIG } from "./config";
import { showError } from "./ui";

// Parse URL parameters for scene source
const params = new URLSearchParams(window.location.search);
const src = params.get("src") ?? CONFIG.DEFAULT_ZARR_PATH;

// Initialize and start the application
const app = new LuxarApp();

app.init(src).catch(error => {
  console.error('Failed to start Luxar application:', error);
  
  // Show error to user if it wasn't already handled by lower-level error handlers
  // This ensures any initialization errors that don't get displayed are still shown
  showError('Failed to start the application. Please check the console for details.');
});
