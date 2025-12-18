/**
 * Custom GUI Library
 * Drop-in replacement for lil-gui with Luxar theme integration
 */

// Main exports
export { GUI } from './core/gui';
export { Controller } from './core/controller';
export { Folder } from './core/folder';

// Types
export type {
  GUIOptions,
  ControllerOptions,
  ChangeCallback,
  FinishChangeCallback,
} from './core/types';
export { ControllerType } from './core/types';

// Specialized controllers (for direct use if needed)
export { NumberController } from './controllers/number-controller';
export { BooleanController } from './controllers/boolean-controller';
export { StringController } from './controllers/string-controller';
export { OptionController } from './controllers/option-controller';
export { FunctionController } from './controllers/function-controller';

// Default export matches lil-gui behavior
export { GUI as default } from './core/gui';
