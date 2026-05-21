/**
 * Custom GUI Library — public entrypoint
 * Drop-in replacement for lil-gui with Luxar theme integration
 */

export { GUI } from './gui/gui';
export { Controller } from './gui/controller';
export { Folder } from './gui/folder';

export type {
  GUIOptions,
  ControllerOptions,
  ChangeCallback,
  FinishChangeCallback,
} from './gui/types';
export { ControllerType } from './gui/types';

export { NumberController } from './gui/controllers/number-controller';
export { BooleanController } from './gui/controllers/boolean-controller';
export { StringController } from './gui/controllers/string-controller';
export { OptionController } from './gui/controllers/option-controller';
export { FunctionController } from './gui/controllers/function-controller';

export { GUI as default } from './gui/gui';
