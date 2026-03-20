import appConfig from './prism.json';
import rawLayers from './layers.json';

// eslint-disable-next-line import/no-unresolved
import boundary from './admin_boundaries.json';

const defined = { layers: rawLayers, ...appConfig };

export default { ...defined, defaultBoundary: boundary };
