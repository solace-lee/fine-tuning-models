export async function checkWebGPUSupport() {
  const result = {
    supported: false,
    adapter: null,
    device: null,
    info: {},
    error: null,
  };

  if (!navigator.gpu) {
    result.error = 'WebGPU API 不可用。请使用 Chrome 113+ 并启用 WebGPU';
    return result;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      result.error = '未找到可用的 GPU 适配器';
      return result;
    }

    result.adapter = adapter;
    result.supported = true;

    const info = adapter.info || {};
    result.info = {
      vendor: info.vendor || 'Unknown',
      architecture: info.architecture || 'Unknown',
      device: info.device || 'Unknown',
      description: info.description || 'Unknown GPU',
    };

    const device = await adapter.requestDevice();
    result.device = device;

    return result;
  } catch (err) {
    result.error = `WebGPU 初始化失败: ${err.message}`;
    return result;
  }
}

export function getBackendPriority() {
  return [
    { name: 'webgpu', label: 'WebGPU (推荐)', icon: '🚀' },
    { name: 'webgl', label: 'WebGL', icon: '🎮' },
    { name: 'wasm', label: 'WebAssembly', icon: '⚙️' },
    { name: 'cpu', label: 'CPU', icon: '🖥️' },
  ];
}
