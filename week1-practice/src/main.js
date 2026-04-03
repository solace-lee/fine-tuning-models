import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { checkWebGPUSupport } from './utils/webgpu-check.js';
import { createMonitorPanel, startAutoRefresh } from './utils/memory-monitor.js';

async function init() {
  const webgpuResult = await checkWebGPUSupport();
  const statusEl = document.getElementById('webgpu-status');

  if (webgpuResult.supported) {
    statusEl.innerHTML = `
      <div class="status-badge status-success">WebGPU 已就绪</div>
      <p style="margin-top: 0.5rem; color: var(--text-secondary);">
        GPU: ${webgpuResult.info.description || webgpuResult.info.vendor}
      </p>
    `;

    await tf.setBackend('webgpu');
    await tf.ready();
  } else {
    statusEl.innerHTML = `
      <div class="status-badge status-danger">WebGPU 不可用</div>
      <p style="margin-top: 0.5rem; color: var(--text-secondary);">
        ${webgpuResult.error}<br>
        请使用 Chrome 113+ 并在 chrome://flags 中启用 "Unsafe WebGPU Support"
      </p>
    `;

    await tf.setBackend('webgl');
    await tf.ready();
  }

  const monitor = createMonitorPanel('memory-monitor');
  if (monitor) {
    monitor.update();
    startAutoRefresh(monitor, 1000);
  }

  const backendSelect = document.getElementById('backend-select');
  const switchBtn = document.getElementById('switch-backend');
  const currentBackendEl = document.getElementById('current-backend');

  currentBackendEl.textContent = `当前后端: ${tf.getBackend()}`;

  switchBtn.addEventListener('click', async () => {
    const backend = backendSelect.value;
    try {
      await tf.setBackend(backend);
      await tf.ready();
      currentBackendEl.textContent = `当前后端: ${tf.getBackend()}`;
      monitor.update();
    } catch (err) {
      currentBackendEl.textContent = `切换失败: ${err.message}`;
    }
  });
}

init();
