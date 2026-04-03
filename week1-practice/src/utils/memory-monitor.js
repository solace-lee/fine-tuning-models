import * as tf from '@tensorflow/tfjs';


export function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export function getMemoryInfo() {
  const mem = tf.memory();
  return {
    numBytes: mem.numBytes,
    numBytesFormatted: formatBytes(mem.numBytes),
    numTensors: mem.numTensors,
    numDataBuffers: mem.numDataBuffers,
    unreliable: mem.unreliable,
    gpuBytes: mem.gpuBytes ? formatBytes(mem.gpuBytes) : 'N/A',
  };
}

export function createMonitorPanel(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return null;

  container.innerHTML = `
    <div class="monitor-panel">
      <div class="monitor-row">
        <span class="monitor-label">已用内存</span>
        <span class="monitor-value" id="mem-bytes">-</span>
      </div>
      <div class="monitor-row">
        <span class="monitor-label">GPU 内存</span>
        <span class="monitor-value" id="mem-gpu">-</span>
      </div>
      <div class="monitor-row">
        <span class="monitor-label">张量数量</span>
        <span class="monitor-value" id="mem-tensors">-</span>
      </div>
      <div class="monitor-row">
        <span class="monitor-label">数据缓冲区</span>
        <span class="monitor-value" id="mem-buffers">-</span>
      </div>
    </div>
  `;

  return {
    update() {
      const info = getMemoryInfo();
      document.getElementById('mem-bytes').textContent = info.numBytesFormatted;
      document.getElementById('mem-gpu').textContent = info.gpuBytes;
      document.getElementById('mem-tensors').textContent = info.numTensors;
      document.getElementById('mem-buffers').textContent = info.numDataBuffers;
    },
  };
}

export function startAutoRefresh(monitor, interval = 1000) {
  return setInterval(() => monitor.update(), interval);
}
