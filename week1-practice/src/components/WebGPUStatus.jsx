import React from 'react';
import { useTensorFlow } from '../context/TensorFlowContext';

export default function WebGPUStatus() {
  const { webgpuSupported, webgpuInfo, isLoading } = useTensorFlow();

  if (isLoading) {
    return (
      <div id="webgpu-status">
        <p>检测中...</p>
      </div>
    );
  }

  if (webgpuSupported) {
    return (
      <div id="webgpu-status">
        <span className="status-badge status-success">WebGPU 已就绪</span>
        <p style={{ marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
          GPU: {webgpuInfo?.info?.description || webgpuInfo?.info?.vendor || 'Unknown'}
        </p>
      </div>
    );
  }

  return (
    <div id="webgpu-status">
      <span className="status-badge status-danger">WebGPU 不可用</span>
      <p style={{ marginTop: '0.5rem', color: 'var(--text-secondary)' }}>
        {webgpuInfo?.error || '未知错误'}<br />
        请使用 Chrome 113+ 并在 chrome://flags 中启用 "Unsafe WebGPU Support"
      </p>
    </div>
  );
}
