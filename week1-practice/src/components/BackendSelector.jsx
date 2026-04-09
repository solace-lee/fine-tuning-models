import React, { useState } from 'react';
import { useTensorFlow } from '../context/TensorFlowContext';

export default function BackendSelector() {
  const { backend, setBackend } = useTensorFlow();
  const [selectedBackend, setSelectedBackend] = useState(backend || 'webgpu');
  const [message, setMessage] = useState('');

  const handleSwitch = async () => {
    const success = await setBackend(selectedBackend);
    if (success) {
      setMessage(`当前后端: ${selectedBackend}`);
    } else {
      setMessage(`切换失败`);
    }
  };

  return (
    <div id="backend-selector">
      <div className="input-group">
        <select
          id="backend-select"
          value={selectedBackend}
          onChange={(e) => setSelectedBackend(e.target.value)}
        >
          <option value="webgpu">WebGPU (推荐)</option>
          <option value="webgl">WebGL</option>
          <option value="wasm">WebAssembly</option>
          <option value="cpu">CPU</option>
        </select>
      </div>
      <button id="switch-backend" className="btn btn-primary" onClick={handleSwitch}>
        切换后端
      </button>
      <p id="current-backend" style={{ marginTop: '0.5rem' }}>
        {message || `当前后端: ${backend}`}
      </p>
    </div>
  );
}
