import React from 'react';
import { Link } from 'react-router-dom';
import WebGPUStatus from '../components/WebGPUStatus';
import BackendSelector from '../components/BackendSelector';
import MemoryMonitor from '../components/MemoryMonitor';

export default function HomePage() {
  return (
    <div>
      <div className="grid">
        <div className="card">
          <h2>WebGPU 支持检测</h2>
          <WebGPUStatus />
        </div>

        <div className="card">
          <h2>后端切换</h2>
          <BackendSelector />
        </div>
      </div>

      <div className="card">
        <h2>显存监控面板</h2>
        <MemoryMonitor />
      </div>

      <div className="card">
        <h2>实战 Demo 导航</h2>
        <div className="nav-links">
          <Link to="/linear-regression">线性回归 Hello World</Link>
          <Link to="/transformers-demo">DistilBERT 文本分类</Link>
        </div>
      </div>
    </div>
  );
}
