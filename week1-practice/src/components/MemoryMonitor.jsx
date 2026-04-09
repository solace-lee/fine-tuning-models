import React, { useEffect, useRef } from 'react';
import { useTensorFlow } from '../context/TensorFlowContext';

export default function MemoryMonitor({ autoRefresh = true, interval = 1000 }) {
  const { memory, updateMemory } = useTensorFlow();
  const intervalRef = useRef(null);

  useEffect(() => {
    updateMemory();

    if (autoRefresh) {
      intervalRef.current = setInterval(updateMemory, interval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [autoRefresh, interval, updateMemory]);

  if (!memory) {
    return (
      <div className="monitor-panel">
        <p>加载中...</p>
      </div>
    );
  }

  return (
    <div className="monitor-panel">
      <div className="monitor-row">
        <span className="monitor-label">已用内存</span>
        <span className="monitor-value">{memory.numBytesFormatted}</span>
      </div>
      <div className="monitor-row">
        <span className="monitor-label">GPU 内存</span>
        <span className="monitor-value">{memory.gpuBytes}</span>
      </div>
      <div className="monitor-row">
        <span className="monitor-label">张量数量</span>
        <span className="monitor-value">{memory.numTensors}</span>
      </div>
      <div className="monitor-row">
        <span className="monitor-label">数据缓冲区</span>
        <span className="monitor-value">{memory.numDataBuffers}</span>
      </div>
    </div>
  );
}
