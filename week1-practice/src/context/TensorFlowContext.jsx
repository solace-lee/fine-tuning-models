import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

const TensorFlowContext = createContext(null);

export function TensorFlowProvider({ children }) {
  const [backend, setBackendState] = useState(null);
  const [webgpuSupported, setWebgpuSupported] = useState(false);
  const [webgpuInfo, setWebgpuInfo] = useState(null);
  const [memory, setMemory] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const updateMemory = useCallback(async () => {
    // await tf.setBackend(backend);
    await tf.ready();
    const mem = tf.memory();
    setMemory({
      numBytes: mem.numBytes,
      numBytesFormatted: formatBytes(mem.numBytes),
      numTensors: mem.numTensors,
      numDataBuffers: mem.numDataBuffers,
      unreliable: mem.unreliable,
      gpuBytes: mem.gpuBytes ? formatBytes(mem.gpuBytes) : 'N/A',
    });
  }, []);

  const checkWebGPU = useCallback(async () => {
    if (!navigator.gpu) {
      return { supported: false, error: 'WebGPU API 不可用。请使用 Chrome 113+ 并启用 WebGPU' };
    }

    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        return { supported: false, error: '未找到可用的 GPU 适配器' };
      }

      const info = adapter.info || {};
      const gpuInfo = {
        vendor: info.vendor || 'Unknown',
        architecture: info.architecture || 'Unknown',
        device: info.device || 'Unknown',
        description: info.description || 'Unknown GPU',
      };

      const device = await adapter.requestDevice();
      return { supported: true, info: gpuInfo, device };
    } catch (err) {
      return { supported: false, error: `WebGPU 初始化失败: ${err.message}` };
    }
  }, []);

  const setBackend = useCallback(async (backendName) => {
    try {
      await tf.setBackend(backendName);
      await tf.ready();
      setBackendState(tf.getBackend());
      updateMemory();
      return true;
    } catch (err) {
      setError(err.message);
      return false;
    }
  }, [updateMemory]);

  useEffect(() => {
    const init = async () => {
      setIsLoading(true);
      try {
        const webgpuResult = await checkWebGPU();
        setWebgpuSupported(webgpuResult.supported);
        setWebgpuInfo(webgpuResult);

        if (webgpuResult.supported) {
          await setBackend('webgpu');
        } else {
          await setBackend('webgl');
        }
      } catch (err) {
        setError(err.message);
        await setBackend('cpu');
      }
      setIsLoading(false);
    };

    init();
  }, [checkWebGPU, setBackend]);

  const value = {
    backend,
    webgpuSupported,
    webgpuInfo,
    memory,
    isLoading,
    error,
    setBackend,
    updateMemory,
  };

  return (
    <TensorFlowContext.Provider value={value}>
      {children}
    </TensorFlowContext.Provider>
  );
}

export function useTensorFlow() {
  const context = useContext(TensorFlowContext);
  if (!context) {
    throw new Error('useTensorFlow must be used within a TensorFlowProvider');
  }
  return context;
}
