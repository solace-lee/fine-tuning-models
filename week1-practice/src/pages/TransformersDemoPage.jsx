import React, { useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { pipeline } from '@huggingface/transformers';
import MemoryMonitor from '../components/MemoryMonitor';

let classifier = null;
export default function TransformersDemoPage() {
  // const [classifier, setClassifier] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadStatus, setLoadStatus] = useState('');
  const [inputText, setInputText] = useState('I absolutely love this product! It works perfectly and exceeded my expectations.');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);

  const loadModel = async () => {
    setIsLoading(true);
    setLoadStatus('模型加载中... (首次加载需要下载约 250MB)');

    try {
      const startTime = performance.now();
      const model = await pipeline(
        'text-classification',
        'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
        { dtype: 'q8' }
      );
      const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);

      console.log(`模型加载时间: ${loadTime}s`, model);
      
      classifier = model;
      // setClassifier(model);
      setLoadStatus(`模型加载完成! 耗时: ${loadTime}s`);
    } catch (err) {
      setLoadStatus(`加载失败: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const analyzeText = async () => {
    if (!classifier) return;

    setIsAnalyzing(true);
    setResult({ status: '分析中...' });

    try {
      const startTime = performance.now();
      const output = await classifier(inputText);
      const inferenceTime = ((performance.now() - startTime) / 1000).toFixed(3);

      const label = output[0].label;
      const score = (output[0].score * 100).toFixed(2);

      const emoji = label === 'POSITIVE' ? '😊' : '😞';
      const labelCN = label === 'POSITIVE' ? '正面' : '负面';

      setResult({
        status: 'completed',
        label,
        labelCN,
        score,
        emoji,
        inferenceTime,
      });
    } catch (err) {
      setResult({ status: 'error', message: err.message });
    } finally {
      setIsAnalyzing(false);
    }
  };

  const testCases = [
    { text: 'This is the best day of my life!', label: '正面示例' },
    { text: 'I hate this terrible experience.', label: '负面示例' },
    { text: 'The weather is okay today.', label: '中性示例' },
  ];

  const handleTestCase = useCallback((text) => {
    setInputText(text);
  }, []);

  return (
    <div>
      <div className="grid">
        <div className="card">
          <h2>模型信息</h2>
          <div>
            <p>模型: distilbert-base-uncased-finetuned-sst-2-english</p>
            <p>参数量: ~66M</p>
            <p>任务: 情感分析 (正面/负面)</p>
          </div>
          <button
            id="load-model"
            className="btn btn-primary"
            onClick={loadModel}
            disabled={isLoading || classifier}
          >
            {classifier ? '已加载' : isLoading ? '加载中...' : '加载模型'}
          </button>
          <div style={{ marginTop: '0.5rem' }}>{loadStatus}</div>
        </div>

        <div className="card">
          <h2>显存监控</h2>
          <MemoryMonitor />
        </div>
      </div>

      <div className="card">
        <h2>情感分析</h2>
        <div className="input-group">
          <label>输入英文文本</label>
          <textarea
            id="input-text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter text to analyze..."
          />
        </div>
        <button
          id="analyze-btn"
          className="btn btn-success"
          onClick={analyzeText}
          disabled={!classifier || isAnalyzing}
        >
          {isAnalyzing ? '分析中...' : '分析情感'}
        </button>

        {result && result.status === 'completed' && (
          <div className="result-box">
            {`${result.emoji} 情感: ${result.labelCN} (${result.label})\n置信度: ${result.score}%\n推理耗时: ${result.inferenceTime}s`}
          </div>
        )}
        {result && result.status === 'error' && (
          <div className="result-box" style={{ color: 'var(--danger)' }}>
            分析失败: {result.message}
          </div>
        )}
      </div>

      <div className="card">
        <h2>预设测试用例</h2>
        <div className="nav-links">
          {testCases.map((tc, index) => (
            <button
              key={index}
              className="nav-link-btn"
              onClick={() => handleTestCase(tc.text)}
            >
              {tc.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
