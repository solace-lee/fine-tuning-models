import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { pipeline } from '@huggingface/transformers';
import { createMonitorPanel } from './utils/memory-monitor.js';

let classifier = null;

async function initBackend() {
  if (navigator.gpu) {
    await tf.setBackend('webgpu');
  } else {
    await tf.setBackend('webgl');
  }
  await tf.ready();
}

async function loadModel() {
  const loadBtn = document.getElementById('load-model');
  const statusEl = document.getElementById('load-status');
  const analyzeBtn = document.getElementById('analyze-btn');

  loadBtn.disabled = true;
  statusEl.innerHTML = '<span style="color: var(--warning);">模型加载中... (首次加载需要下载约 250MB)</span>';

  try {
    const startTime = performance.now();

    classifier = await pipeline(
      'text-classification',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english',
      { dtype: 'q8' }
    );

    const loadTime = ((performance.now() - startTime) / 1000).toFixed(2);

    statusEl.innerHTML = `<span style="color: var(--success);">模型加载完成! 耗时: ${loadTime}s</span>`;
    analyzeBtn.disabled = false;
  } catch (err) {
    statusEl.innerHTML = `<span style="color: var(--danger);">加载失败: ${err.message}</span>`;
    loadBtn.disabled = false;
  }
}

async function analyzeText() {
  if (!classifier) return;

  const text = document.getElementById('input-text').value.trim();
  if (!text) return;

  const resultBox = document.getElementById('analysis-result');
  resultBox.style.display = 'block';
  resultBox.textContent = '分析中...';

  try {
    const startTime = performance.now();
    const result = await classifier(text);
    const inferenceTime = ((performance.now() - startTime) / 1000).toFixed(3);

    const label = result[0].label;
    const score = (result[0].score * 100).toFixed(2);

    const emoji = label === 'POSITIVE' ? '😊' : '😞';
    const labelCN = label === 'POSITIVE' ? '正面' : '负面';

    resultBox.textContent = `${emoji} 情感: ${labelCN} (${label})\n置信度: ${score}%\n推理耗时: ${inferenceTime}s`;
  } catch (err) {
    resultBox.textContent = `分析失败: ${err.message}`;
  }
}

async function init() {
  await initBackend();

  const monitor = createMonitorPanel('memory-monitor');
  if (monitor) {
    monitor.update();
    setInterval(() => monitor.update(), 1000);
  }

  document.getElementById('load-model').addEventListener('click', loadModel);
  document.getElementById('analyze-btn').addEventListener('click', analyzeText);

  document.querySelectorAll('.test-case').forEach(btn => {
    btn.addEventListener('click', () => {
      document.getElementById('input-text').value = btn.dataset.text;
    });
  });
}

init();
