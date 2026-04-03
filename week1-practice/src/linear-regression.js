import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import { createMonitorPanel } from './utils/memory-monitor.js';

let trainedModel = null;

async function initBackend() {
  if (navigator.gpu) {
    await tf.setBackend('webgpu');
  } else {
    await tf.setBackend('webgl');
  }
  await tf.ready();
}

function drawLossChart(losses) {
  const canvas = document.getElementById('loss-chart');
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  if (losses.length === 0) return;

  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const range = maxLoss - minLoss || 1;

  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.beginPath();

  losses.forEach((loss, i) => {
    const x = (i / (losses.length - 1)) * width;
    const y = height - ((loss - minLoss) / range) * (height - 40) - 20;

    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });

  ctx.stroke();

  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px monospace';
  ctx.fillText(`Max: ${maxLoss.toFixed(4)}`, 10, 20);
  ctx.fillText(`Min: ${minLoss.toFixed(4)}`, 10, 35);
  ctx.fillText(`Epochs: ${losses.length}`, width - 100, 20);
}

async function trainModel() {
  const epochs = parseInt(document.getElementById('epochs').value);
  const learningRate = parseFloat(document.getElementById('learning-rate').value);
  const trainBtn = document.getElementById('train-btn');
  const statusEl = document.getElementById('training-status');
  const progressFill = document.getElementById('progress-fill');
  const predictBtn = document.getElementById('predict-btn');

  trainBtn.disabled = true;
  statusEl.textContent = '正在训练...';

  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(learningRate),
    loss: 'meanSquaredError',
  });

  const losses = [];

  await model.fit(xs, ys, {
    epochs,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        losses.push(logs.loss);
        const progress = ((epoch + 1) / epochs) * 100;
        progressFill.style.width = `${progress}%`;
        statusEl.textContent = `Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(6)}`;
        drawLossChart(losses);
      },
    },
  });

  trainedModel = model;
  statusEl.textContent = `训练完成! 最终 Loss: ${losses[losses.length - 1].toFixed(6)}`;
  predictBtn.disabled = false;
  trainBtn.disabled = false;

  xs.dispose();
  ys.dispose();
}

function predict() {
  if (!trainedModel) return;

  const inputVal = parseFloat(document.getElementById('predict-input').value);
  const inputTensor = tf.tensor2d([inputVal], [1, 1]);
  const prediction = trainedModel.predict(inputTensor);
  const result = prediction.dataSync()[0];

  const expected = 2 * inputVal - 1;
  const resultBox = document.getElementById('predict-result');
  resultBox.style.display = 'block';
  resultBox.textContent = `输入: x = ${inputVal}\n预测值: y = ${result.toFixed(4)}\n理论值: y = ${expected} (公式: y = 2x - 1)\n误差: ${Math.abs(result - expected).toFixed(4)}`;

  inputTensor.dispose();
  prediction.dispose();
}

async function init() {
  await initBackend();
  const monitor = createMonitorPanel('memory-monitor');
  if (monitor) {
    monitor.update();
    setInterval(() => monitor.update(), 1000);
  }

  document.getElementById('train-btn').addEventListener('click', trainModel);
  document.getElementById('predict-btn').addEventListener('click', predict);
}

init();
