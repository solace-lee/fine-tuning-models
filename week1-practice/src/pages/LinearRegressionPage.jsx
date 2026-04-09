import React, { useState, useRef, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';
import MemoryMonitor from '../components/MemoryMonitor';
import { useTensorFlow } from '../context/TensorFlowContext';

export default function LinearRegressionPage() {
  const { updateMemory } = useTensorFlow();
  const [epochs, setEpochs] = useState(250);
  const [learningRate, setLearningRate] = useState(0.1);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('等待开始...');
  const [losses, setLosses] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [predictInput, setPredictInput] = useState(10);
  const trainedModelRef = useRef(null);
  const canvasRef = useRef(null);

  const drawLossChart = useCallback((lossData) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    if (lossData.length === 0) return;

    const maxLoss = Math.max(...lossData);
    const minLoss = Math.min(...lossData);
    const range = maxLoss - minLoss || 1;

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2;
    ctx.beginPath();

    lossData.forEach((loss, i) => {
      const x = (i / (lossData.length - 1)) * width;
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
    ctx.fillText(`Epochs: ${lossData.length}`, width - 100, 20);
  }, []);

  const trainModel = async () => {
    setIsTraining(true);
    setStatus('正在训练...');
    setLosses([]);
    setProgress(0);

    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

    model.compile({
      optimizer: tf.train.sgd(learningRate),
      loss: 'meanSquaredError',
    });

    const newLosses = [];

    await model.fit(xs, ys, {
      epochs,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          newLosses.push(logs.loss);
          setLosses([...newLosses]);
          setProgress(((epoch + 1) / epochs) * 100);
          setStatus(`Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(6)}`);
          drawLossChart(newLosses);
          updateMemory();
        },
      },
    });

    trainedModelRef.current = model;
    setStatus(`训练完成! 最终 Loss: ${newLosses[newLosses.length - 1].toFixed(6)}`);
    setIsTraining(false);

    xs.dispose();
    ys.dispose();
  };

  const handlePredict = () => {
    if (!trainedModelRef.current) return;

    const inputTensor = tf.tensor2d([predictInput], [1, 1]);
    const result = trainedModelRef.current.predict(inputTensor);
    const predictedValue = result.dataSync()[0];

    const expected = 2 * predictInput - 1;
    setPrediction({
      input: predictInput,
      predicted: predictedValue,
      expected,
      error: Math.abs(predictedValue - expected),
    });

    inputTensor.dispose();
    result.dispose();
    updateMemory();
  };

  return (
    <div>
      <div className="grid">
        <div className="card">
          <h2>训练配置</h2>
          <div className="input-group">
            <label>训练轮数 (Epochs)</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value) || 0)}
              min={10}
              max={1000}
              disabled={isTraining}
            />
          </div>
          <div className="input-group">
            <label>学习率</label>
            <input
              type="number"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value) || 0)}
              step={0.01}
              min={0.001}
              max={1}
              disabled={isTraining}
            />
          </div>
          <button className="btn btn-primary" onClick={trainModel} disabled={isTraining}>
            {isTraining ? '训练中...' : '开始训练'}
          </button>
        </div>

        <div className="card">
          <h2>显存监控</h2>
          <MemoryMonitor autoRefresh={isTraining} interval={500} />
        </div>
      </div>

      <div className="card">
        <h2>训练过程</h2>
        <div className="progress-bar">
          <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
        </div>
        <p>{status}</p>
      </div>

      <div className="card">
        <h2>损失曲线</h2>
        <canvas ref={canvasRef} width={800} height={300}></canvas>
      </div>

      <div className="card">
        <h2>预测测试</h2>
        <div className="input-group">
          <label>输入 x 值</label>
          <input
            type="number"
            value={predictInput}
            onChange={(e) => setPredictInput(parseFloat(e.target.value) || 0)}
            placeholder="输入任意数字"
          />
        </div>
        <button
          className="btn btn-success"
          onClick={handlePredict}
          disabled={!trainedModelRef.current}
        >
          预测
        </button>
        {prediction && (
          <div className="result-box">
            {`输入: x = ${prediction.input}\n`}
            {`预测值: y = ${prediction.predicted.toFixed(4)}\n`}
            {`理论值: y = ${prediction.expected} (公式: y = 2x - 1)\n`}
            {`误差: ${prediction.error.toFixed(4)}`}
          </div>
        )}
      </div>
    </div>
  );
}
