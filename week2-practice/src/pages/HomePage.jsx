import React from 'react';
import { Link } from 'react-router-dom';

export default function HomePage() {
  return (
    <div>
      <div className="card">
        <h2>Week2 学习目标</h2>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          理解 LoRA（Low-Rank Adaptation）原理，完成第一个浏览器端微调实验。
        </p>
        <ul style={{ marginLeft: '1.5rem', color: 'var(--text-secondary)' }}>
          <li>掌握低秩矩阵分解的数学原理</li>
          <li>理解 LoRA vs 全参数微调的显存对比</li>
          <li>完成第一个文本分类微调实验</li>
          <li>实现训练过程可视化（损失曲线）</li>
        </ul>
      </div>

      <div className="grid">
        <div className="card">
          <h3>LoRA 原理</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            深入理解低秩矩阵分解的数学原理，为什么 LoRA 能在 2G 显存下训练。
          </p>
          <Link to="/lora-theory" className="btn">开始学习</Link>
        </div>

        <div className="card">
          <h3>微调实验</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
            准备 500 条文本分类样本，配置 LoRA 参数，运行浏览器端训练。
          </p>
          <Link to="/fine-tuning" className="btn">开始实验</Link>
        </div>
      </div>

      <div className="card">
        <h3>验收标准</h3>
        <table className="table">
          <thead>
            <tr>
              <th>指标</th>
              <th>目标值</th>
              <th>说明</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>显存占用</td>
              <td>&lt; 1.8G</td>
              <td>LoRA 训练时 GTX1050 显存使用</td>
            </tr>
            <tr>
              <td>准确率提升</td>
              <td>&gt; 20%</td>
              <td>微调后 vs 预训练模型</td>
            </tr>
            <tr>
              <td>权重文件大小</td>
              <td>&lt; 5MB</td>
              <td>保存的 LoRA 权重</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}