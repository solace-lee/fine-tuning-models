import React from 'react';

export default function LoRAPage() {
  return (
    <div>
      <div className="card">
        <h2>LoRA 技术原理</h2>
        <p style={{ color: 'var(--text-secondary)' }}>
          LoRA（Low-Rank Adaptation）是一种高效的模型微调技术，通过低秩矩阵分解实现参数高效微调。
        </p>
      </div>

      <div className="card">
        <h3>1. 核心思想：低秩矩阵分解</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          LoRA 的核心思想是将预训练模型的权重更新分解为两个小矩阵的乘积：
        </p>
        <pre style={{ marginBottom: '1rem' }}>{`原始权重更新: ΔW = B × A
其中: B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)

假设原权重矩阵 W ∈ R^(d×k)：
- 直接微调: 需要训练 d×k 个参数
- LoRA: 只需训练 d×r + r×k 个参数

示例 (d=1024, k=1024, r=8):
- 直接微调: 1,048,576 参数
- LoRA: 8,192 + 8,192 = 16,384 参数 (98.4% 减少!)`}</pre>
      </div>

      <div className="card">
        <h3>2. LoRA 在 Transformer 中的应用</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          在 Transformer 架构中，LoRA 通常应用于 Query、Key、Value 三个投影矩阵：
        </p>
        <pre style={{ marginBottom: '1rem' }}>{`Transformer 注意力机制中的 LoRA:

原始计算: h = W × x
LoRA 计算: h = W × x + (B × A) × x

其中:
- W: 预训练权重（冻结）
- B × A: LoRA 增量权重（可训练）
- r: 秩（通常取 4, 8, 16, 32）
- x: 输入向量

可应用的层:
✓ Q (Query) 投影
✓ K (Key) 投影  
✓ V (Value) 投影
✓ 输出 (Output) 投影`}</pre>
      </div>

      <div className="card">
        <h3>3. 显存优势对比</h3>
        <table className="table">
          <thead>
            <tr>
              <th>微调方法</th>
              <th>可训练参数量</th>
              <th>预估显存 (7B模型)</th>
              <th>GTX1050 适用</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>全参数微调</td>
              <td>7,000,000,000</td>
              <td>~28GB (FP16)</td>
              <td><span className="status-badge error">不支持</span></td>
            </tr>
            <tr>
              <td>LoRA (r=8)</td>
              <td>~4,194,304</td>
              <td>~4GB</td>
              <td><span className="status-badge success">支持</span></td>
            </tr>
            <tr>
              <td>LoRA (r=16)</td>
              <td>~8,388,608</td>
              <td>~6GB</td>
              <td><span className="status-badge warning">勉强</span></td>
            </tr>
            <tr>
              <td>QLoRA (4-bit)</td>
              <td>~4,194,304</td>
              <td>~2GB</td>
              <td><span className="status-badge success">推荐</span></td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>4. LoRA 超参数</h3>
        <table className="table">
          <thead>
            <tr>
              <th>参数</th>
              <th>说明</th>
              <th>推荐值</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><code>rank (r)</code></td>
              <td>低秩矩阵的秩，决定参数量</td>
              <td>4-32，通常 8</td>
            </tr>
            <tr>
              <td><code>alpha</code></td>
              <td>缩放因子，通常设为 rank 的 2 倍</td>
              <td>16-64，通常 16</td>
            </tr>
            <tr>
              <td><code>dropout</code></td>
              <td>LoRA 层的 dropout 率</td>
              <td>0.05-0.1</td>
            </tr>
            <tr>
              <td><code>target_modules</code></td>
              <td>应用 LoRA 的模块</td>
              <td>q_proj, v_proj</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>5. JavaScript 中的 LoRA 实现</h3>
        <p style={{ color: 'var(--text-secondary)', marginBottom: '1rem' }}>
          由于 Transformers.js 尚未原生支持完整的 LoRA 训练，本周我们将使用模拟方式演示 LoRA 原理：
        </p>
        <pre style={{ marginBottom: '1rem' }}>{`// JavaScript 中 LoRA 的模拟实现

// 1. 冻结原始权重
const frozenWeights = model.getWeights();
// 冻结后不参与梯度计算

// 2. 创建可训练的 LoRA 权重
const rank = 8;
const loraA = tf.variable(tf.zeros([rank, hiddenDim]));
const loraB = tf.variable(tf.zeros([hiddenDim, rank]));

// 3. 前向传播时添加 LoRA 效应
function forwardWithLoRA(input, originalWeights) {
  const originalOutput = tf.matMul(input, originalWeights);
  const loraOutput = tf.matMul(tf.matMul(input, loraA), loraB);
  return originalOutput.add(loraOutput);
}

// 4. 只更新 LoRA 权重
optimizer.minimize(() => {
  const output = forwardWithLoRA(input, frozenWeights);
  return lossFn(output, labels);
});`}</pre>
      </div>

      <div className="card">
        <h3>6. 总结</h3>
        <ul style={{ marginLeft: '1.5rem', color: 'var(--text-secondary)' }}>
          <li>LoRA 通过低秩分解将参数量减少 90%+</li>
          <li>显存需求从 28GB 降至 2-4GB</li>
          <li>GTX1050 2GB 显存可以运行 r=8 的 LoRA</li>
          <li>可训练参数少，训练速度快，泛化好</li>
          <li>权重复用性好，可切换不同任务</li>
        </ul>
      </div>
    </div>
  );
}