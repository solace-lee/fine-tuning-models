# AGENTS.md

## Repository Overview

This is a **learning/study repository** for JavaScript-based model fine-tuning (TensorFlow.js, Transformers.js). The main code lives in `week1-practice/`.

**Target hardware**: GTX1050 2GB VRAM - this is a hard constraint. Models >1B params or unquantized 7B+ models will not fit.

---

## Development Commands

```bash
cd week1-practice

# Install dependencies
npm install

# Start dev server (port 3000, auto-opens browser)
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

**Important**: Always run from `week1-practice/` directory.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/App.jsx` | Main app with routing |
| `src/pages/HomePage.jsx` | WebGPU detection, backend switching |
| `src/pages/LinearRegressionPage.jsx` | TF.js hello world - trains model to learn y = 2x - 1 |
| `src/pages/TransformersDemoPage.jsx` | Transformers.js demo - DistilBERT sentiment analysis |
| `src/context/TensorFlowContext.jsx` | TensorFlow backend and memory management |
| `src/components/MemoryMonitor.jsx` | GPU memory monitoring using `tf.memory()` |
| `src/components/BackendSelector.jsx` | Backend switching UI |
| `src/components/WebGPUStatus.jsx` | WebGPU capability detection |
| `src/components/Layout.jsx` | Navigation layout |
| `学习计划.md` | Full 8-week study plan (Chinese) |

---

## Tech Stack

- **React 18** - UI framework
- **React Router 7** - Client-side routing
- **TensorFlow.js** (`@tensorflow/tfjs`) - Core ML framework
- **TensorFlow.js WebGPU backend** (`@tensorflow/tfjs-backend-webgpu`) - GPU acceleration
- **Transformers.js** (`@huggingface/transformers`) - Hugging Face models in JS
- **Vite** - Build tool/dev server

---

## Routing

Routes are defined in `src/App.jsx`:

| Path | Page |
|------|------|
| `/` | HomePage - WebGPU status, backend switch, navigation |
| `/linear-regression` | LinearRegressionPage - TF.js training demo |
| `/transformers-demo` | TransformersDemoPage - DistilBERT sentiment |

---

## TensorFlow Context

Use the `useTensorFlow` hook to access TensorFlow state and methods:

```jsx
import { useTensorFlow } from './context/TensorFlowContext';

function MyComponent() {
  const { backend, webgpuSupported, memory, setBackend, updateMemory } = useTensorFlow();
  // ...
}
```

---

## Backend Priority

The code falls back automatically:

```
webgpu → webgl → wasm → cpu
```

WebGPU requires:
- Chrome 113+ 
- `chrome://flags` → Enable "Unsafe WebGPU Support"

---

## Model Loading

Models download at runtime from HuggingFace. No local model files are committed.

DistilBERT example uses `{ dtype: 'q8' }` (INT8 quantization) to reduce memory footprint.

---

## Memory Constraints

**Target**: Keep VRAM usage under 2GB

- Models: Use quantized versions (q8, q4)
- Monitor with `tf.memory()` - available in all demo pages
- Dispose tensors manually: `tensor.dispose()` or use `tf.tidy()`

---

## Common Patterns

```javascript
// Set backend before using TF.js
import '@tensorflow/tfjs-backend-webgpu';
await tf.setBackend('webgpu');
await tf.ready();

// Load Transformers.js pipeline
import { pipeline } from '@huggingface/transformers';
const classifier = await pipeline('text-classification', 'model-name', { dtype: 'q8' });
```

---

## What NOT to Do

- Do not attempt to load unquantized 7B+ models
- Do not commit `node_modules/` (already in .gitignore)
- Do not commit model weights - they download at runtime
- Do not use Python/Colab workflows - this is JavaScript-only

---

## Related Documentation

- TensorFlow.js: https://www.tensorflow.org/js/tutorials
- Transformers.js: https://huggingface.co/docs/transformers.js
- WebGPU API: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
