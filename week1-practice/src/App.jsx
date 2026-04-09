import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { TensorFlowProvider } from './context/TensorFlowContext';
import HomePage from './pages/HomePage';
import LinearRegressionPage from './pages/LinearRegressionPage';
import TransformersDemoPage from './pages/TransformersDemoPage';
import Layout from './components/Layout';

export default function App() {
  return (
    <TensorFlowProvider>
      <Layout>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/linear-regression" element={<LinearRegressionPage />} />
          <Route path="/transformers-demo" element={<TransformersDemoPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Layout>
    </TensorFlowProvider>
  );
}
