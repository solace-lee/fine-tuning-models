import React from 'react';
import { Link } from 'react-router-dom';

export default function Layout({ children }) {
  return (
    <div className="container">
      <header style={{ marginBottom: '2rem', paddingBottom: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
        <h1>Week2: LoRA 微调实战</h1>
        <nav style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
          <Link to="/">首页</Link>
          <Link to="/lora-theory">LoRA 原理</Link>
          <Link to="/fine-tuning">微调实验</Link>
        </nav>
      </header>
      <main>{children}</main>
    </div>
  );
}