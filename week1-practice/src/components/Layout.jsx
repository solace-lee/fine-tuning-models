import React from 'react';
import { Link, useLocation } from 'react-router-dom';

export default function Layout({ children }) {
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  return (
    <div>
      <header>
        <h1>Week1: JavaScript 深度学习工具链入门</h1>
        <p>GTX1050 2G 显存环境下的 TensorFlow.js 实战</p>
        <nav style={{ marginTop: '1rem' }}>
          <Link 
            to="/" 
            style={{
              marginRight: '1rem',
              color: isActive('/') ? 'var(--accent)' : 'var(--text-secondary)',
              textDecoration: 'none',
            }}
          >
            首页
          </Link>
          <Link 
            to="/linear-regression" 
            style={{
              marginRight: '1rem',
              color: isActive('/linear-regression') ? 'var(--accent)' : 'var(--text-secondary)',
              textDecoration: 'none',
            }}
          >
            线性回归
          </Link>
          <Link 
            to="/transformers-demo" 
            style={{
              color: isActive('/transformers-demo') ? 'var(--accent)' : 'var(--text-secondary)',
              textDecoration: 'none',
            }}
          >
            DistilBERT
          </Link>
        </nav>
      </header>
      <main className="container">
        {children}
      </main>
    </div>
  );
}
