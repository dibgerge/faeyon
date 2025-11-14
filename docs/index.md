<div class="homepage-hero">
  <div class="particles-container"></div>
  
  <div class="homepage-content">
    <h1>✨ Faeyon</h1>
    <p class="subtitle">A PyTorch library for building and training neural networks with a focus on flexibility, expressiveness, and ease of use.</p>
    
    <div class="cta-buttons">
      <a href="getting-started/installation.html" class="cta-button">🚀 Get Started</a>
      <a href="getting-started/quickstart.html" class="cta-button secondary">📖 Quick Start</a>
    </div>
    
    <div class="features-grid">
      <div class="feature-card">
        <h3>✨ Magic System</h3>
        <p>Powerful abstractions for delayed evaluation and lazy operations. Create elegant, expressive code with the <code>X</code> placeholder and operator overloading.</p>
      </div>
      
      <div class="feature-card">
        <h3>💾 Model I/O</h3>
        <p>Easy saving and loading of models with YAML configuration. Seamlessly serialize and deserialize your PyTorch models.</p>
      </div>
      
      <div class="feature-card">
        <h3>🎯 Training</h3>
        <p>Comprehensive training utilities with distributed support. Train models efficiently with built-in recipes and callbacks.</p>
      </div>
      
      <div class="feature-card">
        <h3>📊 Metrics</h3>
        <p>Flexible metric system for tracking training progress. Monitor your model's performance with ease.</p>
      </div>
      
      <div class="feature-card">
        <h3>🧠 Neural Networks</h3>
        <p>Extended PyTorch modules with additional functionality. Build complex architectures with enhanced building blocks.</p>
      </div>
      
      <div class="feature-card">
        <h3>⚡ Performance</h3>
        <p>Optimized for speed and efficiency. Leverage distributed training and advanced optimization techniques.</p>
      </div>
    </div>
    
    <h2>Quick Example</h2>
    
    ```python
    import torch
    from faeyon import X
    from torch import nn
    
    # Create a model with lazy operations
    model = nn.Linear(10, 5) >> nn.ReLU() >> nn.Linear(5, 1)
    
    # Use X for delayed operations
    x = torch.randn(32, 10)
    result = x >> model(X)
    ```
    
    <h2>Installation</h2>
    
    ```bash
    pip install faeyon
    ```
    
    <h2>Getting Started</h2>
    
    <div class="cta-buttons">
      <a href="getting-started/installation.html" class="cta-button">📦 Installation Guide</a>
      <a href="getting-started/quickstart.html" class="cta-button secondary">🚀 Quick Start Tutorial</a>
    </div>
    
    <h2>Documentation</h2>
    
    <p>Browse the documentation to learn more about:</p>
    
    <ul>
      <li><a href="user-guide/magic/overview.html"><strong>Magic System</strong></a> - Delayed evaluation and lazy operations</li>
      <li><a href="user-guide/models/overview.html"><strong>Models</strong></a> - Model creation and I/O</li>
      <li><a href="user-guide/training/overview.html"><strong>Training</strong></a> - Training utilities and recipes</li>
      <li><a href="user-guide/metrics/overview.html"><strong>Metrics</strong></a> - Metric tracking</li>
      <li><a href="user-guide/nn/overview.html"><strong>Neural Networks</strong></a> - Extended PyTorch modules</li>
    </ul>
    
    <hr>
    
    <p style="text-align: center; color: #718096;">
      See <a href="../LICENSE">LICENSE</a> for details.
    </p>
  </div>
</div>
