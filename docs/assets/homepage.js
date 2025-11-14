// Create animated particles for homepage background
(function() {
  function createParticles() {
    const container = document.querySelector('.particles-container');
    if (!container) return;
    
    const particleCount = 80;
    const particles = [];
    
    for (let i = 0; i < particleCount; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      
      // Random size between 2-8px
      const size = Math.random() * 6 + 2;
      particle.style.width = size + 'px';
      particle.style.height = size + 'px';
      
      // Random starting position
      particle.style.left = Math.random() * 100 + '%';
      
      // Random animation duration (10-20 seconds)
      const duration = Math.random() * 10 + 10;
      particle.style.animationDuration = duration + 's';
      
      // Random delay
      particle.style.animationDelay = Math.random() * 5 + 's';
      
      // Random horizontal movement (more variation)
      const horizontalMove = (Math.random() - 0.5) * 300;
      particle.style.setProperty('--horizontal-move', horizontalMove + 'px');
      
      // Add some particles with different colors
      const colors = [
        'rgba(255, 255, 255, 0.3)',
        'rgba(167, 139, 250, 0.4)',
        'rgba(236, 72, 153, 0.3)',
        'rgba(96, 165, 250, 0.3)',
        'rgba(251, 191, 36, 0.3)'
      ];
      particle.style.background = colors[Math.floor(Math.random() * colors.length)];
      
      container.appendChild(particle);
      particles.push(particle);
    }
    
    // Clean up particles that have finished animating
    setInterval(() => {
      particles.forEach(particle => {
        const rect = particle.getBoundingClientRect();
        if (rect.top + rect.height < 0) {
          // Reset particle to bottom
          particle.style.left = Math.random() * 100 + '%';
          particle.style.animationDelay = '0s';
        }
      });
    }, 1000);
  }
  
  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createParticles);
  } else {
    createParticles();
  }
})();

