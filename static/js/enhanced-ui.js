/**
 * Enhanced UI Functionality
 * Consolidated script for:
 * 1. Animated background particles
 * 2. Interactive UI elements (password toggle, etc.)
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all UI enhancements
    initializeParticles();
    initializePasswordToggles();
});

/**
 * Initialize animated background particles
 */
function initializeParticles() {
    // Create particles container if it doesn't exist
    if (!document.querySelector('.particles-container')) {
        const container = document.createElement('div');
        container.className = 'particles-container';
        document.body.appendChild(container);
        
        // Create particles
        createParticles();
        
        // Reposition particles on window resize
        window.addEventListener('resize', function() {
            createParticles();
        });
    }
}

/**
 * Creates floating particles in the background
 */
function createParticles() {
    const container = document.querySelector('.particles-container');
    if (!container) return;
    
    const particleCount = 30; // Number of particles to create
    
    // Clear any existing particles
    container.innerHTML = '';
    
    for (let i = 0; i < particleCount; i++) {
        // Create particle element
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        // Random size between 2px and 8px
        const size = Math.random() * 6 + 2;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        // Random horizontal position
        const horizontalPosition = Math.random() * 100;
        particle.style.left = `${horizontalPosition}%`;
        
        // Random vertical position below the viewport
        const verticalOffset = Math.random() * 50;
        particle.style.bottom = `-${verticalOffset}px`;
        
        // Random drift amount for horizontal movement during animation
        const drift = (Math.random() - 0.5) * 200; // -100px to 100px
        particle.style.setProperty('--particle-drift', `${drift}px`);
        
        // Random opacity
        const opacity = Math.random() * 0.5 + 0.1; // 0.1 to 0.6
        particle.style.setProperty('--particle-opacity', opacity);
        
        // Random animation duration between 8s and 20s
        const duration = Math.random() * 12 + 8;
        particle.style.animation = `float-up ${duration}s infinite`;
        
        // Random animation delay
        const delay = Math.random() * 15;
        particle.style.animationDelay = `${delay}s`;
        
        // Add particle to container
        container.appendChild(particle);
    }
}

/**
 * Initialize password toggle functionality
 */
function initializePasswordToggles() {
    // Password toggle functionality
    const passwordToggles = document.querySelectorAll('.password-toggle');
    
    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', function() {
            const input = this.previousElementSibling;
            const icon = this.querySelector('i');
            
            if (input && input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
                this.classList.add('active');
            } else if (input) {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
                this.classList.remove('active');
            }
        });
    });
}